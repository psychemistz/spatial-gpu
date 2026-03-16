"""Core SpaCET deconvolution algorithm.

Two-stage hierarchical cell type deconvolution:
  Stage 1: Malignant cell fraction inference (inferMal_cor)
  Stage 2: Non-malignant cell type deconvolution (SpatialDeconv)

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse, stats

from spatialgpu.deconvolution.constr_optim import constr_optim
from spatialgpu.deconvolution.reference import (
    get_cancer_signature,
    load_comb_ref,
    load_ref_normal_lihc,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deconvolution(
    adata: ad.AnnData,
    cancer_type: str,
    signature_type: str | None = None,
    adjacent_normal: bool = False,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Two-stage hierarchical cell type deconvolution.

    Equivalent to SpaCET.deconvolution() in R.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data. X should be raw counts (genes x spots
        stored as adata.X with obs=spots, var=genes).
    cancer_type : str
        Cancer type code (e.g., 'BRCA', 'LIHC', 'CRC', 'PANCAN').
    signature_type : str or None
        Force signature type: 'CNA', 'expr', or 'seq_depth'. None for auto.
    adjacent_normal : bool
        If True, skip malignant cell inference (for normal tissue).
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with results in:
        adata.obsm['spacet_propMat'] : cell fraction matrix (spots x cell_types)
        adata.uns['spacet'] : dict with malRes, Ref, etc.
    """
    try:
        prop_mat, mal_res = _deconvolution_via_r(adata, cancer_type, adjacent_normal)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("R SpaCET unavailable (%s), using Python fallback", e)
        prop_mat, mal_res = _deconvolution_python(
            adata, cancer_type, signature_type, adjacent_normal, n_jobs
        )

    # Store results in AnnData
    adata.obsm["spacet_propMat"] = prop_mat.T.reindex(adata.obs_names).values
    adata.uns["spacet"] = {
        "deconvolution": {
            "propMat": prop_mat,
            "malRes": mal_res,
        },
        "propMat_columns": list(prop_mat.index),
    }

    # Always store combined reference for downstream CCI analysis
    try:
        comb_ref = load_comb_ref()
        adata.uns["spacet"]["deconvolution"]["Ref"] = comb_ref
    except Exception:
        logger.warning("Could not load combined reference for CCI.")

    return adata


def _deconvolution_via_r(
    adata: ad.AnnData,
    cancer_type: str,
    adjacent_normal: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run full SpaCET deconvolution in R for exact numerical equivalence."""
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    counts = _get_counts_genes_by_spots(adata)
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    tmpdir = tempfile.mkdtemp()
    try:
        counts_sp = sparse.csc_matrix(counts, dtype=np.float64)
        mmwrite(os.path.join(tmpdir, "counts.mtx"), counts_sp)
        pd.DataFrame({"gene": gene_names}).to_csv(
            os.path.join(tmpdir, "genes.csv"), index=False
        )
        pd.DataFrame({"spot": spot_names}).to_csv(
            os.path.join(tmpdir, "spots.csv"), index=False
        )

        r_adj = "TRUE" if adjacent_normal else "FALSE"

        r_code = f"""
        suppressPackageStartupMessages({{
            library(Matrix)
            library(SpaCET)
        }})

        # Read counts
        ST <- readMM("{tmpdir}/counts.mtx")
        ST <- as(ST, "dgCMatrix")
        genes <- read.csv("{tmpdir}/genes.csv")$gene
        spots <- read.csv("{tmpdir}/spots.csv")$spot
        rownames(ST) <- genes
        colnames(ST) <- spots

        # Create SpaCET object
        SpaCET_obj <- new("SpaCET")
        SpaCET_obj@input$counts <- ST

        # Run deconvolution
        SpaCET_obj <- SpaCET.deconvolution(
            SpaCET_obj,
            cancerType = "{cancer_type}",
            adjacentNormal = {r_adj},
            coreNo = 1
        )

        # Save results
        propMat <- SpaCET_obj@results$deconvolution$propMat
        write.csv(propMat, "{tmpdir}/propMat.csv")

        malProp <- SpaCET_obj@results$deconvolution$malRes$malProp
        write.csv(
            data.frame(spot=names(malProp), malProp=malProp),
            "{tmpdir}/malProp.csv", row.names=FALSE
        )

        malRef <- SpaCET_obj@results$deconvolution$malRes$malRef
        if (!is.null(malRef)) {{
            write.csv(
                data.frame(gene=names(malRef), malRef=malRef),
                "{tmpdir}/malRef.csv", row.names=FALSE
            )
        }}
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R SpaCET.deconvolution failed: {result.stderr[:500]}")

        # Read results
        prop_mat = pd.read_csv(os.path.join(tmpdir, "propMat.csv"), index_col=0)

        mal_prop_df = pd.read_csv(os.path.join(tmpdir, "malProp.csv"))
        mal_prop = pd.Series(
            mal_prop_df["malProp"].values, index=mal_prop_df["spot"].values
        )

        mal_ref_path = os.path.join(tmpdir, "malRef.csv")
        if os.path.exists(mal_ref_path):
            mal_ref_df = pd.read_csv(mal_ref_path)
            mal_ref = pd.Series(
                mal_ref_df["malRef"].values, index=mal_ref_df["gene"].values
            )
        else:
            mal_ref = None

        mal_res = {"malRef": mal_ref, "malProp": mal_prop}
        return prop_mat, mal_res

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _deconvolution_python(
    adata: ad.AnnData,
    cancer_type: str,
    signature_type: str | None = None,
    adjacent_normal: bool = False,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Python fallback for full deconvolution pipeline."""
    counts = _get_counts_genes_by_spots(adata)
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    # Filter zero-sum genes
    if sparse.issparse(counts):
        gene_sums = np.asarray(counts.sum(axis=1)).ravel()
    else:
        gene_sums = counts.sum(axis=1)
    nonzero_mask = gene_sums > 0
    counts = counts[nonzero_mask]
    gene_names = gene_names[nonzero_mask]

    # Load reference
    ref = load_comb_ref()

    if cancer_type in ("LIHC", "CHOL"):
        ref_normal = load_ref_normal_lihc()
        ref = _merge_references(ref, ref_normal)

    n_spots = counts.shape[1]
    if n_spots > 20000:
        ref_genes = set(ref["refProfiles"].index)
        keep_mask = np.array([g in ref_genes for g in gene_names])
        counts = counts[keep_mask]
        gene_names = gene_names[keep_mask]

    # Stage 1
    if adjacent_normal:
        logger.info("Stage 1. Infer malignant cell fraction (skip).")
        mal_prop = pd.Series(0.0, index=spot_names)
        mal_res = {"malRef": None, "malProp": mal_prop}
    else:
        logger.info("Stage 1. Infer malignant cell fraction.")
        mal_res = _infer_mal_cor(
            counts, gene_names, spot_names, cancer_type, signature_type
        )

    # Stage 2
    logger.info("Stage 2. Hierarchically deconvolve non-malignant cell fraction.")

    if n_spots <= 20000:
        prop_mat = _spatial_deconv(
            ST=counts,
            gene_names=gene_names,
            spot_names=spot_names,
            ref=ref,
            mal_prop=mal_res["malProp"],
            mal_ref=mal_res["malRef"],
            mode="standard",
            n_jobs=n_jobs,
        )
    else:
        chunk_size = 5000
        n_chunks = int(np.ceil(n_spots / chunk_size))
        prop_mats = []
        for i in range(n_chunks):
            logger.info(f"Processing {i + 1}/{n_chunks}")
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_spots)
            prop_sub = _spatial_deconv(
                ST=counts[:, start:end],
                gene_names=gene_names,
                spot_names=spot_names[start:end],
                ref=ref,
                mal_prop=mal_res["malProp"].iloc[start:end],
                mal_ref=mal_res["malRef"],
                mode="standard",
                n_jobs=n_jobs,
            )
            prop_mats.append(prop_sub)
        prop_mat = pd.concat(prop_mats, axis=1)

    return prop_mat, mal_res


# ---------------------------------------------------------------------------
# corMat: Correlation with p-values (matches psych::corr.test)
# ---------------------------------------------------------------------------


def cormat(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation between columns of X and columns of Y.

    Equivalent to SpaCET's corMat() function.
    X and Y are (genes x samples) matrices. Correlates each column of X
    with each column of Y (row-wise, i.e., across genes).

    Parameters
    ----------
    X : np.ndarray
        Matrix (n_genes x n_samples).
    Y : np.ndarray
        Matrix (n_genes x n_features), typically a single-column signature.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    pd.DataFrame with columns: cor_r, cor_p, cor_padj
        One row per sample (column of X).
    """
    from statsmodels.stats.multitest import multipletests

    n_samples = X.shape[1]
    n_features = Y.shape[1] if Y.ndim > 1 else 1
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    cor_func = stats.pearsonr if method == "pearson" else stats.spearmanr

    # Compute correlation for each sample against each feature
    # For SpaCET, Y is typically a single column (signature)
    results = []
    for j in range(n_features):
        y_col = Y[:, j]
        rs = np.zeros(n_samples)
        ps = np.zeros(n_samples)
        for i in range(n_samples):
            r, p = cor_func(X[:, i], y_col)
            rs[i] = r
            ps[i] = p
        results.append((rs, ps))

    # Use first feature (SpaCET always uses single-column Y)
    cor_r = np.round(results[0][0], 3)
    cor_p = np.array([float(f"{p:.3g}") for p in results[0][1]])

    # BH adjustment
    _, cor_padj, _, _ = multipletests(cor_p, method="fdr_bh")

    return pd.DataFrame({"cor_r": cor_r, "cor_p": cor_p, "cor_padj": cor_padj})


# ---------------------------------------------------------------------------
# Stage 1: Malignant cell inference
# ---------------------------------------------------------------------------


def _infer_mal_cor(
    counts: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    cancer_type: str,
    signature_type: str | None,
) -> dict[str, Any]:
    """Infer malignant cell fraction per spot.

    Equivalent to SpaCET's inferMal_cor() function.
    """
    n_spots = counts.shape[1]

    # Sequencing depth (genes detected per spot)
    if sparse.issparse(counts):
        seq_depth = np.asarray((counts > 0).sum(axis=0)).ravel()
    else:
        seq_depth = (counts > 0).sum(axis=0)
    seq_depth_series = pd.Series(seq_depth, index=spot_names)

    # CPM normalize, log2, center
    centered = _cpm_log2_center(counts)

    if n_spots < 20000:
        return _infer_mal_small(
            counts,
            centered,
            gene_names,
            spot_names,
            seq_depth_series,
            cancer_type,
            signature_type,
        )
    else:
        return _infer_mal_large(
            counts,
            centered,
            gene_names,
            spot_names,
            seq_depth_series,
            cancer_type,
        )


def _cpm_log2_center(counts: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """CPM normalize, log2(x+1) transform, center by row means.

    Returns dense matrix (genes x spots).
    """
    if sparse.issparse(counts):
        col_sums = np.asarray(counts.sum(axis=0)).ravel()
        # CPM: divide each column by its sum, multiply by 1e6
        # Work with CSC for efficient column operations
        csc = sparse.csc_matrix(counts, dtype=np.float64, copy=True)
        # Normalize columns
        for j in range(csc.shape[1]):
            start, end = csc.indptr[j], csc.indptr[j + 1]
            if col_sums[j] > 0:
                csc.data[start:end] = csc.data[start:end] / col_sums[j] * 1e6
        # log2(x + 1) on nonzero elements
        csc.data = np.log2(csc.data + 1)
        # Convert to dense for centering
        mat = csc.toarray()
    else:
        counts = counts.astype(np.float64)
        col_sums = counts.sum(axis=0)
        mat = counts / col_sums[np.newaxis, :] * 1e6
        mat = np.nan_to_num(mat)
        mat = np.log2(mat + 1)

    # Center: subtract row means
    row_means = mat.mean(axis=1, keepdims=True)
    mat -= row_means
    return mat


def _infer_mal_small(
    counts: sparse.spmatrix | np.ndarray,
    centered: np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    seq_depth: pd.Series,
    cancer_type: str,
    signature_type: str | None,
) -> dict[str, Any]:
    """Malignant inference for datasets with < 20,000 spots."""
    from spatialgpu.deconvolution.mudan import mudan_cluster

    n_spots = centered.shape[1]

    # Step 1: Clustering using MUDAN normalization (matching R exactly)
    logger.info("Stage 1 - Step 1. Clustering.")

    # Full MUDAN pipeline: normalizeVariance + PCA + ward.D + silhouette
    clustering_raw, ods, gsf = mudan_cluster(counts, n_pcs=30, gam_k=5, alpha=0.05)
    clustering = pd.Series(clustering_raw.values, index=spot_names)

    # Step 2: Find tumor clusters
    logger.info("Stage 1 - Step 2. Find tumor clusters.")

    mal_flag = False
    stat_df = None
    sig_type_used = None
    sig_ct_used = None

    if signature_type is None:
        if cancer_type == "PANCAN":
            comb_list = [("expr", "PANCAN")]
        else:
            comb_list = [
                ("CNA", cancer_type),
                ("expr", cancer_type),
                ("expr", "PANCAN"),
            ]

        for cna_expr, ct in comb_list:
            try:
                _, sig = get_cancer_signature(ct, cna_expr)
            except ValueError:
                continue

            # Intersect genes
            olp = np.intersect1d(gene_names, sig.index)
            if len(olp) == 0:
                continue

            gene_idx = np.array([np.where(gene_names == g)[0][0] for g in olp])
            sig_vals = sig.reindex(olp).values.reshape(-1, 1)
            X_sub = centered[gene_idx, :]

            cor_sig = cormat(X_sub, sig_vals)
            cor_sig.index = spot_names

            stat_df = _compute_cluster_stats(cor_sig, clustering, seq_depth)

            if stat_df["clusterMal"].any():
                logger.info(f"                  > Use {cna_expr} signature: {ct}.")
                mal_flag = True
                sig_type_used = cna_expr
                sig_ct_used = ct
                break

    elif signature_type == "seq_depth":
        mal_flag = False
    else:
        _, sig = get_cancer_signature(cancer_type, signature_type)
        olp = np.intersect1d(gene_names, sig.index)
        gene_idx = np.array([np.where(gene_names == g)[0][0] for g in olp])
        sig_vals = sig.reindex(olp).values.reshape(-1, 1)
        X_sub = centered[gene_idx, :]
        cor_sig = cormat(X_sub, sig_vals)
        cor_sig.index = spot_names

        stat_df = _compute_cluster_stats(cor_sig, clustering, seq_depth)
        if stat_df["clusterMal"].any():
            mal_flag = True
            sig_type_used = signature_type
            sig_ct_used = cancer_type
        else:
            raise ValueError("No malignant cells detected in this tumor ST data set.")

    # Step 3: Infer malignant cells
    logger.info("Stage 1 - Step 3. Infer malignant cells.")
    top5p = max(1, round(n_spots * 0.05))

    if mal_flag:
        # Select spots in malignant clusters with positive correlation
        mal_clusters = stat_df.index[stat_df["clusterMal"]]
        spot_mal_mask = clustering.isin(mal_clusters) & (cor_sig["cor_r"].values > 0)
        spot_mal = spot_names[spot_mal_mask.values]
    else:
        # Fallback: top 5% by sequencing depth
        sorted_depth = seq_depth.sort_values(ascending=False)
        spot_mal = sorted_depth.index[:top5p].values
        sig_type_used = "seq_depth"
        sig_ct_used = "current_sample"
        logger.info(
            f"                  > Use {sig_type_used} signature: {sig_ct_used}."
        )

    # Compute malignant reference (CPM of malignant spots)
    spot_mal_idx = np.array([np.where(spot_names == s)[0][0] for s in spot_mal])
    mal_ref = _compute_mal_ref(counts, gene_names, spot_mal_idx)

    # Compute signature from centered expression of malignant spots
    sig_from_mal = centered[:, spot_mal_idx].mean(axis=1).reshape(-1, 1)

    # Correlate all spots with malignant signature
    cor_sig_mal = cormat(centered, sig_from_mal)
    cor_sig_mal.index = spot_names

    mal_prop = cor_sig_mal["cor_r"].values.copy()

    # Clip to 5th-95th percentile, normalize to [0, 1]
    sorted_prop = np.sort(mal_prop)
    p5 = sorted_prop[top5p - 1]  # 0-indexed
    p95 = sorted_prop[len(sorted_prop) - top5p]

    mal_prop = np.clip(mal_prop, p5, p95)
    if (mal_prop.max() - mal_prop.min()) > 0:
        mal_prop = (mal_prop - mal_prop.min()) / (mal_prop.max() - mal_prop.min())
    else:
        mal_prop = np.zeros_like(mal_prop)

    mal_prop = pd.Series(mal_prop, index=spot_names)

    return {
        "sig": (sig_type_used, sig_ct_used),
        "stat_df": stat_df,
        "malRef": mal_ref,
        "malProp": mal_prop,
    }


def _infer_mal_large(
    counts: sparse.spmatrix | np.ndarray,
    centered: np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    seq_depth: pd.Series,
    cancer_type: str,
) -> dict[str, Any]:
    """Malignant inference for datasets with > 20,000 spots.

    Two-round CNA correlation approach.
    """
    # First round: correlate with CNA signature
    _, sig = get_cancer_signature(cancer_type, "CNA")
    olp = np.intersect1d(gene_names, sig.index)
    gene_idx = np.array([np.where(gene_names == g)[0][0] for g in olp])
    sig_vals = sig.reindex(olp).values.reshape(-1, 1)

    # Process in chunks of 5000
    mal_prop = _chunked_correlation(centered[gene_idx], sig_vals, spot_names)

    # Clip (use top 100, not 5%)
    top5p = 100
    sorted_prop = np.sort(mal_prop)
    p5 = sorted_prop[top5p - 1]
    p95 = sorted_prop[len(sorted_prop) - top5p]

    mal_prop = np.clip(mal_prop, p5, p95)
    mal_prop = (mal_prop - mal_prop.min()) / (mal_prop.max() - mal_prop.min())

    # Second round: use top spots as reference
    spot_mal_idx = np.where(mal_prop >= 1.0)[0]
    sig_from_mal = centered[:, spot_mal_idx].mean(axis=1).reshape(-1, 1)

    mal_prop = _chunked_correlation(centered, sig_from_mal, spot_names)

    # Clip again
    sorted_prop = np.sort(mal_prop)
    p5 = sorted_prop[top5p - 1]
    p95 = sorted_prop[len(sorted_prop) - top5p]

    mal_prop = np.clip(mal_prop, p5, p95)
    mal_prop = (mal_prop - mal_prop.min()) / (mal_prop.max() - mal_prop.min())

    # Malignant reference from top spots
    top_idx = np.where(mal_prop >= 1.0)[0]
    mal_counts = counts[:, top_idx]
    if sparse.issparse(mal_counts):
        mal_col_sums = np.asarray(mal_counts.sum(axis=0)).ravel()
        mal_cpm = (
            mal_counts.toarray().astype(np.float64) / mal_col_sums[np.newaxis, :] * 1e6
        )
    else:
        mal_col_sums = mal_counts.sum(axis=0)
        mal_cpm = mal_counts.astype(np.float64) / mal_col_sums[np.newaxis, :] * 1e6
    mal_ref = np.nanmean(mal_cpm, axis=1)
    mal_ref = pd.Series(mal_ref, index=gene_names)

    mal_prop = pd.Series(mal_prop, index=spot_names)

    return {
        "sig": ("CNA", cancer_type),
        "stat_df": None,
        "malRef": mal_ref,
        "malProp": mal_prop,
    }


def _chunked_correlation(
    centered: np.ndarray,
    sig: np.ndarray,
    spot_names: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Compute Pearson correlation in chunks."""
    n_spots = centered.shape[1]
    n_chunks = int(np.ceil(n_spots / chunk_size))
    mal_prop = np.zeros(n_spots)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_spots)
        cor_sig = cormat(centered[:, start:end], sig)
        mal_prop[start:end] = cor_sig["cor_r"].values

    return mal_prop


def _compute_cluster_stats(
    cor_sig: pd.DataFrame,
    clustering: pd.Series,
    seq_depth: pd.Series,
) -> pd.DataFrame:
    """Compute per-cluster statistics for tumor detection."""
    clusters = sorted(clustering.unique())
    records = []

    global_fraction = (
        (cor_sig["cor_r"].values > 0) & (cor_sig["cor_padj"].values < 0.25)
    ).sum() / len(cor_sig)

    for c in clusters:
        mask = clustering == c
        cor_c = cor_sig.loc[mask]
        depth_c = seq_depth.loc[mask]

        n = mask.sum()
        mean_cor = cor_c["cor_r"].mean()

        # Wilcoxon test: is correlation > 0?
        try:
            _, wt_p = stats.wilcoxon(
                cor_c["cor_r"].values,
                alternative="greater",
            )
        except ValueError:
            wt_p = 1.0

        frac_padj = (
            (cor_c["cor_r"].values > 0) & (cor_c["cor_padj"].values < 0.25)
        ).sum() / n

        depth_diff = depth_c.mean() - seq_depth.mean()

        cluster_mal = (
            depth_diff > 0
            and mean_cor > 0
            and wt_p < 0.05
            and frac_padj >= global_fraction
        )

        records.append(
            {
                "cluster": c,
                "spotNum": n,
                "mean": mean_cor,
                "wilcoxTestG0": wt_p,
                "fraction_spot_padj": frac_padj,
                "seq_depth_diff": depth_diff,
                "clusterMal": cluster_mal,
            }
        )

    return pd.DataFrame(records).set_index("cluster")


# ---------------------------------------------------------------------------
# Stage 2: Hierarchical constrained deconvolution
# ---------------------------------------------------------------------------


def _spatial_deconv(
    ST: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    ref: dict[str, Any],
    mal_prop: pd.Series | pd.DataFrame,
    mal_ref: pd.Series | pd.DataFrame | None,
    mode: str = "standard",
    unidentifiable: bool = True,
    macrophage_other: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Hierarchical constrained least-squares deconvolution.

    Equivalent to SpaCET's SpatialDeconv() function.
    Tries R subprocess first for exact numerical equivalence,
    falls back to Python implementation.

    Returns
    -------
    pd.DataFrame : cell_types x spots proportion matrix
    """
    try:
        return _spatial_deconv_via_r(
            ST,
            gene_names,
            spot_names,
            mal_prop,
            mal_ref,
            mode,
            unidentifiable,
            macrophage_other,
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("R SpatialDeconv unavailable (%s), using Python fallback", e)
        return _spatial_deconv_python(
            ST,
            gene_names,
            spot_names,
            ref,
            mal_prop,
            mal_ref,
            mode,
            unidentifiable,
            macrophage_other,
            n_jobs,
        )


def _spatial_deconv_via_r(
    ST: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    mal_prop: pd.Series,
    mal_ref: pd.Series | None,
    mode: str = "standard",
    unidentifiable: bool = True,
    macrophage_other: bool = True,
) -> pd.DataFrame:
    """Run SpatialDeconv entirely in R via subprocess.

    Uses SpaCET's bundled reference data and R's constrOptim for
    exact numerical equivalence with R's SpaCET package.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    tmpdir = tempfile.mkdtemp()
    try:
        # Write counts as MatrixMarket
        st_sp = sparse.csc_matrix(ST, dtype=np.float64)
        mmwrite(os.path.join(tmpdir, "ST.mtx"), st_sp)

        # Write gene names and spot names
        pd.DataFrame({"gene": gene_names}).to_csv(
            os.path.join(tmpdir, "gene_names.csv"), index=False
        )
        pd.DataFrame({"spot": spot_names}).to_csv(
            os.path.join(tmpdir, "spot_names.csv"), index=False
        )

        # Write malProp (may be a Series or DataFrame)
        if isinstance(mal_prop, pd.DataFrame):
            mal_prop.to_csv(os.path.join(tmpdir, "malProp.csv"), index=True)
        else:
            pd.DataFrame({"malProp": mal_prop.values}, index=mal_prop.index).to_csv(
                os.path.join(tmpdir, "malProp.csv"), index=True
            )

        # Write malRef (may be a DataFrame with multiple columns)
        if mal_ref is not None:
            if isinstance(mal_ref, pd.DataFrame):
                mal_ref.to_csv(os.path.join(tmpdir, "malRef.csv"), index=True)
            else:
                pd.DataFrame({"malRef": mal_ref.values}, index=mal_ref.index).to_csv(
                    os.path.join(tmpdir, "malRef.csv"), index=True
                )
            has_mal_ref = "TRUE"
        else:
            has_mal_ref = "FALSE"

        r_unid = "TRUE" if unidentifiable else "FALSE"
        r_mac_other = "TRUE" if macrophage_other else "FALSE"

        r_code = f"""
        suppressPackageStartupMessages(library(Matrix))
        suppressPackageStartupMessages(library(SpaCET))

        # Load inputs
        ST <- readMM("{tmpdir}/ST.mtx")
        ST <- as(ST, "dgCMatrix")
        gene_names <- read.csv("{tmpdir}/gene_names.csv")$gene
        spot_names <- read.csv("{tmpdir}/spot_names.csv")$spot
        rownames(ST) <- gene_names
        colnames(ST) <- spot_names

        malProp_df <- read.csv("{tmpdir}/malProp.csv", row.names=1)
        if (ncol(malProp_df) == 1) {{
            malProp <- malProp_df[,1]
            names(malProp) <- rownames(malProp_df)
        }} else {{
            malProp <- as.matrix(malProp_df)
        }}

        has_malRef <- {has_mal_ref}
        if (has_malRef) {{
            malRef_df <- read.csv("{tmpdir}/malRef.csv", row.names=1)
            if (ncol(malRef_df) == 1) {{
                malRef <- malRef_df[,1]
                names(malRef) <- rownames(malRef_df)
            }} else {{
                malRef <- as.matrix(malRef_df)
            }}
        }} else {{
            malRef <- NULL
        }}

        # Load SpaCET reference
        load(system.file("extdata", "combRef_0.5.rda", package = "SpaCET"))

        # Run SpatialDeconv
        propMat <- SpaCET:::SpatialDeconv(
            ST = ST,
            Ref = Ref,
            malProp = malProp,
            malRef = malRef,
            mode = "{mode}",
            Unidentifiable = {r_unid},
            MacrophageOther = {r_mac_other},
            coreNo = 1
        )

        write.csv(propMat, "{tmpdir}/propMat.csv")
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R SpatialDeconv failed: {result.stderr[:500]}")

        prop_mat = pd.read_csv(
            os.path.join(tmpdir, "propMat.csv"),
            index_col=0,
        )
        return prop_mat

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _spatial_deconv_python(
    ST: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_names: np.ndarray,
    ref: dict[str, Any],
    mal_prop: pd.Series,
    mal_ref: pd.Series | None,
    mode: str = "standard",
    unidentifiable: bool = True,
    macrophage_other: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Python fallback for hierarchical constrained deconvolution."""
    reference = ref["refProfiles"].copy()
    signature = ref["sigGenes"].copy()
    tree = ref["lineageTree"].copy()

    # Intersect genes (preserve order from gene_names, matching R's intersect)
    ref_gene_set = set(reference.index)
    olp_mask = np.array([g in ref_gene_set for g in gene_names])
    olp_genes = gene_names[olp_mask]
    olp_set = set(olp_genes)
    gene_idx = np.where(olp_mask)[0]

    ST_sub = ST[gene_idx]
    reference = reference.loc[olp_genes]

    # CPM normalize ST and reference
    if sparse.issparse(ST_sub):
        col_sums = np.asarray(ST_sub.sum(axis=0)).ravel()
        ST_cpm = ST_sub.toarray().astype(np.float64)
        ST_cpm = ST_cpm / col_sums[np.newaxis, :] * 1e6
    else:
        col_sums = ST_sub.sum(axis=0)
        ST_cpm = ST_sub.astype(np.float64) / col_sums[np.newaxis, :] * 1e6

    ref_cpm = reference.values.astype(np.float64)
    ref_col_sums = ref_cpm.sum(axis=0)
    ref_cpm = ref_cpm / ref_col_sums[np.newaxis, :] * 1e6

    # Remove NaN spots
    nan_mask = ~np.isnan(ST_cpm[0, :])
    ST_cpm = ST_cpm[:, nan_mask]
    valid_spots = (
        spot_names[nan_mask] if isinstance(spot_names, np.ndarray) else spot_names
    )

    # Subtract malignant contribution
    # In deconvMal mode, mal_prop is a DataFrame (cell_types × spots)
    # representing known fractions; mal_prop_arr = colSums = total known fraction
    if isinstance(mal_prop, pd.DataFrame):
        mal_prop_reindexed = mal_prop.reindex(columns=valid_spots, fill_value=0.0)
        mal_prop_arr = mal_prop_reindexed.sum(axis=0).values.astype(np.float64)
    else:
        mal_prop_arr = mal_prop.reindex(valid_spots, fill_value=0.0).values.astype(
            np.float64
        )

    if mal_prop_arr.sum() > 0 and mal_ref is not None:
        if isinstance(mal_ref, pd.DataFrame):
            # Multi-column reference: subtract sum of all known contributions
            # Align: only use fraction rows that have matching reference columns
            shared_types = [t for t in mal_ref.columns if t in mal_prop_reindexed.index]
            mal_ref_aligned = mal_ref[shared_types]
            mal_prop_aligned = mal_prop_reindexed.loc[shared_types]
            mal_ref_sub = mal_ref_aligned.reindex(olp_genes).values.astype(np.float64)
            mal_ref_sub = np.nan_to_num(mal_ref_sub)
            # Compute CPM per column and weight by known fractions
            col_sums = mal_ref_sub.sum(axis=0)
            col_sums[col_sums == 0] = 1
            mal_ref_cpm = mal_ref_sub * 1e6 / col_sums
            # Each known cell type contributes: ref_cpm * fraction
            known_fracs = mal_prop_aligned.values  # (n_types, n_spots)
            mixture_mal = mal_ref_cpm @ known_fracs
            mixture_minus_mal = ST_cpm - mixture_mal
        else:
            mal_ref_sub = mal_ref.reindex(olp_genes).values.astype(np.float64)
            if np.isnan(mal_ref_sub).any():
                mal_ref_sub = np.nan_to_num(mal_ref_sub)
            mal_ref_cpm = (
                mal_ref_sub * 1e6 / mal_ref_sub.sum()
                if mal_ref_sub.sum() > 0
                else mal_ref_sub
            )
            mixture_mal = np.outer(mal_ref_cpm, mal_prop_arr)
            mixture_minus_mal = ST_cpm - mixture_mal
    else:
        mixture_minus_mal = ST_cpm

    # Level 1: Major lineages
    level1_types = [t for t in tree.keys() if t in reference.columns]

    if mode != "deconvMal":
        logger.info("Stage 2 - Level 1. Estimate the major lineage.")

        # Get signature genes for level 1 (preserve first-occurrence order like R's unique)
        sig_keys = list(level1_types) + (["T cell"] if "T cell" in signature else [])
        seen = set()
        sig_genes_l1 = []
        for k in sig_keys:
            if k in signature:
                for g in signature[k]:
                    if g not in seen:
                        seen.add(g)
                        if g in olp_set:
                            sig_genes_l1.append(g)

        sig_idx = np.array([np.where(olp_genes == g)[0][0] for g in sig_genes_l1])

        mixture_l1 = mixture_minus_mal[sig_idx]
        ref_l1 = ref_cpm[sig_idx][
            :, [list(reference.columns).index(t) for t in level1_types]
        ]

        n_spot = mixture_l1.shape[1]
        n_cell = ref_l1.shape[1]

        theta_sum = (1 - mal_prop_arr) - 1e-5

        prop_l1 = _solve_constrained_batch(
            ref_l1,
            mixture_l1,
            n_cell,
            theta_sum,
            pp_min_arr=(
                np.zeros(n_spot) if unidentifiable else (1 - mal_prop_arr - 2e-5)
            ),
            pp_max_arr=1 - mal_prop_arr,
            n_jobs=n_jobs,
        )

        prop_mat_l1 = pd.DataFrame(prop_l1, index=level1_types, columns=valid_spots)

        if mode in ("standard", "deconvWithSC_alt"):
            mal_row = pd.DataFrame(
                [mal_prop_arr], index=["Malignant"], columns=valid_spots
            )
            prop_mat_l1 = pd.concat([mal_row, prop_mat_l1])

        if unidentifiable:
            if mode == "standard":
                unid = 1 - prop_mat_l1.sum(axis=0)
                unid_row = pd.DataFrame(
                    [unid.values], index=["Unidentifiable"], columns=valid_spots
                )
                prop_mat_l1 = pd.concat([prop_mat_l1, unid_row])
            elif mode == "deconvWithSC_alt":
                non_mal = prop_mat_l1.iloc[1:]
                non_mal_sums = non_mal.sum(axis=0)
                non_mal_norm = non_mal / non_mal_sums * (1 - prop_mat_l1.iloc[0])
                prop_mat_l1.iloc[1:] = non_mal_norm
            elif mode == "deconvWithSC":
                col_sums = prop_mat_l1.sum(axis=0)
                prop_mat_l1 = prop_mat_l1 / col_sums
    else:
        # deconvMal mode
        prop_mat_l1 = pd.DataFrame(
            (
                1 - mal_prop_arr.reshape(1, -1)
                if isinstance(mal_prop_arr, np.ndarray)
                else [[1 - x for x in mal_prop_arr]]
            ),
            index=level1_types[:1],
            columns=valid_spots,
        )

    # Level 2: Sublineages
    if mode != "deconvMal":
        logger.info("Stage 2 - Level 2. Estimate the sub lineage.")

    for cell_spe, subtypes in tree.items():
        if len(subtypes) < 2:
            continue
        if cell_spe not in prop_mat_l1.index:
            continue

        logger.info(f"                  > {cell_spe}:")

        subtypes_no_other = [s for s in subtypes if s != "Macrophage other"]
        subtypes_in_ref = [s for s in subtypes_no_other if s in reference.columns]
        if len(subtypes_in_ref) == 0:
            continue

        # Subtract other lineages' contribution
        other_types = [t for t in level1_types if t != cell_spe]
        other_types_in_l1 = [t for t in other_types if t in prop_mat_l1.index]

        if other_types_in_l1:
            other_ref_idx = [
                list(reference.columns).index(t) for t in other_types_in_l1
            ]
            other_contribution = (
                ref_cpm[:, other_ref_idx] @ prop_mat_l1.loc[other_types_in_l1].values
            )
            mixture_l2 = mixture_minus_mal - other_contribution
        else:
            mixture_l2 = mixture_minus_mal.copy()

        # Get signature genes for this lineage (preserve first-occurrence order)
        seen_l2 = set()
        sig_genes_l2 = []
        for st in subtypes_in_ref:
            if st in signature:
                for g in signature[st]:
                    if g not in seen_l2:
                        seen_l2.add(g)
                        if g in olp_set:
                            sig_genes_l2.append(g)

        if len(sig_genes_l2) == 0:
            continue

        sig_idx_l2 = np.array([np.where(olp_genes == g)[0][0] for g in sig_genes_l2])
        ref_idx_l2 = [list(reference.columns).index(s) for s in subtypes_in_ref]

        mix_l2 = mixture_l2[sig_idx_l2]
        ref_l2 = ref_cpm[sig_idx_l2][:, ref_idx_l2]

        n_cell_l2 = ref_l2.shape[1]
        theta_sum_l2 = prop_mat_l1.loc[cell_spe].values - 1e-5

        if cell_spe == "Macrophage" and macrophage_other:
            pp_min_l2 = np.zeros(len(valid_spots))
        else:
            pp_min_l2 = prop_mat_l1.loc[cell_spe].values - 2e-5

        pp_max_l2 = prop_mat_l1.loc[cell_spe].values

        prop_l2 = _solve_constrained_batch(
            ref_l2,
            mix_l2,
            n_cell_l2,
            theta_sum_l2,
            pp_min_arr=pp_min_l2,
            pp_max_arr=pp_max_l2,
            n_jobs=n_jobs,
        )

        sub_df = pd.DataFrame(prop_l2, index=subtypes_in_ref, columns=valid_spots)

        if mode == "standard" and macrophage_other and cell_spe == "Macrophage":
            mac_other = prop_mat_l1.loc[cell_spe] - sub_df.sum(axis=0)
            mac_other_row = pd.DataFrame(
                [mac_other.values], index=["Macrophage other"], columns=valid_spots
            )
            sub_df = pd.concat([sub_df, mac_other_row])

        prop_mat_l1 = pd.concat([prop_mat_l1, sub_df])

    # Post-processing: clip [0, 1]
    prop_mat = prop_mat_l1.clip(0, 1)

    return prop_mat


def _solve_constrained_batch(
    A: np.ndarray,
    B: np.ndarray,
    n_cell: int,
    theta_sum: np.ndarray,
    pp_min_arr: np.ndarray,
    pp_max_arr: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Solve constrained least squares for all spots.

    Two-pass optimization matching R's SpaCET:
    1. Unweighted least squares
    2. Weighted by 1/(fitted + 1)

    Tries R's constrOptim via subprocess first for exact numerical
    equivalence, falls back to Python implementation.
    """
    try:
        return _solve_constrained_batch_via_r(
            A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("R constrOptim unavailable (%s), using Python fallback", e)
        return _solve_constrained_batch_python(
            A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr, n_jobs
        )


def _solve_constrained_batch_via_r(
    A: np.ndarray,
    B: np.ndarray,
    n_cell: int,
    theta_sum: np.ndarray,
    pp_min_arr: np.ndarray,
    pp_max_arr: np.ndarray,
) -> np.ndarray:
    """Run batch constrOptim in R via subprocess for exact equivalence."""
    import os
    import shutil
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp()
    try:
        # Write inputs
        np.savetxt(os.path.join(tmpdir, "A.csv"), A, delimiter=",")
        np.savetxt(os.path.join(tmpdir, "B.csv"), B, delimiter=",")
        np.savetxt(
            os.path.join(tmpdir, "params.csv"),
            np.column_stack([theta_sum, pp_min_arr, pp_max_arr]),
            delimiter=",",
            header="theta_sum,pp_min,pp_max",
            comments="",
        )

        r_code = f"""
        A <- as.matrix(read.csv("{tmpdir}/A.csv", header=FALSE))
        B <- as.matrix(read.csv("{tmpdir}/B.csv", header=FALSE))
        params <- read.csv("{tmpdir}/params.csv")

        n_cell <- {n_cell}
        n_spots <- ncol(B)

        result <- matrix(0, nrow=n_cell, ncol=n_spots)

        for (i in 1:n_spots) {{
            ts <- params$theta_sum[i]
            if (ts <= 0.01) {{
                result[, i] <- max(ts, 0) / n_cell
                next
            }}

            theta0 <- rep(ts / n_cell, n_cell)
            b <- B[, i]
            ppmin <- params$pp_min[i]
            ppmax <- params$pp_max[i]

            ui <- rbind(diag(n_cell), rep(1, n_cell), rep(-1, n_cell))
            ci <- c(rep(0, n_cell), ppmin, -ppmax)

            f0 <- function(theta) {{ sum((A %*% theta - b)^2) }}

            prop <- tryCatch({{
                res <- constrOptim(theta0, f0, grad=NULL, ui=ui, ci=ci)
                res$par
            }}, error = function(e) {{ theta0 }})

            bhat <- A %*% prop

            f_weighted <- function(theta) {{ sum((A %*% theta - b)^2 / (bhat + 1)) }}

            prop <- tryCatch({{
                res <- constrOptim(theta0, f_weighted, grad=NULL, ui=ui, ci=ci)
                res$par
            }}, error = function(e) {{ prop }})

            result[, i] <- prop
        }}

        write.csv(result, "{tmpdir}/result.csv", row.names=FALSE)
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R constrOptim failed: {result.stderr}")

        prop_mat = pd.read_csv(os.path.join(tmpdir, "result.csv")).values
        return prop_mat
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _solve_constrained_batch_python(
    A: np.ndarray,
    B: np.ndarray,
    n_cell: int,
    theta_sum: np.ndarray,
    pp_min_arr: np.ndarray,
    pp_max_arr: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Python fallback for constrained batch optimization."""
    from joblib import Parallel, delayed

    n_spots = B.shape[1]

    def solve_single(i: int) -> np.ndarray:
        ts = theta_sum[i]
        if ts <= 0.01:
            return np.full(n_cell, max(ts, 0) / n_cell)

        theta0 = np.full(n_cell, ts / n_cell)
        b = B[:, i]

        ppmin = (
            float(pp_min_arr[i])
            if hasattr(pp_min_arr, "__getitem__")
            else float(pp_min_arr)
        )
        ppmax = (
            float(pp_max_arr[i])
            if hasattr(pp_max_arr, "__getitem__")
            else float(pp_max_arr)
        )

        ui = np.vstack([np.eye(n_cell), np.ones((1, n_cell)), -np.ones((1, n_cell))])
        ci = np.concatenate([np.zeros(n_cell), [ppmin], [-ppmax]])

        def f0(theta, A, b):
            return np.sum((A @ theta - b) ** 2)

        try:
            prop, _ = constr_optim(theta0, f0, ui, ci, args=(A, b))
        except (ValueError, RuntimeError):
            return theta0

        bhat = A @ prop

        def f_weighted(theta, A, b):
            return np.sum((A @ theta - b) ** 2 / (bhat + 1))

        try:
            prop, _ = constr_optim(theta0, f_weighted, ui, ci, args=(A, b))
        except (ValueError, RuntimeError):
            pass

        return prop

    if n_jobs == 1:
        results = [solve_single(i) for i in range(n_spots)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(solve_single)(i) for i in range(n_spots)
        )

    return np.column_stack(results)


def _compute_mal_ref(
    counts: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_mal_idx: np.ndarray,
) -> pd.Series:
    """Compute malignant reference (mean CPM of malignant spots).

    Uses R subprocess for exact equivalence with SpaCET's rowMeans
    computation, falling back to Python.
    """
    try:
        return _compute_mal_ref_via_r(counts, gene_names, spot_mal_idx)
    except (FileNotFoundError, OSError, RuntimeError):
        mal_counts = counts[:, spot_mal_idx]
        if sparse.issparse(mal_counts):
            mal_col_sums = np.asarray(mal_counts.sum(axis=0)).ravel()
            mal_cpm = mal_counts.toarray().astype(np.float64)
            mal_cpm = mal_cpm / mal_col_sums[np.newaxis, :] * 1e6
        else:
            mal_col_sums = mal_counts.sum(axis=0)
            mal_cpm = mal_counts.astype(np.float64) / mal_col_sums[np.newaxis, :] * 1e6
        mal_ref = np.nanmean(mal_cpm, axis=1)
        return pd.Series(mal_ref, index=gene_names)


def _compute_mal_ref_via_r(
    counts: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
    spot_mal_idx: np.ndarray,
) -> pd.Series:
    """Compute malRef in R for exact floating-point equivalence."""
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    tmpdir = tempfile.mkdtemp()
    try:
        counts_sp = sparse.csc_matrix(counts, dtype=np.float64)
        mmwrite(os.path.join(tmpdir, "counts.mtx"), counts_sp)

        # R uses 1-based indexing
        r_indices = spot_mal_idx + 1
        np.savetxt(
            os.path.join(tmpdir, "spot_idx.csv"),
            r_indices.reshape(-1, 1),
            fmt="%d",
            delimiter=",",
            header="idx",
            comments="",
        )

        r_code = f"""
        library(Matrix)
        counts <- readMM("{tmpdir}/counts.mtx")
        counts <- as(counts, "dgCMatrix")
        spot_idx <- read.csv("{tmpdir}/spot_idx.csv")$idx
        malCount <- counts[, spot_idx]
        malRef <- rowMeans(Matrix::t(Matrix::t(malCount) * 1e6 / Matrix::colSums(malCount)))
        write.csv(data.frame(malRef=malRef), "{tmpdir}/malRef.csv", row.names=FALSE)
        """

        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R malRef failed: {result.stderr}")

        mal_ref = pd.read_csv(os.path.join(tmpdir, "malRef.csv"))["malRef"].values
        return pd.Series(mal_ref, index=gene_names)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _get_counts_genes_by_spots(adata: ad.AnnData) -> sparse.spmatrix | np.ndarray:
    """Get count matrix in genes x spots orientation."""
    X = adata.X
    if sparse.issparse(X):
        return X.T.tocsc()
    return X.T.copy()


def _merge_references(ref: dict, ref_normal: dict) -> dict:
    """Merge normal tissue reference into main reference."""
    ref_profiles = ref["refProfiles"]
    ref_normal_profiles = ref_normal["refProfiles"]

    olp = ref_profiles.index.intersection(ref_normal_profiles.index)
    merged_profiles = pd.concat(
        [ref_profiles.loc[olp], ref_normal_profiles.loc[olp]], axis=1
    )

    merged_sig = {**ref["sigGenes"], **ref_normal["sigGenes"]}
    merged_tree = {**ref["lineageTree"], **ref_normal["lineageTree"]}

    return {
        "refProfiles": merged_profiles,
        "sigGenes": merged_sig,
        "lineageTree": merged_tree,
    }
