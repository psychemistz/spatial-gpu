"""SpaCET extension functions for spatial transcriptomics deconvolution.

Functions for:
  - Malignant cell state discovery (deconvolution_malignant)
  - Deconvolution with matched scRNA-seq (deconvolution_matched_scrnaseq)
  - Malignant deconvolution with custom scRNA-seq (deconvolution_malignant_custom_scrnaseq)
  - Reference generation from scRNA-seq (generate_ref)

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from statsmodels.stats.multitest import multipletests

from spatialgpu.deconvolution.core import (
    _get_counts_genes_by_spots,
    _spatial_deconv,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deconvolution_malignant(
    adata: ad.AnnData,
    malignant: str = "Malignant",
    malignant_cutoff: float = 0.7,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Explore different malignant cell states in tumor ST data.

    Equivalent to SpaCET.deconvolution.malignant() in R.

    Clusters spots with high malignant fraction to identify distinct
    malignant cell states, then re-deconvolves the malignant fraction
    into those states.

    Parameters
    ----------
    adata : AnnData
        Must already have deconvolution results in ``adata.uns['spacet']``.
    malignant : str
        Name of the malignant cell type in the existing deconvolution.
    malignant_cutoff : float
        Fraction cutoff (0-1) for selecting spots with high malignant content.
    n_jobs : int
        Number of parallel jobs for deconvolution.

    Returns
    -------
    AnnData with updated ``adata.uns['spacet']['deconvolution']['propMat']``
    including malignant cell state sub-fractions.
    """
    import scanpy as sc

    # --- Validation ---
    if "spacet" not in adata.uns or "deconvolution" not in adata.uns["spacet"]:
        raise ValueError(
            "Please run deconvolution first using spatialgpu.deconvolution.core.deconvolution."
        )

    deconv = adata.uns["spacet"]["deconvolution"]
    res_deconv: pd.DataFrame = deconv["propMat"]  # cell_types x spots

    if malignant not in res_deconv.index:
        raise ValueError(
            f"Malignant cell type '{malignant}' not found in deconvolution results. "
            f"Available types: {list(res_deconv.index)}"
        )

    lineage_tree = deconv["Ref"]["lineageTree"]
    if malignant in lineage_tree and len(lineage_tree[malignant]) > 1:
        raise ValueError(
            "Deconvolution results already include multiple malignant cell states. "
            "Further deconvolution is not recommended."
        )

    if not 0 <= malignant_cutoff <= 1:
        raise ValueError("malignant_cutoff must be between 0 and 1.")

    # --- Get counts (genes x spots) ---
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

    # --- Select malignant spots ---
    mal_fractions = res_deconv.loc[malignant]
    mal_spot_mask = mal_fractions >= malignant_cutoff
    mal_spots = mal_fractions.index[mal_spot_mask].values

    if len(mal_spots) < 3:
        raise ValueError(
            f"Only {len(mal_spots)} spots have malignant fraction >= {malignant_cutoff}. "
            "Consider lowering the cutoff."
        )

    mal_spot_idx = np.array([np.where(spot_names == s)[0][0] for s in mal_spots])

    # --- CPM normalize malignant spots (1e5, matching R) ---
    counts_mal = counts[:, mal_spot_idx]
    if sparse.issparse(counts_mal):
        counts_mal_dense = counts_mal.toarray().astype(np.float64)
    else:
        counts_mal_dense = counts_mal.astype(np.float64)

    col_sums_mal = counts_mal_dense.sum(axis=0)
    col_sums_mal[col_sums_mal == 0] = 1.0
    cpm_mal = counts_mal_dense / col_sums_mal[np.newaxis, :] * 1e5
    log_mal = np.log2(cpm_mal + 1)

    # --- Clustering within malignant spots ---
    np.random.seed(123)
    logger.info("Clustering malignant spots.")

    # Create temporary AnnData for scanpy processing (spots x genes)
    if sparse.issparse(counts_mal):
        adata_tmp = sc.AnnData(X=counts_mal.T.tocsr())
    else:
        adata_tmp = sc.AnnData(X=counts_mal.T.copy())
    adata_tmp.var_names = pd.Index(gene_names)
    adata_tmp.obs_names = pd.Index(mal_spots)

    # Variance normalization + HVG + PCA (MUDAN equivalent)
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    n_hvg = min(3000, len(gene_names))
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=n_hvg)
    adata_tmp = adata_tmp[:, adata_tmp.var.highly_variable].copy()
    sc.pp.scale(adata_tmp, max_value=10)
    n_comps = min(30, adata_tmp.shape[1] - 1)
    sc.tl.pca(adata_tmp, n_comps=n_comps)
    pcs = adata_tmp.obsm["X_pca"]

    # Hierarchical clustering with Ward's method on correlation distance
    corr_matrix = np.corrcoef(pcs)
    corr_matrix = np.clip(corr_matrix, -1, 1)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    # Silhouette analysis for k=2:9 — use MAX silhouette (not max decrease)
    cluster_numbers = list(range(2, 10))
    sil_scores: list[float] = []
    for k in cluster_numbers:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_samples(dist_matrix, labels, metric="precomputed")
        sil_scores.append(float(np.mean(sil)))

    max_n = cluster_numbers[int(np.argmax(sil_scores))]

    clustering_raw = fcluster(Z, t=max_n, criterion="maxclust")
    # Convert numeric cluster labels to letters (A, B, C, ...)
    clustering_letters = np.array(
        [string.ascii_uppercase[c - 1] for c in clustering_raw]
    )
    content = pd.Series(clustering_letters, index=mal_spots)

    states = sorted(content.unique())
    logger.info(f"Identified {len(states)} malignant cell states.")

    # --- Build new reference ---
    ref_profiles = pd.DataFrame(index=gene_names, dtype=np.float64)
    sig_genes: dict[str, list[str]] = {}

    # Overall malignant reference profile
    ref_profiles["Malignant"] = cpm_mal.mean(axis=1)

    for state in states:
        state_name = f"Malignant cell state {state}"
        state_mask = (content == state).values
        ref_profiles[state_name] = cpm_mal[:, state_mask].mean(axis=1)

        # --- DE analysis: find marker genes for this state ---
        temp_markers: list[str] = []
        for other_state in states:
            if other_state == state:
                continue

            other_mask = (content == other_state).values
            markers = _de_ttest(log_mal, gene_names, state_mask, other_mask, n_top=500)
            temp_markers.extend(markers)

        # Signature genes: appear in exactly 1 comparison
        # (R code: tempMarkers==1, which means unique to one comparison)
        marker_counts = pd.Series(temp_markers).value_counts()
        sig_genes[state_name] = list(marker_counts[marker_counts == 1].index)

    lineage_tree_new: dict[str, list[str]] = {
        malignant: [f"Malignant cell state {s}" for s in states]
    }
    ref_new = {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": lineage_tree_new,
    }

    # --- Re-deconvolve malignant fraction ---
    # Known cell types = everything except the malignant lineage
    known_cell_types = [k for k in lineage_tree.keys() if k != malignant]

    known_fractions = list(known_cell_types)
    if "Unidentifiable" in res_deconv.index:
        known_fractions.append("Unidentifiable")

    mal_prop_known = res_deconv.loc[known_fractions]

    # Known cell reference profiles
    orig_ref = deconv["Ref"]["refProfiles"]
    if isinstance(orig_ref, pd.DataFrame):
        known_cols = [c for c in known_cell_types if c in orig_ref.columns]
        mal_ref_known = orig_ref[known_cols]
    else:
        mal_ref_known = None

    # Re-deconvolve
    logger.info("Re-deconvolving malignant cell states.")

    prop_mat_new = _spatial_deconv(
        ST=counts,
        gene_names=gene_names,
        spot_names=spot_names,
        ref=ref_new,
        mal_prop=mal_prop_known,
        mal_ref=mal_ref_known,
        mode="deconvMal",
        n_jobs=n_jobs,
    )

    # Merge: keep existing rows, add new state rows (exclude "Malignant"
    # row from prop_mat_new since it's already in res_deconv)
    new_rows = prop_mat_new.loc[~prop_mat_new.index.isin([malignant])]
    prop_mat_merged = pd.concat([res_deconv, new_rows])

    # Update adata
    deconv["propMat"] = prop_mat_merged
    adata.uns["spacet"]["deconvolution"] = deconv
    adata.uns["spacet"]["propMat_columns"] = list(prop_mat_merged.index)
    adata.obsm["spacet_propMat"] = prop_mat_merged.T.reindex(adata.obs_names).values

    return adata


def deconvolution_matched_scrnaseq(
    adata: ad.AnnData,
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
    sc_include_malignant: bool = True,
    cancer_type: str | None = None,
    sc_downsampling: bool = True,
    sc_n_cell_each_lineage: int = 100,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Deconvolve ST data with matched scRNA-seq reference.

    Equivalent to SpaCET.deconvolution.matched.scRNAseq() in R.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data with raw counts.
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells). If DataFrame, index = gene
        names and columns = cell IDs.
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Hierarchical lineage tree. Keys = major lineages, values = lists of
        sub-lineages. If a major lineage has no sub-lineages, value = [itself].
    sc_include_malignant : bool
        Whether the scRNA-seq data includes malignant cells. If False,
        ``cancer_type`` must be provided to infer malignant fraction.
    cancer_type : str or None
        Cancer type code. Required when ``sc_include_malignant=False``.
    sc_downsampling : bool
        Whether to downsample cells per type.
    sc_n_cell_each_lineage : int
        Max cells per lineage for downsampling (seed=123).
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with deconvolution results in ``adata.uns['spacet']``.
    """
    from spatialgpu.deconvolution.core import _infer_mal_cor

    # --- Validate inputs ---
    sc_counts, sc_annotation = _validate_sc_inputs(
        sc_counts, sc_annotation, sc_lineage_tree
    )

    # --- Downsampling ---
    if sc_downsampling:
        logger.info(f"Down-sampling: True, n={sc_n_cell_each_lineage}")
        sc_counts, sc_annotation = _downsample_cells(
            sc_counts, sc_annotation, sc_n_cell_each_lineage
        )
    else:
        logger.info("Down-sampling: False")

    # Filter zero-sum genes
    if isinstance(sc_counts, pd.DataFrame):
        row_sums = sc_counts.sum(axis=1)
        sc_counts = sc_counts.loc[row_sums > 0]
    else:
        row_sums = sc_counts.sum(axis=1)
        keep = row_sums > 0
        sc_counts = sc_counts[keep]

    # --- Generate reference ---
    logger.info("1. Generate the cell type reference from the matched scRNAseq data.")
    ref = generate_ref(sc_counts, sc_annotation, sc_lineage_tree, n_jobs=n_jobs)

    # --- Get ST counts ---
    logger.info("2. Hierarchically deconvolve the Spatial Transcriptomics dataset.")
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

    if sc_include_malignant:
        # No malignant inference needed — all cell types are in the reference
        mal_prop = pd.Series(0.0, index=spot_names)

        prop_mat = _spatial_deconv(
            ST=counts,
            gene_names=gene_names,
            spot_names=spot_names,
            ref=ref,
            mal_prop=mal_prop,
            mal_ref=None,
            mode="deconvWithSC",
            unidentifiable=True,
            macrophage_other=False,
            n_jobs=n_jobs,
        )
    else:
        if cancer_type is None:
            raise ValueError("cancer_type is required when sc_include_malignant=False.")

        logger.info("Stage 1. Infer malignant cell fraction.")
        mal_res = _infer_mal_cor(
            counts, gene_names, spot_names, cancer_type, signature_type=None
        )

        logger.info("Stage 2. Deconvolve non-malignant cell fraction.")
        prop_mat = _spatial_deconv(
            ST=counts,
            gene_names=gene_names,
            spot_names=spot_names,
            ref=ref,
            mal_prop=mal_res["malProp"],
            mal_ref=mal_res["malRef"],
            mode="deconvWithSC_alt",
            unidentifiable=True,
            macrophage_other=False,
            n_jobs=n_jobs,
        )

    # Store results
    adata.uns["spacet"] = {
        "deconvolution": {
            "propMat": prop_mat,
            "Ref": ref,
        },
        "propMat_columns": list(prop_mat.index),
    }
    adata.obsm["spacet_propMat"] = prop_mat.T.reindex(adata.obs_names).values

    return adata


def deconvolution_malignant_custom_scrnaseq(
    adata: ad.AnnData,
    malignant: str = "Malignant",
    sc_counts: pd.DataFrame | np.ndarray | None = None,
    sc_annotation: pd.DataFrame | None = None,
    sc_lineage_tree: dict[str, list[str]] | None = None,
    sc_n_cell_each_lineage: int = 100,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Deconvolve malignant fraction using custom scRNA-seq reference.

    Equivalent to SpaCET.deconvolution.malignant.customized.scRNAseq() in R.

    Uses user-provided scRNA-seq data to build a malignant-specific reference,
    then re-deconvolves the malignant fraction into sub-states.

    Parameters
    ----------
    adata : AnnData
        Must already have deconvolution results in ``adata.uns['spacet']``.
    malignant : str
        Name of the malignant cell type in existing results.
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Lineage tree with exactly one entry for the malignant lineage.
    sc_n_cell_each_lineage : int
        Max cells per lineage for downsampling.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with updated deconvolution results.
    """
    # --- Validation ---
    if "spacet" not in adata.uns or "deconvolution" not in adata.uns["spacet"]:
        raise ValueError(
            "Please run deconvolution first using spatialgpu.deconvolution.core.deconvolution."
        )

    deconv = adata.uns["spacet"]["deconvolution"]
    res_deconv: pd.DataFrame = deconv["propMat"]

    if malignant not in res_deconv.index:
        raise ValueError(
            f"Malignant cell type '{malignant}' not found in deconvolution results."
        )

    lineage_tree_orig = deconv["Ref"]["lineageTree"]
    if malignant in lineage_tree_orig and len(lineage_tree_orig[malignant]) > 1:
        raise ValueError(
            "Deconvolution results already include multiple malignant cell states."
        )

    if sc_counts is None or sc_annotation is None or sc_lineage_tree is None:
        raise ValueError(
            "sc_counts, sc_annotation, and sc_lineage_tree are all required."
        )

    if len(sc_lineage_tree) != 1:
        raise ValueError(
            "sc_lineage_tree must have exactly one entry for the malignant lineage."
        )

    sc_counts, sc_annotation = _validate_sc_inputs(
        sc_counts, sc_annotation, sc_lineage_tree
    )

    # --- Downsampling ---
    sc_counts, sc_annotation = _downsample_cells(
        sc_counts, sc_annotation, sc_n_cell_each_lineage
    )

    # Filter zero-sum genes
    if isinstance(sc_counts, pd.DataFrame):
        row_sums = sc_counts.sum(axis=1)
        sc_counts = sc_counts.loc[row_sums > 0]
    else:
        row_sums = sc_counts.sum(axis=1)
        keep = row_sums > 0
        sc_counts = sc_counts[keep]

    # --- Generate reference ---
    logger.info("1. Generate the reference from the input scRNAseq data.")
    ref_new = generate_ref(sc_counts, sc_annotation, sc_lineage_tree, n_jobs=n_jobs)

    # --- Get ST counts ---
    logger.info("2. Deconvolve malignant cells.")
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

    # --- Known cell fractions (non-malignant) ---
    known_cell_types = [k for k in lineage_tree_orig.keys() if k != malignant]
    known_fractions = list(known_cell_types)
    if "Unidentifiable" in res_deconv.index:
        known_fractions.append("Unidentifiable")

    mal_prop_known = res_deconv.loc[known_fractions]

    # Known cell reference
    orig_ref = deconv["Ref"]["refProfiles"]
    if isinstance(orig_ref, pd.DataFrame):
        known_cols = [c for c in known_cell_types if c in orig_ref.columns]
        mal_ref_known = orig_ref[known_cols]
    else:
        mal_ref_known = None

    # --- Deconvolve ---
    prop_mat_new = _spatial_deconv(
        ST=counts,
        gene_names=gene_names,
        spot_names=spot_names,
        ref=ref_new,
        mal_prop=mal_prop_known,
        mal_ref=mal_ref_known,
        mode="deconvMal",
        n_jobs=n_jobs,
    )

    # Merge: keep existing + add new rows (exclude the parent lineage name)
    lineage_parent = list(sc_lineage_tree.keys())[0]
    new_rows = prop_mat_new.loc[~prop_mat_new.index.isin([lineage_parent])]
    prop_mat_merged = pd.concat([res_deconv, new_rows])

    # Update adata
    deconv["propMat"] = prop_mat_merged
    deconv["malRef"] = ref_new
    adata.uns["spacet"]["deconvolution"] = deconv
    adata.uns["spacet"]["propMat_columns"] = list(prop_mat_merged.index)
    adata.obsm["spacet_propMat"] = prop_mat_merged.T.reindex(adata.obs_names).values

    return adata


def generate_ref(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Generate cell type reference from scRNA-seq data.

    Equivalent to SpaCET's generateRef() in R.

    For each major lineage and sub-lineage:
      1. CPM normalize (1e5, not 1e6).
      2. Compute mean expression as refProfile.
      3. DE analysis via t-test (limma equivalent): top 500 by t-stat,
         filter logFC > 0.25 and FDR < 0.01.
      4. Signature genes = significant in >= (n_lineages - 1) comparisons.

    Parameters
    ----------
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Keys = major lineages, values = lists of sub-lineages.
    n_jobs : int
        Number of parallel jobs (reserved for future use).

    Returns
    -------
    dict with keys:
        refProfiles : pd.DataFrame (genes x cell_types)
        sigGenes : dict[str, list[str]]
        lineageTree : dict[str, list[str]]
    """
    # Ensure sc_counts is a DataFrame with gene names as index
    if not isinstance(sc_counts, pd.DataFrame):
        raise TypeError("sc_counts must be a pd.DataFrame with gene names as index.")

    # Build cell ID -> cell type mapping
    if "cellID" not in sc_annotation.columns or "cellType" not in sc_annotation.columns:
        raise ValueError("sc_annotation must have 'cellID' and 'cellType' columns.")
    sc_annotation = sc_annotation.copy()
    sc_annotation.index = sc_annotation["cellID"].astype(str).values

    # Ensure consistent ordering
    cell_types = sc_annotation["cellType"].values.astype(str)

    gene_names = np.array(sc_counts.index)

    # --- CPM normalization (1e5) ---
    if sparse.issparse(sc_counts.values):
        counts_dense = sc_counts.values.toarray().astype(np.float64)
    else:
        counts_dense = sc_counts.values.astype(np.float64)

    col_sums = counts_dense.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    sc_norm = counts_dense / col_sums[np.newaxis, :] * 1e5
    sc_log2 = np.log2(sc_norm + 1)

    # --- Build reference ---
    cell_types_level_1 = list(sc_lineage_tree.keys())
    cell_types_to_be_split = [
        ct
        for ct in cell_types_level_1
        if len(sc_lineage_tree[ct]) != 1 or sc_lineage_tree[ct][0] != ct
    ]

    ref_profiles = pd.DataFrame(index=gene_names, dtype=np.float64)
    sig_genes: dict[str, list[str]] = {}

    for cell_type in cell_types_level_1:
        logger.info(f"  {cell_type}")

        # Cells belonging to this lineage (all subtypes)
        subtypes_of_ct = sc_lineage_tree[cell_type]
        ct_mask = np.isin(cell_types, subtypes_of_ct)
        ct_col_idx = np.where(ct_mask)[0]

        # Reference profile: mean CPM across all cells in this lineage
        ref_profiles[cell_type] = sc_norm[:, ct_col_idx].mean(axis=1)

        # --- DE analysis against each other major lineage ---
        if len(cell_types_level_1) > 1:
            all_markers: list[list[str]] = []

            for other_ct in cell_types_level_1:
                if other_ct == cell_type:
                    continue

                other_subtypes = sc_lineage_tree[other_ct]
                other_mask = np.isin(cell_types, other_subtypes)

                markers = _de_ttest(sc_log2, gene_names, ct_mask, other_mask, n_top=500)
                all_markers.append(markers)

            # Signature genes: present in >= (n_lineages - 1) comparisons
            if all_markers:
                flat_markers = [m for sublist in all_markers for m in sublist]
                marker_counts = pd.Series(flat_markers).value_counts()
                threshold = len(cell_types_level_1) - 1
                sig_genes[cell_type] = list(
                    marker_counts[marker_counts >= threshold].index
                )

        # --- Sub-lineage level ---
        if cell_type in cell_types_to_be_split:
            subtypes = sc_lineage_tree[cell_type]

            for subtype in subtypes:
                sub_mask = cell_types == subtype
                sub_col_idx = np.where(sub_mask)[0]

                ref_profiles[subtype] = sc_norm[:, sub_col_idx].mean(axis=1)

                # DE: subtype vs rest of the same lineage
                other_subtypes_in_lineage = [s for s in subtypes if s != subtype]
                other_sub_mask = np.isin(cell_types, other_subtypes_in_lineage)

                if other_sub_mask.sum() > 0:
                    markers = _de_ttest(
                        sc_log2, gene_names, sub_mask, other_sub_mask, n_top=500
                    )
                    sig_genes[subtype] = markers
                else:
                    sig_genes[subtype] = []

    return {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": sc_lineage_tree,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _de_ttest(
    log_expr: np.ndarray,
    gene_names: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    n_top: int = 500,
    logfc_cutoff: float = 0.25,
    fdr_cutoff: float = 0.01,
) -> list[str]:
    """Differential expression: try limma via R, fall back to t-test.

    SpaCET uses limma (empirical Bayes moderated t-test) for DE analysis.
    This function tries R/limma first for exact equivalence, falling back
    to Welch's t-test if R is unavailable.

    Parameters
    ----------
    log_expr : np.ndarray
        Log2-transformed expression matrix (genes x cells).
    gene_names : np.ndarray
        Gene names corresponding to rows of log_expr.
    group1_mask : np.ndarray
        Boolean mask for group 1 (treatment).
    group2_mask : np.ndarray
        Boolean mask for group 2 (control).
    n_top : int
        Number of top genes by t-statistic to consider.
    logfc_cutoff : float
        Minimum log-fold-change threshold.
    fdr_cutoff : float
        Maximum FDR threshold.

    Returns
    -------
    list[str]
        Gene names passing both logFC and FDR filters among the top n_top.
    """
    try:
        return _de_limma_via_r(
            log_expr,
            gene_names,
            group1_mask,
            group2_mask,
            n_top,
            logfc_cutoff,
            fdr_cutoff,
        )
    except (FileNotFoundError, OSError, RuntimeError):
        return _de_ttest_python(
            log_expr,
            gene_names,
            group1_mask,
            group2_mask,
            n_top,
            logfc_cutoff,
            fdr_cutoff,
        )


def _de_limma_via_r(
    log_expr: np.ndarray,
    gene_names: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    n_top: int = 500,
    logfc_cutoff: float = 0.25,
    fdr_cutoff: float = 0.01,
) -> list[str]:
    """Run limma DE analysis in R for exact SpaCET match."""
    import os
    import shutil
    import subprocess
    import tempfile

    # Subset to the two groups
    col_idx = np.where(group1_mask | group2_mask)[0]
    sub_expr = log_expr[:, col_idx]
    sub_groups = np.where(group1_mask[col_idx], "A", "B")

    tmpdir = tempfile.mkdtemp()
    input_expr = os.path.join(tmpdir, "expr.csv")
    input_groups = os.path.join(tmpdir, "groups.csv")
    output_genes = os.path.join(tmpdir, "de_genes.csv")

    # Write expression matrix
    expr_df = pd.DataFrame(sub_expr, index=gene_names)
    expr_df.to_csv(input_expr)
    pd.DataFrame({"group": sub_groups}).to_csv(input_groups, index=False)

    r_code = f"""
    suppressPackageStartupMessages({{
        library(limma)
    }})

    expr <- as.matrix(read.csv("{input_expr}", row.names=1, check.names=FALSE))
    groups <- read.csv("{input_groups}")$group

    design <- model.matrix(~ 0 + factor(groups))
    colnames(design) <- c("A", "B")

    fit <- lmFit(expr, design)
    contrast.matrix <- makeContrasts(A - B, levels=design)
    fit2 <- contrasts.fit(fit, contrast.matrix)
    fit2 <- eBayes(fit2)

    tt <- topTable(fit2, number={n_top}, sort.by="t")
    sig <- tt[tt$logFC > {logfc_cutoff} & tt$adj.P.Val < {fdr_cutoff}, ]

    write.csv(data.frame(gene=rownames(sig)), "{output_genes}", row.names=FALSE)
    """

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R limma failed: {result.stderr}")

        if os.path.getsize(output_genes) > 0:
            de_df = pd.read_csv(output_genes)
            return list(de_df["gene"].values)
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _de_ttest_python(
    log_expr: np.ndarray,
    gene_names: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    n_top: int = 500,
    logfc_cutoff: float = 0.25,
    fdr_cutoff: float = 0.01,
) -> list[str]:
    """Python fallback DE analysis using Welch's t-test."""
    group1 = log_expr[:, group1_mask]
    group2 = log_expr[:, group2_mask]

    n1 = group1.shape[1]
    n2 = group2.shape[1]

    if n1 < 2 or n2 < 2:
        return []

    mean1 = group1.mean(axis=1)
    mean2 = group2.mean(axis=1)
    var1 = group1.var(axis=1, ddof=1)
    var2 = group2.var(axis=1, ddof=1)

    se = np.sqrt(var1 / n1 + var2 / n2)
    se[se == 0] = np.inf

    t_stat = (mean1 - mean2) / se

    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    denom[denom == 0] = 1.0
    df = num / denom
    df = np.clip(df, 1, np.inf)

    p_values = 2 * stats.t.sf(np.abs(t_stat), df)
    logfc = mean1 - mean2

    sorted_idx = np.argsort(-t_stat)
    top_idx = sorted_idx[: min(n_top, len(sorted_idx))]

    top_genes = gene_names[top_idx]
    top_logfc = logfc[top_idx]
    top_pvals = p_values[top_idx]

    if len(top_pvals) > 0:
        top_pvals = np.where(np.isnan(top_pvals), 1.0, top_pvals)
        _, top_fdr, _, _ = multipletests(top_pvals, method="fdr_bh")
    else:
        return []

    pass_mask = (top_logfc > logfc_cutoff) & (top_fdr < fdr_cutoff)
    return list(top_genes[pass_mask])


def _validate_sc_inputs(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate scRNA-seq inputs and set annotation index.

    Returns
    -------
    tuple of (sc_counts, sc_annotation) with consistent formatting.
    """
    # Ensure annotation has cellID as index
    sc_annotation = sc_annotation.copy()
    if "cellID" in sc_annotation.columns:
        sc_annotation.index = sc_annotation["cellID"].astype(str).values
    else:
        raise ValueError("sc_annotation must have a 'cellID' column.")

    if "cellType" not in sc_annotation.columns:
        raise ValueError("sc_annotation must have a 'cellType' column.")

    # Check dimensions
    if isinstance(sc_counts, pd.DataFrame):
        n_cells_counts = sc_counts.shape[1]
    else:
        n_cells_counts = sc_counts.shape[1]

    n_cells_anno = len(sc_annotation)

    if n_cells_counts != n_cells_anno:
        raise ValueError(
            f"Cell count mismatch: sc_counts has {n_cells_counts} cells, "
            f"sc_annotation has {n_cells_anno} cells."
        )

    # Check cell ID matching
    if isinstance(sc_counts, pd.DataFrame):
        count_ids = {str(c) for c in sc_counts.columns}
        anno_ids = {str(c) for c in sc_annotation.index}
        if count_ids != anno_ids:
            raise ValueError("Cell IDs in sc_counts and sc_annotation do not match.")

    # Validate lineage tree
    if len(sc_lineage_tree) == 0:
        raise ValueError("Lineage tree is empty.")

    all_cell_types = []
    for subtypes in sc_lineage_tree.values():
        all_cell_types.extend(subtypes)

    unique_anno_types = set(sc_annotation["cellType"].astype(str).unique())
    missing = [ct for ct in all_cell_types if ct not in unique_anno_types]
    if missing:
        raise ValueError(
            f"Cell types in lineage tree not found in annotation: {missing}"
        )

    return sc_counts, sc_annotation


def _downsample_cells(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    n_cell_each_lineage: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downsample cells to n_cell_each_lineage per cell type.

    Uses seed=123 for reproducibility, matching R behavior.

    Parameters
    ----------
    sc_counts : pd.DataFrame
        Count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Annotation with 'cellID' and 'cellType' columns.
    n_cell_each_lineage : int
        Maximum number of cells per cell type.

    Returns
    -------
    tuple of downsampled (sc_counts, sc_annotation).
    """
    np.random.seed(123)

    cell_ids = sc_annotation["cellID"].astype(str).values
    cell_types = sc_annotation["cellType"].astype(str).values

    # Group cell IDs by cell type
    type_to_ids: dict[str, list[str]] = {}
    for cid, ctype in zip(cell_ids, cell_types):
        type_to_ids.setdefault(ctype, []).append(cid)

    # Downsample each type
    keep_ids: list[str] = []
    for ctype in sorted(type_to_ids.keys()):
        ids = type_to_ids[ctype]
        n = len(ids)
        if n > n_cell_each_lineage:
            n = n_cell_each_lineage
        sampled = list(np.random.choice(ids, size=n, replace=False))
        keep_ids.extend(sampled)

    # Subset
    if isinstance(sc_counts, pd.DataFrame):
        sc_counts = sc_counts[keep_ids]
    else:
        # If numpy array, need to find column indices
        all_col_ids = np.array(sc_annotation["cellID"].astype(str).values)
        keep_mask = np.isin(all_col_ids, keep_ids)
        sc_counts = sc_counts[:, keep_mask]

    sc_annotation = sc_annotation.loc[keep_ids]

    return sc_counts, sc_annotation
