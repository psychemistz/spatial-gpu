"""Secreted protein activity analysis for spatial transcriptomics.

Wraps SecActPy for activity inference and implements downstream analysis:
  1. Activity inference via ridge regression (delegates to secactpy)
  2. Signaling pattern discovery via NMF
  3. Signaling velocity (source→sink arrows)
  4. Spatial cell-cell communication (CCC)
  5. Cox proportional hazards regression

Reference: SecAct R package (downstream.R)
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import KDTree

from spatialgpu.deconvolution.spatial_correlation import cal_weights

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_secactpy():
    """Import secactpy, raising a clear error if not installed."""
    try:
        return importlib.import_module("secactpy")
    except ModuleNotFoundError:
        raise ImportError(
            "secactpy is required for SecAct analysis. "
            "Install it with: pip install secactpy>=0.2.3 "
            "or: pip install spatial-gpu[secact]"
        ) from None


def _ensure_secact(adata: ad.AnnData) -> dict:
    """Return adata.uns['spacet']['SecAct_output'], creating if absent."""
    if "spacet" not in adata.uns:
        adata.uns["spacet"] = {}
    spacet = adata.uns["spacet"]
    if "SecAct_output" not in spacet:
        spacet["SecAct_output"] = {}
    return spacet["SecAct_output"]


def _get_expression_matrix(adata: ad.AnnData) -> pd.DataFrame:
    """Extract raw counts as DataFrame (genes × spots)."""
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return pd.DataFrame(
        X.T,
        index=adata.var_names.tolist(),
        columns=adata.obs_names.tolist(),
    )


def _normalize_tpm(expr: pd.DataFrame, scale_factor: float = 1e5) -> pd.DataFrame:
    """Normalize to TPM and log2-transform. Matches R: sweep + log2(x+1)."""
    col_sums = expr.sum(axis=0)
    col_sums = col_sums.replace(0, 1)  # avoid division by zero
    normed = expr.div(col_sums, axis=1) * scale_factor
    return np.log2(normed + 1)


def _transfer_symbol(genes: list[str]) -> list[str]:
    """Placeholder for gene symbol transfer (identity for now).

    The R package uses transferSymbol() to map gene aliases.
    """
    return genes


def _rm_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate gene rows, keeping the one with highest total."""
    if not df.index.duplicated().any():
        return df
    row_sums = df.sum(axis=1)
    df = df.copy()
    df["_total"] = row_sums
    df = df.sort_values("_total", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns=["_total"])
    return df


# ---------------------------------------------------------------------------
# 1. Activity Inference
# ---------------------------------------------------------------------------


def secact_inference(
    adata: ad.AnnData,
    sig_matrix: str = "secact",
    scale_factor: float = 1e5,
    is_spot_level: bool = True,
    cell_type_col: Optional[str] = None,
    is_group_sig: Optional[bool] = None,
    is_group_cor: float = 0.9,
    lambda_: float = 5e5,
    n_rand: int = 1000,
    seed: int = 0,
    backend: str = "auto",
    verbose: bool = False,
) -> ad.AnnData:
    """Infer secreted protein activity from spatial transcriptomics data.

    Delegates to ``secactpy.secact_activity_inference_st()`` and stores
    results in ``adata.uns['spacet']['SecAct_output']['SecretedProteinActivity']``.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data with raw counts in X.
    sig_matrix : str
        Signature matrix name: "secact", "cytosig", or path to custom file.
    scale_factor : float
        Normalization scale factor. Default: 1e5.
    is_spot_level : bool
        If True, compute activity per spot. If False, aggregate by cell type.
    cell_type_col : str, optional
        Column in adata.obs for cell type annotations (used when
        is_spot_level=False).
    is_group_sig : bool or None
        Group similar signatures by correlation. None = auto.
    is_group_cor : float
        Correlation threshold for grouping. Default: 0.9.
    lambda_ : float
        Ridge regularization parameter. Default: 5e5.
    n_rand : int
        Number of permutations. Default: 1000.
    seed : int
        Random seed. Default: 0 (exact R compatibility).
    backend : str
        Computation backend: "auto", "numpy", "cupy".
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnnData with results in adata.uns['spacet']['SecAct_output']
    """
    secactpy = _import_secactpy()
    secact_out = _ensure_secact(adata)

    result = secactpy.secact_activity_inference_st(
        input_data=adata,
        is_spot_level=is_spot_level,
        cell_type_col=cell_type_col,
        scale_factor=scale_factor,
        sig_matrix=sig_matrix,
        is_group_sig=is_group_sig,
        is_group_cor=is_group_cor,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
        backend=backend,
        verbose=verbose,
    )

    secact_out["SecretedProteinActivity"] = result
    logger.info(
        "SecAct inference done: %d proteins × %d spots",
        result["zscore"].shape[0],
        result["zscore"].shape[1],
    )
    return adata


# ---------------------------------------------------------------------------
# 2. Signaling Pattern Discovery (NMF)
# ---------------------------------------------------------------------------


def secact_signaling_patterns(
    adata: ad.AnnData,
    k: int | list[int] = 3,
    scale_factor: float = 1e5,
    radius: float = 200.0,
    sigma: float = 100.0,
    seed: int = 123456,
) -> ad.AnnData:
    """Discover signaling patterns via NMF on activity z-scores.

    Equivalent to ``SecAct.signaling.pattern()`` in R.

    Steps:
      1. Filter secreted proteins by Spearman correlation between
         activity and neighbor-aggregated expression (r > 0.05, padj < 0.01).
      2. Run NMF on the filtered, non-negative activity matrix.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct activity results.
    k : int or list[int]
        Number of NMF factors. If a list, selects optimal k by silhouette.
    scale_factor : float
        TPM normalization scale factor.
    radius : float
        Spatial weight radius (micrometers).
    sigma : float
        RBF kernel sigma.
    seed : int
        NMF random seed. Default: 123456 (matches R).

    Returns
    -------
    AnnData with results in adata.uns['spacet']['SecAct_output']['pattern']
    """
    from sklearn.decomposition import NMF
    from sklearn.metrics import silhouette_score

    secact_out = _ensure_secact(adata)
    if "SecretedProteinActivity" not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out["SecretedProteinActivity"]["zscore"].copy()
    act = act.clip(lower=0)  # clip negative z-scores to 0

    # Step 1: Filter by Spearman correlation with neighbor-aggregated expression
    logger.info("Step 1. Filtering secreted proteins")

    expr = _get_expression_matrix(adata)
    expr.index = _transfer_symbol(expr.index.tolist())
    expr = _rm_duplicates(expr)
    expr = _normalize_tpm(expr, scale_factor)

    # Compute spatial weights
    weights = cal_weights(adata, radius=radius, sigma=sigma, diag_as_zero=True)

    # Align spots: only keep spots present in both weight matrix and activity
    spot_names = adata.obs_names.tolist()
    common_spots = [s for s in spot_names if s in act.columns and s in expr.columns]
    act_new = act[common_spots]
    expr_new = expr[common_spots]

    # Neighbor-aggregated expression: expr_new_aggr = expr_new @ weights
    if sparse.issparse(weights):
        weights_dense = weights.toarray()
    else:
        weights_dense = np.asarray(weights)

    # Row-normalize weights so each column sums as in R
    expr_new_aggr = expr_new.values @ weights_dense

    expr_new_aggr = pd.DataFrame(
        expr_new_aggr, index=expr_new.index, columns=common_spots
    )

    # Spearman correlation for each secreted protein
    corr_data = []
    for gene in act_new.index:
        act_gene = act_new.loc[gene].values
        if gene in expr_new.index:
            exp_gene = expr_new_aggr.loc[gene].values
            r, p = stats.spearmanr(act_gene, exp_gene)
            corr_data.append({"gene": gene, "r": r, "p": p})
        else:
            corr_data.append({"gene": gene, "r": np.nan, "p": np.nan})

    corr_df = pd.DataFrame(corr_data).set_index("gene")

    # BH correction
    valid_mask = ~corr_df["p"].isna()
    padj = np.full(len(corr_df), np.nan)
    if valid_mask.any():
        from statsmodels.stats.multitest import multipletests

        _, padj_valid, _, _ = multipletests(
            corr_df.loc[valid_mask, "p"].values, method="fdr_bh"
        )
        padj[valid_mask.values] = padj_valid
    corr_df["padj"] = padj

    # Filter: r > 0.05 and padj < 0.01
    keep_mask = ~corr_df["r"].isna() & (corr_df["r"] > 0.05) & (corr_df["padj"] < 0.01)
    corr_genes = corr_df.index[keep_mask].tolist()

    logger.info(
        "%d/%d secreted proteins kept for signaling patterns.",
        len(corr_genes),
        len(act_new),
    )

    if len(corr_genes) == 0:
        warnings.warn(
            "No secreted proteins passed the correlation filter. "
            "Try adjusting radius or check data quality.",
            stacklevel=2,
        )
        return adata

    # Step 2: NMF
    logger.info("Step 2. NMF")

    # Prepare non-negative matrix (nneg: clip to 0)
    act_nneg = act.loc[corr_genes].clip(lower=0).values

    if isinstance(k, list):
        # Select optimal k by silhouette coefficient
        best_k = k[0]
        best_sil = -1.0
        sil_scores = []
        for ki in k:
            model = NMF(n_components=ki, random_state=seed, max_iter=500)
            W = model.fit_transform(act_nneg)
            labels = W.argmax(axis=1)
            if len(set(labels)) > 1:
                sil = silhouette_score(act_nneg, labels)
            else:
                sil = 0.0
            sil_scores.append(sil)
            if sil > best_sil:
                best_sil = sil
                best_k = ki

        # R uses max drop in silhouette, but we use max silhouette
        k_final = best_k
        logger.info("Optimal k = %d (silhouette = %.3f)", k_final, best_sil)
    else:
        k_final = k

    model = NMF(n_components=k_final, random_state=seed, max_iter=500)
    W = model.fit_transform(act_nneg)
    H = model.components_

    # Create DataFrames matching R's naming
    factor_names = [str(i + 1) for i in range(k_final)]
    weight_W = pd.DataFrame(W, index=corr_genes, columns=factor_names)
    signal_H = pd.DataFrame(H, index=factor_names, columns=act.columns)

    secact_out["pattern"] = {
        "ccc_SP": corr_df,
        "weight_W": weight_W,
        "signal_H": signal_H,
    }

    logger.info("NMF done: %d patterns × %d spots", k_final, signal_H.shape[1])
    return adata


def secact_pattern_genes(
    adata: ad.AnnData,
    n: int,
) -> pd.DataFrame:
    """Enumerate secreted proteins associated with pattern n.

    Equivalent to ``SecAct.signaling.pattern.gene()`` in R.

    Parameters
    ----------
    adata : AnnData
        Must have signaling pattern results.
    n : int
        Pattern number (1-based, matching R convention).

    Returns
    -------
    DataFrame of proteins most associated with pattern n, sorted by weight.
    """
    secact_out = _ensure_secact(adata)
    if "pattern" not in secact_out:
        raise ValueError("Run secact_signaling_patterns() first.")

    weight_W = secact_out["pattern"]["weight_W"].copy()
    n_idx = n - 1  # convert to 0-based

    # R logic: double non-target columns, keep rows where target is max
    temp = weight_W.copy()
    for col_idx in range(temp.shape[1]):
        if col_idx != n_idx:
            temp.iloc[:, col_idx] = 2 * temp.iloc[:, col_idx]

    # Keep rows where pattern n has the max value
    mask = temp.iloc[:, n_idx] == temp.max(axis=1)
    result = weight_W.loc[mask].sort_values(weight_W.columns[n_idx], ascending=False)
    return result


# ---------------------------------------------------------------------------
# 3. Signaling Velocity (spot-level ST)
# ---------------------------------------------------------------------------


def _scalar1(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length (matches R scalar1)."""
    norm = np.sqrt(np.sum(v**2))
    if norm == 0:
        return v
    return v / norm


def secact_signaling_velocity(
    adata: ad.AnnData,
    gene: str,
    signal_mode: str = "receiving",
    scale_factor: float = 1e5,
    radius: float = 200.0,
    sigma: float = 100.0,
) -> dict:
    """Compute signaling velocity arrows for a secreted protein.

    Equivalent to ``SecAct.signaling.velocity.spotST()`` in R.

    The velocity direction starts from source cells producing a secreted
    protein and moves to sink cells receiving the signal. The magnitude
    represents expression × activity product.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct activity results.
    gene : str
        Gene symbol of the secreted protein.
    signal_mode : str
        "receiving" or "sending".
    scale_factor : float
        TPM normalization scale factor.
    radius : float
        Spatial weight radius in micrometers.
    sigma : float
        RBF kernel sigma.

    Returns
    -------
    dict with keys:
        - 'arrows': DataFrame with x_start, y_start, x_end, y_end, vec_len
        - 'points': DataFrame with x, y, value (for background coloring)
        - 'gene': str
        - 'signal_mode': str
    """
    secact_out = _ensure_secact(adata)
    if "SecretedProteinActivity" not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out["SecretedProteinActivity"]["zscore"].copy()
    act = act.clip(lower=0)

    expr = _get_expression_matrix(adata)
    expr.index = _transfer_symbol(expr.index.tolist())
    expr = _rm_duplicates(expr)
    expr = _normalize_tpm(expr, scale_factor)

    # Spatial weights
    weights = cal_weights(adata, radius=radius, sigma=sigma, diag_as_zero=True)
    if sparse.issparse(weights):
        weights_dense = weights.toarray()
    else:
        weights_dense = np.asarray(weights)

    spot_names = adata.obs_names.tolist()
    common_spots = [s for s in spot_names if s in act.columns and s in expr.columns]
    n_spots = len(common_spots)

    act_new = act[common_spots]
    expr_new = expr[common_spots]

    # Build weighted matrix: weights_new[i,j] = weights[i,j] * expr[gene, i] * act[gene, j]
    if gene not in expr_new.index:
        weights_new = np.zeros((n_spots, n_spots))
    else:
        expr_gene = expr_new.loc[gene].values  # (n_spots,)
        # weights_new = weights * expr[gene, :]  (row-wise multiply)
        weights_new = weights_dense[:n_spots, :n_spots] * expr_gene[:, np.newaxis]

    # Then multiply columns by act[gene, :]
    if gene in act_new.index:
        act_gene = act_new.loc[gene].values  # (n_spots,)
        weights_new = weights_new * act_gene[np.newaxis, :]
    else:
        weights_new = np.zeros_like(weights_new)

    # Coordinates
    coords = np.column_stack(
        [
            adata.obs["coordinate_x_um"].values,
            adata.obs["coordinate_y_um"].values,
        ]
    )[:n_spots]

    # Compute arrows
    arrows = []

    if signal_mode == "sending":
        for i in range(n_spots):
            vector_len = np.sum(weights_new[i, :])
            if vector_len == 0:
                continue

            neighbors_mask = weights_new[i, :] > 0
            if not neighbors_mask.any():
                continue

            neighbor_coords = coords[neighbors_mask] - coords[i]
            neighbor_values = weights_new[i, neighbors_mask]

            # Normalize each neighbor direction to unit vector
            norms = np.sqrt(np.sum(neighbor_coords**2, axis=1, keepdims=True))
            norms[norms == 0] = 1
            neighbor_unit = neighbor_coords / norms

            # Weight by value and average
            weighted_dirs = neighbor_unit * neighbor_values[:, np.newaxis]
            avg_dir = np.mean(weighted_dirs, axis=0)
            avg_dir = _scalar1(avg_dir)
            avg_dir = avg_dir * vector_len

            arrows.append(
                {
                    "x_start": coords[i, 0],
                    "y_start": coords[i, 1],
                    "x_change": avg_dir[0],
                    "y_change": avg_dir[1],
                    "x_end": coords[i, 0] + avg_dir[0],
                    "y_end": coords[i, 1] + avg_dir[1],
                    "vec_len": np.sqrt(avg_dir[0] ** 2 + avg_dir[1] ** 2),
                }
            )
    else:  # receiving
        for i in range(n_spots):
            vector_len = np.sum(weights_new[:, i])
            if vector_len == 0:
                continue

            neighbors_mask = weights_new[:, i] > 0
            if not neighbors_mask.any():
                continue

            neighbor_coords = coords[i] - coords[neighbors_mask]
            neighbor_values = weights_new[neighbors_mask, i]

            norms = np.sqrt(np.sum(neighbor_coords**2, axis=1, keepdims=True))
            norms[norms == 0] = 1
            neighbor_unit = neighbor_coords / norms

            weighted_dirs = neighbor_unit * neighbor_values[:, np.newaxis]
            avg_dir = np.mean(weighted_dirs, axis=0)
            avg_dir = _scalar1(avg_dir)
            avg_dir = avg_dir * vector_len

            arrows.append(
                {
                    "x_start": coords[i, 0] - avg_dir[0],
                    "y_start": coords[i, 1] - avg_dir[1],
                    "x_change": avg_dir[0],
                    "y_change": avg_dir[1],
                    "x_end": coords[i, 0],
                    "y_end": coords[i, 1],
                    "vec_len": np.sqrt(avg_dir[0] ** 2 + avg_dir[1] ** 2),
                }
            )

    arrow_df = (
        pd.DataFrame(arrows)
        if arrows
        else pd.DataFrame(
            columns=[
                "x_start",
                "y_start",
                "x_change",
                "y_change",
                "x_end",
                "y_end",
                "vec_len",
            ]
        )
    )

    # Normalize arrow lengths for display (R: scale to max 10)
    if len(arrow_df) > 0:
        max_dx = max(arrow_df["x_change"].abs().max(), 1e-10)
        max_dy = max(arrow_df["y_change"].abs().max(), 1e-10)
        arrow_df["x_change"] = arrow_df["x_change"] * 10 / max_dx
        arrow_df["y_change"] = arrow_df["y_change"] * 10 / max_dy
        arrow_df["x_end"] = arrow_df["x_start"] + arrow_df["x_change"]
        arrow_df["y_end"] = arrow_df["y_start"] + arrow_df["y_change"]

        # Arrow head size: small for weak, large for strong
        arrow_df.loc[arrow_df["vec_len"] < 0.1, "vec_len"] = 0.01
        arrow_df.loc[arrow_df["vec_len"] >= 0.1, "vec_len"] = 0.08

    # Background point values
    if signal_mode == "sending":
        if gene in expr_new.index:
            values = expr_new.loc[gene].values.copy()
            values = np.clip(values, 0, 5)  # R: fig.df[fig.df[,3]>5,3] <- 5
        else:
            values = np.zeros(n_spots)
    else:
        if gene in act_new.index:
            values = act_new.loc[gene].values.copy()
        else:
            values = np.zeros(n_spots)

    points_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": values})

    # Store in adata
    secact_out.setdefault("velocity", {})[gene] = {
        "arrows": arrow_df,
        "points": points_df,
        "signal_mode": signal_mode,
    }

    return {
        "arrows": arrow_df,
        "points": points_df,
        "gene": gene,
        "signal_mode": signal_mode,
    }


# ---------------------------------------------------------------------------
# 4. Spatial Cell-Cell Communication (scST)
# ---------------------------------------------------------------------------


def secact_spatial_ccc(
    adata: ad.AnnData,
    cell_type_col: str,
    scale_factor: float = 1000.0,
    radius: float = 20.0,
    ratio_cutoff: float = 0.2,
    padj_cutoff: float = 0.01,
    n_background: int = 1000,
    seed: int = 123,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Compute spatial cell-cell communication mediated by secreted proteins.

    Equivalent to ``SecAct.CCC.scST()`` in R.

    For each cell-type pair, tests whether neighboring cells communicate
    via secreted proteins (expression × activity > 0) more than expected
    by a permutation background.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct activity results and cell type annotations.
    cell_type_col : str
        Column in adata.obs containing cell type labels.
    scale_factor : float
        TPM normalization scale factor. Default: 1000.
    radius : float
        Neighbor radius in micrometers. Default: 20.
    ratio_cutoff : float
        Minimum ratio of communicating pairs. Default: 0.2.
    padj_cutoff : float
        BH-adjusted p-value cutoff. Default: 0.01.
    n_background : int
        Number of background permutations. Default: 1000.
    seed : int
        Random seed. Default: 123.
    n_jobs : int
        Number of parallel jobs. Default: 1.

    Returns
    -------
    AnnData with CCC results in adata.uns['spacet']['SecAct_output']
    """
    secact_out = _ensure_secact(adata)
    if "SecretedProteinActivity" not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out["SecretedProteinActivity"]["zscore"].copy()
    act = act.clip(lower=0)

    # Expression
    expr = _get_expression_matrix(adata)
    expr.index = _transfer_symbol(expr.index.tolist())
    expr = _rm_duplicates(expr)
    expr = _normalize_tpm(expr, scale_factor)

    # Neighbor graph via KDTree
    logger.info("Step 1. Filtering")
    coords = np.column_stack(
        [adata.obs["coordinate_x_um"].values, adata.obs["coordinate_y_um"].values]
    )
    tree = KDTree(coords)
    pairs = tree.query_pairs(r=radius)

    cell_names = adata.obs_names.tolist()
    cell_types_all = np.array(adata.obs[cell_type_col].values, dtype=str)

    # Binary adjacency
    common_spots = [s for s in cell_names if s in act.columns and s in expr.columns]
    # Remap cell_types to common_spots order (use dict for O(1) lookup)
    name_to_idx = {name: i for i, name in enumerate(cell_names)}
    common_idx = np.array([name_to_idx[s] for s in common_spots], dtype=np.intp)
    cell_types = cell_types_all[common_idx]
    act_new = act[common_spots]
    expr_new = expr[common_spots]
    n_cells = len(common_spots)

    # Remap original indices to common_spots indices
    orig_to_new = {
        name_to_idx[name]: new_idx for new_idx, name in enumerate(common_spots)
    }

    # Build neighbor pair lists remapped to common_spots indices
    i_list, j_list = [], []
    for a, b in pairs:
        if a in orig_to_new and b in orig_to_new:
            na, nb = orig_to_new[a], orig_to_new[b]
            i_list.extend([na, nb])
            j_list.extend([nb, na])
    i_arr = np.array(i_list, dtype=int)
    j_arr = np.array(j_list, dtype=int)

    # Neighbor-aggregated expression for SP filtering
    adj = sparse.csr_matrix(
        (np.ones(len(i_arr)), (i_arr, j_arr)),
        shape=(n_cells, n_cells),
    )

    expr_new_aggr = pd.DataFrame(
        expr_new.values @ adj.toarray(),
        index=expr_new.index,
        columns=common_spots,
    )

    # Filter SPs by Spearman correlation (reuse if available)
    if "ccc_SP" in secact_out and secact_out["ccc_SP"] is not None:
        corr_df = secact_out["ccc_SP"]
    else:
        corr_data = []
        for gene_name in act_new.index:
            act_gene = act_new.loc[gene_name].values
            if gene_name in expr_new.index:
                exp_gene = expr_new_aggr.loc[gene_name].values
                r, p = stats.spearmanr(act_gene, exp_gene)
                corr_data.append({"gene": gene_name, "r": r, "p": p})
            else:
                corr_data.append({"gene": gene_name, "r": np.nan, "p": np.nan})
        corr_df = pd.DataFrame(corr_data).set_index("gene")

        valid_mask = ~corr_df["p"].isna()
        padj = np.full(len(corr_df), np.nan)
        if valid_mask.any():
            from statsmodels.stats.multitest import multipletests

            _, padj_valid, _, _ = multipletests(
                corr_df.loc[valid_mask, "p"].values, method="fdr_bh"
            )
            padj[valid_mask.values] = padj_valid
        corr_df["padj"] = padj

    keep_mask = ~corr_df["r"].isna() & (corr_df["r"] > 0.05) & (corr_df["padj"] < 0.01)
    corr_genes = corr_df.index[keep_mask].tolist()

    logger.info("%d/%d secreted proteins kept for CCC.", len(corr_genes), len(act_new))

    # Step 2: CCC for each cell-type pair
    logger.info("Step 2. CCC")

    unique_types = sorted(set(cell_types))
    cell_groups = {ct: np.where(cell_types == ct)[0] for ct in unique_types}

    # Generate unique cell-type pairs (ct1 > ct2 alphabetically)
    ct_pairs = []
    for ct1 in unique_types:
        for ct2 in unique_types:
            if ct1 > ct2:
                ct_pairs.append((ct1, ct2))

    rng = np.random.RandomState(seed)
    ccc_results = []

    for ct1, ct2 in ct_pairs:
        cells1 = cell_groups[ct1]
        cells2 = cell_groups[ct2]

        # Neighboring pairs of this type
        pair_mask = np.isin(i_arr, cells1) & np.isin(j_arr, cells2)
        pair_i = i_arr[pair_mask]
        pair_j = j_arr[pair_mask]
        n_neighbor = len(pair_i)

        if n_neighbor == 0:
            continue

        # Skip if too few cells neighbor
        unique_i = len(set(pair_i))
        unique_j = len(set(pair_j))
        if unique_i / len(cells1) < 0.05 and unique_j / len(cells2) < 0.05:
            continue

        # Background pairs
        bg_i = rng.choice(cells1, n_neighbor * n_background, replace=True)
        bg_j = rng.choice(cells2, n_neighbor * n_background, replace=True)

        for sp in corr_genes:
            if sp not in expr_new.index or sp not in act_new.index:
                continue

            exp_sp = expr_new.loc[sp].values
            act_sp = act_new.loc[sp].values

            # Direction 1: exp(ct1) * act(ct2)
            ccc_vec = exp_sp[pair_i] * act_sp[pair_j]
            n_comm = np.sum(ccc_vec > 0)
            pos_ratio = n_comm / n_neighbor

            if pos_ratio > ratio_cutoff:
                ccc_raw = np.mean(ccc_vec)
                bg_vec = exp_sp[bg_i] * act_sp[bg_j]
                bg_means = bg_vec.reshape(n_background, n_neighbor).mean(axis=1)
                pv = (np.sum(bg_means >= ccc_raw) + 1) / (n_background + 1)

                ccc_results.append(
                    {
                        "sender": ct1,
                        "secretedProtein": sp,
                        "receiver": ct2,
                        "sender_count": len(cells1),
                        "receiver_count": len(cells2),
                        "neighboringCellPairs": n_neighbor,
                        "communicatingCellPairs": int(n_comm),
                        "ratio": pos_ratio,
                        "pv": pv,
                    }
                )

            # Direction 2: exp(ct2) * act(ct1)
            ccc_vec = exp_sp[pair_j] * act_sp[pair_i]
            n_comm = np.sum(ccc_vec > 0)
            pos_ratio = n_comm / n_neighbor

            if pos_ratio > ratio_cutoff:
                ccc_raw = np.mean(ccc_vec)
                bg_vec = exp_sp[bg_j] * act_sp[bg_i]
                bg_means = bg_vec.reshape(n_background, n_neighbor).mean(axis=1)
                pv = (np.sum(bg_means >= ccc_raw) + 1) / (n_background + 1)

                ccc_results.append(
                    {
                        "sender": ct2,
                        "secretedProtein": sp,
                        "receiver": ct1,
                        "sender_count": len(cells2),
                        "receiver_count": len(cells1),
                        "neighboringCellPairs": n_neighbor,
                        "communicatingCellPairs": int(n_comm),
                        "ratio": pos_ratio,
                        "pv": pv,
                    }
                )

    if ccc_results:
        ccc_df = pd.DataFrame(ccc_results)
        # BH correction
        from statsmodels.stats.multitest import multipletests

        _, pv_adj, _, _ = multipletests(ccc_df["pv"].values, method="fdr_bh")
        ccc_df["pv_adj"] = pv_adj
        ccc_df = ccc_df[ccc_df["pv_adj"] < padj_cutoff]
        ccc_df = ccc_df.sort_values("pv_adj")
    else:
        ccc_df = pd.DataFrame()

    secact_out["ccc_SP"] = corr_df
    secact_out["SecretedProteinCCC"] = ccc_df

    logger.info("CCC done: %d significant interactions.", len(ccc_df))
    return adata


# ---------------------------------------------------------------------------
# 5. Cox Proportional Hazards Regression
# ---------------------------------------------------------------------------


def secact_coxph_regression(
    activity_matrix: pd.DataFrame,
    survival_data: pd.DataFrame,
) -> pd.DataFrame:
    """Cox proportional hazards regression for secreted protein risk scores.

    Equivalent to ``SecAct.coxph.regression()`` in R.

    Parameters
    ----------
    activity_matrix : DataFrame
        Secreted protein activity matrix (proteins × samples).
    survival_data : DataFrame
        Must have columns 'Time' and 'Event' (0/1), indexed by sample ID.

    Returns
    -------
    DataFrame with columns 'risk_score_z' and 'p_value', indexed by protein.
    """
    from lifelines import CoxPHFitter

    # Transpose: R does t(mat) → samples × proteins
    mat = activity_matrix.T

    # Intersect samples
    common = mat.index.intersection(survival_data.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping sample IDs between activity and survival data."
        )

    X = survival_data.loc[common]
    Y = mat.loc[common]

    results = []
    for protein in Y.columns:
        comb = X[["Time", "Event"]].copy()
        comb["Act"] = Y[protein].values

        try:
            cph = CoxPHFitter()
            cph.fit(comb, duration_col="Time", event_col="Event")
            z = cph.summary.loc["Act", "z"]
            p = cph.summary.loc["Act", "p"]
            results.append({"protein": protein, "risk_score_z": z, "p_value": p})
        except Exception:
            results.append(
                {"protein": protein, "risk_score_z": np.nan, "p_value": np.nan}
            )

    return pd.DataFrame(results).set_index("protein")


# ---------------------------------------------------------------------------
# 6. Survival Plot Data (Kaplan-Meier)
# ---------------------------------------------------------------------------


def secact_survival_data(
    activity_matrix: pd.DataFrame,
    survival_data: pd.DataFrame,
    protein: str,
    cutoff: str = "median",
) -> dict:
    """Prepare Kaplan-Meier survival data for a secreted protein.

    Parameters
    ----------
    activity_matrix : DataFrame
        Secreted protein activity matrix (proteins × samples).
    survival_data : DataFrame
        Must have columns 'Time' and 'Event'.
    protein : str
        Protein name.
    cutoff : str
        How to split groups: "median" or "tertile".

    Returns
    -------
    dict with 'high' and 'low' DataFrames (Time, Event columns),
    and 'logrank_p' (log-rank test p-value).
    """
    mat = activity_matrix.T
    common = mat.index.intersection(survival_data.index)

    values = mat.loc[common, protein]
    surv = survival_data.loc[common, ["Time", "Event"]].copy()

    if cutoff == "median":
        threshold = values.median()
        high_mask = values >= threshold
    elif cutoff == "tertile":
        q33 = values.quantile(1 / 3)
        q66 = values.quantile(2 / 3)
        high_mask = values >= q66
        surv = surv.loc[high_mask | (values <= q33)]
        high_mask = values.loc[surv.index] >= q66
    else:
        raise ValueError(f"cutoff must be 'median' or 'tertile', got '{cutoff}'")

    high_group = surv.loc[high_mask]
    low_group = surv.loc[~high_mask]

    # Log-rank test
    from lifelines.statistics import logrank_test

    lr = logrank_test(
        high_group["Time"],
        low_group["Time"],
        event_observed_A=high_group["Event"],
        event_observed_B=low_group["Event"],
    )

    return {
        "high": high_group,
        "low": low_group,
        "logrank_p": lr.p_value,
        "protein": protein,
    }
