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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import KDTree

from spatialgpu.deconvolution._keys import (
    KEY_PATTERN,
    KEY_SECACT,
    KEY_SECRETED_PROTEIN_ACTIVITY,
    UNS_SPACET,
)
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
    if UNS_SPACET not in adata.uns:
        adata.uns[UNS_SPACET] = {}
    spacet = adata.uns[UNS_SPACET]
    if KEY_SECACT not in spacet:
        spacet[KEY_SECACT] = {}
    return spacet[KEY_SECACT]


def _get_expression_matrix(adata: ad.AnnData) -> pd.DataFrame:
    """Extract raw counts as DataFrame (genes × spots)."""
    X = adata.X
    if sparse.issparse(X):
        dense_gb = X.shape[0] * X.shape[1] * 8 / 1e9
        if dense_gb > 4:
            warnings.warn(
                f"Densifying {X.shape} sparse expression matrix ({dense_gb:.1f} GB). "
                f"Consider subsetting genes.",
                ResourceWarning,
                stacklevel=2,
            )
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
    cell_type_col: str | None = None,
    is_group_sig: bool | None = None,
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

    secact_out[KEY_SECRETED_PROTEIN_ACTIVITY] = result
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
    if KEY_SECRETED_PROTEIN_ACTIVITY not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out[KEY_SECRETED_PROTEIN_ACTIVITY]["zscore"].copy()
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

    # Use sparse matmul to avoid dense n×n weight matrix
    if sparse.issparse(weights):
        expr_new_aggr = (
            (expr_new.values @ weights).A
            if sparse.issparse(expr_new.values @ weights)
            else expr_new.values @ weights
        )
    else:
        expr_new_aggr = expr_new.values @ np.asarray(weights)

    expr_new_aggr = pd.DataFrame(
        expr_new_aggr, index=expr_new.index, columns=common_spots
    )

    # Spearman correlation for each secreted protein
    from spatialgpu.core.backend import get_backend as _get_backend

    _backend = _get_backend()

    if _backend.is_gpu_active:
        import cupy as cp

        from spatialgpu.core.gpu_ops import gpu_rankdata

        # Vectorized Spearman for all genes at once
        _common_genes = [g for g in act_new.index if g in expr_new_aggr.index]
        _other_genes = [g for g in act_new.index if g not in expr_new_aggr.index]

        if _common_genes:
            _act_vals = cp.asarray(act_new.loc[_common_genes].values)
            _exp_vals = cp.asarray(expr_new_aggr.loc[_common_genes].values)
            _act_ranked = gpu_rankdata(_act_vals, method="average", axis=1)
            _exp_ranked = gpu_rankdata(_exp_vals, method="average", axis=1)
            _act_c = _act_ranked - _act_ranked.mean(axis=1, keepdims=True)
            _exp_c = _exp_ranked - _exp_ranked.mean(axis=1, keepdims=True)
            _num = (_act_c * _exp_c).sum(axis=1)
            _den = cp.sqrt((_act_c**2).sum(axis=1) * (_exp_c**2).sum(axis=1))
            _den = cp.where(_den == 0, 1.0, _den)
            _r_vals = cp.asnumpy(_num / _den)
            _n_obs = _act_vals.shape[1]
            _t_vals = _r_vals * np.sqrt((_n_obs - 2) / (1 - _r_vals**2 + 1e-300))
            _p_vals = 2 * stats.t.sf(np.abs(_t_vals), df=_n_obs - 2)
            corr_data = [
                {"gene": g, "r": _r_vals[i], "p": _p_vals[i]}
                for i, g in enumerate(_common_genes)
            ]
        else:
            corr_data = []
        corr_data.extend([{"gene": g, "r": np.nan, "p": np.nan} for g in _other_genes])
    else:
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

    if _backend.is_gpu_active:
        import cupy as cp

        from spatialgpu.core.gpu_ops import gpu_nmf

        _act_nneg_gpu = cp.asarray(act_nneg)

        if isinstance(k, list):
            best_k, best_sil = k[0], -1.0
            for ki in k:
                _W_gpu, _H_gpu = gpu_nmf(
                    _act_nneg_gpu, n_components=ki, seed=seed, max_iter=500
                )
                _labels = cp.asnumpy(_W_gpu.argmax(axis=1))
                _sil = (
                    silhouette_score(act_nneg, _labels)
                    if len(set(_labels)) > 1
                    else 0.0
                )
                if _sil > best_sil:
                    best_sil, best_k = _sil, ki
            k_final = best_k
            logger.info("Optimal k = %d (silhouette = %.3f)", k_final, best_sil)
        else:
            k_final = k

        _W_gpu, _H_gpu = gpu_nmf(
            _act_nneg_gpu, n_components=k_final, seed=seed, max_iter=500
        )
        W, H = cp.asnumpy(_W_gpu), cp.asnumpy(_H_gpu)
    else:
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

    secact_out[KEY_PATTERN] = {
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
    if KEY_PATTERN not in secact_out:
        raise ValueError("Run secact_signaling_patterns() first.")

    weight_W = secact_out[KEY_PATTERN]["weight_W"].copy()
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


def _prepare_velocity_data(
    adata: ad.AnnData,
    scale_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract clipped activity z-scores and normalized expression.

    Returns (act, expr) where act has negatives clipped to 0 and
    expr is TPM-normalized + log2-transformed.
    """
    secact_out = _ensure_secact(adata)
    if KEY_SECRETED_PROTEIN_ACTIVITY not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out[KEY_SECRETED_PROTEIN_ACTIVITY]["zscore"].copy()
    act = act.clip(lower=0)

    expr = _get_expression_matrix(adata)
    expr.index = _transfer_symbol(expr.index.tolist())
    expr = _rm_duplicates(expr)
    expr = _normalize_tpm(expr, scale_factor)

    return act, expr


def _densify_weights(weights, is_gpu: bool):
    """Convert spatial weight matrix to dense (GPU or CPU).

    Parameters
    ----------
    weights : sparse or dense array
        Spatial weight matrix from ``cal_weights``.
    is_gpu : bool
        Whether GPU backend is active.

    Returns
    -------
    Dense weight matrix (cupy array on GPU, numpy array on CPU).
    """
    if is_gpu:
        import cupy as cp

        if sparse.issparse(weights):
            return cp.asarray(weights.toarray())
        return cp.asarray(weights)

    if sparse.issparse(weights):
        dense_gb = weights.shape[0] * weights.shape[1] * 8 / 1e9
        if dense_gb > 4:
            warnings.warn(
                f"Densifying {weights.shape} weight matrix ({dense_gb:.1f} GB). "
                f"Consider reducing spot count.",
                ResourceWarning,
                stacklevel=2,
            )
        return weights.toarray()
    return np.asarray(weights)


def _build_weighted_matrix(
    gene: str,
    expr_new: pd.DataFrame,
    act_new: pd.DataFrame,
    weights_dense,
    n_spots: int,
    is_gpu: bool,
) -> np.ndarray:
    """Build weighted matrix: W[i,j] = weights[i,j] * expr[gene,i] * act[gene,j].

    Returns a numpy (n_spots, n_spots) array regardless of backend.
    """
    if gene not in expr_new.index:
        return np.zeros((n_spots, n_spots))

    expr_gene = expr_new.loc[gene].values
    w_sub = weights_dense[:n_spots, :n_spots]

    if is_gpu:
        import cupy as cp

        expr_gpu = cp.asarray(expr_gene)
        result_gpu = w_sub * expr_gpu[:, cp.newaxis]
        if gene in act_new.index:
            act_gpu = cp.asarray(act_new.loc[gene].values)
            result_gpu = result_gpu * act_gpu[cp.newaxis, :]
        else:
            result_gpu = cp.zeros_like(result_gpu)
        return cp.asnumpy(result_gpu)

    # CPU path
    result = w_sub * expr_gene[:, np.newaxis]
    if gene in act_new.index:
        result = result * act_new.loc[gene].values[np.newaxis, :]
    else:
        result = np.zeros_like(result)
    return result


def _compute_velocity_arrows(
    weights_new: np.ndarray,
    coords: np.ndarray,
    n_spots: int,
    signal_mode: str,
) -> pd.DataFrame:
    """Compute source-to-sink velocity arrows for each spot.

    Parameters
    ----------
    weights_new : ndarray (n_spots, n_spots)
        Gene-weighted spatial matrix.
    coords : ndarray (n_spots, 2)
        Spot spatial coordinates.
    n_spots : int
        Number of spots.
    signal_mode : str
        "sending" or "receiving".

    Returns
    -------
    DataFrame with columns x_start, y_start, x_change, y_change,
    x_end, y_end, vec_len.
    """
    _EMPTY_COLS = [
        "x_start",
        "y_start",
        "x_change",
        "y_change",
        "x_end",
        "y_end",
        "vec_len",
    ]
    arrows = []

    for i in range(n_spots):
        if signal_mode == "sending":
            w_slice = weights_new[i, :]
        else:
            w_slice = weights_new[:, i]

        vector_len = np.sum(w_slice)
        if vector_len == 0:
            continue

        neighbors_mask = w_slice > 0
        if not neighbors_mask.any():
            continue

        if signal_mode == "sending":
            neighbor_coords = coords[neighbors_mask] - coords[i]
        else:
            neighbor_coords = coords[i] - coords[neighbors_mask]
        neighbor_values = w_slice[neighbors_mask]

        # Normalize each neighbor direction to unit vector
        norms = np.sqrt(np.sum(neighbor_coords**2, axis=1, keepdims=True))
        norms[norms == 0] = 1
        neighbor_unit = neighbor_coords / norms

        # Weight by value and average
        weighted_dirs = neighbor_unit * neighbor_values[:, np.newaxis]
        avg_dir = np.mean(weighted_dirs, axis=0)
        avg_dir = _scalar1(avg_dir) * vector_len

        if signal_mode == "sending":
            x_start, y_start = coords[i, 0], coords[i, 1]
            x_end = x_start + avg_dir[0]
            y_end = y_start + avg_dir[1]
        else:
            x_end, y_end = coords[i, 0], coords[i, 1]
            x_start = x_end - avg_dir[0]
            y_start = y_end - avg_dir[1]

        arrows.append(
            {
                "x_start": x_start,
                "y_start": y_start,
                "x_change": avg_dir[0],
                "y_change": avg_dir[1],
                "x_end": x_end,
                "y_end": y_end,
                "vec_len": np.sqrt(avg_dir[0] ** 2 + avg_dir[1] ** 2),
            }
        )

    if not arrows:
        return pd.DataFrame(columns=_EMPTY_COLS)
    return pd.DataFrame(arrows)


def _normalize_arrows(arrow_df: pd.DataFrame) -> pd.DataFrame:
    """Scale arrow lengths for display (R convention: max displacement 10).

    Modifies the DataFrame in place and returns it.
    """
    if len(arrow_df) == 0:
        return arrow_df

    max_dx = max(arrow_df["x_change"].abs().max(), 1e-10)
    max_dy = max(arrow_df["y_change"].abs().max(), 1e-10)
    arrow_df["x_change"] = arrow_df["x_change"] * 10 / max_dx
    arrow_df["y_change"] = arrow_df["y_change"] * 10 / max_dy
    arrow_df["x_end"] = arrow_df["x_start"] + arrow_df["x_change"]
    arrow_df["y_end"] = arrow_df["y_start"] + arrow_df["y_change"]

    # Arrow head size: small for weak, large for strong
    arrow_df.loc[arrow_df["vec_len"] < 0.1, "vec_len"] = 0.01
    arrow_df.loc[arrow_df["vec_len"] >= 0.1, "vec_len"] = 0.08
    return arrow_df


def _compute_background_points(
    gene: str,
    signal_mode: str,
    expr_new: pd.DataFrame,
    act_new: pd.DataFrame,
    coords: np.ndarray,
    n_spots: int,
) -> pd.DataFrame:
    """Build background point values for velocity plot coloring.

    For "sending" mode, uses expression (clipped to [0, 5]).
    For "receiving" mode, uses activity z-score.
    """
    if signal_mode == "sending":
        if gene in expr_new.index:
            values = np.clip(expr_new.loc[gene].values, 0, 5)
        else:
            values = np.zeros(n_spots)
    else:
        if gene in act_new.index:
            values = act_new.loc[gene].values
        else:
            values = np.zeros(n_spots)

    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": values})


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
    from spatialgpu.core.backend import get_backend as _get_backend

    # 1. Prepare data
    act, expr = _prepare_velocity_data(adata, scale_factor)

    spot_names = adata.obs_names.tolist()
    common_spots = [s for s in spot_names if s in act.columns and s in expr.columns]
    n_spots = len(common_spots)
    act_new = act[common_spots]
    expr_new = expr[common_spots]

    # 2. Spatial weights → dense
    _backend = _get_backend()
    weights = cal_weights(adata, radius=radius, sigma=sigma, diag_as_zero=True)
    weights_dense = _densify_weights(weights, is_gpu=_backend.is_gpu_active)

    # 3. Gene-weighted matrix: W[i,j] = weights[i,j] * expr[gene,i] * act[gene,j]
    weights_new = _build_weighted_matrix(
        gene,
        expr_new,
        act_new,
        weights_dense,
        n_spots,
        is_gpu=_backend.is_gpu_active,
    )

    # 4. Compute velocity arrows
    coords = np.column_stack(
        [
            adata.obs["coordinate_x_um"].values,
            adata.obs["coordinate_y_um"].values,
        ]
    )[:n_spots]

    arrow_df = _compute_velocity_arrows(weights_new, coords, n_spots, signal_mode)
    arrow_df = _normalize_arrows(arrow_df)

    # 5. Background points for coloring
    points_df = _compute_background_points(
        gene,
        signal_mode,
        expr_new,
        act_new,
        coords,
        n_spots,
    )

    # 6. Store and return
    secact_out = _ensure_secact(adata)
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


def secact_signaling_velocity_scst(
    adata: ad.AnnData,
    sender: str,
    secreted_protein: str,
    receiver: str,
    cell_type_col: str,
    scale_factor: float = 1e5,
    radius: float = 20.0,
) -> dict:
    """Compute single-cell resolution signaling velocity arrows.

    Equivalent to ``SecAct.signaling.velocity.scST()`` in R.

    Draws arrows from each sender cell to neighbouring receiver cells
    where the sender expresses the protein (count > 0) and the receiver
    has positive signalling activity (zscore > 0).

    Parameters
    ----------
    adata : AnnData
        Must have SecAct activity results and cell type annotations.
    sender : str
        Sender cell type name.
    secreted_protein : str
        Gene symbol of the secreted protein.
    receiver : str
        Receiver cell type name.
    cell_type_col : str
        Column in ``adata.obs`` with cell type labels.
    scale_factor : float
        TPM normalisation scale factor. Default: 1e5.
    radius : float
        Neighbour radius in micrometers. Default: 20.

    Returns
    -------
    dict with keys:
        - 'arrows': DataFrame (x_start, y_start, x_end, y_end, vec_len)
        - 'cell_types': DataFrame (x, y, cell_type) for all cells
        - 'sender', 'receiver', 'secreted_protein': str
    """
    secact_out = _ensure_secact(adata)
    if KEY_SECRETED_PROTEIN_ACTIVITY not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out[KEY_SECRETED_PROTEIN_ACTIVITY]["zscore"].copy()
    act = act.clip(lower=0)

    expr = _get_expression_matrix(adata)
    expr.index = _transfer_symbol(expr.index.tolist())
    expr = _rm_duplicates(expr)

    coords = np.column_stack(
        [adata.obs["coordinate_x_um"].values, adata.obs["coordinate_y_um"].values]
    )
    cell_types = adata.obs[cell_type_col].values

    sender_mask = cell_types == sender
    receiver_mask = cell_types == receiver
    sender_idx = np.where(sender_mask)[0]
    receiver_idx = np.where(receiver_mask)[0]

    empty_arrows = pd.DataFrame(
        columns=["x_start", "y_start", "x_end", "y_end", "vec_len"]
    )

    if (
        len(sender_idx) == 0
        or len(receiver_idx) == 0
        or secreted_protein not in expr.index
        or secreted_protein not in act.index
    ):
        arrows = empty_arrows
    else:
        # Vectorised: query sender coords against receiver coords
        sender_tree = KDTree(coords[sender_idx])
        receiver_tree = KDTree(coords[receiver_idx])
        pairs_sr = sender_tree.query_ball_tree(receiver_tree, r=radius)

        # Build index arrays
        s_list, r_list = [], []
        for si, receivers in enumerate(pairs_sr):
            if receivers:
                s_list.extend([si] * len(receivers))
                r_list.extend(receivers)

        if not s_list:
            arrows = empty_arrows
        else:
            s_arr = np.array(s_list)  # indices into sender_idx
            r_arr = np.array(r_list)  # indices into receiver_idx

            # Map to global cell indices
            s_global = sender_idx[s_arr]
            r_global = receiver_idx[r_arr]

            # Vectorised expression / activity check
            cell_names = adata.obs_names
            expr_vals = expr.loc[secreted_protein, cell_names[s_global]].values
            act_vals = act.loc[secreted_protein, cell_names[r_global]].values
            valid = (expr_vals > 0) & (act_vals > 0)

            if valid.sum() == 0:
                arrows = empty_arrows
            else:
                s_valid = s_global[valid]
                r_valid = r_global[valid]
                arrows = pd.DataFrame(
                    {
                        "x_start": coords[s_valid, 0],
                        "y_start": coords[s_valid, 1],
                        "x_end": coords[r_valid, 0],
                        "y_end": coords[r_valid, 1],
                        "vec_len": 1.0,
                    }
                )

    # Cell type DataFrame for plotting (collapse non-sender/receiver to "Other")
    ct_display = np.array(
        [ct if ct in (sender, receiver) else "Other" for ct in cell_types]
    )
    cell_df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "cell_type": ct_display}
    )

    result = {
        "arrows": arrows,
        "cell_types": cell_df,
        "sender": sender,
        "receiver": receiver,
        "secreted_protein": secreted_protein,
    }

    # Store in adata
    secact_out.setdefault("velocity_scst", {})[
        f"{sender}_{secreted_protein}_{receiver}"
    ] = result

    return result


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
    if KEY_SECRETED_PROTEIN_ACTIVITY not in secact_out:
        raise ValueError("Run secact_inference() first.")

    act = secact_out[KEY_SECRETED_PROTEIN_ACTIVITY]["zscore"].copy()
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
        expr_new.values @ adj,
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
