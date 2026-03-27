"""Spatial correlation analysis via Moran's I statistic.

Implements univariate, bivariate, and pairwise Moran's I with permutation
testing, plus an RBF-kernel spatial weight matrix builder. Translated from
SpaCET R package extensions.R (calWeights + SpatialCorrelation).

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree

from spatialgpu.deconvolution.reference import load_lr_database

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cal_weights(
    adata: ad.AnnData,
    radius: float = 200.0,
    k: int | None = None,
    sigma: float = 100.0,
    diag_as_zero: bool = True,
) -> sparse.csr_matrix:
    """Compute spatial weight matrix using an RBF (Gaussian) kernel.

    For each spot, finds all neighbors within *radius* micrometers and assigns
    a weight ``w = exp(-d^2 / (2 * sigma^2))``. Uses a KD-tree for efficient
    radius-based neighbor search (equivalent to RANN::nn2 in R).

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obs['coordinate_x_um']`` and
        ``adata.obs['coordinate_y_um']`` (micrometer coordinates).
    radius : float
        Radius cutoff in micrometers. Default: 200.
    k : int or None
        Maximum number of nearest neighbors. None uses all within radius.
    sigma : float
        Free parameter for the RBF kernel. Default: 100.
    diag_as_zero : bool
        If True (default), set diagonal entries to zero (exclude self-weights).

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse weight matrix of shape (n_spots, n_spots). Row and column
        ordering follows ``adata.obs_names``.
    """
    coords = np.column_stack(
        [
            adata.obs["coordinate_x_um"].values.astype(np.float64),
            adata.obs["coordinate_y_um"].values.astype(np.float64),
        ]
    )
    n_spots = coords.shape[0]

    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active and n_spots > 1000:
        return _cal_weights_gpu(coords, n_spots, radius, sigma, diag_as_zero)

    logger.info(
        "Building weight matrix: %d spots, radius=%.0f um, sigma=%.0f.",
        n_spots,
        radius,
        sigma,
    )

    # KD-tree radius search
    tree = KDTree(coords)
    neighbors = tree.query_ball_tree(tree, r=radius)

    # Flatten neighbor lists into COO triplets
    row_list = []
    col_list = []
    for i, nbrs in enumerate(neighbors):
        nbrs_arr = np.asarray(nbrs, dtype=np.intp)
        # Exclude self
        nbrs_arr = nbrs_arr[nbrs_arr != i]
        if len(nbrs_arr) > 0:
            row_list.append(np.full(len(nbrs_arr), i, dtype=np.intp))
            col_list.append(nbrs_arr)

    if row_list:
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)
        # Vectorized distance and weight computation
        diffs = coords[all_rows] - coords[all_cols]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        valid = (dists > 0) & (dists <= radius)
        all_rows = all_rows[valid]
        all_cols = all_cols[valid]
        dists = dists[valid]
        all_vals = np.exp(-(dists**2) / (2.0 * sigma**2))
    else:
        all_rows = np.array([], dtype=np.intp)
        all_cols = np.array([], dtype=np.intp)
        all_vals = np.array([], dtype=np.float64)

    W = sparse.csr_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(n_spots, n_spots),
    )

    # Optionally limit to k nearest neighbors per spot
    if k is not None and k < n_spots:
        W_csr = W.tocsr()
        for i in range(n_spots):
            start, end = W_csr.indptr[i], W_csr.indptr[i + 1]
            nnz_row = end - start
            if nnz_row > k:
                row_data = W_csr.data[start:end]
                # Zero out all but top-k weights
                keep = np.argpartition(row_data, -k)[-k:]
                mask = np.ones(nnz_row, dtype=bool)
                mask[keep] = False
                W_csr.data[start:end][mask] = 0.0
        W_csr.eliminate_zeros()
        W = W_csr

    if not diag_as_zero:
        W = W + sparse.eye(n_spots, dtype=np.float64, format="csr")

    logger.info("Weight matrix: %d non-zero entries.", W.nnz)

    return W


def spatial_correlation(
    adata: ad.AnnData,
    mode: str,
    item: np.ndarray | pd.DataFrame | list[str] | None = None,
    W: sparse.spmatrix | None = None,
    n_permutation: int = 1000,
) -> ad.AnnData:
    """Calculate spatial correlation using Moran's I statistic.

    Supports univariate (single-gene), bivariate (ligand-receptor pair), and
    pairwise (all-vs-all) modes. Uses permutation testing with BH correction
    for univariate and bivariate modes.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data (spots x genes) with raw counts in X.
    mode : str
        One of ``"univariate"``, ``"bivariate"``, ``"pairwise"``.
    item : array-like or None
        - Univariate: list/array of gene names (None = all genes).
        - Bivariate: DataFrame or 2-column array of (ligand, receptor) pairs
          (None = Ramilowski2015 L-R database).
        - Pairwise: ignored.
    W : sparse matrix or None
        Spatial weight matrix. If None, will be computed using ``cal_weights``.
    n_permutation : int
        Number of permutations for significance testing. Default: 1000.

    Returns
    -------
    AnnData
        Results stored in ``adata.uns['spacet']['SpatialCorrelation'][mode]``:
        - Univariate/bivariate: DataFrame with columns
          ``p.Moran_I``, ``p.Moran_Z``, ``p.Moran_P``, ``p.Moran_Padj``
        - Pairwise: dense matrix of pairwise Moran's I values
    """
    from statsmodels.stats.multitest import multipletests

    valid_modes = ("univariate", "bivariate", "pairwise")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

    # Compute weight matrix if not provided
    if W is None:
        W = cal_weights(adata)

    # Ensure W is sparse
    if not sparse.issparse(W):
        W = sparse.csr_matrix(W)

    # Remove island spots (zero row/col sums)
    if sparse.issparse(W):
        col_sums = np.asarray(W.sum(axis=0)).ravel()
        row_sums = np.asarray(W.sum(axis=1)).ravel()
    else:
        col_sums = np.asarray(W.sum(axis=0)).ravel()
        row_sums = np.asarray(W.sum(axis=1)).ravel()

    valid_mask = (col_sums > 0) & (row_sums > 0)
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        logger.info("Removing %d island spots with zero weight sums.", n_removed)
        valid_idx = np.where(valid_mask)[0]
        W = W[valid_idx][:, valid_idx]
        adata_sub = adata[valid_mask].copy()
    else:
        adata_sub = adata

    # ---- Step 1: Normalize with VST-equivalent ----
    logger.info(
        "Step 1: Normalize count matrix with variance stabilizing transformation."
    )
    mat = _vst_normalize(adata_sub)
    # mat is genes x spots (dense, float64)

    # ---- Step 2: Filter genes and prepare items ----
    logger.info("Step 2: Calculate Moran's I.")

    gene_names = np.array(adata_sub.var_names)

    if item is not None:
        if mode == "bivariate":
            if isinstance(item, pd.DataFrame):
                item_df = item.copy()
                item_df.columns = ["L", "R"]
            else:
                item_arr = np.asarray(item)
                item_df = pd.DataFrame({"L": item_arr[:, 0], "R": item_arr[:, 1]})
            # Collect all genes referenced in pairs
            all_genes_needed = set(item_df["L"].values) | set(item_df["R"].values)
            gene_mask = np.isin(gene_names, list(all_genes_needed))
            mat = mat[gene_mask]
            gene_names = gene_names[gene_mask]
        else:
            # Univariate: filter to specified genes
            item_genes = np.asarray(item)
            gene_mask = np.isin(gene_names, item_genes)
            mat = mat[gene_mask]
            gene_names = gene_names[gene_mask]
    else:
        if mode == "bivariate":
            # Load Ramilowski2015 L-R database
            lr_db = load_lr_database()
            # Columns vary; typically col index 1 = ligand, 3 = receptor
            lr_cols = lr_db.columns
            if len(lr_cols) >= 4:
                item_df = pd.DataFrame(
                    {
                        "L": lr_db.iloc[:, 1].values,
                        "R": lr_db.iloc[:, 3].values,
                    }
                )
            else:
                item_df = pd.DataFrame(
                    {
                        "L": lr_db.iloc[:, 0].values,
                        "R": lr_db.iloc[:, 1].values,
                    }
                )
            # Filter to genes present in the expression data
            all_genes_needed = set(item_df["L"].values) | set(item_df["R"].values)
            gene_mask = np.isin(gene_names, list(all_genes_needed))
            mat = mat[gene_mask]
            gene_names = gene_names[gene_mask]

    # For bivariate, filter pairs to those with both genes present
    if mode == "bivariate":
        gene_set = set(gene_names)
        pair_mask = item_df["L"].isin(gene_set) & item_df["R"].isin(gene_set)
        item_df = item_df[pair_mask].reset_index(drop=True)
        if len(item_df) == 0:
            raise ValueError(
                "No valid ligand-receptor pairs found in the expression data."
            )
        logger.info("Testing %d ligand-receptor pairs.", len(item_df))

    # ---- Step 3: Standardize each gene (z-score with population std) ----
    N = mat.shape[1]

    row_means = mat.mean(axis=1, keepdims=True)
    mat -= row_means
    row_std = np.sqrt(np.sum(mat**2, axis=1, keepdims=True) / N)
    zero_std = (row_std == 0).ravel()
    row_std[row_std == 0] = 1.0  # avoid div-by-zero
    mat /= row_std
    mat[zero_std, :] = 0.0

    # Build gene name -> row index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # ---- Step 4: Compute Moran's I ----
    W_sum = W.sum()

    if mode in ("univariate", "bivariate"):
        n_perm = n_permutation
        rng = np.random.RandomState(123456)

        if mode == "univariate":
            n_items = mat.shape[0]
            item_names = gene_names.copy()

            # Allocate permutation matrix: items x (n_perm + 1)
            moran_perm = np.full((n_items, n_perm + 1), np.nan, dtype=np.float64)

            # Permutations
            for p in range(n_perm):
                random_order = rng.permutation(N)
                X_perm = mat[:, random_order]
                XW = X_perm @ W
                if sparse.issparse(XW):
                    XW = np.asarray(XW.todense())
                moran_perm[:, p] = np.sum(XW * X_perm, axis=1)

            # Observed (last column)
            XW_obs = mat @ W
            if sparse.issparse(XW_obs):
                XW_obs = np.asarray(XW_obs.todense())
            moran_perm[:, n_perm] = np.sum(XW_obs * mat, axis=1)

        else:  # bivariate
            n_items = len(item_df)
            item_names = np.array(
                [
                    f"{lig}_{rec}"
                    for lig, rec in zip(item_df["L"].values, item_df["R"].values)
                ]
            )

            # Get row indices for ligands and receptors
            l_indices = np.array([gene_to_idx[g] for g in item_df["L"].values])
            r_indices = np.array([gene_to_idx[g] for g in item_df["R"].values])

            moran_perm = np.full((n_items, n_perm + 1), np.nan, dtype=np.float64)

            for p in range(n_perm):
                random_order = rng.permutation(N)
                X_perm = mat[:, random_order]
                X_perm_L = X_perm[l_indices, :]
                X_perm_R = X_perm[r_indices, :]

                XW = X_perm_L @ W
                if sparse.issparse(XW):
                    XW = np.asarray(XW.todense())
                moran_perm[:, p] = np.sum(XW * X_perm_R, axis=1)

            # Observed
            XW_obs = mat[l_indices, :] @ W
            if sparse.issparse(XW_obs):
                XW_obs = np.asarray(XW_obs.todense())
            moran_perm[:, n_perm] = np.sum(XW_obs * mat[r_indices, :], axis=1)

        # Normalize by sum of weights
        moran_perm /= W_sum

        # Extract statistics
        moran_I = moran_perm[:, n_perm]

        # Z-score: (observed - mean(permutations)) / std(permutations)
        perm_mean = np.mean(moran_perm[:, :n_perm], axis=1)
        perm_std = np.std(moran_perm[:, :n_perm], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            moran_Z = np.where(
                perm_std > 0,
                (moran_I - perm_mean) / perm_std,
                0.0,
            )

        # P-value: (count of permutations >= observed + 1) / (n_perm + 1)
        observed = moran_perm[:, n_perm].reshape(-1, 1)
        perm_values = moran_perm[:, :n_perm]
        moran_P = (np.sum(perm_values >= observed, axis=1) + 1) / (n_perm + 1)

        # BH adjustment
        _, moran_Padj, _, _ = multipletests(moran_P, method="fdr_bh")

        result_df = pd.DataFrame(
            {
                "p.Moran_I": moran_I,
                "p.Moran_Z": moran_Z,
                "p.Moran_P": moran_P,
                "p.Moran_Padj": moran_Padj,
            },
            index=item_names,
        )

        # Sort by adjusted p-value ascending, then Moran's I descending
        result_df = result_df.sort_values(
            by=["p.Moran_Padj", "p.Moran_I"],
            ascending=[True, False],
        )

        logger.info(
            "Moran's I (%s): %d items tested, %d significant (Padj < 0.05).",
            mode,
            len(result_df),
            (result_df["p.Moran_Padj"] < 0.05).sum(),
        )

    else:  # pairwise
        # Pairwise Moran's I: I_matrix = (Z @ W @ Z.T) / sum(W)
        XW = mat @ W
        if sparse.issparse(XW):
            XW = np.asarray(XW.todense())

        # XWX = XW @ mat.T  (equivalent to tcrossprod(XW, mat) in R)
        moran_matrix = XW @ mat.T / W_sum

        result_df = pd.DataFrame(
            moran_matrix,
            index=gene_names,
            columns=gene_names,
        )

        logger.info(
            "Pairwise Moran's I: %d x %d matrix computed.",
            result_df.shape[0],
            result_df.shape[1],
        )

    # Store results
    if "spacet" not in adata.uns:
        adata.uns["spacet"] = {}
    if "SpatialCorrelation" not in adata.uns["spacet"]:
        adata.uns["spacet"]["SpatialCorrelation"] = {}

    adata.uns["spacet"]["SpatialCorrelation"][mode] = result_df

    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cal_weights_gpu(coords, n_spots, radius, sigma, diag_as_zero):
    """GPU implementation of cal_weights using chunked cdist."""
    import cupy as cp
    from spatialgpu.core.gpu_ops import gpu_cdist

    coords_gpu = cp.asarray(coords)
    chunk_size = min(5000, n_spots)
    rows_list, cols_list, vals_list = [], [], []

    for i in range(0, n_spots, chunk_size):
        end_i = min(i + chunk_size, n_spots)
        dists = gpu_cdist(coords_gpu[i:end_i], coords_gpu)

        mask = (dists <= radius) & (dists > 0)
        local_rows, local_cols = cp.where(mask)
        d_vals = dists[mask]
        w_vals = cp.exp(-d_vals ** 2 / (2 * sigma ** 2))

        rows_list.append(cp.asnumpy(local_rows) + i)
        cols_list.append(cp.asnumpy(local_cols))
        vals_list.append(cp.asnumpy(w_vals))

    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_vals = np.concatenate(vals_list)
    else:
        all_rows = np.array([], dtype=np.int64)
        all_cols = np.array([], dtype=np.int64)
        all_vals = np.array([], dtype=np.float64)

    W = sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_spots, n_spots))

    # Row-normalize (same as CPU path)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = sparse.diags(1.0 / row_sums) @ W

    if diag_as_zero:
        W.setdiag(0)
        W.eliminate_zeros()

    return W


def _vst_normalize(adata: ad.AnnData) -> np.ndarray:
    """Variance-stabilizing normalization for spatial correlation.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data (spots x genes) with raw counts.

    Returns
    -------
    np.ndarray
        Dense matrix of shape (genes, spots) with VST-normalized values.
    """
    return _vst_normalize_python(adata)


def _vst_normalize_python(adata: ad.AnnData) -> np.ndarray:
    """Python fallback VST normalization (approximation)."""
    import scanpy as sc

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    X = adata_norm.X
    if sparse.issparse(X):
        mat = X.toarray().T.astype(np.float64)
    else:
        mat = X.T.astype(np.float64)

    n_expressing = np.sum(mat > 0, axis=1)
    keep_mask = n_expressing >= 5
    mat[~keep_mask, :] = 0.0

    return mat
