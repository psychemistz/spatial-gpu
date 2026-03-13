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

    logger.info(
        "Building weight matrix: %d spots, radius=%.0f um, sigma=%.0f.",
        n_spots,
        radius,
        sigma,
    )

    # KD-tree radius search
    tree = KDTree(coords)
    neighbors = tree.query_ball_tree(tree, r=radius)

    # Build sparse matrix triplets
    rows = []
    cols = []
    vals = []

    for i in range(n_spots):
        for j in neighbors[i]:
            if i == j:
                continue  # skip self
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            if d > 0 and d <= radius:
                w = np.exp(-(d**2) / (2.0 * sigma**2))
                rows.append(i)
                cols.append(j)
                vals.append(w)

    W = sparse.csr_matrix(
        (np.array(vals, dtype=np.float64), (rows, cols)),
        shape=(n_spots, n_spots),
    )

    # Optionally limit to k nearest neighbors per spot
    if k is not None and k < n_spots:
        W_lil = W.tolil()
        for i in range(n_spots):
            row_data = W_lil[i]
            nonzero_cols = row_data.nonzero()[1]
            if len(nonzero_cols) > k:
                weights = np.array(row_data[0, nonzero_cols].toarray()).ravel()
                # Keep top-k by weight (largest weights)
                keep_idx = np.argsort(weights)[-k:]
                remove_idx = np.setdiff1d(np.arange(len(nonzero_cols)), keep_idx)
                for ri in remove_idx:
                    W_lil[i, nonzero_cols[ri]] = 0.0
        W = W_lil.tocsr()
        W.eliminate_zeros()

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

    for i in range(mat.shape[0]):
        x = mat[i, :]
        dx = x - np.mean(x)
        std_x = np.sqrt(np.sum(dx**2) / N)
        if std_x > 0:
            mat[i, :] = dx / std_x
        else:
            mat[i, :] = 0.0

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
                    XW = XW.toarray()
                moran_perm[:, p] = np.sum(XW * X_perm, axis=1)

            # Observed (last column)
            XW_obs = mat @ W
            if sparse.issparse(XW_obs):
                XW_obs = XW_obs.toarray()
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
                    XW = XW.toarray()
                moran_perm[:, p] = np.sum(XW * X_perm_R, axis=1)

            # Observed
            XW_obs = mat[l_indices, :] @ W
            if sparse.issparse(XW_obs):
                XW_obs = XW_obs.toarray()
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
            XW = XW.toarray()

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


def _vst_normalize(adata: ad.AnnData) -> np.ndarray:
    """Variance-stabilizing normalization for spatial correlation.

    Tries R's sctransform::vst via subprocess for exact equivalence with
    SpaCET. Falls back to a Python approximation if R is unavailable.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data (spots x genes) with raw counts.

    Returns
    -------
    np.ndarray
        Dense matrix of shape (genes, spots) with VST-normalized values.
    """
    try:
        return _vst_normalize_via_r(adata)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"R/sctransform not available ({e}), using Python approximation. "
            "Spatial correlation results may differ from SpaCET.",
            stacklevel=2,
        )
        return _vst_normalize_python(adata)


def _vst_normalize_via_r(adata: ad.AnnData) -> np.ndarray:
    """Run sctransform::vst in R via subprocess for exact SpaCET match."""
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    # Get counts as genes x spots sparse matrix
    X = adata.X
    if sparse.issparse(X):
        counts = X.T.tocsc().astype(np.float64)
    else:
        counts = sparse.csc_matrix(X.T, dtype=np.float64)

    gene_names = np.array(adata.var_names)
    n_genes, n_spots = counts.shape

    tmpdir = tempfile.mkdtemp()
    input_mtx = os.path.join(tmpdir, "counts.mtx")
    input_genes = os.path.join(tmpdir, "genes.csv")
    output_mat = os.path.join(tmpdir, "vst_y.csv")

    mmwrite(input_mtx, counts)
    pd.DataFrame({"gene": gene_names}).to_csv(input_genes, index=False)

    r_code = f"""
    suppressPackageStartupMessages({{
        library(Matrix)
        library(sctransform)
    }})
    suppressWarnings({{
        counts <- readMM("{input_mtx}")
        counts <- as(counts, "CsparseMatrix")
    }})
    genes <- read.csv("{input_genes}")$gene
    rownames(counts) <- genes
    colnames(counts) <- paste0("spot_", seq_len(ncol(counts)))

    vst_res <- sctransform::vst(counts, min_cells=5, verbosity=0)
    write.csv(as.matrix(vst_res$y), "{output_mat}", row.names=TRUE)
    """

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R sctransform failed: {result.stderr}")

        vst_df = pd.read_csv(output_mat, index_col=0)
        mat = vst_df.values.astype(np.float64)  # genes x spots

        # Reorder rows to match original gene order
        vst_genes = np.array(vst_df.index)
        if not np.array_equal(vst_genes, gene_names):
            # VST may return a subset of genes; pad with zeros
            full_mat = np.zeros((n_genes, n_spots), dtype=np.float64)
            gene_to_idx = {g: i for i, g in enumerate(gene_names)}
            for i, g in enumerate(vst_genes):
                if g in gene_to_idx:
                    full_mat[gene_to_idx[g], :] = mat[i, :]
            mat = full_mat

        return mat
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
