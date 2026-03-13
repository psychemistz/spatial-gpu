"""I/O utilities for SpaCET deconvolution.

Create AnnData objects from 10X Visium Space Ranger output or from
user-provided count matrices and spot coordinates. Also provides
quality control filtering.

Reference: SpaCET R package utilities.R
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_spacet_object_10x(visium_path: str | Path) -> ad.AnnData:
    """Create an AnnData object from 10X Visium Space Ranger output.

    Reads the filtered feature-barcode matrix and spatial information from a
    Space Ranger output directory. Spot IDs are formatted as
    ``"{array_row}x{array_col}"`` and micrometer coordinates are computed
    from the array indices.

    Parameters
    ----------
    visium_path : str or Path
        Path to the Space Ranger output folder. Must contain
        ``filtered_feature_bc_matrix/`` (or ``filtered_feature_bc_matrix.h5``)
        and ``spatial/`` subdirectories.

    Returns
    -------
    AnnData
        Spatial transcriptomics data with:
        - ``X``: sparse count matrix (spots x genes)
        - ``obs``: spot metadata including pixel and array coordinates
        - ``obsm['spatial']``: pixel coordinates (N x 2)
        - ``uns['spacet_platform']``: platform string
    """
    import anndata as ad

    visium_path = Path(visium_path)
    if not visium_path.exists():
        raise FileNotFoundError(
            f"The visium_path does not exist: {visium_path}"
        )

    # ---- Read count matrix ----
    mtx_dir = visium_path / "filtered_feature_bc_matrix"
    if (mtx_dir / "matrix.mtx.gz").exists():
        counts, gene_names, barcodes = _read_10x_mtx(mtx_dir)
    else:
        # Fallback to scanpy for h5
        import scanpy as sc

        adata_raw = sc.read_10x_h5(
            str(visium_path / "filtered_feature_bc_matrix.h5")
        )
        counts = adata_raw.X.T  # genes x spots
        gene_names = np.array(adata_raw.var_names)
        barcodes = np.array(adata_raw.obs_names)

    # ---- Read spatial metadata ----
    spatial_dir = visium_path / "spatial"

    # Scale factors
    with open(spatial_dir / "scalefactors_json.json") as f:
        scale_factors = json.load(f)

    # Tissue positions
    barcode_df, platform = _read_tissue_positions(spatial_dir)

    # Filter to in-tissue spots
    barcode_df = barcode_df[barcode_df["in_tissue"] == 1].copy()

    # Intersect barcodes with count matrix
    olp = np.intersect1d(barcodes, barcode_df.index)
    if len(olp) == 0:
        raise ValueError(
            "No overlapping barcodes between count matrix and spatial data."
        )

    # Reorder count matrix columns to match barcode order
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    col_idx = np.array([barcode_to_idx[b] for b in olp])
    counts = counts[:, col_idx]
    barcode_df = barcode_df.loc[olp]

    # ---- Compute pixel coordinates ----
    if (spatial_dir / "tissue_lowres_image.png").exists():
        scale = scale_factors["tissue_lowres_scalef"]
    else:
        scale = scale_factors["tissue_hires_scalef"]

    barcode_df["pixel_row"] = np.round(
        barcode_df["pxl_row_in_fullres"].values * scale, 3
    )
    barcode_df["pixel_col"] = np.round(
        barcode_df["pxl_col_in_fullres"].values * scale, 3
    )

    # ---- Build spot IDs: "{array_row}x{array_col}" ----
    spot_ids = [
        f"{r}x{c}"
        for r, c in zip(
            barcode_df["array_row"].values, barcode_df["array_col"].values
        )
    ]

    # ---- Compute micrometer coordinates ----
    coord_x_um = barcode_df["array_col"].values * 0.5 * 100.0
    coord_y_um = barcode_df["array_row"].values * 0.5 * np.sqrt(3) * 100.0
    coord_y_um = coord_y_um.max() - coord_y_um  # flip y-axis

    # ---- Handle duplicate gene names ----
    counts, gene_names = _remove_duplicate_genes(counts, gene_names)

    # ---- Construct AnnData (spots x genes) ----
    counts_csc = sparse.csc_matrix(counts)
    adata = ad.AnnData(
        X=counts_csc.T.tocsr(),
        obs=pd.DataFrame(
            {
                "barcode": barcode_df.index.values,
                "pixel_row": barcode_df["pixel_row"].values,
                "pixel_col": barcode_df["pixel_col"].values,
                "array_row": barcode_df["array_row"].values,
                "array_col": barcode_df["array_col"].values,
                "coordinate_x_um": coord_x_um,
                "coordinate_y_um": coord_y_um,
            },
            index=pd.Index(spot_ids),
        ),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )

    # Spatial coordinates for plotting (pixel_col, pixel_row) = (x, y)
    adata.obsm["spatial"] = np.column_stack(
        [barcode_df["pixel_col"].values, barcode_df["pixel_row"].values]
    )

    adata.uns["spacet_platform"] = platform
    adata.uns["spacet"] = {}

    logger.info(
        "Created SpaCET object: %d spots x %d genes (%s).",
        adata.n_obs,
        adata.n_vars,
        platform,
    )

    return adata


def create_spacet_object(
    counts: np.ndarray | sparse.spmatrix,
    spot_coordinates: pd.DataFrame,
    platform: str,
    image_path: str | Path | None = None,
) -> ad.AnnData:
    """Create an AnnData object from a count matrix and spot coordinates.

    Parameters
    ----------
    counts : array-like
        Count matrix with shape (genes x spots). Can be dense numpy array or
        scipy sparse matrix. Column names (spot IDs) must match the row index
        of *spot_coordinates*.
    spot_coordinates : pd.DataFrame
        DataFrame with at least two columns for X and Y coordinates. The index
        must contain spot IDs matching the columns of *counts*. If the first
        two columns are used as X and Y respectively.
    platform : str
        Platform identifier, e.g. ``"Visium"``, ``"OldST"``, ``"Slide-Seq"``.
    image_path : str, Path, or None
        Optional path to H&E image file.

    Returns
    -------
    AnnData
        Spatial transcriptomics data with spots as obs and genes as var.
    """
    import anndata as ad

    # Validate input types
    if isinstance(counts, pd.DataFrame):
        gene_names = np.array(counts.index)
        spot_names_counts = np.array(counts.columns)
        if sparse.issparse(counts.values):
            counts_mat = counts.values
        else:
            counts_mat = sparse.csc_matrix(counts.values, dtype=np.float64)
    elif sparse.issparse(counts):
        gene_names = None
        spot_names_counts = None
        counts_mat = counts
    else:
        gene_names = None
        spot_names_counts = None
        counts_mat = sparse.csc_matrix(counts, dtype=np.float64)

    # Validate spot ID consistency
    spot_names_coords = np.array(spot_coordinates.index)
    if spot_names_counts is not None:
        if not np.array_equal(spot_names_counts, spot_names_coords):
            raise ValueError(
                "Spot IDs in counts columns and spot_coordinates index "
                "are not identical."
            )
    spot_names = spot_names_coords

    n_genes = counts_mat.shape[0]
    n_spots = counts_mat.shape[1]

    if n_spots != len(spot_names):
        raise ValueError(
            f"Count matrix has {n_spots} spots but spot_coordinates has "
            f"{len(spot_names)} entries."
        )

    # Handle duplicate gene names
    if gene_names is not None:
        counts_mat, gene_names = _remove_duplicate_genes(counts_mat, gene_names)
    else:
        gene_names = np.array([f"Gene_{i}" for i in range(n_genes)])

    # Extract coordinates (first two columns as X, Y)
    coord_cols = spot_coordinates.columns
    if "X" in coord_cols and "Y" in coord_cols:
        coord_x = spot_coordinates["X"].values.astype(np.float64)
        coord_y = spot_coordinates["Y"].values.astype(np.float64)
    else:
        coord_x = spot_coordinates.iloc[:, 0].values.astype(np.float64)
        coord_y = spot_coordinates.iloc[:, 1].values.astype(np.float64)

    # Build AnnData (spots x genes)
    if not sparse.issparse(counts_mat):
        counts_mat = sparse.csc_matrix(counts_mat)
    adata = ad.AnnData(
        X=counts_mat.T.tocsr(),
        obs=pd.DataFrame(
            spot_coordinates.values,
            index=pd.Index(spot_names),
            columns=spot_coordinates.columns,
        ),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )

    # Spatial coordinates
    adata.obsm["spatial"] = np.column_stack([coord_x, coord_y])

    adata.uns["spacet_platform"] = platform
    adata.uns["spacet"] = {}

    if image_path is not None:
        adata.uns["spacet_image_path"] = str(image_path)

    logger.info(
        "Created SpaCET object: %d spots x %d genes (%s).",
        adata.n_obs,
        adata.n_vars,
        platform,
    )

    return adata


def quality_control(
    adata: ad.AnnData,
    min_genes: int = 1,
) -> ad.AnnData:
    """Filter spots by minimum expressed gene count and compute QC metrics.

    Spots with fewer than *min_genes* expressed genes (count > 0) are removed.
    UMI count and expressed gene count are added to ``adata.obs``.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data (spots x genes).
    min_genes : int
        Minimum number of expressed genes per spot. Default: 1.

    Returns
    -------
    AnnData
        Filtered data with QC metrics in ``adata.obs['UMI']`` and
        ``adata.obs['Gene']``.
    """
    logger.info("Removing spots with fewer than %d expressed genes.", min_genes)

    X = adata.X

    # Count expressed genes per spot (spots are rows)
    if sparse.issparse(X):
        genes_per_spot = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        genes_per_spot = (X > 0).sum(axis=1)

    mask = genes_per_spot >= min_genes
    n_removed = (~mask).sum()
    n_kept = mask.sum()

    logger.info("%d spots removed.", n_removed)
    logger.info("%d spots kept.", n_kept)

    # Filter
    adata = adata[mask].copy()

    # Compute QC metrics on filtered data
    X_filt = adata.X
    if sparse.issparse(X_filt):
        umi_counts = np.asarray(X_filt.sum(axis=1)).ravel()
        gene_counts = np.asarray((X_filt > 0).sum(axis=1)).ravel()
    else:
        umi_counts = X_filt.sum(axis=1)
        gene_counts = (X_filt > 0).sum(axis=1)

    adata.obs["UMI"] = umi_counts
    adata.obs["Gene"] = gene_counts

    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_10x_mtx(
    mtx_dir: Path,
) -> tuple[sparse.spmatrix, np.ndarray, np.ndarray]:
    """Read a 10X Genomics filtered_feature_bc_matrix directory.

    Parameters
    ----------
    mtx_dir : Path
        Directory containing matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz.

    Returns
    -------
    tuple of (counts, gene_names, barcodes)
        counts : sparse matrix (genes x spots)
        gene_names : array of gene name strings
        barcodes : array of barcode strings
    """
    from scipy.io import mmread

    # Read sparse matrix (genes x spots in MTX format)
    counts = mmread(str(mtx_dir / "matrix.mtx.gz"))
    counts = sparse.csc_matrix(counts)

    # Read features
    features = pd.read_csv(
        mtx_dir / "features.tsv.gz",
        sep="\t",
        header=None,
        compression="gzip",
    )
    # Use gene symbol (column 1) if available, else column 0
    if features.shape[1] >= 2:
        gene_names = features.iloc[:, 1].values
    else:
        gene_names = features.iloc[:, 0].values

    # Read barcodes
    barcodes_df = pd.read_csv(
        mtx_dir / "barcodes.tsv.gz",
        sep="\t",
        header=None,
        compression="gzip",
    )
    barcodes = barcodes_df.iloc[:, 0].values

    return counts, np.asarray(gene_names, dtype=str), np.asarray(barcodes, dtype=str)


def _read_tissue_positions(
    spatial_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Read tissue position file from spatial directory.

    Tries tissue_positions_list.csv (Visium v1), then
    tissue_positions.csv (Visium v2), then tissue_positions.parquet
    (Visium HD).

    Returns
    -------
    tuple of (barcode_df, platform)
        barcode_df : DataFrame indexed by barcode with columns:
            in_tissue, array_row, array_col, pxl_row_in_fullres,
            pxl_col_in_fullres
        platform : str, either "Visium" or "VisiumHD"
    """
    col_names = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    positions_list_csv = spatial_dir / "tissue_positions_list.csv"
    positions_csv = spatial_dir / "tissue_positions.csv"
    positions_parquet = spatial_dir / "tissue_positions.parquet"

    if positions_list_csv.exists():
        barcode_df = pd.read_csv(positions_list_csv, header=None)
        barcode_df.columns = col_names
        platform = "Visium"
    elif positions_csv.exists():
        barcode_df = pd.read_csv(positions_csv)
        barcode_df.columns = col_names
        platform = "Visium"
    elif positions_parquet.exists():
        import pyarrow.parquet as pq

        barcode_df = pq.read_table(str(positions_parquet)).to_pandas()
        barcode_df.columns = col_names
        platform = "VisiumHD"
    else:
        raise FileNotFoundError(
            f"No tissue position file found in {spatial_dir}. "
            "Expected tissue_positions_list.csv, tissue_positions.csv, "
            "or tissue_positions.parquet."
        )

    barcode_df = barcode_df.set_index("barcode")
    return barcode_df, platform


def _remove_duplicate_genes(
    counts: sparse.spmatrix | np.ndarray,
    gene_names: np.ndarray,
) -> tuple[sparse.spmatrix | np.ndarray, np.ndarray]:
    """Handle duplicate gene names by keeping the gene with the max row sum.

    Matches the R ``rm_duplicates`` function.

    Parameters
    ----------
    counts : sparse or dense matrix
        Gene expression matrix (genes x spots).
    gene_names : np.ndarray
        Gene name array of length n_genes.

    Returns
    -------
    tuple of (filtered_counts, filtered_gene_names)
    """
    unique_names, name_counts = np.unique(gene_names, return_counts=True)
    duplicated = set(unique_names[name_counts > 1])

    if len(duplicated) == 0:
        return counts, gene_names

    logger.info(
        "Found %d duplicated gene names. Keeping gene with max row sum.",
        len(duplicated),
    )

    # Indices for unique (non-duplicated) genes
    unique_mask = np.array([g not in duplicated for g in gene_names])
    unique_indices = np.where(unique_mask)[0]

    # For each duplicated gene, keep the one with the highest row sum
    dup_keep_indices = []
    for gene in sorted(duplicated):
        dup_idx = np.where(gene_names == gene)[0]
        if sparse.issparse(counts):
            row_sums = np.asarray(counts[dup_idx].sum(axis=1)).ravel()
        else:
            row_sums = counts[dup_idx].sum(axis=1)
        best = dup_idx[np.argmax(row_sums)]
        dup_keep_indices.append(best)

    keep_indices = np.sort(
        np.concatenate([unique_indices, np.array(dup_keep_indices)])
    )

    counts = counts[keep_indices]
    gene_names = gene_names[keep_indices]

    return counts, gene_names
