"""
Data readers for various spatial omics platforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def read_visium(
    path: str | Path,
    count_file: str = "filtered_feature_bc_matrix.h5",
    spatial_key: str = "spatial",
    library_id: Optional[str] = None,
) -> ad.AnnData:
    """
    Read 10x Genomics Visium data.

    Parameters
    ----------
    path
        Path to Visium output directory.
    count_file
        Name of count matrix file.
    spatial_key
        Key for spatial coordinates in obsm.
    library_id
        Library ID for the sample.

    Returns
    -------
    AnnData
        Annotated data object with spatial coordinates.

    Examples
    --------
    >>> adata = sp.io.read_visium("path/to/visium_output")
    """
    import scanpy as sc

    path = Path(path)

    # Use scanpy's built-in reader
    adata = sc.read_visium(path, count_file=count_file, library_id=library_id)

    # Ensure spatial coordinates are in obsm
    if spatial_key not in adata.obsm:
        # Try to extract from uns
        if "spatial" in adata.uns:
            lib_id = library_id or list(adata.uns["spatial"].keys())[0]
            if "tissue_positions" in adata.uns["spatial"][lib_id]:
                coords = adata.uns["spatial"][lib_id]["tissue_positions"]
                if isinstance(coords, pd.DataFrame):
                    adata.obsm[spatial_key] = coords[["array_row", "array_col"]].values

    return adata


def read_xenium(
    path: str | Path,
    cells_file: str = "cells.parquet",
    transcripts_file: str = "transcripts.parquet",
    cell_feature_matrix: str = "cell_feature_matrix",
    spatial_key: str = "spatial",
) -> ad.AnnData:
    """
    Read 10x Genomics Xenium data.

    Parameters
    ----------
    path
        Path to Xenium output directory.
    cells_file
        Name of cells file.
    transcripts_file
        Name of transcripts file.
    cell_feature_matrix
        Name of cell feature matrix directory.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data object.

    Examples
    --------
    >>> adata = sp.io.read_xenium("path/to/xenium_output")
    """
    import scanpy as sc

    path = Path(path)

    # Read cell feature matrix
    matrix_path = path / cell_feature_matrix
    if matrix_path.exists():
        adata = sc.read_10x_h5(matrix_path / "cell_feature_matrix.h5")
    else:
        # Try h5 file in main directory
        h5_files = list(path.glob("*.h5"))
        if h5_files:
            adata = sc.read_10x_h5(h5_files[0])
        else:
            raise FileNotFoundError(f"No cell feature matrix found in {path}")

    # Read cell metadata
    cells_path = path / cells_file
    if cells_path.exists():
        if cells_path.suffix == ".parquet":
            cells = pd.read_parquet(cells_path)
        else:
            cells = pd.read_csv(cells_path)

        # Match cell IDs
        if "cell_id" in cells.columns:
            cells = cells.set_index("cell_id")
            cells = cells.loc[adata.obs_names]

        # Add spatial coordinates
        coord_cols = ["x_centroid", "y_centroid"]
        if all(c in cells.columns for c in coord_cols):
            adata.obsm[spatial_key] = cells[coord_cols].values

        # Add other metadata
        for col in cells.columns:
            if col not in coord_cols:
                adata.obs[col] = cells[col].values

    # Store transcripts path for later use
    adata.uns["transcripts_file"] = str(path / transcripts_file)

    return adata


def read_cosmx(
    path: str | Path,
    fov: Optional[int | list[int]] = None,
    spatial_key: str = "spatial",
) -> ad.AnnData:
    """
    Read NanoString CosMx data.

    Parameters
    ----------
    path
        Path to CosMx output directory.
    fov
        Field of view(s) to load. If None, load all.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data object.

    Examples
    --------
    >>> adata = sp.io.read_cosmx("path/to/cosmx_output")
    >>> adata = sp.io.read_cosmx("path/to/cosmx_output", fov=[1, 2, 3])
    """
    import anndata as ad
    from scipy import sparse

    path = Path(path)

    # Find expression matrix
    expr_files = list(path.glob("*exprMat*.csv")) + list(path.glob("*exprMat*.parquet"))
    if not expr_files:
        raise FileNotFoundError(f"No expression matrix found in {path}")

    # Read expression matrix
    expr_file = expr_files[0]
    if expr_file.suffix == ".parquet":
        expr_df = pd.read_parquet(expr_file)
    else:
        expr_df = pd.read_csv(expr_file)

    # Filter by FOV if specified
    if fov is not None:
        if isinstance(fov, int):
            fov = [fov]
        expr_df = expr_df[expr_df["fov"].isin(fov)]

    # Extract cell IDs and gene names
    id_cols = [
        "cell_ID",
        "fov",
        "CenterX_local_px",
        "CenterY_local_px",
        "CenterX_global_px",
        "CenterY_global_px",
    ]
    gene_cols = [
        c for c in expr_df.columns if c not in id_cols and c not in ["cell", "Cell"]
    ]

    # Create count matrix
    X = expr_df[gene_cols].values

    # Create AnnData
    cell_ids = [f"cell_{i}" for i in range(len(expr_df))]

    adata = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_cols),
    )

    # Add metadata
    for col in id_cols:
        if col in expr_df.columns:
            adata.obs[col] = expr_df[col].values

    # Add spatial coordinates
    if "CenterX_global_px" in expr_df.columns:
        adata.obsm[spatial_key] = expr_df[
            ["CenterX_global_px", "CenterY_global_px"]
        ].values
    elif "CenterX_local_px" in expr_df.columns:
        adata.obsm[spatial_key] = expr_df[
            ["CenterX_local_px", "CenterY_local_px"]
        ].values

    return adata


def read_merscope(
    path: str | Path,
    region: Optional[str] = None,
    spatial_key: str = "spatial",
) -> ad.AnnData:
    """
    Read Vizgen MERSCOPE/MERFISH data.

    Parameters
    ----------
    path
        Path to MERSCOPE output directory.
    region
        Specific region to load.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data object.

    Examples
    --------
    >>> adata = sp.io.read_merscope("path/to/merscope_output")
    """
    import anndata as ad
    from scipy import sparse

    path = Path(path)

    # Find cell-by-gene matrix
    matrix_file = path / "cell_by_gene.csv"
    if not matrix_file.exists():
        matrix_files = list(path.glob("*cell_by_gene*.csv"))
        if matrix_files:
            matrix_file = matrix_files[0]
        else:
            raise FileNotFoundError(f"No cell-by-gene matrix found in {path}")

    # Read matrix
    df = pd.read_csv(matrix_file, index_col=0)

    # Find metadata
    meta_file = path / "cell_metadata.csv"
    if not meta_file.exists():
        meta_files = list(path.glob("*metadata*.csv"))
        if meta_files:
            meta_file = meta_files[0]
        else:
            meta_file = None

    # Create AnnData
    X = df.values
    cell_ids = df.index.astype(str)
    gene_names = df.columns.tolist()

    adata = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_names),
    )

    # Add metadata
    if meta_file is not None and meta_file.exists():
        meta_df = pd.read_csv(meta_file, index_col=0)
        meta_df.index = meta_df.index.astype(str)

        # Reindex to match
        meta_df = meta_df.reindex(cell_ids)

        for col in meta_df.columns:
            adata.obs[col] = meta_df[col].values

        # Extract spatial coordinates
        coord_cols = [
            ("center_x", "center_y"),
            ("x", "y"),
            ("X", "Y"),
        ]
        for x_col, y_col in coord_cols:
            if x_col in meta_df.columns and y_col in meta_df.columns:
                adata.obsm[spatial_key] = meta_df[[x_col, y_col]].values
                break

    return adata


def read_spatial_csv(
    count_file: str | Path,
    coord_file: Optional[str | Path] = None,
    gene_col: str = "gene",
    x_col: str = "x",
    y_col: str = "y",
    cell_col: Optional[str] = None,
    spatial_key: str = "spatial",
) -> ad.AnnData:
    """
    Read spatial data from CSV files.

    Parameters
    ----------
    count_file
        Path to count matrix CSV (cells x genes).
    coord_file
        Path to coordinates CSV. If None, assumes coordinates in count_file.
    gene_col
        Column name for genes (if long format).
    x_col
        Column name for x coordinates.
    y_col
        Column name for y coordinates.
    cell_col
        Column name for cell IDs.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data object.

    Examples
    --------
    >>> adata = sp.io.read_spatial_csv(
    ...     "counts.csv", coord_file="coordinates.csv"
    ... )
    """
    import anndata as ad
    from scipy import sparse

    # Read count data
    counts_df = pd.read_csv(count_file, index_col=0)

    # Create AnnData
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts_df.values),
        obs=pd.DataFrame(index=counts_df.index.astype(str)),
        var=pd.DataFrame(index=counts_df.columns.tolist()),
    )

    # Read coordinates
    if coord_file is not None:
        coord_df = pd.read_csv(coord_file, index_col=0)
        coord_df.index = coord_df.index.astype(str)
        coord_df = coord_df.reindex(adata.obs_names)

        adata.obsm[spatial_key] = coord_df[[x_col, y_col]].values

    return adata


def read_spatial_parquet(
    count_file: str | Path,
    coord_file: Optional[str | Path] = None,
    x_col: str = "x",
    y_col: str = "y",
    spatial_key: str = "spatial",
) -> ad.AnnData:
    """
    Read spatial data from Parquet files.

    Parameters
    ----------
    count_file
        Path to count matrix Parquet file.
    coord_file
        Path to coordinates Parquet file.
    x_col
        Column name for x coordinates.
    y_col
        Column name for y coordinates.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    AnnData
        Annotated data object.

    Examples
    --------
    >>> adata = sp.io.read_spatial_parquet("counts.parquet", "coords.parquet")
    """
    import anndata as ad
    from scipy import sparse

    # Read count data
    counts_df = pd.read_parquet(count_file)
    if counts_df.index.name is None:
        counts_df = counts_df.set_index(counts_df.columns[0])

    # Create AnnData
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts_df.values),
        obs=pd.DataFrame(index=counts_df.index.astype(str)),
        var=pd.DataFrame(index=counts_df.columns.tolist()),
    )

    # Read coordinates
    if coord_file is not None:
        coord_df = pd.read_parquet(coord_file)
        if coord_df.index.name is None:
            coord_df = coord_df.set_index(coord_df.columns[0])
        coord_df.index = coord_df.index.astype(str)
        coord_df = coord_df.reindex(adata.obs_names)

        adata.obsm[spatial_key] = coord_df[[x_col, y_col]].values

    return adata
