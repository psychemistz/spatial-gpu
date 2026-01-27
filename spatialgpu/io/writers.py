"""
Data writers for spatial omics data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
    import spatialdata as sd


def write_anndata(
    adata: ad.AnnData,
    path: str | Path,
    compression: str = "gzip",
) -> None:
    """
    Write AnnData to h5ad file.

    Parameters
    ----------
    adata
        Annotated data object.
    path
        Output path.
    compression
        Compression method.

    Examples
    --------
    >>> sp.io.write_anndata(adata, "output.h5ad")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression=compression)


def write_spatial_csv(
    adata: ad.AnnData,
    output_dir: str | Path,
    prefix: str = "spatial",
    spatial_key: str = "spatial",
) -> None:
    """
    Write spatial data to CSV files.

    Creates:
    - {prefix}_counts.csv: Count matrix
    - {prefix}_coordinates.csv: Spatial coordinates
    - {prefix}_metadata.csv: Cell metadata

    Parameters
    ----------
    adata
        Annotated data object.
    output_dir
        Output directory.
    prefix
        Filename prefix.
    spatial_key
        Key for spatial coordinates.

    Examples
    --------
    >>> sp.io.write_spatial_csv(adata, "output/", prefix="my_sample")
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write count matrix
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X

    counts_df = pd.DataFrame(
        X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    counts_df.to_csv(output_dir / f"{prefix}_counts.csv")

    # Write coordinates
    if spatial_key in adata.obsm:
        coords_df = pd.DataFrame(
            adata.obsm[spatial_key],
            index=adata.obs_names,
            columns=(
                ["x", "y"] if adata.obsm[spatial_key].shape[1] == 2 else ["x", "y", "z"]
            ),
        )
        coords_df.to_csv(output_dir / f"{prefix}_coordinates.csv")

    # Write metadata
    if len(adata.obs.columns) > 0:
        adata.obs.to_csv(output_dir / f"{prefix}_metadata.csv")


def export_to_spatialdata(
    adata: ad.AnnData,
    spatial_key: str = "spatial",
    image: Optional[any] = None,
    masks: Optional[any] = None,
) -> sd.SpatialData:
    """
    Export to SpatialData format.

    Parameters
    ----------
    adata
        Annotated data object.
    spatial_key
        Key for spatial coordinates.
    image
        Optional image to include.
    masks
        Optional segmentation masks.

    Returns
    -------
    SpatialData
        SpatialData object.

    Examples
    --------
    >>> sdata = sp.io.export_to_spatialdata(adata)
    """
    try:
        import spatialdata as sd
        from spatialdata.models import ShapesModel, TableModel
    except ImportError:
        raise ImportError(
            "spatialdata not installed. Install with: pip install spatialdata"
        ) from None

    import geopandas as gpd
    from shapely.geometry import Point

    # Create shapes from coordinates
    if spatial_key in adata.obsm:
        coords = adata.obsm[spatial_key]
        points = [Point(x, y) for x, y in coords[:, :2]]
        shapes = gpd.GeoDataFrame(
            {"cell_id": adata.obs_names},
            geometry=points,
        )
    else:
        shapes = None

    # Build SpatialData
    elements = {}

    if shapes is not None:
        elements["cells"] = ShapesModel.parse(shapes)

    # Add table
    elements["table"] = TableModel.parse(
        adata,
        region="cells",
        region_key="cell_id",
        instance_key="cell_id",
    )

    # Add image if provided
    if image is not None:
        from spatialdata.models import Image2DModel

        elements["image"] = Image2DModel.parse(image)

    # Add masks if provided
    if masks is not None:
        from spatialdata.models import Labels2DModel

        elements["masks"] = Labels2DModel.parse(masks)

    return sd.SpatialData(**elements)
