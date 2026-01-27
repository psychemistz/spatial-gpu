"""
Input/output functions for spatial omics data.

Supports reading data from major spatial platforms:
- 10x Genomics (Visium, Xenium)
- NanoString (CosMx)
- Vizgen (MERSCOPE/MERFISH)
- Generic formats (CSV, Parquet, AnnData)
"""

from spatialgpu.io.readers import (
    read_visium,
    read_xenium,
    read_cosmx,
    read_merscope,
    read_spatial_csv,
    read_spatial_parquet,
)
from spatialgpu.io.writers import (
    write_anndata,
    write_spatial_csv,
    export_to_spatialdata,
)

__all__ = [
    # Readers
    "read_visium",
    "read_xenium",
    "read_cosmx",
    "read_merscope",
    "read_spatial_csv",
    "read_spatial_parquet",
    # Writers
    "write_anndata",
    "write_spatial_csv",
    "export_to_spatialdata",
]
