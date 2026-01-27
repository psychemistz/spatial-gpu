"""
Transcript assignment to cells.

Functions for assigning spatial transcripts (e.g., from MERFISH, Xenium)
to segmented cells with GPU acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    from numpy.typing import NDArray

from spatialgpu.segmentation.core import SegmentationResult


def segment_transcripts(
    transcripts: pd.DataFrame,
    segmentation: SegmentationResult,
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "gene",
    pixel_size: float = 1.0,
) -> pd.DataFrame:
    """
    Assign transcripts to cells based on segmentation.

    Parameters
    ----------
    transcripts
        DataFrame with transcript coordinates and gene information.
    segmentation
        Cell segmentation result with masks.
    x_col
        Column name for x coordinates.
    y_col
        Column name for y coordinates.
    gene_col
        Column name for gene identifiers.
    pixel_size
        Conversion factor from transcript coordinates to pixels.

    Returns
    -------
    DataFrame
        Original transcripts with added 'cell_id' column.

    Examples
    --------
    >>> import pandas as pd
    >>> transcripts = pd.read_csv("transcripts.csv")
    >>> result = sp.segmentation.segment_cells(image)
    >>> assigned = sp.segmentation.segment_transcripts(
    ...     transcripts, result, x_col="x_um", y_col="y_um"
    ... )
    >>> assigned.groupby("cell_id")[gene_col].count()
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    # Get coordinates and convert to pixel space
    x = (transcripts[x_col].values / pixel_size).astype(np.int32)
    y = (transcripts[y_col].values / pixel_size).astype(np.int32)

    # Clip to image bounds
    h, w = segmentation.masks.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    if backend.is_gpu_active:
        cell_ids = _assign_transcripts_gpu(segmentation.masks, x, y)
    else:
        cell_ids = _assign_transcripts_cpu(segmentation.masks, x, y)

    # Add cell IDs to transcript dataframe
    result = transcripts.copy()
    result["cell_id"] = cell_ids

    return result


def _assign_transcripts_cpu(
    masks: NDArray,
    x: NDArray,
    y: NDArray,
) -> NDArray:
    """CPU implementation of transcript assignment."""
    return masks[y, x]


def _assign_transcripts_gpu(
    masks: NDArray,
    x: NDArray,
    y: NDArray,
) -> NDArray:
    """GPU implementation of transcript assignment."""

    from spatialgpu.core.array_utils import to_cpu, to_gpu

    masks_gpu = to_gpu(masks)
    x_gpu = to_gpu(x)
    y_gpu = to_gpu(y)

    cell_ids = masks_gpu[y_gpu, x_gpu]

    return to_cpu(cell_ids)


def assign_transcripts_to_cells(
    adata: ad.AnnData,
    transcripts: pd.DataFrame,
    masks: NDArray,
    x_col: str = "x",
    y_col: str = "y",
    gene_col: str = "gene",
    pixel_size: float = 1.0,
    min_transcripts: int = 10,
    key_added: str = "spatial",
) -> ad.AnnData:
    """
    Create AnnData from transcripts and segmentation.

    Parameters
    ----------
    adata
        Existing AnnData or None to create new.
    transcripts
        DataFrame with transcript coordinates and gene information.
    masks
        Cell segmentation mask array.
    x_col
        Column name for x coordinates.
    y_col
        Column name for y coordinates.
    gene_col
        Column name for gene identifiers.
    pixel_size
        Conversion factor from transcript coordinates to pixels.
    min_transcripts
        Minimum transcripts per cell to include.
    key_added
        Key for storing spatial data.

    Returns
    -------
    AnnData
        Annotated data with transcript counts per cell.

    Examples
    --------
    >>> adata = sp.segmentation.assign_transcripts_to_cells(
    ...     None, transcripts, masks,
    ...     x_col="x_um", y_col="y_um", gene_col="gene"
    ... )
    """
    import anndata as ad
    from scipy import sparse

    # Create segmentation result wrapper
    segmentation = SegmentationResult.from_masks(masks)

    # Assign transcripts to cells
    assigned = segment_transcripts(
        transcripts,
        segmentation,
        x_col=x_col,
        y_col=y_col,
        gene_col=gene_col,
        pixel_size=pixel_size,
    )

    # Filter out unassigned transcripts (cell_id == 0)
    assigned = assigned[assigned["cell_id"] > 0]

    # Get unique genes and cells
    genes = sorted(assigned[gene_col].unique())
    cell_ids = sorted(assigned["cell_id"].unique())

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cell_to_idx = {c: i for i, c in enumerate(cell_ids)}

    # Build count matrix
    n_cells = len(cell_ids)
    n_genes = len(genes)

    # Count transcripts per cell-gene pair
    counts = assigned.groupby(["cell_id", gene_col]).size().reset_index(name="count")

    row_idx = [cell_to_idx[c] for c in counts["cell_id"]]
    col_idx = [gene_to_idx[g] for g in counts[gene_col]]
    values = counts["count"].values

    X = sparse.csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(n_cells, n_genes),
    )

    # Filter cells with too few transcripts
    cell_sums = np.array(X.sum(axis=1)).flatten()
    keep_cells = cell_sums >= min_transcripts

    X = X[keep_cells]
    kept_cell_ids = [c for c, keep in zip(cell_ids, keep_cells) if keep]

    # Create AnnData
    import pandas as pd

    adata_new = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_id": kept_cell_ids},
            index=[f"cell_{c}" for c in kept_cell_ids],
        ),
        var=pd.DataFrame(
            {"gene": genes},
            index=genes,
        ),
    )

    # Add spatial coordinates (cell centroids)
    centroids = segmentation.centroids
    cell_id_to_centroid = {
        cid: centroids[i] for i, cid in enumerate(segmentation.cell_ids)
    }

    spatial_coords = np.array(
        [cell_id_to_centroid.get(cid, [np.nan, np.nan]) for cid in kept_cell_ids]
    )

    adata_new.obsm[key_added] = spatial_coords

    # Add cell areas
    cell_id_to_area = {
        cid: segmentation.areas[i] for i, cid in enumerate(segmentation.cell_ids)
    }
    adata_new.obs["cell_area"] = [cell_id_to_area.get(cid, 0) for cid in kept_cell_ids]

    return adata_new


def transcript_density(
    transcripts: pd.DataFrame,
    masks: NDArray,
    x_col: str = "x",
    y_col: str = "y",
    pixel_size: float = 1.0,
) -> NDArray:
    """
    Compute transcript density per cell.

    Parameters
    ----------
    transcripts
        DataFrame with transcript coordinates.
    masks
        Cell segmentation mask.
    x_col
        Column name for x coordinates.
    y_col
        Column name for y coordinates.
    pixel_size
        Conversion factor.

    Returns
    -------
    array
        Transcript density (count/area) per cell ID.
    """

    # Get segmentation result
    segmentation = SegmentationResult.from_masks(masks)

    # Assign transcripts
    assigned = segment_transcripts(
        transcripts,
        segmentation,
        x_col=x_col,
        y_col=y_col,
        pixel_size=pixel_size,
    )

    # Count per cell
    counts = assigned.groupby("cell_id").size()

    # Compute density
    density = np.zeros(len(segmentation.cell_ids) + 1)
    for cid in segmentation.cell_ids:
        idx = np.where(segmentation.cell_ids == cid)[0]
        if len(idx) > 0:
            area = segmentation.areas[idx[0]]
            count = counts.get(cid, 0)
            density[cid] = count / max(area, 1)

    return density
