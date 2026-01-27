"""
Utility functions for segmentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_centroids(
    masks: NDArray,
    cell_ids: NDArray | None = None,
) -> NDArray:
    """
    Compute cell centroids from mask array.

    Parameters
    ----------
    masks
        Integer mask array.
    cell_ids
        Cell IDs to compute centroids for.

    Returns
    -------
    array
        Centroids, shape (n_cells, 2) as (y, x).
    """
    if cell_ids is None:
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]

    centroids = ndimage.center_of_mass(
        masks > 0,
        labels=masks,
        index=cell_ids,
    )

    return np.array(centroids)


def compute_areas(
    masks: NDArray,
    cell_ids: NDArray | None = None,
) -> NDArray:
    """
    Compute cell areas from mask array.

    Parameters
    ----------
    masks
        Integer mask array.
    cell_ids
        Cell IDs to compute areas for.

    Returns
    -------
    array
        Areas in pixels.
    """
    if cell_ids is None:
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]

    areas = ndimage.sum(
        np.ones_like(masks),
        labels=masks,
        index=cell_ids,
    )

    return np.array(areas)


def compute_boundaries(
    masks: NDArray,
    connectivity: int = 1,
) -> NDArray:
    """
    Compute cell boundaries from mask array.

    Parameters
    ----------
    masks
        Integer mask array.
    connectivity
        Connectivity for boundary detection.

    Returns
    -------
    array
        Boolean boundary mask.
    """
    # Erode masks
    from scipy.ndimage import binary_erosion

    eroded = np.zeros_like(masks)
    for cid in np.unique(masks):
        if cid == 0:
            continue
        cell_mask = masks == cid
        if connectivity == 1:
            struct = ndimage.generate_binary_structure(2, 1)
        else:
            struct = ndimage.generate_binary_structure(2, 2)
        eroded_mask = binary_erosion(cell_mask, structure=struct)
        eroded[eroded_mask] = cid

    # Boundary is original minus eroded
    boundaries = (masks > 0) & (eroded != masks)

    return boundaries


def compute_circularity(
    masks: NDArray,
    cell_ids: NDArray | None = None,
) -> NDArray:
    """
    Compute cell circularity (4 * pi * area / perimeter^2).

    Parameters
    ----------
    masks
        Integer mask array.
    cell_ids
        Cell IDs to compute circularity for.

    Returns
    -------
    array
        Circularity values (1.0 = perfect circle).
    """
    if cell_ids is None:
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]

    areas = compute_areas(masks, cell_ids)

    # Compute perimeters
    perimeters = []
    boundaries = compute_boundaries(masks)

    for cid in cell_ids:
        boundary_mask = boundaries & (masks == cid)
        perimeter = np.sum(boundary_mask)
        perimeters.append(perimeter)

    perimeters = np.array(perimeters)

    # Circularity
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = 4 * np.pi * areas / (perimeters ** 2)
        circularity = np.nan_to_num(circularity, nan=0.0, posinf=1.0, neginf=0.0)

    return circularity


def merge_tiled_masks(
    tile_masks: list[tuple[NDArray, tuple[int, int]]],
    output_shape: tuple[int, int],
    overlap: int,
    min_overlap_area: float = 0.5,
) -> NDArray:
    """
    Merge tiled segmentation masks into a single mask.

    Handles cells that span tile boundaries by matching based on overlap.

    Parameters
    ----------
    tile_masks
        List of (mask_array, (y_offset, x_offset)) tuples.
    output_shape
        Shape of output mask (H, W).
    overlap
        Overlap between tiles in pixels.
    min_overlap_area
        Minimum overlap fraction to consider cells as same.

    Returns
    -------
    array
        Merged mask array.
    """
    output = np.zeros(output_shape, dtype=np.int32)
    next_cell_id = 1

    # Map from tile cell IDs to global cell IDs
    cell_id_maps = []

    for tile_idx, (tile_mask, (y_off, x_off)) in enumerate(tile_masks):
        h, w = tile_mask.shape
        cell_id_map = {}

        # Get region in output that this tile covers
        y_end = min(y_off + h, output_shape[0])
        x_end = min(x_off + w, output_shape[1])

        # Check existing cells in overlapping region
        existing = output[y_off:y_end, x_off:x_end]

        for cell_id in np.unique(tile_mask):
            if cell_id == 0:
                continue

            cell_mask = tile_mask[:y_end-y_off, :x_end-x_off] == cell_id
            cell_area = np.sum(cell_mask)

            # Check for matching existing cells
            matched = False
            for existing_id in np.unique(existing[cell_mask]):
                if existing_id == 0:
                    continue

                # Check overlap
                existing_mask = existing == existing_id
                overlap_area = np.sum(cell_mask & existing_mask)
                overlap_ratio = overlap_area / cell_area

                if overlap_ratio >= min_overlap_area:
                    # Match to existing cell
                    cell_id_map[cell_id] = existing_id
                    matched = True
                    break

            if not matched:
                # New cell
                cell_id_map[cell_id] = next_cell_id
                next_cell_id += 1

        cell_id_maps.append(cell_id_map)

        # Write to output using mapped IDs
        for orig_id, new_id in cell_id_map.items():
            mask = tile_mask[:y_end-y_off, :x_end-x_off] == orig_id
            output[y_off:y_end, x_off:x_end][mask] = new_id

    return output


def expand_masks(
    masks: NDArray,
    expansion: int = 5,
) -> NDArray:
    """
    Expand cell masks by a fixed number of pixels.

    Useful for creating cytoplasm masks from nuclear masks.

    Parameters
    ----------
    masks
        Integer mask array.
    expansion
        Number of pixels to expand each cell.

    Returns
    -------
    array
        Expanded mask array.
    """
    from scipy.ndimage import distance_transform_edt

    expanded = np.zeros_like(masks)

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id

        # Compute distance transform from cell boundary
        dist = distance_transform_edt(~cell_mask)

        # Expand where distance is within expansion and no other cell
        expand_region = (dist <= expansion) & (expanded == 0)
        expanded[expand_region] = cell_id

    return expanded


def filter_by_size(
    masks: NDArray,
    min_size: int = 0,
    max_size: int | None = None,
) -> NDArray:
    """
    Filter cells by size.

    Parameters
    ----------
    masks
        Integer mask array.
    min_size
        Minimum cell size in pixels.
    max_size
        Maximum cell size in pixels.

    Returns
    -------
    array
        Filtered mask array.
    """
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]

    areas = compute_areas(masks, cell_ids)

    # Find cells to keep
    keep = areas >= min_size
    if max_size is not None:
        keep &= areas <= max_size

    # Create filtered mask
    filtered = np.zeros_like(masks)
    for cid, keep_cell in zip(cell_ids, keep):
        if keep_cell:
            filtered[masks == cid] = cid

    # Relabel to be contiguous
    unique_kept = np.unique(filtered)
    unique_kept = unique_kept[unique_kept != 0]

    relabeled = np.zeros_like(filtered)
    for new_id, old_id in enumerate(unique_kept, start=1):
        relabeled[filtered == old_id] = new_id

    return relabeled


def remove_edge_cells(
    masks: NDArray,
    edge_buffer: int = 0,
) -> NDArray:
    """
    Remove cells touching image edges.

    Parameters
    ----------
    masks
        Integer mask array.
    edge_buffer
        Additional buffer from edges.

    Returns
    -------
    array
        Mask with edge cells removed.
    """
    h, w = masks.shape

    # Find cells touching edges
    edge_cells = set()

    # Top and bottom edges
    edge_cells.update(np.unique(masks[:edge_buffer+1, :]))
    edge_cells.update(np.unique(masks[-(edge_buffer+1):, :]))

    # Left and right edges
    edge_cells.update(np.unique(masks[:, :edge_buffer+1]))
    edge_cells.update(np.unique(masks[:, -(edge_buffer+1):]))

    # Remove background from set
    edge_cells.discard(0)

    # Create filtered mask
    filtered = masks.copy()
    for cid in edge_cells:
        filtered[masks == cid] = 0

    return filtered
