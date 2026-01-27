"""
Synthetic data generation for benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad


def generate_synthetic_data(
    n_cells: int = 10000,
    n_genes: int = 500,
    n_clusters: int = 10,
    spatial_dims: int = 2,
    extent: tuple[float, float] = (0, 1000),
    sparsity: float = 0.9,
    seed: Optional[int] = None,
) -> ad.AnnData:
    """
    Generate synthetic spatial data for benchmarking.

    Parameters
    ----------
    n_cells
        Number of cells.
    n_genes
        Number of genes.
    n_clusters
        Number of cell clusters.
    spatial_dims
        Number of spatial dimensions (2 or 3).
    extent
        Spatial extent (min, max).
    sparsity
        Sparsity of count matrix (fraction of zeros).
    seed
        Random seed.

    Returns
    -------
    AnnData
        Synthetic annotated data object.

    Examples
    --------
    >>> adata = sp.benchmarks.generate_synthetic_data(n_cells=100000)
    >>> print(adata)
    AnnData object with n_obs × n_vars = 100000 × 500
    """
    import anndata as ad

    if seed is not None:
        np.random.seed(seed)

    # Generate count matrix
    counts = np.random.negative_binomial(2, 0.5, size=(n_cells, n_genes))

    # Apply sparsity
    mask = np.random.random((n_cells, n_genes)) < sparsity
    counts[mask] = 0

    X = sparse.csr_matrix(counts.astype(np.float32))

    # Generate spatial coordinates
    coords = np.random.uniform(extent[0], extent[1], size=(n_cells, spatial_dims))

    # Generate cluster labels
    clusters = np.random.choice(
        [f"cluster_{i}" for i in range(n_clusters)],
        size=n_cells,
    )

    # Create observation dataframe
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical(clusters),
            "total_counts": np.array(X.sum(axis=1)).flatten(),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    # Create variable dataframe
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords

    return adata


def generate_spatial_clusters(
    n_cells_per_cluster: int = 1000,
    n_clusters: int = 10,
    n_genes: int = 500,
    cluster_radius: float = 50.0,
    extent: tuple[float, float] = (0, 1000),
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> ad.AnnData:
    """
    Generate synthetic data with spatially coherent clusters.

    Creates data where each cluster is localized in space, useful for
    testing spatial analysis methods.

    Parameters
    ----------
    n_cells_per_cluster
        Number of cells per cluster.
    n_clusters
        Number of clusters.
    n_genes
        Number of genes.
    cluster_radius
        Radius of each cluster.
    extent
        Spatial extent.
    noise
        Noise level for cluster positions.
    seed
        Random seed.

    Returns
    -------
    AnnData
        Synthetic data with spatial clusters.

    Examples
    --------
    >>> adata = sp.benchmarks.generate_spatial_clusters(
    ...     n_cells_per_cluster=5000,
    ...     n_clusters=10
    ... )
    """
    import anndata as ad

    if seed is not None:
        np.random.seed(seed)

    n_cells = n_cells_per_cluster * n_clusters

    # Generate cluster centers
    cluster_centers = np.random.uniform(
        extent[0] + cluster_radius,
        extent[1] - cluster_radius,
        size=(n_clusters, 2),
    )

    # Generate cells around cluster centers
    all_coords = []
    all_clusters = []

    for i, center in enumerate(cluster_centers):
        # Generate cells in a circle around center
        angles = np.random.uniform(0, 2 * np.pi, n_cells_per_cluster)
        radii = np.random.uniform(0, cluster_radius, n_cells_per_cluster)

        x = center[0] + radii * np.cos(angles)
        y = center[1] + radii * np.sin(angles)

        # Add noise
        x += np.random.normal(0, noise * cluster_radius, n_cells_per_cluster)
        y += np.random.normal(0, noise * cluster_radius, n_cells_per_cluster)

        coords = np.stack([x, y], axis=1)
        all_coords.append(coords)
        all_clusters.extend([f"cluster_{i}"] * n_cells_per_cluster)

    coords = np.vstack(all_coords)
    clusters = np.array(all_clusters)

    # Clip to extent
    coords = np.clip(coords, extent[0], extent[1])

    # Generate count matrix with cluster-specific expression
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)

    for i in range(n_clusters):
        mask = clusters == f"cluster_{i}"

        # Cluster-specific marker genes
        n_markers = n_genes // n_clusters
        marker_start = i * n_markers
        marker_end = (i + 1) * n_markers

        # High expression for marker genes
        counts[mask, marker_start:marker_end] = np.random.negative_binomial(
            5, 0.3, size=(mask.sum(), n_markers)
        )

        # Low background expression for other genes
        counts[mask] += np.random.negative_binomial(1, 0.8, size=(mask.sum(), n_genes))

    X = sparse.csr_matrix(counts)

    # Create AnnData
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical(clusters),
            "total_counts": np.array(X.sum(axis=1)).flatten(),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords

    return adata


def generate_image_with_cells(
    n_cells: int = 100,
    image_size: tuple[int, int] = (512, 512),
    cell_radius_range: tuple[int, int] = (10, 30),
    intensity_range: tuple[float, float] = (0.3, 1.0),
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic image with cells for segmentation testing.

    Parameters
    ----------
    n_cells
        Number of cells.
    image_size
        Size of output image (H, W).
    cell_radius_range
        Range of cell radii.
    intensity_range
        Range of cell intensities.
    noise_level
        Gaussian noise level.
    seed
        Random seed.

    Returns
    -------
    image
        Synthetic image.
    masks
        Ground truth segmentation masks.

    Examples
    --------
    >>> image, masks = sp.benchmarks.generate_image_with_cells(n_cells=50)
    >>> result = sp.segmentation.segment_cells(image)
    >>> metrics = sp.segmentation.evaluate_segmentation(result, masks)
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = image_size
    image = np.zeros((h, w), dtype=np.float32)
    masks = np.zeros((h, w), dtype=np.int32)

    # Generate non-overlapping cell positions
    positions = []
    radii = []

    attempts = 0
    max_attempts = n_cells * 100

    while len(positions) < n_cells and attempts < max_attempts:
        attempts += 1

        r = np.random.randint(*cell_radius_range)
        y = np.random.randint(r, h - r)
        x = np.random.randint(r, w - r)

        # Check for overlap
        overlaps = False
        for pos, rad in zip(positions, radii):
            dist = np.sqrt((y - pos[0]) ** 2 + (x - pos[1]) ** 2)
            if dist < r + rad + 5:  # 5 pixel buffer
                overlaps = True
                break

        if not overlaps:
            positions.append((y, x))
            radii.append(r)

    # Draw cells
    yy, xx = np.ogrid[:h, :w]

    for idx, ((cy, cx), r) in enumerate(zip(positions, radii), start=1):
        # Create circular mask
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        cell_mask = dist <= r

        # Add to masks
        masks[cell_mask] = idx

        # Add to image with intensity gradient
        intensity = np.random.uniform(*intensity_range)
        cell_intensity = intensity * (1 - dist[cell_mask] / r * 0.3)
        image[cell_mask] = np.maximum(image[cell_mask], cell_intensity)

    # Add noise
    image += np.random.normal(0, noise_level, image.shape)
    image = np.clip(image, 0, 1)

    return image, masks
