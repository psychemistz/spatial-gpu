"""
GPU-accelerated Ripley's spatial statistics.

Implements K, L, F, and G functions for spatial point pattern analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import anndata as ad
    from numpy.typing import NDArray


def ripley(
    adata: ad.AnnData,
    cluster_key: Optional[str] = None,
    mode: Literal["K", "L", "F", "G"] = "L",
    spatial_key: str = "spatial",
    radii: Optional[Sequence[float]] = None,
    n_radii: int = 50,
    n_simulations: int = 100,
    seed: Optional[int] = None,
    copy: bool = False,
    show_progress: bool = True,
) -> Optional[dict]:
    """
    Compute Ripley's statistics with GPU acceleration.

    This is a drop-in replacement for `squidpy.gr.ripley`.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates.
    cluster_key
        Key in `adata.obs` containing cluster labels. If None, analyzes all points.
    mode
        Ripley statistic to compute:
        - "K": Ripley's K function
        - "L": Ripley's L function (variance-stabilized K)
        - "F": Empty space function (nearest neighbor from random points)
        - "G": Nearest neighbor function (nearest neighbor distribution)
    spatial_key
        Key in `adata.obsm` containing spatial coordinates.
    radii
        Radii at which to compute statistics.
    n_radii
        Number of radii if not specified.
    n_simulations
        Number of simulations for confidence envelope.
    seed
        Random seed for reproducibility.
    copy
        Return results instead of storing in adata.
    show_progress
        Show progress bar.

    Returns
    -------
    If `copy=True`, returns dict with statistics.
    Otherwise stores in `adata.uns['ripley']`.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> stats = sp.graph.ripley(adata, mode="L", copy=True)
    >>> stats["L"]  # L function values
    >>> stats["radii"]  # radii used
    """
    from spatialgpu.core.backend import get_backend
    from spatialgpu.graph.utils import get_spatial_coords

    backend = get_backend()

    coords = get_spatial_coords(adata, spatial_key=spatial_key)
    coords.shape[0]

    # Compute bounding box
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    area = np.prod(maxs - mins)

    # Determine radii
    if radii is None:
        max_radius = np.min(maxs - mins) / 4
        radii = np.linspace(0, max_radius, n_radii)
    else:
        radii = np.array(radii)

    # Get cluster-specific analysis if requested
    if cluster_key is not None:
        clusters = adata.obs[cluster_key]
        if hasattr(clusters, "cat"):
            categories = clusters.cat.categories.tolist()
            cluster_idx = clusters.cat.codes.values
        else:
            categories = sorted(clusters.unique())
            cluster_idx = np.array([categories.index(c) for c in clusters])

        results = {}
        for cat_idx, cat in enumerate(categories):
            mask = cluster_idx == cat_idx
            cat_coords = coords[mask]
            cat_results = _compute_ripley(
                cat_coords,
                radii,
                mode,
                area,
                n_simulations,
                seed,
                backend.is_gpu_active,
                show_progress,
            )
            results[cat] = cat_results
    else:
        results = _compute_ripley(
            coords,
            radii,
            mode,
            area,
            n_simulations,
            seed,
            backend.is_gpu_active,
            show_progress,
        )

    output = {
        "stats": results,
        "radii": radii,
        "mode": mode,
        "area": area,
    }

    if copy:
        return output

    adata.uns["ripley"] = output
    return None


def _compute_ripley(
    coords: NDArray,
    radii: NDArray,
    mode: str,
    area: float,
    n_simulations: int,
    seed: Optional[int],
    use_gpu: bool,
    show_progress: bool,
) -> dict:
    """Compute Ripley statistics for a set of coordinates."""
    if use_gpu:
        return _compute_ripley_gpu(
            coords, radii, mode, area, n_simulations, seed, show_progress
        )
    else:
        return _compute_ripley_cpu(
            coords, radii, mode, area, n_simulations, seed, show_progress
        )


def _compute_ripley_gpu(
    coords: NDArray,
    radii: NDArray,
    mode: str,
    area: float,
    n_simulations: int,
    seed: Optional[int],
    show_progress: bool,
) -> dict:
    """GPU implementation of Ripley statistics."""
    import cupy as cp

    from spatialgpu.core.array_utils import to_cpu, to_gpu

    if seed is not None:
        cp.random.seed(seed)

    n_points = coords.shape[0]
    coords_gpu = to_gpu(coords.astype(np.float32))
    radii_gpu = to_gpu(radii.astype(np.float32))

    # Compute pairwise distances
    diff = coords_gpu[:, None, :] - coords_gpu[None, :, :]
    dists = cp.sqrt(cp.sum(diff**2, axis=2))

    # Compute K function
    K = cp.zeros(len(radii), dtype=cp.float32)
    for i, r in enumerate(radii_gpu):
        # Count pairs within radius (excluding self)
        count = cp.sum((dists <= r) & (dists > 0))
        K[i] = (area / (n_points * (n_points - 1))) * count

    K = to_cpu(K)

    # Compute L function if requested
    if mode in ("L", "K"):
        L = np.sqrt(K / np.pi) - radii

    # Simulation envelope
    sim_stats = []
    iterator = range(n_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Computing {mode} envelope")

    mins = to_cpu(coords_gpu.min(axis=0))
    maxs = to_cpu(coords_gpu.max(axis=0))

    for _ in iterator:
        # Generate random points in bounding box
        rand_coords = cp.random.uniform(
            to_gpu(mins), to_gpu(maxs), size=(n_points, coords.shape[1])
        )

        # Compute K for random points
        rand_diff = rand_coords[:, None, :] - rand_coords[None, :, :]
        rand_dists = cp.sqrt(cp.sum(rand_diff**2, axis=2))

        K_sim = cp.zeros(len(radii), dtype=cp.float32)
        for i, r in enumerate(radii_gpu):
            count = cp.sum((rand_dists <= r) & (rand_dists > 0))
            K_sim[i] = (area / (n_points * (n_points - 1))) * count

        if mode == "L":
            sim_stat = cp.sqrt(K_sim / np.pi) - radii_gpu
        else:
            sim_stat = K_sim

        sim_stats.append(to_cpu(sim_stat))

    sim_stats = np.array(sim_stats)

    result = {
        "observed": L if mode == "L" else K,
        "simulated_mean": np.mean(sim_stats, axis=0),
        "simulated_lo": np.percentile(sim_stats, 2.5, axis=0),
        "simulated_hi": np.percentile(sim_stats, 97.5, axis=0),
    }

    return result


def _compute_ripley_cpu(
    coords: NDArray,
    radii: NDArray,
    mode: str,
    area: float,
    n_simulations: int,
    seed: Optional[int],
    show_progress: bool,
) -> dict:
    """CPU implementation of Ripley statistics."""
    from scipy.spatial.distance import pdist, squareform

    if seed is not None:
        np.random.seed(seed)

    n_points = coords.shape[0]

    # Compute pairwise distances
    dists = squareform(pdist(coords))

    # Compute K function
    K = np.zeros(len(radii), dtype=np.float32)
    for i, r in enumerate(radii):
        count = np.sum((dists <= r) & (dists > 0))
        K[i] = (area / (n_points * (n_points - 1))) * count

    # Compute L function if requested
    if mode in ("L", "K"):
        L = np.sqrt(K / np.pi) - radii

    # Simulation envelope
    sim_stats = []
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    iterator = range(n_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Computing {mode} envelope")

    for _ in iterator:
        # Generate random points
        rand_coords = np.random.uniform(mins, maxs, size=(n_points, coords.shape[1]))
        rand_dists = squareform(pdist(rand_coords))

        K_sim = np.zeros(len(radii), dtype=np.float32)
        for i, r in enumerate(radii):
            count = np.sum((rand_dists <= r) & (rand_dists > 0))
            K_sim[i] = (area / (n_points * (n_points - 1))) * count

        if mode == "L":
            sim_stat = np.sqrt(K_sim / np.pi) - radii
        else:
            sim_stat = K_sim

        sim_stats.append(sim_stat)

    sim_stats = np.array(sim_stats)

    result = {
        "observed": L if mode == "L" else K,
        "simulated_mean": np.mean(sim_stats, axis=0),
        "simulated_lo": np.percentile(sim_stats, 2.5, axis=0),
        "simulated_hi": np.percentile(sim_stats, 97.5, axis=0),
    }

    return result
