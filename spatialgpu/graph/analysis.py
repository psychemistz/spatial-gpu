"""
GPU-accelerated spatial graph analysis functions.

Provides neighborhood enrichment, co-occurrence, and interaction analysis
with significant speedups on GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    from numpy.typing import NDArray


def nhood_enrichment(
    adata: ad.AnnData,
    cluster_key: str,
    connectivity_key: str = "spatial_connectivities",
    n_perms: int = 1000,
    seed: Optional[int] = None,
    copy: bool = False,
    show_progress: bool = True,
) -> Optional[tuple[NDArray, NDArray]]:
    """
    Compute neighborhood enrichment with GPU acceleration.

    This is a drop-in replacement for `squidpy.gr.nhood_enrichment` with
    significant speedup on GPU.

    Parameters
    ----------
    adata
        Annotated data object with spatial graph.
    cluster_key
        Key in `adata.obs` containing cluster labels.
    connectivity_key
        Key in `adata.obsp` containing connectivity matrix.
    n_perms
        Number of permutations for statistical testing.
    seed
        Random seed for reproducibility.
    copy
        Return results instead of storing in adata.

    Returns
    -------
    If `copy=True`, returns (zscore, count) arrays.
    Otherwise stores results in `adata.uns[f'{cluster_key}_nhood_enrichment']`.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> sp.graph.spatial_neighbors(adata, n_neighbors=6)
    >>> zscore, count = sp.graph.nhood_enrichment(
    ...     adata, cluster_key="cell_type", copy=True
    ... )
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    # Get cluster labels
    clusters = adata.obs[cluster_key]
    if hasattr(clusters, "cat"):
        categories = clusters.cat.categories.tolist()
        cluster_idx = clusters.cat.codes.values
    else:
        categories = sorted(clusters.unique())
        cluster_idx = np.array([categories.index(c) for c in clusters])

    n_clusters = len(categories)
    n_cells = len(cluster_idx)

    # Get connectivity matrix
    adj = adata.obsp[connectivity_key]
    if not sparse.issparse(adj):
        adj = sparse.csr_matrix(adj)

    if backend.is_gpu_active:
        zscore, count = _nhood_enrichment_gpu(
            adj, cluster_idx, n_clusters, n_perms, seed, show_progress
        )
    else:
        zscore, count = _nhood_enrichment_cpu(
            adj, cluster_idx, n_clusters, n_perms, seed, show_progress
        )

    if copy:
        return zscore, count

    # Store in adata
    adata.uns[f"{cluster_key}_nhood_enrichment"] = {
        "zscore": zscore,
        "count": count,
        "categories": categories,
    }
    return None


def _nhood_enrichment_gpu(
    adj: sparse.csr_matrix,
    cluster_idx: NDArray,
    n_clusters: int,
    n_perms: int,
    seed: Optional[int],
    show_progress: bool,
) -> tuple[NDArray, NDArray]:
    """GPU implementation of neighborhood enrichment."""
    from spatialgpu.core.backend import get_backend
    from spatialgpu.core.array_utils import to_gpu, to_cpu

    backend = get_backend()
    import cupy as cp

    if seed is not None:
        cp.random.seed(seed)

    n_cells = len(cluster_idx)

    # Transfer to GPU
    cluster_idx_gpu = to_gpu(cluster_idx.astype(np.int32))

    # Convert sparse matrix to GPU
    adj_data = to_gpu(adj.data.astype(np.float32))
    adj_indices = to_gpu(adj.indices.astype(np.int32))
    adj_indptr = to_gpu(adj.indptr.astype(np.int32))

    # Compute observed counts
    count = _compute_interaction_count_gpu(
        adj_data, adj_indices, adj_indptr, cluster_idx_gpu, n_clusters
    )
    count = to_cpu(count)

    # Permutation test
    perm_counts = cp.zeros((n_perms, n_clusters, n_clusters), dtype=cp.float32)

    iterator = range(n_perms)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutation test")

    for i in iterator:
        # Permute cluster labels
        perm_idx = cp.random.permutation(cluster_idx_gpu)
        perm_count = _compute_interaction_count_gpu(
            adj_data, adj_indices, adj_indptr, perm_idx, n_clusters
        )
        perm_counts[i] = perm_count

    # Compute z-scores
    perm_mean = to_cpu(cp.mean(perm_counts, axis=0))
    perm_std = to_cpu(cp.std(perm_counts, axis=0))

    # Avoid division by zero
    perm_std[perm_std == 0] = 1

    zscore = (count - perm_mean) / perm_std

    return zscore, count


def _compute_interaction_count_gpu(
    adj_data,
    adj_indices,
    adj_indptr,
    cluster_idx,
    n_clusters: int,
):
    """Compute interaction count matrix on GPU using custom kernel."""
    import cupy as cp

    n_cells = len(cluster_idx)
    count = cp.zeros((n_clusters, n_clusters), dtype=cp.float32)

    # CUDA kernel for counting interactions
    interaction_kernel = cp.ElementwiseKernel(
        "raw int32 adj_indices, raw int32 adj_indptr, raw int32 cluster_idx, int32 n_clusters",
        "raw float32 count",
        """
        int cell_i = i;
        int cluster_i = cluster_idx[cell_i];
        int start = adj_indptr[cell_i];
        int end = adj_indptr[cell_i + 1];

        for (int j = start; j < end; j++) {
            int cell_j = adj_indices[j];
            int cluster_j = cluster_idx[cell_j];
            atomicAdd(&count[cluster_i * n_clusters + cluster_j], 1.0f);
        }
        """,
        "interaction_kernel",
    )

    interaction_kernel(adj_indices, adj_indptr, cluster_idx, n_clusters, count, size=n_cells)

    return count


def _nhood_enrichment_cpu(
    adj: sparse.csr_matrix,
    cluster_idx: NDArray,
    n_clusters: int,
    n_perms: int,
    seed: Optional[int],
    show_progress: bool,
) -> tuple[NDArray, NDArray]:
    """CPU implementation of neighborhood enrichment."""
    if seed is not None:
        np.random.seed(seed)

    n_cells = len(cluster_idx)

    # Compute observed counts
    count = _compute_interaction_count_cpu(adj, cluster_idx, n_clusters)

    # Permutation test
    perm_counts = np.zeros((n_perms, n_clusters, n_clusters), dtype=np.float32)

    iterator = range(n_perms)
    if show_progress:
        iterator = tqdm(iterator, desc="Permutation test")

    for i in iterator:
        perm_idx = np.random.permutation(cluster_idx)
        perm_counts[i] = _compute_interaction_count_cpu(adj, perm_idx, n_clusters)

    # Compute z-scores
    perm_mean = np.mean(perm_counts, axis=0)
    perm_std = np.std(perm_counts, axis=0)
    perm_std[perm_std == 0] = 1

    zscore = (count - perm_mean) / perm_std

    return zscore, count


def _compute_interaction_count_cpu(
    adj: sparse.csr_matrix,
    cluster_idx: NDArray,
    n_clusters: int,
) -> NDArray:
    """Compute interaction count matrix on CPU."""
    count = np.zeros((n_clusters, n_clusters), dtype=np.float32)

    rows, cols = adj.nonzero()
    for i, j in zip(rows, cols):
        count[cluster_idx[i], cluster_idx[j]] += 1

    return count


def co_occurrence(
    adata: ad.AnnData,
    cluster_key: str,
    spatial_key: str = "spatial",
    interval: Optional[Sequence[float]] = None,
    n_splits: int = 50,
    copy: bool = False,
    show_progress: bool = True,
) -> Optional[tuple[NDArray, NDArray]]:
    """
    Compute spatial co-occurrence with GPU acceleration.

    This is a drop-in replacement for `squidpy.gr.co_occurrence`.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates.
    cluster_key
        Key in `adata.obs` containing cluster labels.
    spatial_key
        Key in `adata.obsm` containing spatial coordinates.
    interval
        Distance interval (min, max) for analysis.
    n_splits
        Number of distance bins.
    copy
        Return results instead of storing in adata.

    Returns
    -------
    If `copy=True`, returns (occurrence, interval) arrays.
    Otherwise stores in `adata.uns[f'{cluster_key}_co_occurrence']`.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> occurrence, intervals = sp.graph.co_occurrence(
    ...     adata, cluster_key="cell_type", copy=True
    ... )
    """
    from spatialgpu.core.backend import get_backend
    from spatialgpu.graph.utils import get_spatial_coords

    backend = get_backend()

    coords = get_spatial_coords(adata, spatial_key=spatial_key)

    # Get cluster labels
    clusters = adata.obs[cluster_key]
    if hasattr(clusters, "cat"):
        categories = clusters.cat.categories.tolist()
        cluster_idx = clusters.cat.codes.values
    else:
        categories = sorted(clusters.unique())
        cluster_idx = np.array([categories.index(c) for c in clusters])

    n_clusters = len(categories)

    # Determine interval
    if interval is None:
        # Auto-compute based on data extent
        dists = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
        interval = (0, np.percentile(dists, 95) * 2)

    # Create distance bins
    bins = np.linspace(interval[0], interval[1], n_splits + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if backend.is_gpu_active:
        occurrence = _co_occurrence_gpu(
            coords, cluster_idx, n_clusters, bins, show_progress
        )
    else:
        occurrence = _co_occurrence_cpu(
            coords, cluster_idx, n_clusters, bins, show_progress
        )

    if copy:
        return occurrence, bin_centers

    adata.uns[f"{cluster_key}_co_occurrence"] = {
        "occurrence": occurrence,
        "interval": bin_centers,
        "categories": categories,
    }
    return None


def _co_occurrence_gpu(
    coords: NDArray,
    cluster_idx: NDArray,
    n_clusters: int,
    bins: NDArray,
    show_progress: bool,
) -> NDArray:
    """GPU implementation of co-occurrence."""
    from spatialgpu.core.array_utils import to_gpu, to_cpu
    import cupy as cp

    n_cells = len(cluster_idx)
    n_bins = len(bins) - 1

    coords_gpu = to_gpu(coords.astype(np.float32))
    cluster_idx_gpu = to_gpu(cluster_idx.astype(np.int32))
    bins_gpu = to_gpu(bins.astype(np.float32))

    occurrence = cp.zeros((n_clusters, n_clusters, n_bins), dtype=cp.float32)
    counts = cp.zeros(n_bins, dtype=cp.float32)

    # Process in chunks to manage memory
    chunk_size = min(5000, n_cells)

    iterator = range(0, n_cells, chunk_size)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing co-occurrence")

    for i in iterator:
        end_i = min(i + chunk_size, n_cells)
        chunk_coords = coords_gpu[i:end_i]
        chunk_clusters = cluster_idx_gpu[i:end_i]

        # Compute pairwise distances
        diff = chunk_coords[:, None, :] - coords_gpu[None, :, :]
        dists = cp.sqrt(cp.sum(diff ** 2, axis=2))

        # Bin distances and count co-occurrences
        for b in range(n_bins):
            mask = (dists >= bins_gpu[b]) & (dists < bins_gpu[b + 1])

            for ci in range(n_clusters):
                for cj in range(n_clusters):
                    cluster_mask = (
                        (chunk_clusters[:, None] == ci) &
                        (cluster_idx_gpu[None, :] == cj)
                    )
                    occurrence[ci, cj, b] += cp.sum(mask & cluster_mask)

            counts[b] += cp.sum(mask)

    # Normalize
    counts = cp.maximum(counts, 1)
    occurrence = occurrence / counts[None, None, :]

    return to_cpu(occurrence)


def _co_occurrence_cpu(
    coords: NDArray,
    cluster_idx: NDArray,
    n_clusters: int,
    bins: NDArray,
    show_progress: bool,
) -> NDArray:
    """CPU implementation of co-occurrence."""
    from scipy.spatial.distance import cdist

    n_cells = len(cluster_idx)
    n_bins = len(bins) - 1

    occurrence = np.zeros((n_clusters, n_clusters, n_bins), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.float32)

    # Process in chunks
    chunk_size = min(2000, n_cells)

    iterator = range(0, n_cells, chunk_size)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing co-occurrence")

    for i in iterator:
        end_i = min(i + chunk_size, n_cells)
        dists = cdist(coords[i:end_i], coords)

        for b in range(n_bins):
            mask = (dists >= bins[b]) & (dists < bins[b + 1])

            for ci in range(n_clusters):
                for cj in range(n_clusters):
                    cluster_mask = (
                        (cluster_idx[i:end_i, None] == ci) &
                        (cluster_idx[None, :] == cj)
                    )
                    occurrence[ci, cj, b] += np.sum(mask & cluster_mask)

            counts[b] += np.sum(mask)

    # Normalize
    counts = np.maximum(counts, 1)
    occurrence = occurrence / counts[None, None, :]

    return occurrence


def interaction_matrix(
    adata: ad.AnnData,
    cluster_key: str,
    connectivity_key: str = "spatial_connectivities",
    normalized: bool = True,
    copy: bool = False,
) -> Optional[NDArray]:
    """
    Compute cell-cell interaction matrix.

    Parameters
    ----------
    adata
        Annotated data object with spatial graph.
    cluster_key
        Key in `adata.obs` containing cluster labels.
    connectivity_key
        Key in `adata.obsp` containing connectivity matrix.
    normalized
        Normalize by expected frequencies.
    copy
        Return results instead of storing in adata.

    Returns
    -------
    If `copy=True`, returns interaction matrix.
    Otherwise stores in `adata.uns[f'{cluster_key}_interaction_matrix']`.
    """
    # Get cluster labels
    clusters = adata.obs[cluster_key]
    if hasattr(clusters, "cat"):
        categories = clusters.cat.categories.tolist()
        cluster_idx = clusters.cat.codes.values
    else:
        categories = sorted(clusters.unique())
        cluster_idx = np.array([categories.index(c) for c in clusters])

    n_clusters = len(categories)

    # Get connectivity matrix
    adj = adata.obsp[connectivity_key]
    if not sparse.issparse(adj):
        adj = sparse.csr_matrix(adj)

    # Compute interaction counts
    interaction = _compute_interaction_count_cpu(adj, cluster_idx, n_clusters)

    if normalized:
        # Normalize by expected frequencies
        cluster_counts = np.bincount(cluster_idx, minlength=n_clusters)
        expected = np.outer(cluster_counts, cluster_counts) / len(cluster_idx)
        expected = np.maximum(expected, 1)
        interaction = interaction / expected

    if copy:
        return interaction

    adata.uns[f"{cluster_key}_interaction_matrix"] = {
        "interaction": interaction,
        "categories": categories,
    }
    return None


def centrality_scores(
    adata: ad.AnnData,
    cluster_key: str,
    connectivity_key: str = "spatial_connectivities",
    score_types: Sequence[str] = ("degree", "closeness", "betweenness"),
    copy: bool = False,
) -> Optional[dict[str, NDArray]]:
    """
    Compute graph centrality scores per cluster.

    Parameters
    ----------
    adata
        Annotated data object with spatial graph.
    cluster_key
        Key in `adata.obs` containing cluster labels.
    connectivity_key
        Key in `adata.obsp` containing connectivity matrix.
    score_types
        Types of centrality scores to compute.
    copy
        Return results instead of storing in adata.

    Returns
    -------
    If `copy=True`, returns dict of score arrays.
    Otherwise stores in `adata.uns[f'{cluster_key}_centrality_scores']`.
    """
    from spatialgpu.core.backend import get_backend
    import networkx as nx

    backend = get_backend()

    adj = adata.obsp[connectivity_key]

    # Get cluster labels
    clusters = adata.obs[cluster_key]
    if hasattr(clusters, "cat"):
        categories = clusters.cat.categories.tolist()
        cluster_idx = clusters.cat.codes.values
    else:
        categories = sorted(clusters.unique())
        cluster_idx = np.array([categories.index(c) for c in clusters])

    n_clusters = len(categories)
    n_cells = len(cluster_idx)

    # Build networkx graph
    G = nx.from_scipy_sparse_array(adj)

    scores = {}

    if "degree" in score_types:
        degree = np.array([d for n, d in G.degree()])
        scores["degree"] = np.array([
            np.mean(degree[cluster_idx == c]) for c in range(n_clusters)
        ])

    if "closeness" in score_types:
        closeness = np.array(list(nx.closeness_centrality(G).values()))
        scores["closeness"] = np.array([
            np.mean(closeness[cluster_idx == c]) for c in range(n_clusters)
        ])

    if "betweenness" in score_types:
        # Sample for large graphs
        k = min(1000, n_cells)
        betweenness = np.array(list(
            nx.betweenness_centrality(G, k=k).values()
        ))
        scores["betweenness"] = np.array([
            np.mean(betweenness[cluster_idx == c]) for c in range(n_clusters)
        ])

    if copy:
        return scores

    adata.uns[f"{cluster_key}_centrality_scores"] = {
        "scores": scores,
        "categories": categories,
    }
    return None
