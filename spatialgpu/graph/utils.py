"""
Utility functions for spatial graph operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad
    from numpy.typing import NDArray


def get_spatial_coords(
    adata: ad.AnnData,
    spatial_key: str = "spatial",
    library_id: Optional[str] = None,
) -> NDArray:
    """
    Extract spatial coordinates from AnnData object.

    Parameters
    ----------
    adata
        Annotated data object.
    spatial_key
        Key in `adata.obsm` containing spatial coordinates.
    library_id
        Library ID for Visium data (if applicable).

    Returns
    -------
    coords
        Spatial coordinates array, shape (n_cells, n_dims).

    Raises
    ------
    KeyError
        If spatial coordinates not found.
    """
    # Try obsm first
    if spatial_key in adata.obsm:
        return np.array(adata.obsm[spatial_key])

    # Try uns for Visium-style data
    if "spatial" in adata.uns:
        spatial_uns = adata.uns["spatial"]
        if library_id is None and len(spatial_uns) == 1:
            library_id = list(spatial_uns.keys())[0]

        if library_id and library_id in spatial_uns:
            if "metadata" in spatial_uns[library_id]:
                # Visium format
                pass

    # Check for X_spatial
    if "X_spatial" in adata.obsm:
        return np.array(adata.obsm["X_spatial"])

    raise KeyError(
        f"Spatial coordinates not found. Tried: "
        f"adata.obsm['{spatial_key}'], adata.obsm['X_spatial'], adata.uns['spatial']"
    )


def adjacency_to_edge_list(
    adj: sparse.csr_matrix,
    weighted: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
    """
    Convert adjacency matrix to edge list.

    Parameters
    ----------
    adj
        Sparse adjacency matrix.
    weighted
        If True, also return edge weights.

    Returns
    -------
    sources : array
        Source node indices.
    targets : array
        Target node indices.
    weights : array (optional)
        Edge weights if `weighted=True`.

    Examples
    --------
    >>> sources, targets = adjacency_to_edge_list(adj)
    >>> sources, targets, weights = adjacency_to_edge_list(adj, weighted=True)
    """
    if not sparse.issparse(adj):
        adj = sparse.csr_matrix(adj)

    coo = adj.tocoo()

    if weighted:
        return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
    return coo.row.astype(np.int64), coo.col.astype(np.int64)


def edge_list_to_adjacency(
    sources: NDArray,
    targets: NDArray,
    weights: Optional[NDArray] = None,
    n_nodes: Optional[int] = None,
) -> sparse.csr_matrix:
    """
    Convert edge list to adjacency matrix.

    Parameters
    ----------
    sources
        Source node indices.
    targets
        Target node indices.
    weights
        Edge weights (default: all 1).
    n_nodes
        Number of nodes. If None, inferred from max index.

    Returns
    -------
    adj
        Sparse adjacency matrix in CSR format.

    Examples
    --------
    >>> adj = edge_list_to_adjacency(sources, targets)
    >>> adj = edge_list_to_adjacency(sources, targets, weights=distances)
    """
    if n_nodes is None:
        n_nodes = max(sources.max(), targets.max()) + 1

    if weights is None:
        weights = np.ones(len(sources))

    return sparse.csr_matrix(
        (weights, (sources, targets)),
        shape=(n_nodes, n_nodes),
    )


def compute_graph_laplacian(
    adj: sparse.csr_matrix,
    normalized: bool = True,
) -> sparse.csr_matrix:
    """
    Compute graph Laplacian from adjacency matrix.

    Parameters
    ----------
    adj
        Sparse adjacency matrix.
    normalized
        If True, compute normalized Laplacian.

    Returns
    -------
    laplacian
        Graph Laplacian matrix.
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        return _compute_laplacian_gpu(adj, normalized)
    else:
        return _compute_laplacian_cpu(adj, normalized)


def _compute_laplacian_cpu(
    adj: sparse.csr_matrix,
    normalized: bool,
) -> sparse.csr_matrix:
    """CPU implementation of graph Laplacian."""
    from scipy.sparse.csgraph import laplacian

    return laplacian(adj, normed=normalized)


def _compute_laplacian_gpu(
    adj: sparse.csr_matrix,
    normalized: bool,
) -> sparse.csr_matrix:
    """GPU implementation of graph Laplacian."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    # Convert to GPU
    adj_gpu = cp_sparse.csr_matrix(adj)

    # Compute degree
    degrees = cp.array(adj_gpu.sum(axis=1)).flatten()

    if normalized:
        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = 1.0 / cp.sqrt(cp.maximum(degrees, 1))
        d_inv_sqrt_diag = cp_sparse.diags(d_inv_sqrt)

        laplacian = (
            cp_sparse.eye(adj_gpu.shape[0])
            - d_inv_sqrt_diag @ adj_gpu @ d_inv_sqrt_diag
        )
    else:
        # Unnormalized Laplacian: D - A
        D = cp_sparse.diags(degrees)
        laplacian = D - adj_gpu

    # Convert back to CPU
    return sparse.csr_matrix(laplacian.get())


def subsample_graph(
    adj: sparse.csr_matrix,
    n_samples: int,
    seed: Optional[int] = None,
) -> tuple[sparse.csr_matrix, NDArray]:
    """
    Subsample nodes from a graph.

    Parameters
    ----------
    adj
        Sparse adjacency matrix.
    n_samples
        Number of nodes to sample.
    seed
        Random seed.

    Returns
    -------
    subgraph
        Subsampled adjacency matrix.
    indices
        Indices of sampled nodes.
    """
    if seed is not None:
        np.random.seed(seed)

    n_nodes = adj.shape[0]
    indices = np.random.choice(n_nodes, size=min(n_samples, n_nodes), replace=False)
    indices = np.sort(indices)

    subgraph = adj[indices][:, indices]

    return subgraph, indices


def graph_connected_components(
    adj: sparse.csr_matrix,
) -> tuple[int, NDArray]:
    """
    Find connected components in graph.

    Parameters
    ----------
    adj
        Sparse adjacency matrix.

    Returns
    -------
    n_components
        Number of connected components.
    labels
        Component label for each node.
    """
    from scipy.sparse.csgraph import connected_components

    return connected_components(adj, directed=False)


def filter_edges_by_distance(
    adj: sparse.csr_matrix,
    distances: sparse.csr_matrix,
    min_dist: Optional[float] = None,
    max_dist: Optional[float] = None,
) -> sparse.csr_matrix:
    """
    Filter graph edges by distance.

    Parameters
    ----------
    adj
        Sparse adjacency matrix.
    distances
        Sparse distance matrix.
    min_dist
        Minimum distance threshold.
    max_dist
        Maximum distance threshold.

    Returns
    -------
    filtered_adj
        Filtered adjacency matrix.
    """
    mask = np.ones(distances.data.shape, dtype=bool)

    if min_dist is not None:
        mask &= distances.data >= min_dist
    if max_dist is not None:
        mask &= distances.data <= max_dist

    adj = adj.tocoo()
    filtered = sparse.coo_matrix(
        (adj.data[mask], (adj.row[mask], adj.col[mask])),
        shape=adj.shape,
    )

    return filtered.tocsr()
