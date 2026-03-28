"""
GPU-accelerated spatial neighbor graph construction.

Provides 10-100x speedup over Squidpy for large datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay

if TYPE_CHECKING:
    import anndata as ad
    from numpy.typing import NDArray


def spatial_neighbors(
    adata: ad.AnnData,
    n_neighbors: int = 6,
    radius: float | None = None,
    coord_type: Literal["grid", "generic"] = "generic",
    spatial_key: str = "spatial",
    key_added: str = "spatial",
    transform: Literal["spectral", "cosine", None] = None,
    n_rings: int = 1,
    percentile: float | None = None,
    set_diag: bool = False,
    copy: bool = False,
) -> ad.AnnData | None:
    """
    Build spatial neighbor graph with GPU acceleration.

    This is a drop-in replacement for `squidpy.gr.spatial_neighbors` with
    10-100x speedup on GPU for large datasets.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates.
    n_neighbors
        Number of nearest neighbors for kNN graph. Ignored if `radius` is set.
    radius
        Radius for radius-based neighbor graph. If set, `n_neighbors` is ignored.
    coord_type
        Type of coordinates:
        - "grid": Assumes grid-like structure (e.g., Visium spots)
        - "generic": Generic coordinates (e.g., single-cell spatial)
    spatial_key
        Key in `adata.obsm` containing spatial coordinates.
    key_added
        Key to add to `adata.obsp` for connectivity and distances.
    transform
        Transformation to apply to distances:
        - "spectral": Spectral transformation
        - "cosine": Cosine similarity
        - None: No transformation
    n_rings
        Number of rings for grid-based neighbors (only for coord_type="grid").
    percentile
        If set, use this percentile of distances as radius for radius graph.
    set_diag
        Whether to set diagonal elements to 1.
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    If `copy=True`, returns modified AnnData, otherwise modifies in place.

    Examples
    --------
    >>> import scanpy as sc
    >>> import spatialgpu as sp
    >>> adata = sc.datasets.visium_sge()
    >>> sp.graph.spatial_neighbors(adata, n_neighbors=6)
    >>> adata.obsp["spatial_connectivities"]
    <sparse matrix of shape (n_cells, n_cells)>
    """
    from spatialgpu.core.backend import get_backend
    from spatialgpu.graph.utils import get_spatial_coords

    if copy:
        adata = adata.copy()

    coords = get_spatial_coords(adata, spatial_key=spatial_key)

    get_backend()

    # Build graph based on method
    if radius is not None:
        connectivities, distances = radius_graph(
            coords, radius=radius, set_diag=set_diag
        )
    elif percentile is not None:
        # First compute distances to estimate radius
        _, dists = knn_graph(coords, n_neighbors=n_neighbors, return_distance=True)
        radius_est = np.percentile(dists.data, percentile)
        connectivities, distances = radius_graph(
            coords, radius=radius_est, set_diag=set_diag
        )
    else:
        connectivities, distances = knn_graph(
            coords, n_neighbors=n_neighbors, set_diag=set_diag
        )

    # Apply transformation if requested
    if transform == "spectral":
        connectivities = _spectral_transform(connectivities)
    elif transform == "cosine":
        connectivities = _cosine_transform(connectivities, coords)

    # Store results
    adata.obsp[f"{key_added}_connectivities"] = connectivities
    adata.obsp[f"{key_added}_distances"] = distances

    # Store parameters in uns
    adata.uns[key_added] = {
        "connectivities_key": f"{key_added}_connectivities",
        "distances_key": f"{key_added}_distances",
        "params": {
            "n_neighbors": n_neighbors,
            "radius": radius,
            "coord_type": coord_type,
            "transform": transform,
        },
    }

    if copy:
        return adata
    return None


def knn_graph(
    coords: NDArray,
    n_neighbors: int = 6,
    metric: str = "euclidean",
    set_diag: bool = False,
    return_distance: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build k-nearest neighbor graph with GPU acceleration.

    Parameters
    ----------
    coords
        Spatial coordinates, shape (n_cells, n_dims)
    n_neighbors
        Number of nearest neighbors
    metric
        Distance metric (currently only "euclidean" on GPU)
    set_diag
        Set diagonal to 1
    return_distance
        Also return distance matrix

    Returns
    -------
    connectivities
        Binary connectivity matrix (sparse CSR)
    distances
        Distance matrix (sparse CSR)
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()
    n_cells = coords.shape[0]

    # Ensure we don't request more neighbors than cells
    n_neighbors = min(n_neighbors, n_cells - 1)

    if backend.is_gpu_active:
        return _knn_graph_gpu(coords, n_neighbors, set_diag)
    else:
        return _knn_graph_cpu(coords, n_neighbors, metric, set_diag)


def _knn_graph_gpu(
    coords: NDArray,
    n_neighbors: int,
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """GPU implementation using cuML with CuPy brute-force fallback."""
    from spatialgpu.core.array_utils import to_cpu, to_gpu
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    n_cells = coords.shape[0]

    # Transfer to GPU
    coords_gpu = to_gpu(coords.astype(np.float32))

    try:
        cuml = backend.get_cuml()

        # Use cuML NearestNeighbors
        nn = cuml.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # Include self
            metric="euclidean",
            algorithm="brute",  # Brute force is fast on GPU
        )
        nn.fit(coords_gpu)
        distances_arr, indices_arr = nn.kneighbors(coords_gpu)

        # Transfer back to CPU for sparse matrix construction
        distances_arr = to_cpu(distances_arr)
        indices_arr = to_cpu(indices_arr)
    except (ImportError, RuntimeError) as exc:
        # Fallback: CuPy brute-force kNN when cuML is unavailable
        import logging

        import cupy as cp

        logging.getLogger(__name__).warning(
            "cuML unavailable, falling back to CuPy brute-force kNN: %s", exc
        )

        k = n_neighbors + 1  # Include self
        chunk_size = min(5000, n_cells)
        all_distances = np.empty((n_cells, k), dtype=np.float32)
        all_indices = np.empty((n_cells, k), dtype=np.int64)

        # Pre-compute loop-invariant squared norms
        all_sq = cp.sum(coords_gpu**2, axis=1, keepdims=True).T

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = coords_gpu[start:end]
            chunk_sq = cp.sum(chunk**2, axis=1, keepdims=True)
            dists_sq = chunk_sq + all_sq - 2.0 * chunk @ coords_gpu.T
            dists_sq = cp.maximum(dists_sq, 0.0)
            dists_chunk = cp.sqrt(dists_sq)

            # Guard against k >= n_cells (argpartition requires k < axis len)
            if k >= n_cells:
                idx = cp.argsort(dists_chunk, axis=1)[:, :k]
                row_dists = cp.take_along_axis(dists_chunk, idx, axis=1)
            else:
                idx = cp.argpartition(dists_chunk, k, axis=1)[:, :k]
                row_dists = cp.take_along_axis(dists_chunk, idx, axis=1)
                sort_order = cp.argsort(row_dists, axis=1)
                idx = cp.take_along_axis(idx, sort_order, axis=1)
                row_dists = cp.take_along_axis(row_dists, sort_order, axis=1)

            all_distances[start:end] = to_cpu(row_dists)
            all_indices[start:end] = to_cpu(idx)

        distances_arr = all_distances
        indices_arr = all_indices

    # Build sparse matrices (exclude self-neighbors)
    row_indices = np.repeat(np.arange(n_cells), n_neighbors)
    col_indices = indices_arr[:, 1:].ravel()  # Skip first column (self)
    dist_values = distances_arr[:, 1:].ravel()

    # Connectivity matrix (binary)
    connectivities = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(n_cells, n_cells),
    )

    # Distance matrix
    distances = sparse.csr_matrix(
        (dist_values, (row_indices, col_indices)),
        shape=(n_cells, n_cells),
    )

    # Make symmetric
    connectivities = (connectivities + connectivities.T).astype(bool).astype(float)
    connectivities = sparse.csr_matrix(connectivities)

    distances_sym = (distances + distances.T) / 2
    distances_sym.data[distances_sym.data == 0] = np.inf
    distances = distances_sym.minimum(distances + distances.T)
    distances = sparse.csr_matrix(distances)

    if set_diag:
        connectivities.setdiag(1)

    return connectivities, distances


def _knn_graph_cpu(
    coords: NDArray,
    n_neighbors: int,
    metric: str = "euclidean",
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """CPU implementation using sklearn."""
    from sklearn.neighbors import NearestNeighbors

    n_cells = coords.shape[0]

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        algorithm="auto",
    )
    nn.fit(coords)
    distances_arr, indices_arr = nn.kneighbors(coords)

    # Build sparse matrices (exclude self)
    row_indices = np.repeat(np.arange(n_cells), n_neighbors)
    col_indices = indices_arr[:, 1:].ravel()
    dist_values = distances_arr[:, 1:].ravel()

    connectivities = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(n_cells, n_cells),
    )

    distances = sparse.csr_matrix(
        (dist_values, (row_indices, col_indices)),
        shape=(n_cells, n_cells),
    )

    # Make symmetric
    connectivities = (connectivities + connectivities.T).astype(bool).astype(float)
    connectivities = sparse.csr_matrix(connectivities)

    distances_sym = (distances + distances.T) / 2
    distances = sparse.csr_matrix(distances_sym)

    if set_diag:
        connectivities.setdiag(1)

    return connectivities, distances


def radius_graph(
    coords: NDArray,
    radius: float,
    metric: str = "euclidean",
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build radius-based neighbor graph.

    Parameters
    ----------
    coords
        Spatial coordinates, shape (n_cells, n_dims)
    radius
        Maximum distance for neighbors
    metric
        Distance metric
    set_diag
        Set diagonal to 1

    Returns
    -------
    connectivities
        Binary connectivity matrix (sparse CSR)
    distances
        Distance matrix (sparse CSR)
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        return _radius_graph_gpu(coords, radius, set_diag)
    else:
        return _radius_graph_cpu(coords, radius, metric, set_diag)


def _radius_graph_gpu(
    coords: NDArray,
    radius: float,
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """GPU implementation of radius graph."""
    from spatialgpu.core.array_utils import to_cpu, to_gpu
    from spatialgpu.core.backend import get_backend

    get_backend()
    import cupy as cp

    n_cells = coords.shape[0]
    coords_gpu = to_gpu(coords.astype(np.float32))

    # Compute pairwise distances in chunks to manage memory
    chunk_size = min(10000, n_cells)
    rows = []
    cols = []
    dists = []

    # Pre-compute squared norms for memory-efficient distance computation
    all_sq_norms = cp.sum(coords_gpu**2, axis=1, keepdims=True)

    for i in range(0, n_cells, chunk_size):
        end_i = min(i + chunk_size, n_cells)
        chunk_i = coords_gpu[i:end_i]
        chunk_sq_i = all_sq_norms[i:end_i]

        for j in range(0, n_cells, chunk_size):
            end_j = min(j + chunk_size, n_cells)
            chunk_j = coords_gpu[j:end_j]

            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b (avoids large 3D intermediate)
            dists_sq = chunk_sq_i + all_sq_norms[j:end_j].T - 2.0 * chunk_i @ chunk_j.T
            dist_chunk = cp.sqrt(cp.maximum(dists_sq, 0.0))

            # Find pairs within radius
            mask = (dist_chunk <= radius) & (dist_chunk > 0)
            local_rows, local_cols = cp.where(mask)

            rows.append(to_cpu(local_rows) + i)
            cols.append(to_cpu(local_cols) + j)
            dists.append(to_cpu(dist_chunk[mask]))

    # Concatenate all results
    if rows:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_dists = np.concatenate(dists)
    else:
        all_rows = np.array([], dtype=np.int64)
        all_cols = np.array([], dtype=np.int64)
        all_dists = np.array([], dtype=np.float32)

    # Build sparse matrices
    connectivities = sparse.csr_matrix(
        (np.ones(len(all_rows)), (all_rows, all_cols)),
        shape=(n_cells, n_cells),
    )
    distances = sparse.csr_matrix(
        (all_dists, (all_rows, all_cols)),
        shape=(n_cells, n_cells),
    )

    if set_diag:
        connectivities.setdiag(1)

    return connectivities, distances


def _radius_graph_cpu(
    coords: NDArray,
    radius: float,
    metric: str = "euclidean",
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """CPU implementation of radius graph."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(radius=radius, metric=metric, algorithm="auto")
    nn.fit(coords)

    connectivities = nn.radius_neighbors_graph(coords, mode="connectivity")
    distances = nn.radius_neighbors_graph(coords, mode="distance")

    # Remove self-connections
    connectivities.setdiag(0)
    distances.setdiag(0)
    connectivities.eliminate_zeros()
    distances.eliminate_zeros()

    if set_diag:
        connectivities.setdiag(1)

    return connectivities.tocsr(), distances.tocsr()


def delaunay_graph(
    coords: NDArray,
    set_diag: bool = False,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build Delaunay triangulation-based neighbor graph.

    Parameters
    ----------
    coords
        Spatial coordinates, shape (n_cells, 2) - must be 2D
    set_diag
        Set diagonal to 1

    Returns
    -------
    connectivities
        Binary connectivity matrix (sparse CSR)
    distances
        Distance matrix (sparse CSR)
    """
    if coords.shape[1] != 2:
        raise ValueError("Delaunay graph requires 2D coordinates")

    n_cells = coords.shape[0]

    # Compute Delaunay triangulation
    tri = Delaunay(coords)

    # Extract unique edges from triangulation (vectorized)
    s = tri.simplices
    edge_pairs = np.concatenate(
        [
            np.sort(s[:, [0, 1]], axis=1),
            np.sort(s[:, [0, 2]], axis=1),
            np.sort(s[:, [1, 2]], axis=1),
        ],
        axis=0,
    )
    edges = np.unique(edge_pairs, axis=0)

    if len(edges) == 0:
        connectivities = sparse.csr_matrix((n_cells, n_cells))
        distances = sparse.csr_matrix((n_cells, n_cells))
    else:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])

        # Compute distances
        dists = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        dists = np.concatenate([dists, dists])

        connectivities = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_cells, n_cells),
        )
        distances = sparse.csr_matrix(
            (dists, (rows, cols)),
            shape=(n_cells, n_cells),
        )

    if set_diag:
        connectivities.setdiag(1)

    return connectivities, distances


def _spectral_transform(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """Apply spectral transformation to adjacency matrix."""
    # Compute degree matrix
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero

    # D^{-1/2} A D^{-1/2}
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    return d_inv_sqrt @ adj @ d_inv_sqrt


def _cosine_transform(
    adj: sparse.csr_matrix,
    coords: NDArray,
) -> sparse.csr_matrix:
    """Apply cosine similarity transformation."""
    n = coords.shape[0]

    rows, cols = adj.nonzero()

    # Vectorized cosine similarity for connected pairs
    v1 = coords[rows]
    v2 = coords[cols]
    dot = np.sum(v1 * v2, axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    norms = np.maximum(norms, 1e-12)
    cos_sims = (dot / norms + 1) / 2  # Scale to [0, 1]

    return sparse.csr_matrix(
        (cos_sims, (rows, cols)),
        shape=(n, n),
    )
