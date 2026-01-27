"""
GPU-accelerated spatial graph operations.

This module provides Squidpy-compatible API for spatial graph construction
and analysis, with 10-100x speedup on GPU.

Functions
---------
spatial_neighbors
    Build spatial neighbor graph (kNN or radius-based)
nhood_enrichment
    Neighborhood enrichment analysis
co_occurrence
    Spatial co-occurrence analysis
interaction_matrix
    Cell-cell interaction matrix
ripley
    Ripley's statistics (K, L, F, G functions)
centrality_scores
    Graph centrality measures
"""

from spatialgpu.graph.neighbors import (
    spatial_neighbors,
    knn_graph,
    radius_graph,
    delaunay_graph,
)
from spatialgpu.graph.analysis import (
    nhood_enrichment,
    co_occurrence,
    interaction_matrix,
    centrality_scores,
)
from spatialgpu.graph.ripley import ripley
from spatialgpu.graph.utils import (
    get_spatial_coords,
    adjacency_to_edge_list,
    edge_list_to_adjacency,
)

__all__ = [
    # Graph construction
    "spatial_neighbors",
    "knn_graph",
    "radius_graph",
    "delaunay_graph",
    # Analysis
    "nhood_enrichment",
    "co_occurrence",
    "interaction_matrix",
    "centrality_scores",
    "ripley",
    # Utilities
    "get_spatial_coords",
    "adjacency_to_edge_list",
    "edge_list_to_adjacency",
]
