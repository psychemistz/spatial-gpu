"""
GPU-accelerated spatial visualization.

Provides efficient visualization for large spatial datasets with
GPU-accelerated rendering and interactive exploration.

Functions
---------
spatial_scatter
    Scatter plot of spatial coordinates with gene/cluster coloring
spatial_heatmap
    Heatmap visualization of spatial features
nhood_enrichment
    Visualize neighborhood enrichment results
co_occurrence
    Visualize co-occurrence patterns
segmentation_overlay
    Overlay segmentation masks on images
interactive_viewer
    Launch interactive napari viewer
"""

from spatialgpu.visualization.plotting import (
    spatial_scatter,
    spatial_heatmap,
    nhood_enrichment_plot,
    co_occurrence_plot,
    interaction_matrix_plot,
    ripley_plot,
)
from spatialgpu.visualization.segmentation import (
    segmentation_overlay,
    show_masks,
    show_boundaries,
)

__all__ = [
    # Spatial plots
    "spatial_scatter",
    "spatial_heatmap",
    # Analysis plots
    "nhood_enrichment_plot",
    "co_occurrence_plot",
    "interaction_matrix_plot",
    "ripley_plot",
    # Segmentation visualization
    "segmentation_overlay",
    "show_masks",
    "show_boundaries",
]
