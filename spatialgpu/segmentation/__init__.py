"""
GPU-accelerated cell segmentation for spatial omics.

Provides unified interface to multiple segmentation models with
GPU acceleration and tiled processing for large images.

Models Supported
----------------
- Cellpose: General-purpose cell segmentation
- StarDist: Star-convex polygon segmentation
- SAM: Segment Anything Model adaptation
- Custom: User-defined models

Functions
---------
segment_cells
    Segment cells from images with automatic model selection
segment_nuclei
    Segment nuclei specifically
segment_transcripts
    Assign transcripts to cells
evaluate_segmentation
    Evaluate segmentation quality
"""

from spatialgpu.segmentation.core import (
    CellSegmenter,
    segment_cells,
    segment_nuclei,
)
from spatialgpu.segmentation.evaluation import (
    compute_segmentation_metrics,
    evaluate_segmentation,
)
from spatialgpu.segmentation.models import (
    CellposeModel,
    StarDistModel,
    get_available_models,
)
from spatialgpu.segmentation.transcript import (
    assign_transcripts_to_cells,
    segment_transcripts,
)

__all__ = [
    # Core functions
    "segment_cells",
    "segment_nuclei",
    "CellSegmenter",
    # Models
    "CellposeModel",
    "StarDistModel",
    "get_available_models",
    # Transcript assignment
    "segment_transcripts",
    "assign_transcripts_to_cells",
    # Evaluation
    "evaluate_segmentation",
    "compute_segmentation_metrics",
]
