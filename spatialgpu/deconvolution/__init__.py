"""
SpaCET-compatible cell type deconvolution for spatial transcriptomics.

GPU-accelerated Python implementation of the SpaCET algorithm
(Nature Communications 14, 568, 2023) for estimating cell type fractions
in tumor spatial transcriptomics data.
"""

from __future__ import annotations

from spatialgpu.deconvolution.core import cormat, deconvolution
from spatialgpu.deconvolution.extensions import (
    deconvolution_malignant,
    deconvolution_malignant_custom_scrnaseq,
    deconvolution_matched_scrnaseq,
    generate_ref,
)
from spatialgpu.deconvolution.gene_set_score import gene_set_score
from spatialgpu.deconvolution.interaction import (
    cci_cell_type_pair,
    cci_colocalization,
    cci_lr_network_score,
    combine_interface,
    distance_to_interface,
    identify_interface,
)
from spatialgpu.deconvolution.io import (
    create_spacet_object,
    create_spacet_object_10x,
    quality_control,
)
from spatialgpu.deconvolution.reference import (
    get_cancer_signature,
    load_comb_ref,
    load_gene_set,
    load_lr_database,
    read_gmt,
    write_gmt,
)
from spatialgpu.deconvolution.spatial_correlation import (
    cal_weights,
    spatial_correlation,
)
from spatialgpu.deconvolution.visualization import (
    visualize_cell_type_pair,
    visualize_colocalization,
    visualize_distance_to_interface,
    visualize_spatial_feature,
)

__all__ = [
    # Core deconvolution
    "deconvolution",
    "deconvolution_malignant",
    "deconvolution_matched_scrnaseq",
    "deconvolution_malignant_custom_scrnaseq",
    "generate_ref",
    "cormat",
    # Gene set scoring
    "gene_set_score",
    # Cell-cell interaction
    "cci_colocalization",
    "cci_lr_network_score",
    "cci_cell_type_pair",
    "identify_interface",
    "combine_interface",
    "distance_to_interface",
    # Spatial correlation
    "cal_weights",
    "spatial_correlation",
    # Reference data
    "load_comb_ref",
    "load_gene_set",
    "load_lr_database",
    "get_cancer_signature",
    "read_gmt",
    "write_gmt",
    # Visualization
    "visualize_spatial_feature",
    "visualize_colocalization",
    "visualize_cell_type_pair",
    "visualize_distance_to_interface",
    # I/O
    "create_spacet_object",
    "create_spacet_object_10x",
    "quality_control",
]
