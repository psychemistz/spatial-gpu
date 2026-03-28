"""
Benchmarking utilities for spatial-gpu.

Provides tools to measure and compare performance between CPU and GPU
implementations, and against other libraries like Squidpy.
"""

from spatialgpu.benchmarks.pseudobulk import (
    compare_methods,
    evaluate_deconvolution,
    export_for_cibersortx,
    export_for_music,
    generate_pseudobulk_dirichlet,
    generate_pseudobulk_titration,
    generate_semi_synthetic_scrna,
    import_external_results,
)
from spatialgpu.benchmarks.runner import (
    BenchmarkResult,
    benchmark,
    benchmark_suite,
    compare_backends,
)
from spatialgpu.benchmarks.synthetic import (
    generate_spatial_clusters,
    generate_synthetic_data,
)

__all__ = [
    "benchmark",
    "compare_backends",
    "benchmark_suite",
    "BenchmarkResult",
    "generate_synthetic_data",
    "generate_spatial_clusters",
    # Pseudobulk benchmark
    "generate_semi_synthetic_scrna",
    "generate_pseudobulk_dirichlet",
    "generate_pseudobulk_titration",
    "evaluate_deconvolution",
    "export_for_music",
    "export_for_cibersortx",
    "import_external_results",
    "compare_methods",
]
