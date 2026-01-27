"""
Benchmarking utilities for spatial-gpu.

Provides tools to measure and compare performance between CPU and GPU
implementations, and against other libraries like Squidpy.
"""

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
]
