"""
Benchmarking utilities for spatial-gpu.

Provides tools to measure and compare performance between CPU and GPU
implementations, and against other libraries like Squidpy.
"""

from spatialgpu.benchmarks.runner import (
    benchmark,
    compare_backends,
    benchmark_suite,
    BenchmarkResult,
)
from spatialgpu.benchmarks.synthetic import (
    generate_synthetic_data,
    generate_spatial_clusters,
)

__all__ = [
    "benchmark",
    "compare_backends",
    "benchmark_suite",
    "BenchmarkResult",
    "generate_synthetic_data",
    "generate_spatial_clusters",
]
