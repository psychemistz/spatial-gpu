"""
spatial-gpu: GPU-accelerated spatial omics analysis framework.

A high-performance library for spatial transcriptomics and spatial omics analysis,
designed for seamless integration with the scverse ecosystem (Scanpy, Squidpy, AnnData).

Key Features:
- 10-100x speedup on common spatial analysis operations
- Drop-in replacement for Squidpy functions
- GPU-accelerated cell segmentation
- Memory-efficient handling of 100M+ cell datasets
- Full compatibility with AnnData and SpatialData formats
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spatial-gpu")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

from spatialgpu.core.backend import get_backend, set_backend, Backend
from spatialgpu.core.config import config, GPUConfig

# Graph operations (Squidpy-compatible API)
from spatialgpu import graph
from spatialgpu import segmentation
from spatialgpu import visualization as viz
from spatialgpu import io
from spatialgpu import benchmarks

__all__ = [
    "__version__",
    # Backend management
    "get_backend",
    "set_backend",
    "Backend",
    "config",
    "GPUConfig",
    # Submodules
    "graph",
    "segmentation",
    "viz",
    "io",
    "benchmarks",
]
