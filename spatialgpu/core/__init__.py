"""Core functionality for spatial-gpu."""

from spatialgpu.core.backend import Backend, get_backend, set_backend
from spatialgpu.core.config import config, GPUConfig
from spatialgpu.core.array_utils import to_gpu, to_cpu, get_array_module

__all__ = [
    "Backend",
    "get_backend",
    "set_backend",
    "config",
    "GPUConfig",
    "to_gpu",
    "to_cpu",
    "get_array_module",
]
