"""Core functionality for spatial-gpu."""

from spatialgpu.core.array_utils import get_array_module, to_cpu, to_gpu
from spatialgpu.core.backend import Backend, get_backend, set_backend
from spatialgpu.core.config import GPUConfig, config

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
