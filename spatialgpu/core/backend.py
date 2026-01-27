"""
Backend management for CPU/GPU computation.

Provides automatic detection and seamless switching between CPU (NumPy/SciPy)
and GPU (CuPy/cuML/cuGraph) backends.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from typing import Optional


class BackendType(Enum):
    """Available computation backends."""

    CPU = auto()
    CUDA = auto()
    # Future: ROCm support
    # ROCM = auto()


@dataclass
class GPUInfo:
    """Information about available GPU device."""

    device_id: int
    name: str
    total_memory: int  # bytes
    free_memory: int   # bytes
    compute_capability: tuple[int, int]

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024**3)

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024**3)

    def __repr__(self) -> str:
        return (
            f"GPUInfo(id={self.device_id}, name='{self.name}', "
            f"memory={self.total_memory_gb:.1f}GB, "
            f"compute_capability={self.compute_capability})"
        )


class Backend:
    """
    Manages computation backend selection and GPU resources.

    The Backend class provides:
    - Automatic GPU detection and initialization
    - Seamless fallback to CPU when GPU unavailable
    - Memory management and device selection
    - Array module abstraction (numpy/cupy)

    Examples
    --------
    >>> import spatialgpu as sp
    >>> backend = sp.get_backend()
    >>> backend.is_gpu_available
    True
    >>> backend.device_info
    GPUInfo(id=0, name='NVIDIA A100', memory=40.0GB, ...)

    >>> # Force CPU backend
    >>> sp.set_backend("cpu")
    >>> backend = sp.get_backend()
    >>> backend.backend_type
    <BackendType.CPU: 1>
    """

    _instance: Optional[Backend] = None
    _initialized: bool = False

    def __new__(cls) -> Backend:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if Backend._initialized:
            return

        self._backend_type: BackendType = BackendType.CPU
        self._gpu_available: bool = False
        self._gpu_info: Optional[GPUInfo] = None
        self._device_id: int = 0

        # Lazy-loaded modules
        self._cupy = None
        self._cuml = None
        self._cugraph = None

        # Try to initialize GPU
        self._detect_gpu()
        Backend._initialized = True

    def _detect_gpu(self) -> None:
        """Detect available GPU and initialize CUDA if available."""
        try:
            import cupy as cp

            # Check if CUDA is actually available
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                self._gpu_available = False
                return

            # Get device info
            device = cp.cuda.Device(self._device_id)
            props = cp.cuda.runtime.getDeviceProperties(device.id)

            mem_info = device.mem_info
            self._gpu_info = GPUInfo(
                device_id=device.id,
                name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                total_memory=mem_info[1],
                free_memory=mem_info[0],
                compute_capability=(props["major"], props["minor"]),
            )

            self._gpu_available = True
            self._backend_type = BackendType.CUDA
            self._cupy = cp

        except ImportError:
            self._gpu_available = False
            warnings.warn(
                "CuPy not installed. GPU acceleration unavailable. "
                "Install with: pip install spatial-gpu[cuda]",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:
            self._gpu_available = False
            warnings.warn(
                f"GPU detection failed: {e}. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def backend_type(self) -> BackendType:
        """Current active backend type."""
        return self._backend_type

    @property
    def is_gpu_available(self) -> bool:
        """Whether GPU computation is available."""
        return self._gpu_available

    @property
    def is_gpu_active(self) -> bool:
        """Whether GPU is currently the active backend."""
        return self._backend_type == BackendType.CUDA

    @property
    def device_info(self) -> Optional[GPUInfo]:
        """Information about the active GPU device."""
        return self._gpu_info

    @property
    def device_id(self) -> int:
        """Active GPU device ID."""
        return self._device_id

    @device_id.setter
    def device_id(self, value: int) -> None:
        """Set active GPU device."""
        if not self._gpu_available:
            raise RuntimeError("No GPU available")

        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if value < 0 or value >= device_count:
            raise ValueError(f"Invalid device ID {value}. Available: 0-{device_count-1}")

        self._device_id = value
        cp.cuda.Device(value).use()
        self._detect_gpu()  # Refresh device info

    def set_backend(self, backend: Literal["cpu", "cuda", "gpu", "auto"]) -> None:
        """
        Set the computation backend.

        Parameters
        ----------
        backend : {"cpu", "cuda", "gpu", "auto"}
            Backend to use:
            - "cpu": Force CPU computation
            - "cuda" or "gpu": Force GPU computation (raises if unavailable)
            - "auto": Use GPU if available, otherwise CPU
        """
        backend = backend.lower()

        if backend == "cpu":
            self._backend_type = BackendType.CPU
        elif backend in ("cuda", "gpu"):
            if not self._gpu_available:
                raise RuntimeError(
                    "GPU backend requested but no GPU available. "
                    "Install CUDA toolkit and cupy: pip install spatial-gpu[cuda]"
                )
            self._backend_type = BackendType.CUDA
        elif backend == "auto":
            self._backend_type = BackendType.CUDA if self._gpu_available else BackendType.CPU
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'cpu', 'cuda', or 'auto'")

    @property
    def xp(self):
        """
        Get the array module (numpy or cupy) for current backend.

        Returns
        -------
        module
            numpy if CPU backend, cupy if GPU backend
        """
        if self._backend_type == BackendType.CUDA:
            if self._cupy is None:
                import cupy as cp
                self._cupy = cp
            return self._cupy
        return np

    @property
    def scipy(self):
        """
        Get scipy or cupyx.scipy for current backend.

        Returns
        -------
        module
            scipy if CPU backend, cupyx.scipy if GPU backend
        """
        if self._backend_type == BackendType.CUDA:
            import cupyx.scipy
            return cupyx.scipy
        import scipy
        return scipy

    def get_cuml(self):
        """Get cuML module (GPU machine learning)."""
        if not self._gpu_available:
            raise RuntimeError("cuML requires GPU. No GPU available.")
        if self._cuml is None:
            try:
                import cuml
                self._cuml = cuml
            except ImportError:
                raise ImportError(
                    "cuML not installed. Install with: pip install spatial-gpu[cuda]"
                )
        return self._cuml

    def get_cugraph(self):
        """Get cuGraph module (GPU graph analytics)."""
        if not self._gpu_available:
            raise RuntimeError("cuGraph requires GPU. No GPU available.")
        if self._cugraph is None:
            try:
                import cugraph
                self._cugraph = cugraph
            except ImportError:
                raise ImportError(
                    "cuGraph not installed. Install with: pip install spatial-gpu[cuda]"
                )
        return self._cugraph

    def memory_info(self) -> dict[str, Any]:
        """
        Get current GPU memory usage.

        Returns
        -------
        dict
            Memory information with keys: total, free, used (all in bytes)
        """
        if not self._gpu_available:
            return {"total": 0, "free": 0, "used": 0, "backend": "cpu"}

        import cupy as cp
        device = cp.cuda.Device(self._device_id)
        free, total = device.mem_info

        return {
            "total": total,
            "free": free,
            "used": total - free,
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "used_gb": (total - free) / (1024**3),
            "backend": "cuda",
        }

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self._gpu_available and self._cupy is not None:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def __repr__(self) -> str:
        status = "active" if self.is_gpu_active else "inactive"
        if self._gpu_available:
            return (
                f"Backend(type={self._backend_type.name}, "
                f"gpu={status}, device={self._gpu_info})"
            )
        return f"Backend(type={self._backend_type.name}, gpu=unavailable)"


# Global backend instance
_backend: Optional[Backend] = None


def get_backend() -> Backend:
    """
    Get the global Backend instance.

    Returns
    -------
    Backend
        The global backend manager

    Examples
    --------
    >>> import spatialgpu as sp
    >>> backend = sp.get_backend()
    >>> print(backend.is_gpu_available)
    True
    """
    global _backend
    if _backend is None:
        _backend = Backend()
    return _backend


def set_backend(backend: Literal["cpu", "cuda", "gpu", "auto"]) -> None:
    """
    Set the global computation backend.

    Parameters
    ----------
    backend : {"cpu", "cuda", "gpu", "auto"}
        Backend to use. "auto" uses GPU if available.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> sp.set_backend("gpu")  # Force GPU
    >>> sp.set_backend("cpu")  # Force CPU
    >>> sp.set_backend("auto") # Auto-detect
    """
    get_backend().set_backend(backend)
