"""
Configuration management for spatial-gpu.

Provides global settings for GPU memory, parallelization, and computation options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class GPUConfig:
    """
    Configuration for GPU computation.

    Attributes
    ----------
    device_id : int
        GPU device to use (default: 0)
    memory_limit : float or None
        Maximum GPU memory to use in GB (None = no limit)
    memory_pool : bool
        Use CuPy memory pool for faster allocations
    pinned_memory : bool
        Use pinned (page-locked) memory for faster CPU-GPU transfers
    unified_memory : bool
        Use CUDA unified memory for out-of-core computation
    stream_count : int
        Number of CUDA streams for parallel kernel execution
    chunk_size : int
        Default chunk size for batched operations (number of cells)
    """

    device_id: int = 0
    memory_limit: Optional[float] = None  # GB
    memory_pool: bool = True
    pinned_memory: bool = True
    unified_memory: bool = False
    stream_count: int = 4
    chunk_size: int = 100_000

    def apply(self) -> None:
        """Apply this configuration to the GPU backend."""
        from spatialgpu.core.backend import get_backend

        backend = get_backend()
        if not backend.is_gpu_available:
            return

        backend.device_id = self.device_id

        if self.memory_limit is not None:
            import cupy as cp

            # Set memory limit
            mempool = cp.get_default_memory_pool()
            limit_bytes = int(self.memory_limit * 1024**3)
            mempool.set_limit(size=limit_bytes)


@dataclass
class ComputeConfig:
    """
    Configuration for computation behavior.

    Attributes
    ----------
    n_jobs : int
        Number of parallel jobs for CPU computation (-1 = all cores)
    backend : str
        Preferred backend ("auto", "cpu", "cuda")
    precision : str
        Floating point precision ("float32", "float64")
    seed : int or None
        Random seed for reproducibility
    verbose : bool
        Enable verbose output
    progress_bar : bool
        Show progress bars for long operations
    """

    n_jobs: int = -1
    backend: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["float32", "float64"] = "float32"
    seed: Optional[int] = None
    verbose: bool = False
    progress_bar: bool = True


@dataclass
class GraphConfig:
    """
    Configuration for spatial graph operations.

    Attributes
    ----------
    default_coord_type : str
        Default coordinate type for spatial operations
    default_n_neighbors : int
        Default number of neighbors for kNN graphs
    default_radius : float or None
        Default radius for radius graphs
    use_kd_tree : bool
        Use KD-tree for CPU neighbor search (faster for low dimensions)
    use_ball_tree : bool
        Use Ball tree for radius queries
    """

    default_coord_type: Literal["generic", "grid"] = "generic"
    default_n_neighbors: int = 6
    default_radius: Optional[float] = None
    use_kd_tree: bool = True
    use_ball_tree: bool = True


@dataclass
class SegmentationConfig:
    """
    Configuration for cell segmentation.

    Attributes
    ----------
    default_model : str
        Default segmentation model
    gpu_batch_size : int
        Batch size for GPU inference
    tile_size : int
        Tile size for processing large images
    tile_overlap : int
        Overlap between tiles
    min_cell_size : int
        Minimum cell size in pixels
    """

    default_model: str = "cellpose"
    gpu_batch_size: int = 8
    tile_size: int = 2048
    tile_overlap: int = 128
    min_cell_size: int = 15


@dataclass
class Config:
    """
    Global configuration for spatial-gpu.

    Access via `spatialgpu.config`.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> sp.config.compute.backend = "cuda"
    >>> sp.config.gpu.memory_limit = 8.0  # Limit to 8GB
    >>> sp.config.compute.precision = "float32"
    """

    gpu: GPUConfig = field(default_factory=GPUConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    def reset(self) -> None:
        """Reset all configuration to defaults."""
        self.gpu = GPUConfig()
        self.compute = ComputeConfig()
        self.graph = GraphConfig()
        self.segmentation = SegmentationConfig()

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        from dataclasses import asdict
        return {
            "gpu": asdict(self.gpu),
            "compute": asdict(self.compute),
            "graph": asdict(self.graph),
            "segmentation": asdict(self.segmentation),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Create configuration from dictionary."""
        return cls(
            gpu=GPUConfig(**d.get("gpu", {})),
            compute=ComputeConfig(**d.get("compute", {})),
            graph=GraphConfig(**d.get("graph", {})),
            segmentation=SegmentationConfig(**d.get("segmentation", {})),
        )


# Global configuration instance
config = Config()
