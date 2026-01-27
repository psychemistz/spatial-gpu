"""
Benchmark runner for performance comparison.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import anndata as ad


@dataclass
class BenchmarkResult:
    """
    Results from a benchmark run.

    Attributes
    ----------
    name : str
        Name of the benchmark.
    backend : str
        Backend used ("cpu" or "gpu").
    times : list[float]
        Execution times for each run.
    mean_time : float
        Mean execution time.
    std_time : float
        Standard deviation of execution time.
    min_time : float
        Minimum execution time.
    max_time : float
        Maximum execution time.
    n_cells : int
        Number of cells in dataset.
    memory_peak : float
        Peak memory usage (if tracked).
    metadata : dict
        Additional metadata.
    """

    name: str
    backend: str
    times: list[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_cells: int = 0
    memory_peak: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def speedup_vs(self) -> dict[str, float]:
        """Speedup ratios vs other results (if available)."""
        return self.metadata.get("speedup_vs", {})

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name='{self.name}', backend='{self.backend}', "
            f"mean={self.mean_time:.4f}s, std={self.std_time:.4f}s, "
            f"n_cells={self.n_cells})"
        )


def benchmark(
    func: Callable,
    *args,
    n_runs: int = 5,
    warmup: int = 1,
    name: Optional[str] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Benchmark a function.

    Parameters
    ----------
    func
        Function to benchmark.
    *args
        Arguments to pass to function.
    n_runs
        Number of benchmark runs.
    warmup
        Number of warmup runs (not counted).
    name
        Name for the benchmark.
    **kwargs
        Keyword arguments to pass to function.

    Returns
    -------
    BenchmarkResult
        Benchmark results.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> result = sp.benchmarks.benchmark(
    ...     sp.graph.spatial_neighbors,
    ...     adata, n_neighbors=6,
    ...     n_runs=5
    ... )
    >>> print(f"Mean time: {result.mean_time:.4f}s")
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if name is None:
        name = func.__name__

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
        gc.collect()
        if backend.is_gpu_active:
            backend.clear_memory()

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        gc.collect()
        if backend.is_gpu_active:
            backend.clear_memory()
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        _ = func(*args, **kwargs)

        if backend.is_gpu_active:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    # Get n_cells if adata provided
    n_cells = 0
    for arg in args:
        if hasattr(arg, "n_obs"):
            n_cells = arg.n_obs
            break

    return BenchmarkResult(
        name=name,
        backend="gpu" if backend.is_gpu_active else "cpu",
        times=times,
        mean_time=np.mean(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        n_cells=n_cells,
    )


def compare_backends(
    func: Callable,
    *args,
    n_runs: int = 5,
    warmup: int = 1,
    name: Optional[str] = None,
    **kwargs,
) -> dict[str, BenchmarkResult]:
    """
    Compare function performance between CPU and GPU backends.

    Parameters
    ----------
    func
        Function to benchmark.
    *args
        Arguments to pass to function.
    n_runs
        Number of benchmark runs.
    warmup
        Number of warmup runs.
    name
        Name for the benchmark.
    **kwargs
        Keyword arguments to pass to function.

    Returns
    -------
    dict
        Results for each backend: {"cpu": result, "gpu": result, "speedup": float}

    Examples
    --------
    >>> results = sp.benchmarks.compare_backends(
    ...     sp.graph.spatial_neighbors,
    ...     adata, n_neighbors=6
    ... )
    >>> print(f"GPU speedup: {results['speedup']:.1f}x")
    """
    from spatialgpu.core.backend import get_backend, set_backend

    if name is None:
        name = func.__name__

    results = {}

    # CPU benchmark
    set_backend("cpu")
    results["cpu"] = benchmark(
        func, *args,
        n_runs=n_runs, warmup=warmup, name=name,
        **kwargs,
    )

    # GPU benchmark (if available)
    backend = get_backend()
    if backend.is_gpu_available:
        set_backend("gpu")
        results["gpu"] = benchmark(
            func, *args,
            n_runs=n_runs, warmup=warmup, name=name,
            **kwargs,
        )

        # Calculate speedup
        results["speedup"] = results["cpu"].mean_time / results["gpu"].mean_time
        results["gpu"].metadata["speedup_vs"] = {"cpu": results["speedup"]}
    else:
        results["speedup"] = 1.0

    return results


def benchmark_suite(
    adata: ad.AnnData,
    operations: Optional[Sequence[str]] = None,
    n_runs: int = 5,
    compare: bool = True,
) -> dict[str, Any]:
    """
    Run comprehensive benchmark suite.

    Parameters
    ----------
    adata
        Annotated data object to use for benchmarks.
    operations
        Specific operations to benchmark. If None, runs all.
        Options: "neighbors", "nhood_enrichment", "co_occurrence", "ripley"
    n_runs
        Number of runs per benchmark.
    compare
        Compare CPU vs GPU.

    Returns
    -------
    dict
        Benchmark results for all operations.

    Examples
    --------
    >>> adata = sp.benchmarks.generate_synthetic_data(n_cells=100000)
    >>> results = sp.benchmarks.benchmark_suite(adata)
    >>> for op, res in results.items():
    ...     print(f"{op}: {res['speedup']:.1f}x speedup")
    """
    import spatialgpu as sp

    if operations is None:
        operations = ["neighbors", "nhood_enrichment", "co_occurrence", "ripley"]

    results = {}

    # Ensure cluster labels exist
    if "cluster" not in adata.obs.columns:
        import numpy as np
        adata.obs["cluster"] = np.random.randint(0, 10, size=adata.n_obs).astype(str)
        adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    if "neighbors" in operations:
        if compare:
            results["neighbors"] = compare_backends(
                sp.graph.spatial_neighbors,
                adata, n_neighbors=6,
                n_runs=n_runs,
                name="spatial_neighbors",
            )
        else:
            results["neighbors"] = benchmark(
                sp.graph.spatial_neighbors,
                adata, n_neighbors=6,
                n_runs=n_runs,
                name="spatial_neighbors",
            )

    # Ensure spatial graph exists for remaining operations
    if "spatial_connectivities" not in adata.obsp:
        sp.graph.spatial_neighbors(adata, n_neighbors=6)

    if "nhood_enrichment" in operations:
        if compare:
            results["nhood_enrichment"] = compare_backends(
                sp.graph.nhood_enrichment,
                adata, cluster_key="cluster", n_perms=100,
                n_runs=n_runs,
                name="nhood_enrichment",
            )
        else:
            results["nhood_enrichment"] = benchmark(
                sp.graph.nhood_enrichment,
                adata, cluster_key="cluster", n_perms=100,
                n_runs=n_runs,
                name="nhood_enrichment",
            )

    if "co_occurrence" in operations:
        if compare:
            results["co_occurrence"] = compare_backends(
                sp.graph.co_occurrence,
                adata, cluster_key="cluster", n_splits=20,
                n_runs=n_runs,
                name="co_occurrence",
            )
        else:
            results["co_occurrence"] = benchmark(
                sp.graph.co_occurrence,
                adata, cluster_key="cluster", n_splits=20,
                n_runs=n_runs,
                name="co_occurrence",
            )

    if "ripley" in operations:
        if compare:
            results["ripley"] = compare_backends(
                sp.graph.ripley,
                adata, mode="L", n_simulations=50,
                n_runs=n_runs,
                name="ripley",
            )
        else:
            results["ripley"] = benchmark(
                sp.graph.ripley,
                adata, mode="L", n_simulations=50,
                n_runs=n_runs,
                name="ripley",
            )

    return results


def format_benchmark_results(
    results: dict[str, Any],
    format: Literal["table", "markdown", "dict"] = "table",
) -> str | dict:
    """
    Format benchmark results for display.

    Parameters
    ----------
    results
        Benchmark results from benchmark_suite.
    format
        Output format.

    Returns
    -------
    str or dict
        Formatted results.
    """
    if format == "dict":
        return {
            op: {
                "cpu_time": res.get("cpu", {}).mean_time if isinstance(res, dict) else res.mean_time,
                "gpu_time": res.get("gpu", {}).mean_time if isinstance(res, dict) and "gpu" in res else None,
                "speedup": res.get("speedup", 1.0) if isinstance(res, dict) else 1.0,
            }
            for op, res in results.items()
        }

    lines = []

    if format == "markdown":
        lines.append("| Operation | CPU (s) | GPU (s) | Speedup |")
        lines.append("|-----------|---------|---------|---------|")
    else:
        lines.append(f"{'Operation':<20} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
        lines.append("-" * 56)

    for op, res in results.items():
        if isinstance(res, dict):
            cpu_time = res.get("cpu", BenchmarkResult("", "", [], 0, 0, 0, 0)).mean_time
            gpu_time = res.get("gpu", BenchmarkResult("", "", [], 0, 0, 0, 0)).mean_time
            speedup = res.get("speedup", 1.0)
        else:
            cpu_time = res.mean_time
            gpu_time = 0
            speedup = 1.0

        if format == "markdown":
            lines.append(f"| {op} | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.1f}x |")
        else:
            lines.append(f"{op:<20} {cpu_time:>12.4f} {gpu_time:>12.4f} {speedup:>9.1f}x")

    return "\n".join(lines)
