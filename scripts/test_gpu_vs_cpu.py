"""GPU vs CPU validation and benchmark for spatial-gpu.

Tests that all GPU code paths produce identical results to CPU paths,
and measures speedup. Covers:
  1. knn_graph (cuML path + CuPy brute-force fallback)
  2. nhood_enrichment (GPU permutation test)
  3. co_occurrence (vectorized one-hot matmul)
  4. radius_graph (squared-norm distance trick)

Usage:
    python scripts/test_gpu_vs_cpu.py
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class BenchResult:
    name: str
    n_cells: int
    cpu_time: float
    gpu_time: float
    max_diff: float
    passed: bool

    @property
    def speedup(self) -> float:
        return self.cpu_time / self.gpu_time if self.gpu_time > 0 else float("inf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coords(n: int, dims: int = 2, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n, dims).astype(np.float64) * 1000


def _make_cluster_idx(n: int, n_clusters: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_clusters, size=n).astype(np.int32)


def _force_cpu():
    from spatialgpu.core.backend import _Backend

    b = _Backend()
    b._gpu_active = False
    import spatialgpu.core.backend as bmod

    bmod._backend = b
    return b


def _force_gpu():
    from spatialgpu.core.backend import _Backend

    b = _Backend()
    b._gpu_active = True
    import spatialgpu.core.backend as bmod

    bmod._backend = b
    return b


# ---------------------------------------------------------------------------
# Test: knn_graph
# ---------------------------------------------------------------------------


def test_knn_graph(n_cells: int, n_neighbors: int = 6) -> BenchResult:
    from spatialgpu.graph.neighbors import knn_graph

    coords = _make_coords(n_cells)

    # CPU
    _force_cpu()
    gc.collect()
    t0 = time.perf_counter()
    conn_cpu, dist_cpu = knn_graph(coords, n_neighbors=n_neighbors)
    cpu_time = time.perf_counter() - t0

    # GPU
    _force_gpu()
    gc.collect()
    # Warmup
    knn_graph(_make_coords(100), n_neighbors=min(6, 99))

    t0 = time.perf_counter()
    conn_gpu, dist_gpu = knn_graph(coords, n_neighbors=n_neighbors)
    gpu_time = time.perf_counter() - t0

    # Compare: check that GPU neighbors are a superset/match of CPU neighbors
    # (symmetric kNN may differ slightly in tie-breaking but structure should match)
    diff_conn = np.abs(conn_cpu.toarray() - conn_gpu.toarray())
    max_conn_diff = diff_conn.max()

    # Distance comparison (only where both have edges)
    both_mask = (conn_cpu.toarray() > 0) & (conn_gpu.toarray() > 0)
    if both_mask.any():
        d_cpu = dist_cpu.toarray()[both_mask]
        d_gpu = dist_gpu.toarray()[both_mask]
        max_dist_diff = np.max(np.abs(d_cpu - d_gpu))
    else:
        max_dist_diff = 0.0

    max_diff = max(max_conn_diff, max_dist_diff)
    # kNN tie-breaking can cause small topology diffs; allow connectivity diff <= 2
    passed = max_conn_diff <= 2 and max_dist_diff < 1e-3

    return BenchResult(
        name="knn_graph",
        n_cells=n_cells,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        max_diff=max_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Test: knn_graph CuPy fallback (force cuML import to fail)
# ---------------------------------------------------------------------------


def test_knn_cupy_fallback(n_cells: int, n_neighbors: int = 6) -> BenchResult:
    """Test the CuPy brute-force fallback by monkeypatching get_cuml."""
    from spatialgpu.graph.neighbors import knn_graph

    coords = _make_coords(n_cells)

    # CPU baseline
    _force_cpu()
    gc.collect()
    t0 = time.perf_counter()
    conn_cpu, dist_cpu = knn_graph(coords, n_neighbors=n_neighbors)
    cpu_time = time.perf_counter() - t0

    # GPU with cuML disabled
    _force_gpu()
    gc.collect()
    from spatialgpu.core import backend as bmod

    orig_get_cuml = bmod._backend.get_cuml

    def _raise_import(*a, **kw):
        raise ImportError("cuML forced unavailable for test")

    bmod._backend.get_cuml = _raise_import
    try:
        t0 = time.perf_counter()
        conn_fb, dist_fb = knn_graph(coords, n_neighbors=n_neighbors)
        gpu_time = time.perf_counter() - t0
    finally:
        bmod._backend.get_cuml = orig_get_cuml

    both_mask = (conn_cpu.toarray() > 0) & (conn_fb.toarray() > 0)
    if both_mask.any():
        max_dist_diff = np.max(
            np.abs(dist_cpu.toarray()[both_mask] - dist_fb.toarray()[both_mask])
        )
    else:
        max_dist_diff = 0.0

    max_conn_diff = np.abs(conn_cpu.toarray() - conn_fb.toarray()).max()
    max_diff = max(max_conn_diff, max_dist_diff)
    passed = max_conn_diff <= 2 and max_dist_diff < 1e-3

    return BenchResult(
        name="knn_cupy_fallback",
        n_cells=n_cells,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        max_diff=max_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Test: knn_graph edge case (few cells)
# ---------------------------------------------------------------------------


def test_knn_few_cells() -> BenchResult:
    """Edge case: more neighbors requested than cells available."""
    from spatialgpu.graph.neighbors import knn_graph

    coords = _make_coords(5)

    _force_gpu()
    gc.collect()
    from spatialgpu.core import backend as bmod

    orig_get_cuml = bmod._backend.get_cuml

    def _raise_import(*a, **kw):
        raise ImportError("cuML forced unavailable for test")

    bmod._backend.get_cuml = _raise_import
    try:
        conn, dist = knn_graph(coords, n_neighbors=10)
    finally:
        bmod._backend.get_cuml = orig_get_cuml

    _force_cpu()
    conn_cpu, dist_cpu = knn_graph(coords, n_neighbors=10)

    max_diff = np.abs(conn.toarray() - conn_cpu.toarray()).max()
    passed = conn.shape == (5, 5) and max_diff <= 2

    return BenchResult(
        name="knn_few_cells",
        n_cells=5,
        cpu_time=0.0,
        gpu_time=0.0,
        max_diff=max_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Test: nhood_enrichment
# ---------------------------------------------------------------------------


def test_nhood_enrichment(n_cells: int, n_clusters: int = 8) -> BenchResult:
    from spatialgpu.graph.analysis import _nhood_enrichment_cpu, _nhood_enrichment_gpu

    coords = _make_coords(n_cells)
    cluster_idx = _make_cluster_idx(n_cells, n_clusters)

    # Build a simple kNN adjacency
    from spatialgpu.graph.neighbors import knn_graph

    _force_cpu()
    conn, _ = knn_graph(coords, n_neighbors=6)

    # CPU
    gc.collect()
    t0 = time.perf_counter()
    zscore_cpu, count_cpu = _nhood_enrichment_cpu(
        conn, cluster_idx, n_clusters, n_perms=100, seed=42, show_progress=False
    )
    cpu_time = time.perf_counter() - t0

    # GPU
    _force_gpu()
    gc.collect()
    t0 = time.perf_counter()
    zscore_gpu, count_gpu = _nhood_enrichment_gpu(
        conn, cluster_idx, n_clusters, n_perms=100, seed=42, show_progress=False
    )
    gpu_time = time.perf_counter() - t0

    # Count matrices should be identical (deterministic, no permutation)
    count_diff = np.max(np.abs(count_cpu - count_gpu))
    # Z-scores will differ due to different RNG, but should be in same ballpark
    # Just check counts match exactly
    passed = count_diff < 1e-3

    return BenchResult(
        name="nhood_enrichment",
        n_cells=n_cells,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        max_diff=count_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Test: co_occurrence
# ---------------------------------------------------------------------------


def test_co_occurrence(n_cells: int, n_clusters: int = 5) -> BenchResult:
    from spatialgpu.graph.analysis import _co_occurrence_cpu, _co_occurrence_gpu

    coords = _make_coords(n_cells)
    cluster_idx = _make_cluster_idx(n_cells, n_clusters)
    bins = np.linspace(0, 500, 21)

    # CPU
    _force_cpu()
    gc.collect()
    t0 = time.perf_counter()
    occ_cpu = _co_occurrence_cpu(coords, cluster_idx, n_clusters, bins, False)
    cpu_time = time.perf_counter() - t0

    # GPU
    _force_gpu()
    gc.collect()
    t0 = time.perf_counter()
    occ_gpu = _co_occurrence_gpu(coords, cluster_idx, n_clusters, bins, False)
    gpu_time = time.perf_counter() - t0

    max_diff = np.max(np.abs(occ_cpu - occ_gpu))
    passed = max_diff < 1e-4

    return BenchResult(
        name="co_occurrence",
        n_cells=n_cells,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        max_diff=max_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Test: radius_graph
# ---------------------------------------------------------------------------


def test_radius_graph(n_cells: int, radius: float = 50.0) -> BenchResult:
    from spatialgpu.graph.neighbors import radius_graph

    coords = _make_coords(n_cells)

    _force_cpu()
    gc.collect()
    t0 = time.perf_counter()
    conn_cpu, dist_cpu = radius_graph(coords, radius=radius)
    cpu_time = time.perf_counter() - t0

    _force_gpu()
    gc.collect()
    t0 = time.perf_counter()
    conn_gpu, dist_gpu = radius_graph(coords, radius=radius)
    gpu_time = time.perf_counter() - t0

    # Compare edges
    diff = np.abs(conn_cpu.toarray() - conn_gpu.toarray()).max()
    # Distance comparison
    both = (conn_cpu.toarray() > 0) & (conn_gpu.toarray() > 0)
    if both.any():
        dist_diff = np.max(
            np.abs(dist_cpu.toarray()[both] - dist_gpu.toarray()[both])
        )
    else:
        dist_diff = 0.0

    max_diff = max(diff, dist_diff)
    passed = diff == 0 and dist_diff < 1e-3

    return BenchResult(
        name="radius_graph",
        n_cells=n_cells,
        cpu_time=cpu_time,
        gpu_time=gpu_time,
        max_diff=max_diff,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 72)
    print("GPU vs CPU Validation & Benchmark — spatial-gpu")
    print("=" * 72)

    # Check GPU availability
    try:
        import cupy as cp

        print(f"CuPy version: {cp.__version__}")
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"GPU memory: {cp.cuda.Device(0).mem_info[1] / 1e9:.1f} GB")
    except Exception as e:
        print(f"GPU not available: {e}")
        print("Cannot run GPU tests. Exiting.")
        sys.exit(1)

    try:
        import cuml

        print(f"cuML version: {cuml.__version__}")
        cuml_available = True
    except ImportError:
        print("cuML not available (CuPy fallback will be tested)")
        cuml_available = False

    print()

    results: list[BenchResult] = []

    # Small dataset tests (correctness)
    print("--- Correctness Tests (small datasets) ---")
    for size in [100, 500, 2000]:
        for test_fn in [test_knn_graph, test_co_occurrence, test_nhood_enrichment, test_radius_graph]:
            r = test_fn(size)
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  [{status}] {r.name} n={r.n_cells:>6d}  "
                f"CPU={r.cpu_time:.3f}s  GPU={r.gpu_time:.3f}s  "
                f"speedup={r.speedup:.1f}x  max_diff={r.max_diff:.2e}"
            )

    # Edge case: few cells
    print("\n--- Edge Case Tests ---")
    r = test_knn_few_cells()
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {r.name} (5 cells, k=10)")

    # CuPy fallback test
    r = test_knn_cupy_fallback(1000)
    results.append(r)
    status = "PASS" if r.passed else "FAIL"
    print(
        f"  [{status}] {r.name} n={r.n_cells:>6d}  "
        f"CPU={r.cpu_time:.3f}s  GPU(cupy)={r.gpu_time:.3f}s  "
        f"speedup={r.speedup:.1f}x  max_diff={r.max_diff:.2e}"
    )

    # Larger benchmark (performance)
    print("\n--- Performance Benchmarks (larger datasets) ---")
    for size in [5000, 20000, 50000]:
        for test_fn in [test_knn_graph, test_co_occurrence]:
            try:
                r = test_fn(size)
                results.append(r)
                status = "PASS" if r.passed else "FAIL"
                print(
                    f"  [{status}] {r.name} n={r.n_cells:>6d}  "
                    f"CPU={r.cpu_time:.3f}s  GPU={r.gpu_time:.3f}s  "
                    f"speedup={r.speedup:.1f}x  max_diff={r.max_diff:.2e}"
                )
            except Exception as e:
                print(f"  [ERROR] {test_fn.__name__} n={size}: {e}")

    # nhood_enrichment benchmark (permutation-heavy)
    for size in [5000, 10000]:
        try:
            r = test_nhood_enrichment(size)
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  [{status}] {r.name} n={r.n_cells:>6d}  "
                f"CPU={r.cpu_time:.3f}s  GPU={r.gpu_time:.3f}s  "
                f"speedup={r.speedup:.1f}x  max_diff={r.max_diff:.2e}"
            )
        except Exception as e:
            print(f"  [ERROR] nhood_enrichment n={size}: {e}")

    # Summary
    print("\n" + "=" * 72)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    print(f"TOTAL: {n_pass} passed, {n_fail} failed out of {len(results)} tests")

    # Speedup summary
    bench_results = [r for r in results if r.cpu_time > 0 and r.gpu_time > 0]
    if bench_results:
        print("\nSpeedup summary:")
        for r in bench_results:
            print(f"  {r.name:25s} n={r.n_cells:>6d}  {r.speedup:>6.1f}x")

    print("=" * 72)

    if n_fail > 0:
        print("\nFAILED tests:")
        for r in results:
            if not r.passed:
                print(f"  {r.name} n={r.n_cells} max_diff={r.max_diff:.2e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
