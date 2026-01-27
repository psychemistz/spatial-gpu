#!/usr/bin/env python
"""
Benchmark Comparison: spatial-gpu vs Squidpy
=============================================

This script compares the performance of spatial-gpu against Squidpy
across different dataset sizes and operations.
"""

import time
import numpy as np
import pandas as pd

import spatialgpu as sp

# Check for Squidpy
try:
    import squidpy as sq
    HAS_SQUIDPY = True
except ImportError:
    HAS_SQUIDPY = False
    print("Squidpy not installed. Skipping Squidpy comparisons.")


def benchmark_function(func, *args, n_runs=3, **kwargs):
    """Benchmark a function and return mean time."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def run_benchmarks(n_cells_list=[10000, 50000, 100000]):
    """Run comprehensive benchmarks."""
    results = []

    for n_cells in n_cells_list:
        print(f"\n{'='*60}")
        print(f"Benchmarking with {n_cells:,} cells")
        print(f"{'='*60}")

        # Generate data
        print("Generating synthetic data...")
        adata = sp.benchmarks.generate_synthetic_data(
            n_cells=n_cells,
            n_genes=200,
            n_clusters=10,
            seed=42,
        )

        # Deep copy for Squidpy tests
        if HAS_SQUIDPY:
            import copy
            adata_sq = adata.copy()

        # -----------------------------------------
        # spatial_neighbors benchmark
        # -----------------------------------------
        print("\n1. spatial_neighbors")

        # spatial-gpu (CPU)
        sp.set_backend("cpu")
        mean_time, std_time = benchmark_function(
            sp.graph.spatial_neighbors,
            adata.copy(), n_neighbors=6
        )
        print(f"   spatial-gpu (CPU): {mean_time:.4f}s ± {std_time:.4f}s")
        results.append({
            "n_cells": n_cells,
            "operation": "spatial_neighbors",
            "library": "spatial-gpu",
            "backend": "CPU",
            "mean_time": mean_time,
            "std_time": std_time,
        })

        # spatial-gpu (GPU)
        if sp.get_backend().is_gpu_available:
            sp.set_backend("gpu")
            mean_time, std_time = benchmark_function(
                sp.graph.spatial_neighbors,
                adata.copy(), n_neighbors=6
            )
            print(f"   spatial-gpu (GPU): {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "spatial_neighbors",
                "library": "spatial-gpu",
                "backend": "GPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

        # Squidpy
        if HAS_SQUIDPY:
            mean_time, std_time = benchmark_function(
                sq.gr.spatial_neighbors,
                adata_sq.copy(), n_neighs=6
            )
            print(f"   Squidpy:           {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "spatial_neighbors",
                "library": "Squidpy",
                "backend": "CPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

        # -----------------------------------------
        # nhood_enrichment benchmark
        # -----------------------------------------
        print("\n2. nhood_enrichment")

        # Prepare data
        sp.set_backend("cpu")
        adata_prepared = adata.copy()
        sp.graph.spatial_neighbors(adata_prepared, n_neighbors=6)

        if HAS_SQUIDPY:
            adata_sq_prepared = adata_sq.copy()
            sq.gr.spatial_neighbors(adata_sq_prepared, n_neighs=6)

        # spatial-gpu (CPU)
        sp.set_backend("cpu")
        mean_time, std_time = benchmark_function(
            sp.graph.nhood_enrichment,
            adata_prepared.copy(),
            cluster_key="cluster",
            n_perms=100,
            show_progress=False,
        )
        print(f"   spatial-gpu (CPU): {mean_time:.4f}s ± {std_time:.4f}s")
        results.append({
            "n_cells": n_cells,
            "operation": "nhood_enrichment",
            "library": "spatial-gpu",
            "backend": "CPU",
            "mean_time": mean_time,
            "std_time": std_time,
        })

        # spatial-gpu (GPU)
        if sp.get_backend().is_gpu_available:
            sp.set_backend("gpu")
            mean_time, std_time = benchmark_function(
                sp.graph.nhood_enrichment,
                adata_prepared.copy(),
                cluster_key="cluster",
                n_perms=100,
                show_progress=False,
            )
            print(f"   spatial-gpu (GPU): {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "nhood_enrichment",
                "library": "spatial-gpu",
                "backend": "GPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

        # Squidpy
        if HAS_SQUIDPY:
            mean_time, std_time = benchmark_function(
                sq.gr.nhood_enrichment,
                adata_sq_prepared.copy(),
                cluster_key="cluster",
                n_perms=100,
                show_progress_bar=False,
            )
            print(f"   Squidpy:           {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "nhood_enrichment",
                "library": "Squidpy",
                "backend": "CPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

        # -----------------------------------------
        # co_occurrence benchmark
        # -----------------------------------------
        print("\n3. co_occurrence")

        # spatial-gpu (CPU)
        sp.set_backend("cpu")
        mean_time, std_time = benchmark_function(
            sp.graph.co_occurrence,
            adata_prepared.copy(),
            cluster_key="cluster",
            n_splits=20,
            show_progress=False,
        )
        print(f"   spatial-gpu (CPU): {mean_time:.4f}s ± {std_time:.4f}s")
        results.append({
            "n_cells": n_cells,
            "operation": "co_occurrence",
            "library": "spatial-gpu",
            "backend": "CPU",
            "mean_time": mean_time,
            "std_time": std_time,
        })

        # spatial-gpu (GPU)
        if sp.get_backend().is_gpu_available:
            sp.set_backend("gpu")
            mean_time, std_time = benchmark_function(
                sp.graph.co_occurrence,
                adata_prepared.copy(),
                cluster_key="cluster",
                n_splits=20,
                show_progress=False,
            )
            print(f"   spatial-gpu (GPU): {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "co_occurrence",
                "library": "spatial-gpu",
                "backend": "GPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

        # Squidpy
        if HAS_SQUIDPY:
            mean_time, std_time = benchmark_function(
                sq.gr.co_occurrence,
                adata_sq_prepared.copy(),
                cluster_key="cluster",
                n_splits=20,
                show_progress_bar=False,
            )
            print(f"   Squidpy:           {mean_time:.4f}s ± {std_time:.4f}s")
            results.append({
                "n_cells": n_cells,
                "operation": "co_occurrence",
                "library": "Squidpy",
                "backend": "CPU",
                "mean_time": mean_time,
                "std_time": std_time,
            })

    return pd.DataFrame(results)


def print_summary(df):
    """Print benchmark summary with speedups."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for operation in df["operation"].unique():
        print(f"\n{operation}")
        print("-" * 40)

        for n_cells in df["n_cells"].unique():
            mask = (df["operation"] == operation) & (df["n_cells"] == n_cells)
            subset = df[mask]

            print(f"\n  {n_cells:,} cells:")

            # Get baseline (Squidpy or CPU)
            if "Squidpy" in subset["library"].values:
                baseline_time = subset[subset["library"] == "Squidpy"]["mean_time"].values[0]
                baseline_name = "Squidpy"
            else:
                baseline_time = subset[
                    (subset["library"] == "spatial-gpu") & (subset["backend"] == "CPU")
                ]["mean_time"].values[0]
                baseline_name = "CPU"

            for _, row in subset.iterrows():
                speedup = baseline_time / row["mean_time"]
                speedup_str = f"({speedup:.1f}x vs {baseline_name})" if speedup != 1.0 else ""
                print(f"    {row['library']:15s} ({row['backend']:3s}): "
                      f"{row['mean_time']:.4f}s {speedup_str}")


if __name__ == "__main__":
    # Print system info
    print("=" * 60)
    print("BENCHMARK: spatial-gpu vs Squidpy")
    print("=" * 60)

    backend = sp.get_backend()
    print(f"\nGPU available: {backend.is_gpu_available}")
    if backend.device_info:
        print(f"GPU device: {backend.device_info.name}")
        print(f"GPU memory: {backend.device_info.total_memory_gb:.1f} GB")

    # Run benchmarks
    results_df = run_benchmarks(n_cells_list=[10000, 50000, 100000])

    # Print summary
    print_summary(results_df)

    # Save results
    results_df.to_csv("benchmark_results.csv", index=False)
    print(f"\nResults saved to benchmark_results.csv")
