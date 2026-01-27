#!/usr/bin/env python
"""
spatial-gpu Quick Start Example
===============================

This example demonstrates the basic usage of spatial-gpu for
GPU-accelerated spatial omics analysis.
"""

import numpy as np

# Import spatial-gpu
import spatialgpu as sp

print(f"spatial-gpu version: {sp.__version__}")

# Check GPU availability
backend = sp.get_backend()
print(f"GPU available: {backend.is_gpu_available}")
print(f"GPU active: {backend.is_gpu_active}")
if backend.device_info:
    print(f"Device: {backend.device_info.name}")

# ============================================
# Generate Synthetic Data
# ============================================

print("\n" + "=" * 50)
print("Generating synthetic data...")
print("=" * 50)

# Generate synthetic spatial data with 50,000 cells
adata = sp.benchmarks.generate_synthetic_data(
    n_cells=50000,
    n_genes=200,
    n_clusters=10,
    seed=42,
)

print(f"Created AnnData: {adata}")
print(f"Spatial coordinates shape: {adata.obsm['spatial'].shape}")
print(f"Clusters: {adata.obs['cluster'].cat.categories.tolist()}")

# ============================================
# Spatial Graph Construction
# ============================================

print("\n" + "=" * 50)
print("Building spatial neighbor graph...")
print("=" * 50)

# Build k-nearest neighbors graph
sp.graph.spatial_neighbors(adata, n_neighbors=6)

print(f"Connectivities shape: {adata.obsp['spatial_connectivities'].shape}")
print(f"Number of edges: {adata.obsp['spatial_connectivities'].nnz}")

# ============================================
# Neighborhood Enrichment Analysis
# ============================================

print("\n" + "=" * 50)
print("Computing neighborhood enrichment...")
print("=" * 50)

# Compute neighborhood enrichment with 100 permutations
zscore, count = sp.graph.nhood_enrichment(
    adata,
    cluster_key="cluster",
    n_perms=100,
    copy=True,
    show_progress=True,
)

print(f"Z-score matrix shape: {zscore.shape}")
print(f"Count matrix shape: {count.shape}")

# Find strongest positive enrichments
mask = ~np.eye(zscore.shape[0], dtype=bool)  # Exclude diagonal
top_idx = np.unravel_index(np.argmax(zscore * mask), zscore.shape)
clusters = adata.obs["cluster"].cat.categories
print(f"Strongest enrichment: {clusters[top_idx[0]]} <-> {clusters[top_idx[1]]} (z={zscore[top_idx]:.2f})")

# ============================================
# Co-occurrence Analysis
# ============================================

print("\n" + "=" * 50)
print("Computing co-occurrence...")
print("=" * 50)

occurrence, intervals = sp.graph.co_occurrence(
    adata,
    cluster_key="cluster",
    n_splits=20,
    copy=True,
    show_progress=True,
)

print(f"Co-occurrence shape: {occurrence.shape}")
print(f"Distance intervals: {intervals[0]:.1f} to {intervals[-1]:.1f}")

# ============================================
# Ripley's Statistics
# ============================================

print("\n" + "=" * 50)
print("Computing Ripley's L function...")
print("=" * 50)

ripley_result = sp.graph.ripley(
    adata,
    mode="L",
    n_simulations=50,
    n_radii=30,
    copy=True,
    show_progress=True,
)

stats = ripley_result["stats"]
radii = ripley_result["radii"]
print(f"Number of radii: {len(radii)}")
print(f"L(r) range: [{stats['observed'].min():.2f}, {stats['observed'].max():.2f}]")

# Check for clustering (L > expected)
if np.any(stats["observed"] > stats["simulated_hi"]):
    print("Significant spatial clustering detected!")
else:
    print("No significant clustering detected.")

# ============================================
# Benchmarking
# ============================================

print("\n" + "=" * 50)
print("Running benchmarks...")
print("=" * 50)

# Benchmark spatial_neighbors
result = sp.benchmarks.benchmark(
    sp.graph.spatial_neighbors,
    adata,
    n_neighbors=6,
    n_runs=3,
    name="spatial_neighbors",
)

print(f"spatial_neighbors: {result.mean_time:.4f}s ± {result.std_time:.4f}s")

# Compare CPU vs GPU (if GPU available)
if backend.is_gpu_available:
    print("\nComparing CPU vs GPU performance...")
    comparison = sp.benchmarks.compare_backends(
        sp.graph.spatial_neighbors,
        adata,
        n_neighbors=6,
        n_runs=3,
    )
    print(f"CPU time: {comparison['cpu'].mean_time:.4f}s")
    print(f"GPU time: {comparison['gpu'].mean_time:.4f}s")
    print(f"Speedup: {comparison['speedup']:.1f}x")

print("\n" + "=" * 50)
print("Quick start complete!")
print("=" * 50)
