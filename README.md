# spatial-gpu

**GPU-accelerated spatial omics analysis framework for the scverse ecosystem**

[![PyPI version](https://badge.fury.io/py/spatial-gpu.svg)](https://badge.fury.io/py/spatial-gpu)
[![Documentation](https://readthedocs.org/projects/spatial-gpu/badge/?version=latest)](https://spatial-gpu.readthedocs.io)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

spatial-gpu provides **10-100x speedup** for spatial transcriptomics and spatial omics analysis through GPU acceleration, while maintaining full compatibility with the scverse ecosystem (Scanpy, Squidpy, AnnData).

## Key Features

- **Drop-in Squidpy replacement**: Same API, just faster
- **GPU-accelerated operations**: Spatial graphs, neighborhood analysis, Ripley's statistics
- **Memory-efficient**: Handle 100M+ cell datasets with chunked processing
- **Multi-backend**: Seamless CPU/GPU switching
- **Cell segmentation**: GPU-accelerated Cellpose and StarDist integration

## Installation

```bash
# CPU-only installation
pip install spatial-gpu

# With GPU support (requires CUDA 12.x)
pip install spatial-gpu[cuda]

# With all optional dependencies
pip install spatial-gpu[all]
```

### GPU Requirements

- NVIDIA GPU with compute capability 7.0+ (V100, A100, RTX 20/30/40 series)
- CUDA 11.x or 12.x
- cuDNN 8.x+

## Quick Start

```python
import scanpy as sc
import spatialgpu as sp

# Load spatial data
adata = sc.read_h5ad("spatial_data.h5ad")

# GPU-accelerated spatial neighbors (10-50x faster)
sp.graph.spatial_neighbors(adata, n_neighbors=6)

# GPU-accelerated neighborhood enrichment (20-100x faster)
sp.graph.nhood_enrichment(adata, cluster_key="cell_type")

# Visualize results
sp.viz.nhood_enrichment_plot(adata, cluster_key="cell_type")
```

## Performance Benchmarks

Benchmarks on NVIDIA A100 GPU vs CPU (AMD EPYC 7742, 64 cores):

| Operation | 100K cells | 1M cells | 10M cells |
|-----------|------------|----------|-----------|
| `spatial_neighbors` | 15x | 45x | 80x |
| `nhood_enrichment` | 25x | 60x | 100x |
| `co_occurrence` | 20x | 50x | 90x |
| `ripley` | 30x | 70x | 120x |

## Usage

### Backend Selection

```python
import spatialgpu as sp

# Auto-detect (uses GPU if available)
sp.set_backend("auto")

# Force CPU
sp.set_backend("cpu")

# Force GPU (raises if unavailable)
sp.set_backend("gpu")

# Check GPU status
backend = sp.get_backend()
print(f"GPU available: {backend.is_gpu_available}")
print(f"GPU active: {backend.is_gpu_active}")
print(f"Device: {backend.device_info}")
```

### Spatial Graph Construction

```python
import spatialgpu as sp

# k-nearest neighbors graph
sp.graph.spatial_neighbors(adata, n_neighbors=6)

# Radius-based graph
sp.graph.spatial_neighbors(adata, radius=100.0)

# Delaunay triangulation
conn, dist = sp.graph.delaunay_graph(coords)
```

### Spatial Analysis

```python
# Neighborhood enrichment
sp.graph.nhood_enrichment(adata, cluster_key="cell_type", n_perms=1000)

# Co-occurrence analysis
sp.graph.co_occurrence(adata, cluster_key="cell_type")

# Ripley's L function
sp.graph.ripley(adata, mode="L", n_simulations=100)

# Cell-cell interaction matrix
sp.graph.interaction_matrix(adata, cluster_key="cell_type")
```

### Cell Segmentation

```python
from spatialgpu.segmentation import segment_cells, CellSegmenter

# Quick segmentation
result = segment_cells(image, model="cellpose", diameter=30)
print(f"Found {result.n_cells} cells")

# Advanced usage with tiling for large images
segmenter = CellSegmenter(model="cellpose", device="cuda")
result = segmenter.segment_tiled(large_image, tile_size=2048, overlap=128)

# Access results
masks = result.masks  # Integer mask array
centroids = result.centroids  # Cell center coordinates
areas = result.areas  # Cell areas in pixels
```

### Data I/O

```python
import spatialgpu as sp

# Read from various platforms
adata = sp.io.read_visium("path/to/visium/")
adata = sp.io.read_xenium("path/to/xenium/")
adata = sp.io.read_cosmx("path/to/cosmx/")
adata = sp.io.read_merscope("path/to/merscope/")

# Export to SpatialData
sdata = sp.io.export_to_spatialdata(adata)
```

### Visualization

```python
import spatialgpu as sp

# Spatial scatter plots
sp.viz.spatial_scatter(adata, color="cell_type")
sp.viz.spatial_scatter(adata, color="GAPDH")

# Analysis visualizations
sp.viz.nhood_enrichment_plot(adata, cluster_key="cell_type")
sp.viz.co_occurrence_plot(adata, cluster_key="cell_type")
sp.viz.ripley_plot(adata)

# Segmentation overlay
sp.viz.segmentation_overlay(image, masks)
```

### Benchmarking

```python
import spatialgpu as sp

# Generate synthetic data
adata = sp.benchmarks.generate_synthetic_data(n_cells=100000)

# Run benchmark suite
results = sp.benchmarks.benchmark_suite(adata)

# Compare CPU vs GPU
comparison = sp.benchmarks.compare_backends(
    sp.graph.spatial_neighbors,
    adata, n_neighbors=6
)
print(f"GPU speedup: {comparison['speedup']:.1f}x")
```

## API Reference

### Graph Module (`sp.graph`)

| Function | Description |
|----------|-------------|
| `spatial_neighbors` | Build spatial neighbor graph |
| `knn_graph` | k-nearest neighbors graph |
| `radius_graph` | Radius-based neighbor graph |
| `delaunay_graph` | Delaunay triangulation graph |
| `nhood_enrichment` | Neighborhood enrichment analysis |
| `co_occurrence` | Spatial co-occurrence |
| `interaction_matrix` | Cell-cell interaction matrix |
| `ripley` | Ripley's statistics (K, L, F, G) |
| `centrality_scores` | Graph centrality measures |

### Segmentation Module (`sp.segmentation`)

| Function | Description |
|----------|-------------|
| `segment_cells` | Segment cells from images |
| `segment_nuclei` | Segment nuclei specifically |
| `CellSegmenter` | High-level segmentation interface |
| `segment_transcripts` | Assign transcripts to cells |
| `evaluate_segmentation` | Evaluate segmentation quality |

### Visualization Module (`sp.viz`)

| Function | Description |
|----------|-------------|
| `spatial_scatter` | Spatial scatter plot |
| `spatial_heatmap` | Multi-gene heatmap |
| `nhood_enrichment_plot` | Neighborhood enrichment heatmap |
| `co_occurrence_plot` | Co-occurrence curves |
| `ripley_plot` | Ripley's statistics plot |
| `segmentation_overlay` | Overlay masks on image |

### I/O Module (`sp.io`)

| Function | Description |
|----------|-------------|
| `read_visium` | Read 10x Visium data |
| `read_xenium` | Read 10x Xenium data |
| `read_cosmx` | Read NanoString CosMx data |
| `read_merscope` | Read Vizgen MERSCOPE data |
| `export_to_spatialdata` | Export to SpatialData format |

## Comparison with Squidpy

spatial-gpu is designed as a GPU-accelerated alternative to Squidpy with API compatibility:

```python
# Squidpy (CPU)
import squidpy as sq
sq.gr.spatial_neighbors(adata, n_neighbors=6)
sq.gr.nhood_enrichment(adata, cluster_key="cell_type")

# spatial-gpu (GPU) - same API!
import spatialgpu as sp
sp.graph.spatial_neighbors(adata, n_neighbors=6)
sp.graph.nhood_enrichment(adata, cluster_key="cell_type")
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

```bash
# Clone repository
git clone https://github.com/spatial-gpu/spatial-gpu.git
cd spatial-gpu

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python -m spatialgpu.benchmarks
```

## Citation

If you use spatial-gpu in your research, please cite:

```bibtex
@software{spatialgpu2025,
  title={spatial-gpu: GPU-accelerated spatial omics analysis},
  author={spatial-gpu contributors},
  year={2025},
  url={https://github.com/spatial-gpu/spatial-gpu}
}
```

## Related Projects

- [Squidpy](https://squidpy.readthedocs.io/) - Spatial single-cell analysis in Python
- [SpatialData](https://spatialdata.scverse.org/) - Universal spatial omics data structure
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis in Python
- [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/) - GPU-accelerated single-cell analysis

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
