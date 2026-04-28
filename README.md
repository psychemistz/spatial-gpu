# spatial-gpu

**GPU-accelerated spatial omics analysis framework for the scverse ecosystem**

[![CI](https://github.com/psychemistz/spatial-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/psychemistz/spatial-gpu/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://psychemistz.github.io/spatial-gpu/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

spatial-gpu is a pure-Python reimplementation of [SpaCET](https://github.com/data2intelligence/SpaCET) with full GPU acceleration. The entire pipeline -- deconvolution, cell-cell interaction, gene set scoring, spatial correlation, and SecAct signaling -- runs on GPU when available, with automatic CPU fallback. Zero R dependency.

## Installation

```bash
pip install spatial-gpu            # CPU
pip install spatial-gpu[cuda]      # GPU (CUDA 12.x)
pip install spatial-gpu[secact]    # + SecAct
pip install spatial-gpu[all]       # Everything
```

### Docker

```bash
docker run -it --rm -v $(pwd):/workspace psychemistz/spatial-gpu:latest       # CPU
docker run -it --rm --gpus all -v $(pwd):/workspace psychemistz/spatial-gpu:gpu  # GPU
```

Available tags: `latest`, `gpu`, `with-r`, `gpu-with-r`. See [DOCKER.md](DOCKER.md) for details.

## Quick Start

```python
import spatialgpu.deconvolution as spacet

# Load and deconvolve
adata = spacet.create_spacet_object_10x("path/to/visium/")
adata = spacet.quality_control(adata)
adata = spacet.deconvolution(adata, cancer_type="BRCA")

# Cell-cell interaction
adata = spacet.cci_colocalization(adata)
adata = spacet.cci_lr_network_score(adata)

# Visualize
spacet.visualize_spatial_feature(adata, spatial_type="CellFraction", spatial_features=["All"])
```

## Tutorials

Full documentation at **[psychemistz.github.io/spatial-gpu](https://psychemistz.github.io/spatial-gpu/)**

| Tutorial | Topic |
|----------|-------|
| [T1: Visium BC](https://psychemistz.github.io/spatial-gpu/visium_BC.html) | Deconvolution + CCI + interface analysis |
| [T2: Old ST PDAC](https://psychemistz.github.io/spatial-gpu/oldST_PDAC.html) | Matched scRNA-seq deconvolution |
| [T3: Hi-Res ST CRC](https://psychemistz.github.io/spatial-gpu/hiresST_CRC.html) | High-resolution deconvolution |
| [T4: Gene Set Score](https://psychemistz.github.io/spatial-gpu/GeneSetScore.html) | Hallmark, CancerCellState, TLS scoring |
| [T5: Spatial Correlation](https://psychemistz.github.io/spatial-gpu/SpatialCorrelation.html) | Moran's I (univariate, bivariate, pairwise) |
| [T6: Signaling Patterns](https://psychemistz.github.io/spatial-gpu/stPattern.html) | SecAct + NMF patterns + velocity |
| [T7: Cell-Cell Communication](https://psychemistz.github.io/spatial-gpu/stCCC.html) | Single-cell CCC (CosMx 443K cells) |

## GPU Acceleration

GPU is used automatically when CuPy is installed. Every pipeline step has a GPU code path:

| Function | What it accelerates |
|----------|-------------------|
| `deconvolution` | NNLS + constrained QP solver |
| `cormat` | Spearman/Pearson correlation |
| `cci_colocalization` | Pairwise Spearman |
| `cci_lr_network_score` | Bipartite edge swap + permutation scoring |
| `distance_to_interface` | Pairwise distance + permutation test |
| `gene_set_score` | UCell rank-based scoring |
| `secact_signaling_patterns` | Spearman filtering + NMF |
| `secact_signaling_velocity` | Spatial weight matrix ops |
| `cal_weights` | RBF kernel spatial weights |
| `spatial_neighbors` | kNN graph construction |
| `nhood_enrichment` | Permutation test with CUDA kernel |

```python
import spatialgpu as sp
sp.set_backend("auto")  # GPU if available, else CPU
sp.set_backend("gpu")   # Force GPU (raises if unavailable)
sp.set_backend("cpu")   # Force CPU
```

## Citation

If you use spatial-gpu in your research, please cite:

Beibei Ru, Jinlin Huang, Yu Zhang, Kenneth Aldape, Peng Jiang. Estimation of cell lineages in tumors from spatial transcriptomics data. *Nature Communications* 14, 568 (2023). [Full Text](https://www.nature.com/articles/s41467-023-36062-6)

```bibtex
@article{ru2023spacet,
  title={Estimation of cell lineages in tumors from spatial transcriptomics data},
  author={Ru, Beibei and Huang, Jinlin and Zhang, Yu and Aldape, Kenneth and Jiang, Peng},
  journal={Nature Communications},
  volume={14},
  pages={568},
  year={2023},
  doi={10.1038/s41467-023-36062-6}
}
```

## Related Projects

- [SpaCET](https://github.com/data2intelligence/SpaCET) -- Original R package
- [SecActPy](https://github.com/data2intelligence/SecActpy) -- Secreted protein activity
- [Squidpy](https://squidpy.readthedocs.io/) -- Spatial single-cell analysis
- [Scanpy](https://scanpy.readthedocs.io/) -- Single-cell analysis

## License

BSD 3-Clause. See [LICENSE](LICENSE).
