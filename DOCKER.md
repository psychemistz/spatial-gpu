# spatial-gpu Docker Images

GPU-accelerated spatial omics analysis with SpaCET deconvolution and SecAct signaling.

## Quick Start

```bash
# CPU (works on Windows, Mac, Linux)
docker run -it --rm -v $(pwd):/workspace psychemistz/spatial-gpu:latest

# GPU (requires NVIDIA Container Toolkit)
docker run -it --rm --gpus all -v $(pwd):/workspace psychemistz/spatial-gpu:gpu

# Jupyter Lab
docker run -it --rm -p 8888:8888 -v $(pwd):/workspace psychemistz/spatial-gpu:latest \
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

## Available Tags

| Tag | Python | GPU (CuPy) | R (SpaCET) | SecAct | Use Case |
|-----|--------|-----------|------------|--------|----------|
| `latest`, `cpu` | 3.x | - | - | Yes | Standard analysis |
| `gpu` | 3.x | CUDA 12.2 | - | Yes | GPU-accelerated analysis |
| `with-r`, `cpu-with-r` | 3.x | - | Yes | Yes | R cross-validation |
| `gpu-with-r` | 3.x | CUDA 12.2 | Yes | Yes | Full stack |

## Docker Compose

```bash
# Start CPU container
docker compose up -d spatial-gpu

# Start GPU container
docker compose up -d spatial-gpu-gpu

# Jupyter Lab (CPU: port 8888, GPU: port 8889, R: port 8890)
docker compose up spatial-gpu-jupyter
docker compose up spatial-gpu-jupyter-gpu

# Enter running container
docker compose exec spatial-gpu bash
```

## Build from Source

```bash
# CPU (default)
docker build -t spatial-gpu:latest .

# GPU
docker build -t spatial-gpu:gpu --build-arg USE_GPU=true .

# With R SpaCET
docker build -t spatial-gpu:with-r --build-arg INSTALL_R=true .

# GPU + R (full stack)
docker build -t spatial-gpu:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
```

## Platform Support

| Platform | CPU | GPU |
|----------|-----|-----|
| Linux (x86_64) | Yes | Yes (NVIDIA GPU + Container Toolkit) |
| macOS (Intel) | Yes | No |
| macOS (Apple Silicon) | Yes (Rosetta 2) | No |
| Windows (WSL2) | Yes | Yes (NVIDIA GPU + WSL2 CUDA) |

### Windows GPU Setup

1. Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install [NVIDIA drivers for WSL](https://developer.nvidia.com/cuda/wsl)
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
4. Run: `docker run --gpus all psychemistz/spatial-gpu:gpu nvidia-smi`

## Included Packages

**Python:** spatial-gpu (spatialgpu), secactpy, scanpy, squidpy, anndata, scikit-learn, matplotlib, jupyter

**GPU (gpu tag):** CuPy (CUDA 12.2)

**R (with-r tags):** SpaCET, Seurat, Matrix, Rcpp, RcppArmadillo, rhdf5
