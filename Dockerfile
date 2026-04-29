# =============================================================================
# spatial-gpu: GPU-Accelerated Spatial Omics Analysis
#
# Unified Dockerfile for CPU and GPU versions, with optional R SpaCET package.
# Includes secactpy for secreted protein activity analysis.
#
# Build CPU version (default):
#   docker build -t spatial-gpu:latest .
#
# Build GPU version:
#   docker build -t spatial-gpu:gpu --build-arg USE_GPU=true .
#
# Build with R SpaCET package (for R cross-validation):
#   docker build -t spatial-gpu:with-r --build-arg INSTALL_R=true .
#   docker build -t spatial-gpu:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
#
# Run CPU:
#   docker run -it --rm -v $(pwd):/workspace spatial-gpu:latest
#
# Run GPU:
#   docker run -it --rm --gpus all -v $(pwd):/workspace spatial-gpu:gpu
#
# Run Jupyter:
#   docker run -it --rm -p 8888:8888 -v $(pwd):/workspace spatial-gpu:latest \
#       jupyter lab --ip=0.0.0.0 --no-browser --allow-root
# =============================================================================

# Build arguments
ARG USE_GPU=false
ARG INSTALL_R=false

# =============================================================================
# Base Image Selection
# =============================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false

# Select base image based on USE_GPU argument
FROM base-${USE_GPU} AS base

# Re-declare ARGs after FROM (required by Docker)
ARG USE_GPU=false
ARG INSTALL_R=false

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Build tools (required for Rcpp, RcppArmadillo, and C++ extensions)
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    # Libraries for R packages and HDF5
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libxt-dev \
    # Required by R `fs` package (transitively pulled by SpaCET deps
    # scatterpie/shiny/plotly/DT/factoextra). Without libuv-dev, `fs`
    # fails to compile and the SpaCET source install aborts.
    libuv1-dev \
    # Required by R `igraph` (used by SpaCET's lineage tree code).
    libglpk-dev \
    libgmp-dev \
    # Linear algebra (required for scipy, sklearn, SpaCET)
    liblapack-dev \
    libblas-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# R: Install R and SpaCET (optional)
# =============================================================================

ARG INSTALL_R
# Install R 4.4+ from CRAN's apt repo (Ubuntu 22.04's default r-base is 4.1.2,
# which is ABI-incompatible with Posit Package Manager's "latest" binaries —
# they get rejected and rebuilt from source, which is slow and breaks on
# modern Bioconductor deps). The CRAN signing key (51716619E084DAB9, full
# fingerprint E298A3A825C0D65DFD57CBB651716619E084DAB9) is fetched directly
# from CRAN's published key file rather than a GPG keyserver — keyservers
# are flaky inside CI runners and the key fingerprint has rotated over
# time, so pulling from CRAN's authoritative URL is the durable approach.
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing R 4.4+ from CRAN apt repo..." && \
        echo "========================================" && \
        apt-get update && \
        apt-get install -y --no-install-recommends ca-certificates wget && \
        wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
            | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null && \
        echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
            > /etc/apt/sources.list.d/cran.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            r-base \
            r-base-dev && \
        rm -rf /var/lib/apt/lists/* && \
        R -e "cat('R version:', R.version.string, '\n')"; \
    else \
        echo "Skipping R installation"; \
    fi

# Use Posit Package Manager for pre-compiled R binaries (much faster).
# Requires R >= 4.2 to match RSPM's binary ABI (handled by the CRAN apt repo
# step above).
ENV RSPM="https://packagemanager.posit.co/cran/__linux__/jammy/latest"

# Bootstrap BiocManager + install Bioconductor packages FIRST.
# Seurat / NMF and SpaCET pull Bioc-only transitive deps (Biobase, BPCells,
# DESeq2, DelayedArray, GenomicRanges, GenomeInfoDb, glmGamPoi, harmony,
# IRanges, limma, monocle, presto, rtracklayer, multtest, etc.). If we install
# them in the CRAN step before BiocManager is on the repo list, Seurat
# installs with broken deps and the final SpaCET step exits 1.
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing BiocManager + Bioconductor packages..." && \
        echo "========================================" && \
        R -e "options(repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
              install.packages('BiocManager', Ncpus = parallel::detectCores())" && \
        R -e "BiocManager::install(ask = FALSE, update = FALSE)" && \
        R -e "BiocManager::install(c( \
                  'rhdf5', \
                  'BiocGenerics', 'S4Vectors', 'IRanges', 'BiocParallel', \
                  'Biobase', 'SingleCellExperiment', 'SummarizedExperiment', \
                  'DelayedArray', 'GenomeInfoDb', 'GenomicRanges', \
                  'limma', 'DESeq2', 'glmGamPoi', 'rtracklayer', 'multtest', \
                  'MAST' \
              ), ask = FALSE, update = FALSE, Ncpus = parallel::detectCores())"; \
    fi

# CRAN dependencies, with BiocManager::repositories() prepended so any
# remaining Bioc deps (e.g. transitively pulled by Seurat extras) resolve.
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing CRAN packages (binary)..." && \
        echo "========================================" && \
        R -e "options(repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'), \
                                BiocManager::repositories())); \
              install.packages(c( \
                  'remotes', 'devtools', \
                  'Matrix', 'Rcpp', 'RcppArmadillo', \
                  'ggplot2', 'sctransform', \
                  'testthat', 'data.table', \
                  'Seurat', 'hdf5r', \
                  'NMF', 'psych', 'pheatmap' \
              ), dependencies = TRUE, Ncpus = parallel::detectCores())"; \
    fi

# Install SpaCET from GitHub.
# Set repos = BiocManager::repositories() so remotes::install_github with
# dependencies = TRUE searches Bioconductor in addition to CRAN. Without
# this, Bioc-only deps fail to resolve and the install silently no-ops,
# making library(SpaCET) below throw and aborting the build.
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing SpaCET from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600, \
                      repos = BiocManager::repositories()); \
              remotes::install_github('data2intelligence/SpaCET', \
                  dependencies = TRUE, \
                  upgrade = 'never', \
                  force = TRUE, \
                  Ncpus = parallel::detectCores()); \
              if (!requireNamespace('SpaCET', quietly = TRUE)) \
                  stop('SpaCET install failed'); \
              library(SpaCET); \
              cat('SpaCET version:', as.character(packageVersion('SpaCET')), '\n')"; \
    fi

# Verify R installation
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Verifying R installation..." && \
        echo "========================================" && \
        R -e "cat('R version:', R.version.string, '\n'); \
              required <- c('SpaCET', 'Matrix', 'Rcpp', 'RcppArmadillo'); \
              all_ok <- TRUE; \
              for (pkg in required) { \
                  if (requireNamespace(pkg, quietly = TRUE)) { \
                      cat(pkg, as.character(packageVersion(pkg)), 'OK\n') \
                  } else { \
                      cat(pkg, 'MISSING\n'); \
                      all_ok <- FALSE \
                  } \
              }; \
              if (!all_ok) stop('Required R packages are missing!')"; \
    fi

# =============================================================================
# Python: Install spatial-gpu + secactpy
# =============================================================================

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install base Python packages
RUN pip3 install --no-cache-dir \
    "numpy>=1.21" \
    "scipy>=1.7" \
    "pandas>=1.3" \
    "anndata>=0.8" \
    "scanpy>=1.9" \
    "squidpy>=1.2" \
    "scikit-learn>=1.0" \
    "networkx>=2.6" \
    "tqdm>=4.62" \
    "matplotlib>=3.5" \
    "seaborn>=0.12" \
    "h5py>=3.0" \
    "statsmodels>=0.13" \
    jupyter \
    jupyterlab

# Install CuPy for GPU version only
ARG USE_GPU
RUN if [ "$USE_GPU" = "true" ]; then \
        echo "Installing CuPy for GPU support..." && \
        pip3 install --no-cache-dir cupy-cuda12x; \
    else \
        echo "Skipping CuPy (CPU-only build)"; \
    fi

# Install secactpy (secreted protein activity inference)
RUN pip3 install --no-cache-dir secactpy>=0.2.3

# Install spatial-gpu from GitHub
RUN pip3 install --no-cache-dir git+https://github.com/psychemistz/spatial-gpu.git

# Verify Python installation
ARG USE_GPU
RUN python3 -c "\
import spatialgpu; \
b = spatialgpu.get_backend(); \
print(f'spatial-gpu {spatialgpu.__version__} OK'); \
print(f'GPU available: {b.is_gpu_available}, GPU active: {b.is_gpu_active}'); \
import secactpy; \
print(f'secactpy OK')"

# =============================================================================
# Environment
# =============================================================================

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU environment variables (harmless if not using GPU)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Entry Point
# =============================================================================

CMD ["/bin/bash"]

# =============================================================================
# Labels
# =============================================================================

LABEL maintainer="Seongyong Park <https://github.com/psychemistz>"
LABEL description="spatial-gpu - GPU-Accelerated Spatial Omics (SpaCET + SecAct)"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/psychemistz/spatial-gpu"
