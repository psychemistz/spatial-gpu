#!/bin/bash
#SBATCH --job-name=sgpu_t8r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_real_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_real_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "=== Tutorial 8b: Real BRCA Pseudobulk Benchmark (GPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

pip install -e . --quiet 2>/dev/null || true

# Verify data
ls data/BRCA_scRNA/BRCA_scRNA_full.h5ad || { echo "ERROR: BRCA scRNA data not found. Run slurm_download_brca_scrna.sh first."; exit 1; }

# Verify GPU backend
echo "Checking GPU backend..."
python -c "import spatialgpu; b = spatialgpu.get_backend(); print(f'GPU available: {b.is_gpu_available}, GPU active: {b.is_gpu_active}'); assert b.is_gpu_active, 'GPU not active!'"
echo ""

python docs/run_full_tutorial_t8_real_brca.py 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t8_real_*.txt 2>/dev/null || true
echo "--- Figure files ---"
ls -lh docs/figures/benchmark_real_brca_*.png 2>/dev/null || true

echo ""
echo "End: $(date)"
