#!/bin/bash
#SBATCH --job-name=sgpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/gpu_test_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/gpu_test_%j.err

set -euo pipefail

# Environment setup
source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== GPU vs CPU Validation ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo ""

# Install package in editable mode if needed
pip install -e . --quiet 2>/dev/null || true

python scripts/test_gpu_vs_cpu.py

echo ""
echo "End: $(date)"
