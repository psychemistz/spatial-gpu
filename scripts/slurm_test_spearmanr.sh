#!/bin/bash
#SBATCH --job-name=sgpu_spearmanr_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/test_spearmanr_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/test_spearmanr_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== GPU Spearman correlation primitive tests ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

pip install -e . --quiet 2>/dev/null || true

echo ""
echo "--- Running TestGPUSpearmanr ---"
python -m pytest tests/test_gpu_ops.py::TestGPUSpearmanr -x -v 2>&1 | tail -30

echo ""
echo "Done: $(date)"
