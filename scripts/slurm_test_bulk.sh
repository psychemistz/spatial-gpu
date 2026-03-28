#!/bin/bash
#SBATCH --job-name=sgpu_bulk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/bulk_test_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/bulk_test_%j.err

set -euo pipefail

# Environment setup
source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Bulk Deconvolution Tests ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo ""

pip install -e . --quiet 2>/dev/null || true

# Run CPU-only tests first (no GPU needed)
echo "--- CPU-only tests ---"
python -m pytest tests/test_deconvolution/test_bulk.py -v \
    -k "not GPUEquivalence" \
    --tb=short 2>&1

echo ""
echo "--- GPU vs CPU equivalence tests ---"
python -m pytest tests/test_deconvolution/test_bulk.py -v \
    -k "GPUEquivalence" \
    --tb=short 2>&1

echo ""
echo "End: $(date)"
