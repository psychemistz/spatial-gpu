#!/bin/bash
#SBATCH --job-name=sgpu_t8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_%j.err

set -euo pipefail

# Environment setup
source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorial T8: Bulk Deconvolution Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo ""

pip install -e . --quiet 2>/dev/null || true

python docs/run_full_tutorial_t8_bulk_benchmark.py

echo ""
echo "End: $(date)"
