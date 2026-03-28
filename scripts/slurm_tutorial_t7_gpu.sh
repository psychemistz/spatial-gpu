#!/bin/bash
#SBATCH --job-name=sgpu_t7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t7_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t7_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorial 7: stCCC / CosMx LIHC (Full Dataset, GPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

pip install -e . --quiet 2>/dev/null || true
pip install pycirclize --quiet 2>/dev/null || true

# Verify data
ls data/LIHC_CosMx/LIHC_CosMx.h5ad || { echo "ERROR: LIHC_CosMx h5ad not found. Run slurm_extract_cosmx.sh first."; exit 1; }

python docs/run_full_tutorial_t7.py 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t7_*.txt 2>/dev/null || true
echo "--- Figure files ---"
ls -lh docs/figures/cosmx_*.png docs/figures/cosmx_*.html 2>/dev/null || true

echo ""
echo "End: $(date)"
