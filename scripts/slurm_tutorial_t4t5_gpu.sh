#!/bin/bash
#SBATCH --job-name=sgpu_t4t5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t4t5_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t4t5_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorials 4+5: GeneSetScore + SpatialCorrelation (Full Dataset, GPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

pip install -e . --quiet 2>/dev/null || true

# Verify data
ls data/Visium_BC/filtered_feature_bc_matrix.h5 || { echo "ERROR: Visium_BC data not found"; exit 1; }

# T4+T5 includes genome-wide Moran's I (all ~33K genes) and pairwise (33K x 33K)
# These are the heaviest computations — GPU accelerated
python docs/run_full_tutorial_t4_t5.py 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t5_*.txt 2>/dev/null || true
echo "--- Figure files ---"
ls -lh docs/figures/gs_*.png docs/figures/sc_coexpression.png 2>/dev/null

echo ""
echo "End: $(date)"
