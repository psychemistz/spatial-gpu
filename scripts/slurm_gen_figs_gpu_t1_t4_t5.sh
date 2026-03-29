#!/bin/bash
#SBATCH --job-name=sgpu_fig145
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/fig_t1_t4_t5_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/fig_t1_t4_t5_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorial Figure Generation (T1, T4, T5) — GPU ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

# Environment setup
source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo ""

# Install package in editable mode
pip install -e . --quiet 2>/dev/null || true

# Verify data symlinks
echo "--- Verifying data ---"
ls -la data/Visium_BC/filtered_feature_bc_matrix.h5 || { echo "ERROR: Visium_BC data not found"; exit 1; }

# Tutorial 1: Visium Breast Cancer (full pipeline)
echo ""
echo "=== Tutorial 1: Visium BC ==="
python docs/generate_figures.py 2>&1

# Tutorial 4: Gene Set Score (uses Visium_BC data)
echo ""
echo "=== Tutorial 4: GeneSetScore ==="
python docs/generate_all_figures.py 4 2>&1

# Tutorial 5: Spatial Correlation (uses Visium_BC data)
echo ""
echo "=== Tutorial 5: SpatialCorrelation ==="
python docs/generate_all_figures.py 5 2>&1

echo ""
echo "--- Generated figures ---"
ls -lh docs/figures/qc_umi_gene.png docs/figures/fraction_*.png docs/figures/composition_pie.png \
       docs/figures/most_abundant.png docs/figures/colocalization.png docs/figures/lr_network_score.png \
       docs/figures/cell_type_pair_caf_m2.png docs/figures/interface*.png docs/figures/distance_to_interface.png \
       docs/figures/malignant_states.png \
       docs/figures/gs_*.png docs/figures/sc_coexpression.png 2>/dev/null

echo ""
echo "End: $(date)"
echo "=== T1/T4/T5 figure generation complete ==="
