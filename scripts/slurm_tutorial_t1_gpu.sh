#!/bin/bash
#SBATCH --job-name=sgpu_t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t1_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t1_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorial 1: Visium BC (Full Dataset, GPU) ==="
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

# Verify GPU backend is active
echo "Checking GPU backend..."
python -c "import spatialgpu; b = spatialgpu.get_backend(); print(f'GPU available: {b.is_gpu_available}, GPU active: {b.is_gpu_active}'); assert b.is_gpu_active, 'GPU not active!'"
echo ""

python docs/run_full_tutorial_t1.py 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t1_*.txt 2>/dev/null
echo "--- Figure files ---"
ls -lh docs/figures/qc_umi_gene.png docs/figures/fraction_*.png docs/figures/composition_pie.png \
       docs/figures/most_abundant.png docs/figures/colocalization.png docs/figures/lr_network_score.png \
       docs/figures/cell_type_pair_caf_m2.png docs/figures/interface*.png \
       docs/figures/distance_to_interface.png docs/figures/malignant_states.png 2>/dev/null

echo ""
echo "End: $(date)"
