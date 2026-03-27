#!/bin/bash
#SBATCH --job-name=sgpu_t2t3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t2t3_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t2t3_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorials 2+3: oldST PDAC + hiresST CRC (Full Dataset, GPU) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"

pip install -e . --quiet 2>/dev/null || true

# Verify h5ad data (requires prior R extraction job)
ls data/oldST_PDAC/st_PDAC.h5ad || { echo "ERROR: oldST_PDAC h5ad not found. Run slurm_extract_rdata.sh first."; exit 1; }
ls data/hiresST_CRC/hiresST_CRC.h5ad || { echo "ERROR: hiresST_CRC h5ad not found. Run slurm_extract_rdata.sh first."; exit 1; }

# Verify GPU backend is active
echo "Checking GPU backend..."
python -c "import spatialgpu; b = spatialgpu.get_backend(); print(f'GPU available: {b.is_gpu_available}, GPU active: {b.is_gpu_active}'); assert b.is_gpu_active, 'GPU not active!'"
echo ""

python docs/run_full_tutorial_t2_t3.py 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t2_*.txt docs/outputs/t3_*.txt 2>/dev/null
echo "--- Figure files ---"
ls -lh docs/figures/pdac_*.png docs/figures/crc_*.png 2>/dev/null

echo ""
echo "End: $(date)"
