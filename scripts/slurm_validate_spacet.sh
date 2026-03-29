#!/bin/bash
#SBATCH --job-name=spacet_val
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_val_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_val_%j.err

set -euo pipefail

# Load R for SpaCET
module load R/4.3

# Load Python
source /data/parks34/conda/etc/profile.d/conda.sh
conda activate secactpy

# Force native Python mode (no R subprocess in spatial-gpu)
export SPATIALGPU_FORCE_PYTHON=1

cd /data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== SpaCET R vs Python (native) Validation ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python scripts/validate_r_vs_python_native.py \
    --samples \
        "BRCA_10x_Datasets/Version1.0.0_Breast.Cancer_rep1" \
        "BRCA_10x_Datasets/Version1.0.0_Breast.Cancer_rep2"

echo ""
echo "End: $(date)"
