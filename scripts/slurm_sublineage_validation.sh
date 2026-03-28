#!/bin/bash
#SBATCH --job-name=sublin_val
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/sublineage_val_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/sublineage_val_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Sublineage Deconvolution Concordance: R SpaCET vs spatial-gpu ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo ""

# ---- Step 1: Run R SpaCET and save intermediates ----
echo "==== Step 1: Running R SpaCET deconvolution ===="
module load R/4.3
Rscript scripts/r_save_sublineage_intermediates.R
echo ""

# ---- Step 2: Run Python comparison ----
echo "==== Step 2: Running Python deconvolution + comparison ===="
source ~/bin/myconda
conda activate secactpy

pip install -e . --quiet 2>/dev/null || true

python scripts/validate_sublineage_concordance.py

echo ""
echo "End: $(date)"
