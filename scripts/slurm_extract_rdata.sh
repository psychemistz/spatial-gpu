#!/bin/bash
#SBATCH --job-name=sgpu_extract
#SBATCH --partition=norm
#SBATCH --mem=16g
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/extract_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/extract_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== R Data Extraction + h5ad Conversion ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

# Load R
module load R/4.3

echo ""
echo "--- Step 1: Extract R data to CSV/MTX ---"
Rscript scripts/extract_spacet_data.R

# Load Python
source ~/bin/myconda
conda activate secactpy

echo ""
echo "--- Step 2: Convert CSV/MTX to h5ad ---"
pip install -e . --quiet 2>/dev/null || true
python scripts/convert_to_h5ad.py

echo ""
echo "--- Step 3: Verify data files ---"
echo "oldST_PDAC:"
ls -lh data/oldST_PDAC/*.h5ad 2>/dev/null || echo "  MISSING h5ad files"
echo "hiresST_CRC:"
ls -lh data/hiresST_CRC/*.h5ad 2>/dev/null || echo "  MISSING h5ad files"

echo ""
echo "End: $(date)"
echo "=== Extraction complete ==="
