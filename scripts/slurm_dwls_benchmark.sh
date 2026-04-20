#!/bin/bash
#SBATCH --job-name=dwls_bm
#SBATCH --partition=norm
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/dwls_benchmark_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/dwls_benchmark_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy
module load R/4.3

echo "=== DWLS Benchmark on Real BRCA Pseudobulk ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "R: $(R --version | head -1)"
echo "Rscript: $(which Rscript)"

# Verify exported data exists
ls docs/outputs/t8_real_sc_counts.csv || { echo "ERROR: Run tutorial t8 real BRCA first."; exit 1; }
ls docs/outputs/t8_real_bulk_uniform.csv || { echo "ERROR: Run tutorial t8 real BRCA first."; exit 1; }

# DWLS/MAST install handled by the R script itself (idempotent).
Rscript scripts/run_dwls_benchmark.R 2>&1

echo ""
echo "--- DWLS output files ---"
ls -lh docs/outputs/t8_dwls_*.csv 2>/dev/null || true
ls -lh docs/outputs/t8_dwls_signature.rds 2>/dev/null || true

echo ""
echo "End: $(date)"
