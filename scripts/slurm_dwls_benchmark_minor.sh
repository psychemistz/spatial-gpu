#!/bin/bash
#SBATCH --job-name=dwls_minor
#SBATCH --partition=norm
#SBATCH --mem=128g
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/dwls_minor_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/dwls_minor_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy
module load R/4.3

echo "=== DWLS (minor) Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "R: $(R --version | head -1)"

ls docs/outputs/t8_real_sc_counts.csv      || { echo "ERROR: missing counts."; exit 1; }
ls docs/outputs/t8_real_sc_meta_minor.csv  || { echo "ERROR: run export_dwls_minor_meta.py first."; exit 1; }
ls docs/outputs/t8_real_bulk_uniform.csv   || { echo "ERROR: run tutorial t8 real BRCA first."; exit 1; }

Rscript scripts/run_dwls_benchmark_minor.R 2>&1

echo ""
echo "--- DWLS minor output files ---"
ls -lh docs/outputs/t8_dwls_minor_*.csv 2>/dev/null || true
ls -lh docs/outputs/t8_dwls_minor_signature.rds 2>/dev/null || true

echo ""
echo "End: $(date)"
