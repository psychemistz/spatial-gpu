#!/bin/bash
#SBATCH --job-name=spacet_wbench
#SBATCH --partition=norm
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_wbench_%x_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_wbench_%x_%j.err

# Usage: sbatch --export=ALL,METHOD=v2_absolute --job-name=wbench_v2 scripts/slurm_bench_spacet_weighting.sh
# METHOD: v0_none | v1_ratio | v2_absolute | v3_irwls | v4_per_type

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy

: "${METHOD:?METHOD must be set via --export=ALL,METHOD=...}"

echo "=== SpaCET weighting benchmark — METHOD=${METHOD} ==="
echo "Start: $(date)  Node: $(hostname)"

pip install -e . --quiet 2>/dev/null || true

python scripts/bench_spacet_weighting.py --method "${METHOD}" 2>&1

echo ""
echo "--- Output files ---"
ls -lh docs/outputs/t8_spacet_${METHOD}_*.csv docs/outputs/t8_spacet_${METHOD}_*.txt 2>/dev/null || true
echo "End: $(date)"
