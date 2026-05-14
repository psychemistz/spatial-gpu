#!/bin/bash
#SBATCH --job-name=spacet_dream
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_dream_%A_%a.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/spacet_dream_%A_%a.err

# Array tasks index (dataset, method) pairs — 4 in-vitro datasets x 2 methods = 8.
# Customize the CONFIGS list below before submitting.

set -euo pipefail
source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

DATA=/vf/users/parks34/projects/0sigdiscov/data/dream
OUT=/vf/users/parks34/projects/0sigdiscov/bench_outputs/dream
REF=$DATA/reference/reference_coarse.pkl
mkdir -p "$OUT"

CONFIGS=(
    "DS1:$DATA/in_vitro_expression/DS1_hugo_tpm.tsv:v0_none"
    "DS1:$DATA/in_vitro_expression/DS1_hugo_tpm.tsv:v3_irwls"
    "DS2:$DATA/in_vitro_expression/DS2_hugo_tpm.tsv:v0_none"
    "DS2:$DATA/in_vitro_expression/DS2_hugo_tpm.tsv:v3_irwls"
    "DS3:$DATA/in_vitro_expression/DS3_hugo_tpm.tsv:v0_none"
    "DS3:$DATA/in_vitro_expression/DS3_hugo_tpm.tsv:v3_irwls"
    "DS4:$DATA/in_vitro_expression/DS4_hugo_tpm.tsv:v0_none"
    "DS4:$DATA/in_vitro_expression/DS4_hugo_tpm.tsv:v3_irwls"
)
IFS=':' read -r DATASET BULK METHOD <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "=== task $SLURM_ARRAY_TASK_ID: dataset=$DATASET method=$METHOD ==="
echo "bulk: $BULK"
echo "node: $(hostname)"
date

python scripts/bench_real/run_spacet_dream.py \
    --bulk "$BULK" \
    --reference "$REF" \
    --method "$METHOD" \
    --out "$OUT/${DATASET}_${METHOD}.csv" \
    --no-malignant \
    --n-jobs 8

echo "done: $(date)"
