#!/bin/bash
#SBATCH --job-name=score_dream
#SBATCH --partition=norm
#SBATCH --mem=8g
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/score_dream_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/score_dream_%j.err

set -euo pipefail
source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

DATA=/vf/users/parks34/projects/0sigdiscov/data/dream
OUT=/vf/users/parks34/projects/0sigdiscov/bench_outputs/dream

python scripts/bench_real/score_dream.py \
    --predictions "$OUT" \
    --gt "$DATA/gt_coarse/coarse.csv" \
    --resolution coarse \
    --out "$OUT/scores" \
    --plot
