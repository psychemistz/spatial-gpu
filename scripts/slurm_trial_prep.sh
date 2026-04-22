#!/bin/bash
# Pre-generate trial data for one trial (SLURM array index = trial number).
#
# Usage: sbatch --array=0-9 scripts/slurm_trial_prep.sh

#SBATCH --job-name=trial_prep
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/trial_prep_%A_%a.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/trial_prep_%A_%a.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy

TRIAL=${SLURM_ARRAY_TASK_ID}
echo "=== Trial prep T$(printf %02d ${TRIAL}) ==="
echo "Start: $(date)  Node: $(hostname)"

python scripts/prep_trial_data.py --trial "${TRIAL}"

echo "End: $(date)"
