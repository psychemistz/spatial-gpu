#!/bin/bash
# Generic per-method trial array script.
#
# Usage:
#   sbatch --array=0-9 --export=ALL,METHOD=spacet_v3_irwls --job-name=trial_v3 scripts/slurm_trial_array.sh
#
# METHOD options:
#   spacet_v0_none | spacet_v1_ratio | spacet_v3_irwls
#   music
#   dwls_minor

#SBATCH --partition=norm
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/trial_%x_%A_%a.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/trial_%x_%A_%a.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy

: "${METHOD:?METHOD must be set via --export=ALL,METHOD=...}"
TRIAL=${SLURM_ARRAY_TASK_ID}

echo "=== ${METHOD} trial T$(printf %02d ${TRIAL}) ==="
echo "Start: $(date)  Node: $(hostname)"

# Ensure trial data exists; generate if missing.
TRIAL_DIR="docs/outputs/trials/T$(printf %02d ${TRIAL})"
if [ ! -f "${TRIAL_DIR}/sc_counts.csv" ]; then
    echo "Trial data not found, generating..."
    python scripts/prep_trial_data.py --trial "${TRIAL}"
fi

case "${METHOD}" in
    spacet_*)
        VARIANT="${METHOD#spacet_}"
        pip install -e . --quiet 2>/dev/null || true
        python scripts/bench_spacet_trial.py --method "${VARIANT}" --trial "${TRIAL}" 2>&1
        ;;
    music)
        module load R/4.3
        # Avoid BiocParallel's _R_CHECK_LIMIT_CORES_ strict-2-core cap
        unset _R_CHECK_LIMIT_CORES_
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
        Rscript scripts/run_music_trial.R "${TRIAL}" 2>&1
        ;;
    dwls_minor)
        module load R/4.3
        unset _R_CHECK_LIMIT_CORES_
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
        Rscript scripts/run_dwls_minor_trial.R "${TRIAL}" 2>&1
        ;;
    *)
        echo "ERROR: unknown METHOD=${METHOD}"; exit 1
        ;;
esac

echo ""
echo "End: $(date)"
