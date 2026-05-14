#!/bin/bash
#SBATCH --job-name=fetch_dream
#SBATCH --partition=norm
#SBATCH --mem=4g
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/fetch_dream_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/fetch_dream_%j.err

set -euo pipefail

# Auth: SYNAPSE_AUTH_TOKEN env var or ~/.synapseConfig
# Requires synapseclient in the conda env (pip install synapseclient).

source ~/bin/myconda
conda activate spatialgpu  # or whatever your active env is — see scripts/bench_real/README.md

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

DEST=/vf/users/parks34/projects/0sigdiscov/data/dream
mkdir -p "$DEST"

echo "=== Fetch DREAM Challenge data ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Dest: $DEST"
echo ""

python scripts/bench_real/fetch_dream.py --dest "$DEST"

echo ""
echo "Done: $(date)"
