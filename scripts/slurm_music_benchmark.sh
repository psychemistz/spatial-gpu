#!/bin/bash
#SBATCH --job-name=music_bm
#SBATCH --partition=norm
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/music_benchmark_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/music_benchmark_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy
module load R/4.3

echo "=== MuSiC Benchmark on Real BRCA Pseudobulk ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "R: $(R --version | head -1)"
echo "Rscript: $(which Rscript)"

# Verify exported data exists
ls docs/outputs/t8_real_sc_counts.csv || { echo "ERROR: Run tutorial t8 real BRCA first."; exit 1; }
ls docs/outputs/t8_real_bulk_uniform.csv || { echo "ERROR: Run tutorial t8 real BRCA first."; exit 1; }

# Install TOAST (Bioconductor dependency for MuSiC) and MuSiC
Rscript -e "
if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cloud.r-project.org/')
BiocManager::install('TOAST', ask=FALSE, update=FALSE)
if (!requireNamespace('MuSiC', quietly=TRUE)) {
  if (!requireNamespace('devtools', quietly=TRUE)) install.packages('devtools', repos='https://cloud.r-project.org/')
  devtools::install_github('xuranw/MuSiC', upgrade='never')
}
cat('MuSiC version:', as.character(packageVersion('MuSiC')), '\n')
" 2>&1

# Run MuSiC
Rscript scripts/run_music_benchmark.R 2>&1

echo ""
echo "--- MuSiC output files ---"
ls -lh docs/outputs/t8_music_*.csv 2>/dev/null || true

echo ""
echo "End: $(date)"
