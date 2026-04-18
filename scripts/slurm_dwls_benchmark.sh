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

# Install DWLS + MAST (idempotent)
Rscript -e "
if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', repos='https://cloud.r-project.org/')
if (!requireNamespace('MAST', quietly=TRUE)) BiocManager::install('MAST', ask=FALSE, update=FALSE)
if (!requireNamespace('DWLS', quietly=TRUE)) {
  tryCatch(
    install.packages('DWLS', repos='https://cloud.r-project.org/'),
    error = function(e) {
      if (!requireNamespace('remotes', quietly=TRUE)) install.packages('remotes', repos='https://cloud.r-project.org/')
      remotes::install_github('dtsoucas/DWLS', upgrade='never')
    }
  )
}
cat('DWLS version:', as.character(packageVersion('DWLS')), '\n')
cat('MAST version:', as.character(packageVersion('MAST')), '\n')
" 2>&1

# Run DWLS
Rscript scripts/run_dwls_benchmark.R 2>&1

echo ""
echo "--- DWLS output files ---"
ls -lh docs/outputs/t8_dwls_*.csv 2>/dev/null || true
ls -lh docs/outputs/t8_dwls_signature.rds 2>/dev/null || true

echo ""
echo "End: $(date)"
