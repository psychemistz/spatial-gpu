#!/usr/bin/env Rscript
# =============================================================================
# MuSiC benchmark for one trial.
#
# Usage: Rscript scripts/run_music_trial.R <trial>
# Reads docs/outputs/trials/T{NN}/{sc_counts,sc_meta,bulk_*}.csv
# Writes docs/outputs/trials/T{NN}/music_<scenario>.csv
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript run_music_trial.R <trial>")
trial <- as.integer(args[1])
trial_dir <- sprintf("docs/outputs/trials/T%02d", trial)

cat(sprintf("=== MuSiC trial T%02d ===\n", trial))
cat(sprintf("Start: %s  trial_dir: %s\n", format(Sys.time()), trial_dir))

# Force-remove BiocParallel's core-cap (see run_dwls_minor_trial.R for detail)
Sys.unsetenv("_R_CHECK_LIMIT_CORES_")

if (!requireNamespace("MuSiC", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", repos = "https://cloud.r-project.org/")
  devtools::install_github("xuranw/MuSiC", upgrade = "never")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos = "https://cloud.r-project.org/")
  BiocManager::install("SingleCellExperiment", ask = FALSE, update = FALSE)
}

suppressPackageStartupMessages({
  library(MuSiC)
  library(SingleCellExperiment)
  library(Biobase)
})

sc_counts <- as.matrix(read.csv(file.path(trial_dir, "sc_counts.csv"), row.names = 1, check.names = FALSE))
sc_meta <- read.csv(file.path(trial_dir, "sc_meta.csv"), row.names = 1, stringsAsFactors = FALSE)
cat(sprintf("  Reference: %d genes x %d cells (%d subjects)\n",
            nrow(sc_counts), ncol(sc_counts), length(unique(sc_meta$subject_id))))

sc_sce <- SingleCellExperiment(assays = list(counts = sc_counts), colData = sc_meta)

scenarios <- c("uniform", "sparse", "tumor_purity", "titration")
for (scenario in scenarios) {
  bulk_file <- file.path(trial_dir, paste0("bulk_", scenario, ".csv"))
  out_file  <- file.path(trial_dir, paste0("music_", scenario, ".csv"))
  if (!file.exists(bulk_file)) {
    cat(sprintf("  skip %s — missing %s\n", scenario, bulk_file))
    next
  }
  bulk_mtx <- t(as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE)))
  common_genes <- intersect(rownames(bulk_mtx), rownames(sc_counts))
  bulk_mtx <- bulk_mtx[common_genes, , drop = FALSE]

  res <- music_prop(
    bulk.mtx     = bulk_mtx,
    sc.sce       = sc_sce,
    clusters     = "cell_type",
    samples      = "subject_id",
    select.ct    = unique(sc_meta$cell_type),
    verbose      = FALSE
  )

  props <- as.data.frame(res$Est.prop.weighted)
  write.csv(props, out_file)
  cat(sprintf("  %s: %d samples x %d types -> %s\n",
              scenario, nrow(props), ncol(props), out_file))
}

cat(sprintf("\nDone: %s\n", format(Sys.time())))
