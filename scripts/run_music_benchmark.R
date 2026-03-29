#!/usr/bin/env Rscript
# =============================================================================
# MuSiC Deconvolution Benchmark on Real BRCA Pseudobulk
#
# Reads:
#   - data/BRCA_scRNA/BRCA_scRNA_full.h5ad  (scRNA-seq reference)
#   - docs/outputs/t8_real_bulk_*.csv        (pseudobulk, exported by Python)
#
# Outputs:
#   - docs/outputs/t8_music_*.csv            (MuSiC proportions)
#
# Requires: MuSiC, Biobase, SingleCellExperiment, anndata (R), zellkonverter
#
# Usage: Rscript scripts/run_music_benchmark.R
# =============================================================================

cat("=== MuSiC Deconvolution Benchmark ===\n")
cat("Start:", format(Sys.time()), "\n\n")

# ---- Install MuSiC if needed ----
if (!requireNamespace("MuSiC", quietly = TRUE)) {
  cat("Installing MuSiC from GitHub...\n")
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  devtools::install_github("xuranw/MuSiC", upgrade = "never")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  BiocManager::install("SingleCellExperiment", ask = FALSE, update = FALSE)
}

library(MuSiC)
library(SingleCellExperiment)
library(Biobase)
library(Matrix)

output_dir <- "docs/outputs"

# ---- Step 1: Load scRNA-seq reference ----
cat("1. Loading scRNA-seq reference...\n")

# Read the exported CSV files (Python script exports these)
sc_counts <- as.matrix(read.csv(
  file.path(output_dir, "t8_real_sc_counts.csv"),
  row.names = 1, check.names = FALSE
))
sc_meta <- read.csv(
  file.path(output_dir, "t8_real_sc_meta.csv"),
  row.names = 1, stringsAsFactors = FALSE
)

cat(sprintf("   scRNA-seq: %d genes x %d cells\n", nrow(sc_counts), ncol(sc_counts)))
cat(sprintf("   Cell types: %s\n", paste(unique(sc_meta$cell_type), collapse=", ")))
cat(sprintf("   Subjects: %d\n", length(unique(sc_meta$subject_id))))

# Build SingleCellExperiment
sc_sce <- SingleCellExperiment(
  assays = list(counts = sc_counts),
  colData = sc_meta
)

# ---- Step 2: Run MuSiC on each scenario ----
scenarios <- c("uniform", "sparse", "tumor_purity", "titration")

for (scenario in scenarios) {
  bulk_file <- file.path(output_dir, paste0("t8_real_bulk_", scenario, ".csv"))

  if (!file.exists(bulk_file)) {
    cat(sprintf("   Skipping %s (file not found: %s)\n", scenario, bulk_file))
    next
  }

  cat(sprintf("\n2. Running MuSiC — %s...\n", scenario))

  bulk_mtx <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
  # Transpose: CSV is samples x genes, MuSiC wants genes x samples
  bulk_mtx <- t(bulk_mtx)
  cat(sprintf("   Bulk: %d genes x %d samples\n", nrow(bulk_mtx), ncol(bulk_mtx)))

  # Intersect genes
  common_genes <- intersect(rownames(bulk_mtx), rownames(sc_counts))
  cat(sprintf("   Common genes: %d\n", length(common_genes)))

  tryCatch({
    result <- MuSiC::music_prop(
      bulk.mtx = bulk_mtx[common_genes, ],
      sc.sce   = sc_sce[common_genes, ],
      clusters = "cell_type",
      samples  = "subject_id",
      verbose  = FALSE
    )

    # Extract proportions (samples x cell_types)
    props <- result$Est.prop.weighted
    cat(sprintf("   MuSiC result: %d samples x %d types\n", nrow(props), ncol(props)))

    # Save
    out_file <- file.path(output_dir, paste0("t8_music_", scenario, ".csv"))
    write.csv(props, out_file)
    cat(sprintf("   Saved: %s\n", out_file))

  }, error = function(e) {
    cat(sprintf("   ERROR in %s: %s\n", scenario, e$message))
  })
}

cat(sprintf("\nDone: %s\n", format(Sys.time())))
