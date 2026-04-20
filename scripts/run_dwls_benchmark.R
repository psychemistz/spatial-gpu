#!/usr/bin/env Rscript
# =============================================================================
# DWLS Deconvolution Benchmark on Real BRCA Pseudobulk (major types).
#
# DWLS (Dampened Weighted Least Squares), Tsoucas et al. 2019, Nat Commun.
# Builds a cell-type signature matrix via MAST DE, then solves damped WLS
# per bulk sample.
#
# Reads (same exports as the MuSiC benchmark):
#   - docs/outputs/t8_real_sc_counts.csv    (scRNA-seq reference, genes x cells)
#   - docs/outputs/t8_real_sc_meta.csv      (cell_type, subject_id per cell)
#   - docs/outputs/t8_real_bulk_*.csv       (pseudobulk, samples x genes)
#
# Outputs:
#   - docs/outputs/t8_dwls_signature.rds    (cached signature matrix)
#   - docs/outputs/t8_dwls_sigmat/          (MAST DE cache, auto-reused)
#   - docs/outputs/t8_dwls_*.csv            (DWLS proportions, samples x types)
#
# Usage: Rscript scripts/run_dwls_benchmark.R
# =============================================================================

cat("=== DWLS Deconvolution Benchmark ===\n")
cat("Start:", format(Sys.time()), "\n\n")

source("scripts/_dwls_common.R")
ensure_dwls_packages()

suppressPackageStartupMessages({
  library(DWLS)
  library(MAST)
})
cat("DWLS version:", as.character(packageVersion("DWLS")), "\n")
cat("MAST version:", as.character(packageVersion("MAST")), "\n\n")

output_dir <- "docs/outputs"
sig_cache_dir <- file.path(output_dir, "t8_dwls_sigmat")
sig_rds       <- file.path(output_dir, "t8_dwls_signature.rds")
dir.create(sig_cache_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Step 1: Load scRNA-seq reference ----
cat("1. Loading scRNA-seq reference...\n")
sc_counts <- as.matrix(read.csv(
  file.path(output_dir, "t8_real_sc_counts.csv"),
  row.names = 1, check.names = FALSE
))
sc_meta <- read.csv(
  file.path(output_dir, "t8_real_sc_meta.csv"),
  row.names = 1, stringsAsFactors = FALSE
)
stopifnot(identical(colnames(sc_counts), rownames(sc_meta)))
cat(sprintf("   scRNA-seq: %d genes x %d cells\n", nrow(sc_counts), ncol(sc_counts)))
cat(sprintf("   Cell types: %s\n",
            paste(sort(unique(sc_meta$cell_type)), collapse = ", ")))

ct <- make_ct_maps(sc_meta$cell_type)
sc_meta$cell_type_clean <- ct$to_clean[sc_meta$cell_type]
cat(sprintf("   Sanitized types: %s\n", paste(ct$clean, collapse = ", ")))

# ---- Step 2: Build (or load cached) signature matrix ----
cat("\n2. Building DWLS signature matrix via MAST DE...\n")
Signature <- build_or_load_signature(
  sc_counts, sc_meta$cell_type_clean, sig_rds, sig_cache_dir
)
cat(sprintf("   Signature matrix: %d genes x %d cell types\n",
            nrow(Signature), ncol(Signature)))
cat(sprintf("   Cell types in signature: %s\n",
            paste(colnames(Signature), collapse = ", ")))

# ---- Step 3: Deconvolve each scenario ----
deconvolve_scenarios(
  Signature, ct$to_original,
  scenarios   = c("uniform", "sparse", "tumor_purity", "titration"),
  output_dir  = output_dir,
  bulk_prefix = "t8_real_bulk_",
  out_prefix  = "t8_dwls_",
  label       = "DWLS"
)

cat(sprintf("\nDone: %s\n", format(Sys.time())))
