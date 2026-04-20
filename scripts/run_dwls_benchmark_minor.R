#!/usr/bin/env Rscript
# =============================================================================
# DWLS Deconvolution Benchmark — MINOR (subtype) resolution.
#
# Fair-to-SpaCET variant: DWLS is given the same train-set cells but at
# celltype_minor resolution (~22 kept minors after MIN_CELLS=30 filter), so
# the dampened-WLS regression works at the subtype level SpaCET models
# internally. Minor-to-major collapse happens at evaluation time in Python.
#
# Reads:
#   - docs/outputs/t8_real_sc_counts.csv         (same cells as broad DWLS)
#   - docs/outputs/t8_real_sc_meta_minor.csv     (celltype_minor per cell, subset)
#   - docs/outputs/t8_real_bulk_*.csv            (pseudobulk, samples x genes)
#
# Outputs:
#   - docs/outputs/t8_dwls_minor_signature.rds
#   - docs/outputs/t8_dwls_minor_sigmat/
#   - docs/outputs/t8_dwls_minor_*.csv           (samples x ~22 minor types)
#
# Usage: Rscript scripts/run_dwls_benchmark_minor.R
# =============================================================================

cat("=== DWLS Benchmark (minor resolution) ===\n")
cat("Start:", format(Sys.time()), "\n\n")

source("scripts/_dwls_common.R")
ensure_dwls_packages()

suppressPackageStartupMessages({
  library(DWLS)
  library(MAST)
})
cat("DWLS version:", as.character(packageVersion("DWLS")), "\n")
cat("MAST version:", as.character(packageVersion("MAST")), "\n\n")

output_dir    <- "docs/outputs"
sig_cache_dir <- file.path(output_dir, "t8_dwls_minor_sigmat")
sig_rds       <- file.path(output_dir, "t8_dwls_minor_signature.rds")
dir.create(sig_cache_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Step 1: Load scRNA-seq reference + minor meta ----
cat("1. Loading scRNA-seq reference (minor-level meta)...\n")
sc_counts <- as.matrix(read.csv(
  file.path(output_dir, "t8_real_sc_counts.csv"),
  row.names = 1, check.names = FALSE
))
sc_meta <- read.csv(
  file.path(output_dir, "t8_real_sc_meta_minor.csv"),
  row.names = 1, stringsAsFactors = FALSE
)

# Keep only cells that survived the minor-type filter
keep_cells <- intersect(colnames(sc_counts), rownames(sc_meta))
sc_counts <- sc_counts[, keep_cells]
sc_meta <- sc_meta[keep_cells, , drop = FALSE]
stopifnot(identical(colnames(sc_counts), rownames(sc_meta)))

cat(sprintf("   scRNA-seq: %d genes x %d cells (pre-filter)\n",
            nrow(sc_counts), ncol(sc_counts)))

# Pre-filter near-zero genes before MAST. Halves peak DE memory; these
# genes have no per-cell-type signal DWLS could use anyway.
nz <- rowSums(sc_counts > 0) >= 5
sc_counts <- sc_counts[nz, , drop = FALSE]
cat(sprintf("   After gene filter (>=5 non-zero cells): %d genes x %d cells\n",
            nrow(sc_counts), ncol(sc_counts)))
cat(sprintf("   Minor types (%d): %s\n",
            length(unique(sc_meta$celltype_minor)),
            paste(sort(unique(sc_meta$celltype_minor)), collapse = ", ")))

ct <- make_ct_maps(sc_meta$celltype_minor)
sc_meta$ct_clean <- ct$to_clean[sc_meta$celltype_minor]
cat(sprintf("   Sanitized: %s\n", paste(ct$clean, collapse = ", ")))

# ---- Step 2: Build / load signature ----
cat("\n2. Building DWLS signature via MAST DE (per minor type)...\n")
Signature <- build_or_load_signature(
  sc_counts, sc_meta$ct_clean, sig_rds, sig_cache_dir
)
cat(sprintf("   Signature: %d genes x %d minor types\n",
            nrow(Signature), ncol(Signature)))

# ---- Step 3: Deconvolve each scenario at minor resolution ----
deconvolve_scenarios(
  Signature, ct$to_original,
  scenarios   = c("uniform", "sparse", "tumor_purity", "titration"),
  output_dir  = output_dir,
  bulk_prefix = "t8_real_bulk_",
  out_prefix  = "t8_dwls_minor_",
  label       = "DWLS (minor)"
)

cat(sprintf("\nDone: %s\n", format(Sys.time())))
