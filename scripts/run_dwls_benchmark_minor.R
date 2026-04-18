#!/usr/bin/env Rscript
# =============================================================================
# DWLS Deconvolution Benchmark — MINOR (subtype) resolution
#
# Fair-to-SpaCET variant: DWLS is given the same train-set cells but at
# celltype_minor resolution (22 kept minors, filtered at MIN_CELLS=30), so
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
#   - docs/outputs/t8_dwls_minor_*.csv           (samples x 22 minor types)
#
# Usage: Rscript scripts/run_dwls_benchmark_minor.R
# =============================================================================

cat("=== DWLS Benchmark (minor resolution) ===\n")
cat("Start:", format(Sys.time()), "\n\n")

suppressPackageStartupMessages({
  library(DWLS)
  library(MAST)
})

cat("DWLS version:", as.character(packageVersion("DWLS")), "\n")
cat("MAST version:", as.character(packageVersion("MAST")), "\n\n")

output_dir <- "docs/outputs"
sig_cache_dir <- file.path(output_dir, "t8_dwls_minor_sigmat")
sig_rds <- file.path(output_dir, "t8_dwls_minor_signature.rds")
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

# Pre-filter: drop genes expressed in <5 cells. Cuts MAST memory per DE call
# roughly in half without changing what DWLS could use (these genes carry no
# per-cell-type signal anyway).
nz <- rowSums(sc_counts > 0) >= 5
sc_counts <- sc_counts[nz, , drop = FALSE]
cat(sprintf("   After gene filter (>=5 non-zero cells): %d genes x %d cells\n",
            nrow(sc_counts), ncol(sc_counts)))
cat(sprintf("   Minor types (%d): %s\n",
            length(unique(sc_meta$celltype_minor)),
            paste(sort(unique(sc_meta$celltype_minor)), collapse = ", ")))

# DWLS's eval(parse(...)) breaks on spaces/hyphens/plus/etc. Sanitize.
sanitize_ct <- function(x) gsub("[^A-Za-z0-9]", "_", x)
ct_original <- sort(unique(sc_meta$celltype_minor))
ct_clean <- sanitize_ct(ct_original)
names(ct_original) <- ct_clean
ct_map <- setNames(ct_clean, ct_original)
if (any(duplicated(ct_clean))) {
  stop("Sanitized minor names collide: ",
       paste(ct_clean[duplicated(ct_clean)], collapse = ", "))
}
sc_meta$ct_clean <- ct_map[sc_meta$celltype_minor]
cat(sprintf("   Sanitized: %s\n", paste(ct_clean, collapse = ", ")))

# ---- Step 2: Build / load signature ----
cat("\n2. Building DWLS signature via MAST DE (per minor type)...\n")
if (file.exists(sig_rds)) {
  cat("   Loading cached signature from", sig_rds, "\n")
  Signature <- readRDS(sig_rds)
} else {
  t0 <- Sys.time()
  Signature <- DWLS::buildSignatureMatrixMAST(
    scdata      = sc_counts,
    id          = sc_meta$ct_clean,
    path        = sig_cache_dir,
    diff.cutoff = 0.5,
    pval.cutoff = 0.01
  )
  dt <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("   Signature built in %.1f min\n", as.numeric(dt)))
  saveRDS(Signature, sig_rds)
}
cat(sprintf("   Signature: %d genes x %d minor types\n",
            nrow(Signature), ncol(Signature)))

# ---- Step 3: Deconvolve each scenario at minor resolution ----
scenarios <- c("uniform", "sparse", "tumor_purity", "titration")

for (scenario in scenarios) {
  bulk_file <- file.path(output_dir, paste0("t8_real_bulk_", scenario, ".csv"))
  out_file  <- file.path(output_dir, paste0("t8_dwls_minor_", scenario, ".csv"))
  if (!file.exists(bulk_file)) {
    cat(sprintf("\n   Skipping %s — missing: %s\n", scenario, bulk_file))
    next
  }
  cat(sprintf("\n3. DWLS (minor) — %s\n", scenario))

  bulk_mtx <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
  bulk_mtx <- t(bulk_mtx)
  cat(sprintf("   Bulk: %d genes x %d samples\n", nrow(bulk_mtx), ncol(bulk_mtx)))

  ct_names <- colnames(Signature)
  props <- matrix(
    NA_real_,
    nrow = ncol(bulk_mtx), ncol = length(ct_names),
    dimnames = list(colnames(bulk_mtx), ct_names)
  )

  t0 <- Sys.time()
  for (i in seq_len(ncol(bulk_mtx))) {
    tr <- tryCatch(
      DWLS::trimData(Signature, bulk_mtx[, i]),
      error = function(e) { cat("   trimData error:", e$message, "\n"); NULL }
    )
    if (is.null(tr)) next
    est <- tryCatch(
      DWLS::solveDampenedWLS(tr$sig, tr$bulk),
      error = function(e) { cat("   solveDampenedWLS error:", e$message, "\n"); NULL }
    )
    if (is.null(est)) next
    est[est < 0] <- 0
    if (sum(est) > 0) est <- est / sum(est)
    props[i, names(est)] <- est

    if (i %% 25 == 0 || i == ncol(bulk_mtx)) {
      cat(sprintf("     %d / %d samples\n", i, ncol(bulk_mtx)))
    }
  }
  dt <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("   Done in %.1f min\n", as.numeric(dt)))

  colnames(props) <- ct_original[colnames(props)]
  write.csv(props, out_file)
  cat(sprintf("   Saved: %s\n", out_file))
}

cat(sprintf("\nDone: %s\n", format(Sys.time())))
