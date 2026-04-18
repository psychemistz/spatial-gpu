#!/usr/bin/env Rscript
# =============================================================================
# DWLS Deconvolution Benchmark on Real BRCA Pseudobulk
#
# DWLS (Dampened Weighted Least Squares), Tsoucas et al. 2019, Nat Commun.
# Uses a single-cell reference to derive a cell-type signature matrix via
# MAST differential expression, then solves damped WLS per bulk sample.
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

# ---- Install DWLS if needed ----
# DWLS is on CRAN (package 'DWLS', Sistig 2024). Fallback to GitHub source.
if (!requireNamespace("DWLS", quietly = TRUE)) {
  cat("Installing DWLS from CRAN...\n")
  install.packages("DWLS", repos = "https://cloud.r-project.org/")
}
if (!requireNamespace("DWLS", quietly = TRUE)) {
  cat("CRAN install failed, falling back to GitHub source (dtsoucas/DWLS)...\n")
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/")
  }
  remotes::install_github("dtsoucas/DWLS", upgrade = "never")
}

# MAST is required by buildSignatureMatrixMAST
if (!requireNamespace("MAST", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org/")
  }
  BiocManager::install("MAST", ask = FALSE, update = FALSE)
}

library(DWLS)
library(MAST)

cat("DWLS version:", as.character(packageVersion("DWLS")), "\n")
cat("MAST version:", as.character(packageVersion("MAST")), "\n\n")

output_dir <- "docs/outputs"
sig_cache_dir <- file.path(output_dir, "t8_dwls_sigmat")
sig_rds <- file.path(output_dir, "t8_dwls_signature.rds")
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
cat(sprintf("   Cell types: %s\n", paste(sort(unique(sc_meta$cell_type)), collapse = ", ")))

# DWLS builds variable/file names by pasting cell-type strings into
# eval(parse(text=...)), so it breaks on hyphens, spaces, and other
# non-syntactic characters. Sanitize here, track the mapping, and
# reverse-map before writing output.
sanitize_ct <- function(x) gsub("[^A-Za-z0-9]", "_", x)
ct_original <- sort(unique(sc_meta$cell_type))
ct_clean    <- sanitize_ct(ct_original)
names(ct_original) <- ct_clean            # clean -> original
ct_map <- setNames(ct_clean, ct_original) # original -> clean
if (any(duplicated(ct_clean))) {
  stop("Sanitized cell-type names collide: ",
       paste(ct_clean[duplicated(ct_clean)], collapse = ", "))
}
sc_meta$cell_type_clean <- ct_map[sc_meta$cell_type]
cat(sprintf("   Sanitized types: %s\n", paste(ct_clean, collapse = ", ")))

# ---- Step 2: Build (or load cached) signature matrix ----
cat("\n2. Building DWLS signature matrix via MAST DE...\n")
if (file.exists(sig_rds)) {
  cat("   Loading cached signature from", sig_rds, "\n")
  Signature <- readRDS(sig_rds)
} else {
  cat("   No cache found — running buildSignatureMatrixMAST (slow; per-cell-type DE)\n")
  t0 <- Sys.time()
  Signature <- DWLS::buildSignatureMatrixMAST(
    scdata     = sc_counts,
    id         = sc_meta$cell_type_clean,
    path       = sig_cache_dir,
    diff.cutoff = 0.5,
    pval.cutoff = 0.01
  )
  dt <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("   Signature built in %.1f min\n", as.numeric(dt)))
  saveRDS(Signature, sig_rds)
  cat("   Cached to", sig_rds, "\n")
}
cat(sprintf("   Signature matrix: %d genes x %d cell types\n",
            nrow(Signature), ncol(Signature)))
cat(sprintf("   Cell types in signature: %s\n",
            paste(colnames(Signature), collapse = ", ")))

# ---- Step 3: Deconvolve each scenario ----
scenarios <- c("uniform", "sparse", "tumor_purity", "titration")

for (scenario in scenarios) {
  bulk_file <- file.path(output_dir, paste0("t8_real_bulk_", scenario, ".csv"))
  out_file  <- file.path(output_dir, paste0("t8_dwls_", scenario, ".csv"))

  if (!file.exists(bulk_file)) {
    cat(sprintf("\n   Skipping %s — file not found: %s\n", scenario, bulk_file))
    next
  }
  cat(sprintf("\n3. DWLS — %s\n", scenario))

  bulk_mtx <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
  # CSV is samples x genes; DWLS expects a per-sample gene vector (named)
  bulk_mtx <- t(bulk_mtx)  # now: genes x samples
  cat(sprintf("   Bulk: %d genes x %d samples\n", nrow(bulk_mtx), ncol(bulk_mtx)))

  ct_names <- colnames(Signature)
  props <- matrix(
    NA_real_,
    nrow = ncol(bulk_mtx), ncol = length(ct_names),
    dimnames = list(colnames(bulk_mtx), ct_names)
  )

  t0 <- Sys.time()
  for (i in seq_len(ncol(bulk_mtx))) {
    bulk_vec <- bulk_mtx[, i]
    # trimData intersects signature and bulk on genes
    tr <- tryCatch(
      DWLS::trimData(Signature, bulk_vec),
      error = function(e) { cat("   trimData error:", e$message, "\n"); NULL }
    )
    if (is.null(tr)) next
    est <- tryCatch(
      DWLS::solveDampenedWLS(tr$sig, tr$bulk),
      error = function(e) { cat("   solveDampenedWLS error:", e$message, "\n"); NULL }
    )
    if (is.null(est)) next
    # est is a named numeric over the signature's cell types
    est[est < 0] <- 0
    if (sum(est) > 0) est <- est / sum(est)
    props[i, names(est)] <- est

    if (i %% 25 == 0 || i == ncol(bulk_mtx)) {
      cat(sprintf("     %d / %d samples\n", i, ncol(bulk_mtx)))
    }
  }
  dt <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("   Done in %.1f min\n", as.numeric(dt)))

  # Restore original (un-sanitized) cell type names before writing
  colnames(props) <- ct_original[colnames(props)]
  write.csv(props, out_file)
  cat(sprintf("   Saved: %s\n", out_file))
}

cat(sprintf("\nDone: %s\n", format(Sys.time())))
