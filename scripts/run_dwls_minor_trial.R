#!/usr/bin/env Rscript
# =============================================================================
# DWLS (minor-resolution) benchmark for one trial.
#
# Usage: Rscript scripts/run_dwls_minor_trial.R <trial>
# Reads docs/outputs/trials/T{NN}/{sc_counts,sc_meta_minor,bulk_*}.csv
# Writes docs/outputs/trials/T{NN}/dwls_minor_<scenario>.csv
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript run_dwls_minor_trial.R <trial>")
trial <- as.integer(args[1])
trial_dir <- sprintf("docs/outputs/trials/T%02d", trial)

cat(sprintf("=== DWLS (minor) trial T%02d ===\n", trial))
cat(sprintf("Start: %s  trial_dir: %s\n", format(Sys.time()), trial_dir))

# Diagnostic: print the env var that BiocParallel.check_ncores reads.
cat(sprintf("[diag] _R_CHECK_LIMIT_CORES_ before unset = '%s' (nzchar=%s)\n",
            Sys.getenv("_R_CHECK_LIMIT_CORES_"),
            as.character(nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")))))
Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
cat(sprintf("[diag] _R_CHECK_LIMIT_CORES_ after  unset = '%s' (nzchar=%s)\n",
            Sys.getenv("_R_CHECK_LIMIT_CORES_"),
            as.character(nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_")))))

suppressPackageStartupMessages(library(BiocParallel))
register(SerialParam())
options(mc.cores = 1)
cat(sprintf("[diag] bpparam() = %s\n", class(bpparam())[1]))

source("scripts/_dwls_common.R")
ensure_dwls_packages()

suppressPackageStartupMessages({
  library(DWLS)
  library(MAST)
})

sc_counts <- as.matrix(read.csv(file.path(trial_dir, "sc_counts.csv"), row.names = 1, check.names = FALSE))
sc_meta   <- read.csv(file.path(trial_dir, "sc_meta_minor.csv"), row.names = 1, stringsAsFactors = FALSE)

# Keep only cells present in minor meta (drops minors with <30 cells)
keep_cells <- intersect(colnames(sc_counts), rownames(sc_meta))
sc_counts <- sc_counts[, keep_cells]
sc_meta <- sc_meta[keep_cells, , drop = FALSE]
stopifnot(identical(colnames(sc_counts), rownames(sc_meta)))

# Pre-filter near-zero genes (halves MAST memory)
nz <- rowSums(sc_counts > 0) >= 5
sc_counts <- sc_counts[nz, , drop = FALSE]

cat(sprintf("  Reference: %d genes x %d cells\n", nrow(sc_counts), ncol(sc_counts)))
cat(sprintf("  Minor types: %d\n", length(unique(sc_meta$celltype_minor))))

ct <- make_ct_maps(sc_meta$celltype_minor)
sc_meta$ct_clean <- ct$to_clean[sc_meta$celltype_minor]

sig_cache_dir <- file.path(trial_dir, "dwls_minor_sigmat")
sig_rds <- file.path(trial_dir, "dwls_minor_signature.rds")
dir.create(sig_cache_dir, showWarnings = FALSE, recursive = TRUE)

Signature <- build_or_load_signature(sc_counts, sc_meta$ct_clean, sig_rds, sig_cache_dir)
cat(sprintf("  Signature: %d genes x %d minor types\n", nrow(Signature), ncol(Signature)))

deconvolve_scenarios(
  Signature, ct$to_original,
  scenarios   = c("uniform", "sparse", "tumor_purity", "titration"),
  output_dir  = trial_dir,
  bulk_prefix = "bulk_",
  out_prefix  = "dwls_minor_",
  label       = "DWLS (minor)"
)

cat(sprintf("\nDone: %s\n", format(Sys.time())))
