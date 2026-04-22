# =============================================================================
# Shared helpers for DWLS benchmark scripts.
# Sourced by run_dwls_benchmark.R (major types) and run_dwls_benchmark_minor.R.
# =============================================================================

ensure_dwls_packages <- function() {
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
  if (!requireNamespace("MAST", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) {
      install.packages("BiocManager", repos = "https://cloud.r-project.org/")
    }
    BiocManager::install("MAST", ask = FALSE, update = FALSE)
  }
}

# DWLS uses eval(parse(text=cell_type)) internally, so non-syntactic
# characters (spaces, hyphens) break it. Sanitize and reverse-map at write.
sanitize_ct <- function(x) gsub("[^A-Za-z0-9]", "_", x)

make_ct_maps <- function(cell_types) {
  original <- sort(unique(cell_types))
  clean <- sanitize_ct(original)
  if (any(duplicated(clean))) {
    stop("Sanitized cell-type names collide: ",
         paste(clean[duplicated(clean)], collapse = ", "))
  }
  list(
    clean       = clean,
    original    = original,
    to_clean    = setNames(clean, original),
    to_original = setNames(original, clean)
  )
}

build_or_load_signature <- function(sc_counts, ids, sig_rds, sig_cache_dir) {
  if (file.exists(sig_rds)) {
    cat("   Loading cached signature from", sig_rds, "\n")
    return(readRDS(sig_rds))
  }
  t0 <- Sys.time()
  Signature <- DWLS::buildSignatureMatrixMAST(
    scdata      = sc_counts,
    id          = ids,
    path        = sig_cache_dir,
    diff.cutoff = 0.5,
    pval.cutoff = 0.01
  )
  dt <- difftime(Sys.time(), t0, units = "mins")
  cat(sprintf("   Signature built in %.1f min\n", as.numeric(dt)))
  saveRDS(Signature, sig_rds)
  cat("   Cached to", sig_rds, "\n")
  Signature
}

deconvolve_scenarios <- function(Signature, to_original, scenarios, output_dir,
                                 bulk_prefix = "t8_real_bulk_",
                                 out_prefix  = "t8_dwls_",
                                 label       = "DWLS") {
  for (scenario in scenarios) {
    bulk_file <- file.path(output_dir, paste0(bulk_prefix, scenario, ".csv"))
    out_file  <- file.path(output_dir, paste0(out_prefix, scenario, ".csv"))
    if (!file.exists(bulk_file)) {
      cat(sprintf("\n   Skipping %s — missing: %s\n", scenario, bulk_file))
      next
    }
    if (file.exists(out_file)) {
      cat(sprintf("\n   Skipping %s — already done: %s\n", scenario, out_file))
      next
    }
    cat(sprintf("\n3. %s — %s\n", label, scenario))

    bulk_mtx <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
    bulk_mtx <- t(bulk_mtx)
    cat(sprintf("   Bulk: %d genes x %d samples\n", nrow(bulk_mtx), ncol(bulk_mtx)))

    # Gene intersection is invariant across samples, so hoist it out of the
    # per-sample loop. DWLS::trimData does exactly this (verified in source).
    common_genes <- intersect(rownames(Signature), rownames(bulk_mtx))
    sig_trim <- Signature[common_genes, , drop = FALSE]
    bulk_trim <- bulk_mtx[common_genes, , drop = FALSE]

    ct_names <- colnames(Signature)
    props <- matrix(
      NA_real_,
      nrow = ncol(bulk_trim), ncol = length(ct_names),
      dimnames = list(colnames(bulk_trim), ct_names)
    )

    t0 <- Sys.time()
    for (i in seq_len(ncol(bulk_trim))) {
      est <- tryCatch(
        DWLS::solveDampenedWLS(sig_trim, bulk_trim[, i]),
        error = function(e) { cat("   solveDampenedWLS error:", e$message, "\n"); NULL }
      )
      if (is.null(est)) next
      est[est < 0] <- 0
      if (sum(est) > 0) est <- est / sum(est)
      props[i, names(est)] <- est

      if (i %% 25 == 0 || i == ncol(bulk_trim)) {
        cat(sprintf("     %d / %d samples\n", i, ncol(bulk_trim)))
      }
    }
    dt <- difftime(Sys.time(), t0, units = "mins")
    cat(sprintf("   Done in %.1f min\n", as.numeric(dt)))

    colnames(props) <- to_original[colnames(props)]
    write.csv(props, out_file)
    cat(sprintf("   Saved: %s\n", out_file))
  }
}
