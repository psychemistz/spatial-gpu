#!/usr/bin/env Rscript
# Extract LIHC_CosMx_data.rda to CSV/MTX for Python.
#
# Usage: Rscript scripts/extract_cosmx_data.R

library(Matrix)
library(jsonlite)

cat("=== Extracting LIHC_CosMx_data ===\n")

load("data/LIHC_CosMx_data.rda")

out_dir <- "data/LIHC_CosMx"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# List all objects loaded
loaded_objs <- ls()
cat("Loaded objects:", paste(loaded_objs, collapse = ", "), "\n")

# Counts
if (exists("counts")) {
  cat("  counts:", nrow(counts), "genes x", ncol(counts), "cells\n")
  if (inherits(counts, "dgCMatrix") || inherits(counts, "dgTMatrix")) {
    writeMM(counts, file.path(out_dir, "counts.mtx"))
  } else {
    writeMM(as(as.matrix(counts), "dgCMatrix"), file.path(out_dir, "counts.mtx"))
  }
  write.csv(data.frame(gene = rownames(counts)), file.path(out_dir, "genes.csv"), row.names = FALSE)
  write.csv(data.frame(cell = colnames(counts)), file.path(out_dir, "cells.csv"), row.names = FALSE)
}

# Coordinates
if (exists("spotCoordinates")) {
  cat("  spotCoordinates:", nrow(spotCoordinates), "cells\n")
  write.csv(spotCoordinates, file.path(out_dir, "coordinates.csv"), row.names = TRUE)
}

# Metadata
if (exists("metaData")) {
  cat("  metaData:", nrow(metaData), "cells,", ncol(metaData), "cols\n")
  cat("  metaData columns:", paste(colnames(metaData), collapse = ", "), "\n")
  write.csv(metaData, file.path(out_dir, "metadata.csv"), row.names = TRUE)
}

cat("=== Done ===\n")
