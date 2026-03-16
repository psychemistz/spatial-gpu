#!/usr/bin/env Rscript
# Extract all SpaCET tutorial datasets to CSV/MTX for Python consumption.
#
# Usage: Rscript scripts/extract_spacet_data.R
#
# Output structure:
#   data/oldST_PDAC/
#     st_counts.mtx, st_genes.csv, st_spots.csv, st_coordinates.csv
#     sc_counts.mtx, sc_genes.csv, sc_cells.csv, sc_annotation.csv
#     sc_lineage_tree.json
#     colors_vector.json
#   data/hiresST_CRC/
#     counts.mtx, genes.csv, spots.csv, coordinates.csv
#     colors_vector.json

library(Matrix)
library(jsonlite)

spacet_path <- system.file("extdata", package = "SpaCET")

# ============================================================================
# 1. oldST_PDAC — ST data + matched scRNA-seq
# ============================================================================
cat("=== Extracting oldST_PDAC ===\n")
out_dir <- "data/oldST_PDAC"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

oldST_Path <- file.path(spacet_path, "oldST_PDAC")

# --- ST data ---
load(file.path(oldST_Path, "st_PDAC.rda"))

cat("  ST counts:", nrow(counts), "genes x", ncol(counts), "spots\n")
cat("  ST coordinates:", nrow(spotCoordinates), "spots\n")

# Save counts as MatrixMarket (genes x spots)
if (inherits(counts, "dgCMatrix") || inherits(counts, "dgTMatrix")) {
  writeMM(counts, file.path(out_dir, "st_counts.mtx"))
} else {
  writeMM(as(as.matrix(counts), "dgCMatrix"), file.path(out_dir, "st_counts.mtx"))
}
write.csv(data.frame(gene = rownames(counts)), file.path(out_dir, "st_genes.csv"), row.names = FALSE)
write.csv(data.frame(spot = colnames(counts)), file.path(out_dir, "st_spots.csv"), row.names = FALSE)
write.csv(spotCoordinates, file.path(out_dir, "st_coordinates.csv"), row.names = TRUE)

# --- scRNA-seq data ---
load(file.path(oldST_Path, "sc_PDAC.rda"))

cat("  SC counts:", nrow(sc_counts), "genes x", ncol(sc_counts), "cells\n")
cat("  SC annotation:", nrow(sc_annotation), "cells\n")

if (inherits(sc_counts, "dgCMatrix") || inherits(sc_counts, "dgTMatrix")) {
  writeMM(sc_counts, file.path(out_dir, "sc_counts.mtx"))
} else {
  writeMM(as(as.matrix(sc_counts), "dgCMatrix"), file.path(out_dir, "sc_counts.mtx"))
}
write.csv(data.frame(gene = rownames(sc_counts)), file.path(out_dir, "sc_genes.csv"), row.names = FALSE)
write.csv(data.frame(cell = colnames(sc_counts)), file.path(out_dir, "sc_cells.csv"), row.names = FALSE)
write.csv(as.data.frame(sc_annotation), file.path(out_dir, "sc_annotation.csv"), row.names = TRUE)

# Lineage tree as JSON
write_json(sc_lineageTree, file.path(out_dir, "sc_lineage_tree.json"), auto_unbox = TRUE, pretty = TRUE)

# --- Colors ---
load(file.path(oldST_Path, "colors_vector.rda"))
write_json(as.list(colors_vector), file.path(out_dir, "colors_vector.json"), auto_unbox = TRUE, pretty = TRUE)

cat("  Done: oldST_PDAC\n\n")

# ============================================================================
# 2. hiresST_CRC — Slide-seq CRC data
# ============================================================================
cat("=== Extracting hiresST_CRC ===\n")
out_dir <- "data/hiresST_CRC"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

hiresST_Path <- file.path(spacet_path, "hiresST_CRC")

# --- Counts ---
load(file.path(hiresST_Path, "counts.rda"))
cat("  Counts:", nrow(counts), "genes x", ncol(counts), "spots\n")

if (inherits(counts, "dgCMatrix") || inherits(counts, "dgTMatrix")) {
  writeMM(counts, file.path(out_dir, "counts.mtx"))
} else {
  writeMM(as(as.matrix(counts), "dgCMatrix"), file.path(out_dir, "counts.mtx"))
}
write.csv(data.frame(gene = rownames(counts)), file.path(out_dir, "genes.csv"), row.names = FALSE)
write.csv(data.frame(spot = colnames(counts)), file.path(out_dir, "spots.csv"), row.names = FALSE)

# --- Coordinates ---
load(file.path(hiresST_Path, "spotCoordinates.rda"))
cat("  Coordinates:", nrow(spotCoordinates), "spots\n")
write.csv(spotCoordinates, file.path(out_dir, "coordinates.csv"), row.names = TRUE)

# --- Colors ---
load(file.path(hiresST_Path, "colors_vector.rda"))
write_json(as.list(colors_vector), file.path(out_dir, "colors_vector.json"), auto_unbox = TRUE, pretty = TRUE)

cat("  Done: hiresST_CRC\n\n")

cat("=== All extractions complete ===\n")
