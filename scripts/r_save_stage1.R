#!/usr/bin/env Rscript
# Save R's Stage 1 intermediate results for all 3 VST datasets.
# Saves: malProp, malRef for each dataset.

library(SpaCET)
library(Matrix)

vst_dir <- "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
out_dir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/validation"

for (i in 1:3) {
  cat(sprintf("\n========== Dataset %d ==========\n", i))

  counts_file <- file.path(vst_dir, paste0(i, "_counts.tsv"))
  counts <- read.table(counts_file, header = TRUE, row.names = 1, sep = "\t",
                        check.names = FALSE)

  spot_ids <- colnames(counts)
  parts <- strsplit(spot_ids, "x")
  array_row <- as.numeric(sapply(parts, "[", 1))
  array_col <- as.numeric(sapply(parts, "[", 2))

  coord_x_um <- array_col * 0.5 * 100
  coord_y_um <- max(array_row * 0.5 * sqrt(3) * 100) - (array_row * 0.5 * sqrt(3) * 100)

  spotCoordinates <- data.frame(
    pixel_row = array_row, pixel_col = array_col,
    array_row = array_row, array_col = array_col,
    coordinate_x_um = coord_x_um, coordinate_y_um = coord_y_um,
    row.names = spot_ids
  )

  counts_sparse <- as(as.matrix(counts), "dgCMatrix")
  obj <- new("SpaCET",
    input = list(counts = counts_sparse, spotCoordinates = spotCoordinates,
                 platform = "Visium", image = list(path = NA)),
    results = list()
  )

  obj <- SpaCET.quality.control(obj, min.genes = 1)
  obj <- SpaCET.deconvolution(obj, cancerType = "BRCA", coreNo = 1)

  # Save malProp
  malProp <- obj@results$deconvolution$malRes$malProp
  write.csv(data.frame(spot = names(malProp), malProp = as.numeric(malProp)),
            file.path(out_dir, sprintf("vst%d_malProp_full.csv", i)), row.names = FALSE)

  # Save malRef (gene-level)
  malRef <- obj@results$deconvolution$malRes$malRef
  write.csv(data.frame(gene = names(malRef), malRef = as.numeric(malRef)),
            file.path(out_dir, sprintf("vst%d_malRef.csv", i)), row.names = FALSE)

  cat(sprintf("  Saved malProp (%d spots) and malRef (%d genes)\n",
              length(malProp), length(malRef)))
}

cat("\n========== DONE ==========\n")
