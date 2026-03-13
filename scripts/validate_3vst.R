#!/usr/bin/env Rscript
# Run SpaCET deconvolution on 3 Visium VST datasets and save results
# for Python numerical equivalence validation.

library(SpaCET)
library(Matrix)

vst_dir <- "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
out_dir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/validation"

for (i in 1:3) {
  cat(sprintf("\n========== Dataset %d ==========\n", i))

  counts_file <- file.path(vst_dir, paste0(i, "_counts.tsv"))

  # Read counts (genes x spots, tab-separated)
  counts <- read.table(counts_file, header = TRUE, row.names = 1, sep = "\t",
                        check.names = FALSE)
  cat(sprintf("  Loaded counts: %d genes x %d spots\n", nrow(counts), ncol(counts)))

  # Build spot coordinates from spot IDs (format: "rowxcol")
  spot_ids <- colnames(counts)
  parts <- strsplit(spot_ids, "x")
  array_row <- as.numeric(sapply(parts, "[", 1))
  array_col <- as.numeric(sapply(parts, "[", 2))

  # Compute pixel and micrometer coordinates (matching SpaCET convention)
  pixel_row <- array_row  # simplified; just need consistent coords
  pixel_col <- array_col
  coord_x_um <- array_col * 0.5 * 100
  coord_y_um <- max(array_row * 0.5 * sqrt(3) * 100) - (array_row * 0.5 * sqrt(3) * 100)

  spotCoordinates <- data.frame(
    pixel_row = pixel_row,
    pixel_col = pixel_col,
    array_row = array_row,
    array_col = array_col,
    coordinate_x_um = coord_x_um,
    coordinate_y_um = coord_y_um,
    row.names = spot_ids
  )

  # Create SpaCET object
  counts_sparse <- as(as.matrix(counts), "dgCMatrix")

  obj <- new("SpaCET",
    input = list(
      counts = counts_sparse,
      spotCoordinates = spotCoordinates,
      platform = "Visium",
      image = list(path = NA)
    ),
    results = list()
  )

  # Quality control
  obj <- SpaCET.quality.control(obj, min.genes = 1)
  cat(sprintf("  After QC: %d spots\n", ncol(obj@input$counts)))

  # Deconvolution
  cat("  Running deconvolution (cancer_type=BRCA)...\n")
  obj <- SpaCET.deconvolution(obj, cancerType = "BRCA", coreNo = 1)

  # Extract results
  propMat <- obj@results$deconvolution$propMat
  cat(sprintf("  propMat: %d cell types x %d spots\n", nrow(propMat), ncol(propMat)))
  cat(sprintf("  Cell types: %s\n", paste(rownames(propMat), collapse = ", ")))

  # Identify major vs minor lineages
  lineageTree <- obj@results$deconvolution$Ref$lineageTree
  major_types <- names(lineageTree)
  minor_types <- unlist(lineageTree)

  # Add Malignant if present
  if ("Malignant" %in% rownames(propMat) && !"Malignant" %in% major_types) {
    major_types <- c("Malignant", major_types)
  }

  # Major lineage propMat
  major_types_present <- major_types[major_types %in% rownames(propMat)]
  propMat_major <- propMat[major_types_present, , drop = FALSE]

  # Minor lineage propMat
  minor_types_present <- minor_types[minor_types %in% rownames(propMat)]
  propMat_minor <- propMat[minor_types_present, , drop = FALSE]

  cat(sprintf("  Major lineages (%d): %s\n", length(major_types_present),
              paste(major_types_present, collapse = ", ")))
  cat(sprintf("  Minor lineages (%d): %s\n", length(minor_types_present),
              paste(minor_types_present, collapse = ", ")))

  # Save
  prefix <- file.path(out_dir, paste0("vst", i))
  write.csv(propMat, paste0(prefix, "_propMat.csv"))
  write.csv(propMat_major, paste0(prefix, "_propMat_major.csv"))
  write.csv(propMat_minor, paste0(prefix, "_propMat_minor.csv"))

  # Save malProp
  malProp <- obj@results$deconvolution$malRes$malProp
  write.csv(malProp, paste0(prefix, "_malProp.csv"))

  # Save spot coordinates
  write.csv(spotCoordinates, paste0(prefix, "_spotCoordinates.csv"))

  cat(sprintf("  Saved to %s_*.csv\n", prefix))
}

cat("\n========== DONE ==========\n")
