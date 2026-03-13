#!/usr/bin/env Rscript
# Save intermediate deconvolution values from R SpaCET for diagnostic comparison.
# Replicates SpatialDeconv step-by-step and saves A, B, theta, propMat at each stage.

library(SpaCET)
library(Matrix)

vst_dir <- "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
out_dir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/validation/intermediates"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

i <- 1  # VST1 dataset

cat("========== Loading VST1 ==========\n")
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
  pixel_row = array_row,
  pixel_col = array_col,
  array_row = array_row,
  array_col = array_col,
  coordinate_x_um = coord_x_um,
  coordinate_y_um = coord_y_um,
  row.names = spot_ids
)

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

obj <- SpaCET.quality.control(obj, min.genes = 1)
cat(sprintf("After QC: %d genes x %d spots\n", nrow(obj@input$counts), ncol(obj@input$counts)))

# Run full deconvolution
obj <- SpaCET.deconvolution(obj, cancerType = "BRCA", coreNo = 1)

# Now replicate SpatialDeconv step by step to save intermediates
ST <- obj@input$counts

# Filter zero-sum genes
keepGene <- rowSums(ST) > 0
ST <- ST[keepGene, ]
geneNames <- rownames(ST)
spotNames <- colnames(ST)

# Load reference
data_dir <- system.file("extdata", package = "SpaCET")
load(file.path(data_dir, "combRef_0.5.rda"))
Reference <- Ref$refProfiles
Signature <- Ref$sigGenes
Tree <- Ref$lineageTree

cat(sprintf("Reference: %d genes x %d cell types\n", nrow(Reference), ncol(Reference)))
cat(sprintf("Tree keys: %s\n", paste(names(Tree), collapse=", ")))

# Intersect genes
olp <- intersect(geneNames, rownames(Reference))
cat(sprintf("Overlapping genes: %d\n", length(olp)))

# Save overlapping genes
write.csv(data.frame(gene = olp), file.path(out_dir, "olp_genes.csv"), row.names = FALSE)

# Get gene indices for ST
gene_idx_st <- match(olp, geneNames)
ST_sub <- ST[olp, ]
Reference_sub <- Reference[olp, ]

# CPM normalize ST
ST_colSums <- colSums(ST_sub)
ST_cpm <- as.matrix(ST_sub)
for (j in 1:ncol(ST_cpm)) {
  ST_cpm[, j] <- ST_cpm[, j] / ST_colSums[j] * 1e6
}

# CPM normalize Reference
Ref_colSums <- colSums(Reference_sub)
Ref_cpm <- as.matrix(Reference_sub)
for (j in 1:ncol(Ref_cpm)) {
  Ref_cpm[, j] <- Ref_cpm[, j] / Ref_colSums[j] * 1e6
}

# Save CPM matrices
write.csv(ST_cpm[1:20, 1:10], file.path(out_dir, "ST_cpm_sample.csv"))
write.csv(Ref_cpm[1:20, ], file.path(out_dir, "Ref_cpm_sample.csv"))

# Get malProp and malRef from results
malProp <- obj@results$deconvolution$malRes$malProp
malRef <- obj@results$deconvolution$malRes$malRef

# Save malProp
write.csv(data.frame(spot = names(malProp), malProp = as.numeric(malProp)),
          file.path(out_dir, "malProp.csv"), row.names = FALSE)

# malRef CPM
malRef_olp <- malRef[olp]
malRef_CPM <- malRef_olp * 1e6 / sum(malRef_olp)

# Subtract malignant contribution
mixture_minus_mal <- ST_cpm - outer(as.numeric(malRef_CPM), as.numeric(malProp))
write.csv(mixture_minus_mal[1:20, 1:10], file.path(out_dir, "mixture_minus_mal_sample.csv"))

# Level 1 setup
cellName <- names(Tree)
nCell <- length(cellName)
cat(sprintf("Level 1 cell types (%d): %s\n", nCell, paste(cellName, collapse=", ")))

# Signature genes for Level 1
sigName <- c(names(Tree), "T cell")
sigName_valid <- sigName[sigName %in% names(Signature)]
sigGenes_L1 <- unique(unlist(Signature[sigName_valid]))
sigGenes_L1 <- sigGenes_L1[sigGenes_L1 %in% olp]
cat(sprintf("Level 1 signature genes: %d\n", length(sigGenes_L1)))

# Save signature genes
write.csv(data.frame(gene = sigGenes_L1), file.path(out_dir, "sigGenes_L1.csv"), row.names = FALSE)

# Level 1 A and B matrices
A_L1 <- Ref_cpm[sigGenes_L1, cellName]
B_L1 <- mixture_minus_mal[sigGenes_L1, ]

# Save A matrix (reference for L1) - full
write.csv(A_L1, file.path(out_dir, "A_L1.csv"))

# Save B matrix (mixture for L1) - sample of first 20 spots
write.csv(B_L1[, 1:min(20, ncol(B_L1))], file.path(out_dir, "B_L1_sample.csv"))

# Level 1 constraints
cat(sprintf("malProp range: [%f, %f]\n", min(malProp), max(malProp)))

# Run optimization for first 5 spots and save details
cat("Running Level 1 optimization for first 5 spots...\n")

f0 <- function(theta, A, b) {
  sum((A %*% theta - b)^2)
}

# Find 5 spots with reasonable malProp (not too close to 0 or 1) for diagnostics
good_spots <- which(malProp > 0.1 & malProp < 0.9)
if (length(good_spots) < 5) {
  good_spots <- which(malProp > 0.01 & malProp < 0.99)
}
diag_spots <- good_spots[1:min(5, length(good_spots))]
cat(sprintf("Diagnostic spots: %s\n", paste(diag_spots, collapse=", ")))

for (spot_i in diag_spots) {
  thetaSum <- (1 - malProp[spot_i]) - 1e-5
  theta0 <- rep(thetaSum / nCell, nCell)

  ppmin <- 0  # Unidentifiable = TRUE
  ppmax <- 1 - malProp[spot_i]

  ui <- rbind(diag(nCell), rep(1, nCell), rep(-1, nCell))
  ci <- c(rep(0, nCell), ppmin, -ppmax)

  b_spot <- B_L1[, spot_i]

  tryCatch({
    # Pass 1
    res1 <- constrOptim(theta0, f0, NULL, ui, ci, A = A_L1, b = b_spot)

    # Pass 2
    bhat <- A_L1 %*% res1$par
    f_weighted <- function(theta, A, b) {
      sum((A %*% theta - b)^2 / (bhat + 1))
    }
    res2 <- constrOptim(theta0, f_weighted, NULL, ui, ci, A = A_L1, b = b_spot)

    result <- data.frame(
      cellType = cellName,
      theta0 = theta0,
      pass1 = as.numeric(res1$par),
      pass2 = as.numeric(res2$par),
      row.names = cellName
    )

    cat(sprintf("  Spot %d (%s): thetaSum=%.6f, ppmin=%.6f, ppmax=%.6f, malProp=%.6f\n",
                spot_i, spotNames[spot_i], thetaSum, ppmin, ppmax, malProp[spot_i]))
    cat(sprintf("    Pass1 obj: %.6e, Pass2 obj: %.6e\n", res1$value, res2$value))
    cat(sprintf("    Pass2 result: %s\n", paste(sprintf("%.6f", res2$par), collapse=", ")))

    write.csv(result, file.path(out_dir, sprintf("spot%d_L1_result.csv", spot_i)))

    # Also save b_spot
    write.csv(data.frame(gene = sigGenes_L1, value = b_spot),
              file.path(out_dir, sprintf("spot%d_b_L1.csv", spot_i)), row.names = FALSE)
  }, error = function(e) {
    cat(sprintf("  Spot %d: ERROR: %s\n", spot_i, e$message))
  })
}

# Save the FULL Level 1 propMat from R results
propMat <- obj@results$deconvolution$propMat
propMat_L1 <- propMat[cellName, ]
if ("Malignant" %in% rownames(propMat)) {
  propMat_L1 <- rbind(propMat["Malignant", ], propMat_L1)
}
write.csv(propMat_L1, file.path(out_dir, "propMat_L1_R.csv"))

# Level 2: save for one lineage (e.g., "B cell")
for (cellSpe in names(Tree)) {
  subtypes <- Tree[[cellSpe]]
  if (length(subtypes) < 2) next

  subtypes_no_other <- subtypes[subtypes != "Macrophage other"]
  subtypes_in_ref <- subtypes_no_other[subtypes_no_other %in% colnames(Reference_sub)]
  if (length(subtypes_in_ref) == 0) next

  # Signature genes for this lineage
  sigGenes_L2_list <- Signature[subtypes_in_ref[subtypes_in_ref %in% names(Signature)]]
  sigGenes_L2 <- unique(unlist(sigGenes_L2_list))
  sigGenes_L2 <- sigGenes_L2[sigGenes_L2 %in% olp]

  if (length(sigGenes_L2) == 0) next

  cat(sprintf("Level 2 - %s: %d subtypes, %d sig genes\n",
              cellSpe, length(subtypes_in_ref), length(sigGenes_L2)))

  # Save signature genes for this lineage
  write.csv(data.frame(gene = sigGenes_L2),
            file.path(out_dir, sprintf("sigGenes_L2_%s.csv", gsub(" ", "_", cellSpe))),
            row.names = FALSE)
}

# Save Level 1 cell type names in order
write.csv(data.frame(cellType = cellName, idx = 1:nCell),
          file.path(out_dir, "L1_cellTypes.csv"), row.names = FALSE)

# Save Tree structure
for (k in names(Tree)) {
  subtypes <- Tree[[k]]
  write.csv(data.frame(subtype = subtypes),
            file.path(out_dir, sprintf("tree_%s.csv", gsub(" ", "_", k))),
            row.names = FALSE)
}

cat("\n========== DONE ==========\n")
cat(sprintf("Intermediate files saved to: %s\n", out_dir))
