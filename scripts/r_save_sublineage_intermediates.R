#!/usr/bin/env Rscript
# Run R SpaCET deconvolution and save intermediates for sublineage concordance test.
# Saves: full propMat, malProp, malRef, counts, lineageTree, refProfiles

library(SpaCET)
library(Matrix)

out_dir <- "/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/sublineage_concordance"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

visium_path <- file.path(system.file("extdata", package = "SpaCET"), "Visium_BC")
cat(sprintf("Using Visium_BC data from: %s\n", visium_path))

# Create SpaCET object from 10X Visium
obj <- create.SpaCET.object.10X(visiumPath = visium_path)
obj <- SpaCET.quality.control(obj, min.genes = 1)
cat(sprintf("After QC: %d genes x %d spots\n", nrow(obj@input$counts), ncol(obj@input$counts)))

# Run full deconvolution
cat("Running SpaCET.deconvolution (BRCA)...\n")
obj <- SpaCET.deconvolution(obj, cancerType = "BRCA", coreNo = 4)

# --- Save outputs ---

# Full propMat (cell_types x spots)
propMat <- obj@results$deconvolution$propMat
write.csv(as.data.frame(propMat), file.path(out_dir, "r_propMat_full.csv"))
cat(sprintf("Full propMat: %d cell types x %d spots\n", nrow(propMat), ncol(propMat)))
cat(sprintf("Cell types: %s\n", paste(rownames(propMat), collapse=", ")))

# Malignant proportion — extract from propMat (most reliable)
malProp <- propMat["Malignant",]
write.csv(data.frame(spot = names(malProp), malProp = as.numeric(malProp)),
          file.path(out_dir, "r_malProp.csv"), row.names = FALSE)
cat(sprintf("malProp range: [%f, %f]\n", min(malProp), max(malProp)))

# Malignant reference — from malRes
malRes <- obj@results$deconvolution$malRes
if (!is.null(malRes$malRef)) {
  malRef <- malRes$malRef
  if (is.matrix(malRef) || is.data.frame(malRef)) {
    write.csv(malRef, file.path(out_dir, "r_malRef.csv"))
  } else {
    write.csv(data.frame(gene = names(malRef), value = as.numeric(malRef)),
              file.path(out_dir, "r_malRef.csv"), row.names = FALSE)
  }
  cat(sprintf("malRef: %d genes\n", length(malRef)))
} else {
  cat("malRef is NULL\n")
}

# Counts matrix (sparse, save as triplet format for Python)
counts <- obj@input$counts
counts_dgT <- as(counts, "dgTMatrix")
write.csv(
  data.frame(row = counts_dgT@i, col = counts_dgT@j, val = counts_dgT@x),
  file.path(out_dir, "r_counts_triplet.csv"), row.names = FALSE
)
write.csv(data.frame(gene = rownames(counts)), file.path(out_dir, "r_gene_names.csv"), row.names = FALSE)
write.csv(data.frame(spot = colnames(counts)), file.path(out_dir, "r_spot_names.csv"), row.names = FALSE)
cat(sprintf("Counts: %d genes x %d spots, %d nonzeros\n",
            nrow(counts), ncol(counts), length(counts_dgT@x)))

# Lineage tree
tree <- obj@results$deconvolution$Ref$lineageTree
for (k in names(tree)) {
  write.csv(data.frame(subtype = tree[[k]]),
            file.path(out_dir, sprintf("r_tree_%s.csv", gsub(" ", "_", k))),
            row.names = FALSE)
}
# Also save tree keys in order
write.csv(data.frame(lineage = names(tree)), file.path(out_dir, "r_tree_keys.csv"), row.names = FALSE)

# Level 1 propMat (major lineages only)
l1_types <- names(tree)
propMat_L1 <- propMat[intersect(c("Malignant", l1_types), rownames(propMat)), ]
write.csv(as.data.frame(propMat_L1), file.path(out_dir, "r_propMat_L1.csv"))

# Level 2 propMat (sublineages only)
all_subtypes <- unlist(tree)
propMat_L2 <- propMat[intersect(all_subtypes, rownames(propMat)), ]
write.csv(as.data.frame(propMat_L2), file.path(out_dir, "r_propMat_L2.csv"))

cat(sprintf("\nLevel 1 types (%d): %s\n", nrow(propMat_L1), paste(rownames(propMat_L1), collapse=", ")))
cat(sprintf("Level 2 types (%d): %s\n", nrow(propMat_L2), paste(rownames(propMat_L2), collapse=", ")))

cat(sprintf("\nAll files saved to: %s\n", out_dir))
cat("R done.\n")
