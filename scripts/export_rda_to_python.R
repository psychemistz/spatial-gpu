#!/usr/bin/env Rscript
# Export SpaCET .rda reference data to Python-readable formats (CSV/JSON)

library(jsonlite)

spacet_extdata <- system.file("extdata", package = "SpaCET")
if (spacet_extdata == "") {
  spacet_extdata <- "/Users/seongyongpark/project/psychemist/SpaCET/inst/extdata"
}

outdir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/spatialgpu/data"

cat("=== Exporting combRef_0.5.rda ===\n")
load(file.path(spacet_extdata, "combRef_0.5.rda"))

# refProfiles: genes x cell_types matrix
write.csv(Ref$refProfiles, file.path(outdir, "combRef_refProfiles.csv"), quote = TRUE)

# sigGenes: list of gene vectors per cell type
write(toJSON(Ref$sigGenes, auto_unbox = FALSE, pretty = FALSE),
      file.path(outdir, "combRef_sigGenes.json"))

# lineageTree: list of sublineage vectors per major lineage
write(toJSON(Ref$lineageTree, auto_unbox = FALSE, pretty = FALSE),
      file.path(outdir, "combRef_lineageTree.json"))

cat("  refProfiles:", nrow(Ref$refProfiles), "genes x", ncol(Ref$refProfiles), "cell types\n")
cat("  sigGenes:", length(Ref$sigGenes), "cell types\n")
cat("  lineageTree:", length(Ref$lineageTree), "major lineages\n")


cat("\n=== Exporting cancerDictionary.rda ===\n")
load(file.path(spacet_extdata, "cancerDictionary.rda"))

# CNA signatures
for (name in names(cancerDictionary$CNA)) {
  sig <- cancerDictionary$CNA[[name]]
  fname <- paste0("cancerDict_CNA_", gsub("[^a-zA-Z0-9_]", "_", name), ".csv")
  if (is.matrix(sig) || is.data.frame(sig)) {
    write.csv(data.frame(gene = rownames(sig), value = sig[,1]),
              file.path(outdir, fname), row.names = FALSE, quote = TRUE)
  } else {
    write.csv(data.frame(gene = names(sig), value = as.numeric(sig)),
              file.path(outdir, fname), row.names = FALSE, quote = TRUE)
  }
}

# expr signatures
for (name in names(cancerDictionary$expr)) {
  sig <- cancerDictionary$expr[[name]]
  fname <- paste0("cancerDict_expr_", gsub("[^a-zA-Z0-9_]", "_", name), ".csv")
  if (is.matrix(sig) || is.data.frame(sig)) {
    write.csv(data.frame(gene = rownames(sig), value = sig[,1]),
              file.path(outdir, fname), row.names = FALSE, quote = TRUE)
  } else {
    write.csv(data.frame(gene = names(sig), value = as.numeric(sig)),
              file.path(outdir, fname), row.names = FALSE, quote = TRUE)
  }
}

# Save index of available signatures
index <- list(
  CNA = names(cancerDictionary$CNA),
  expr = names(cancerDictionary$expr)
)
write(toJSON(index, auto_unbox = FALSE, pretty = TRUE),
      file.path(outdir, "cancerDictionary_index.json"))

cat("  CNA signatures:", length(cancerDictionary$CNA), "\n")
cat("  expr signatures:", length(cancerDictionary$expr), "\n")


cat("\n=== Exporting Ref_Normal_LIHC.rda ===\n")
load(file.path(spacet_extdata, "Ref_Normal_LIHC.rda"))

write.csv(Ref_Normal$refProfiles, file.path(outdir, "Ref_Normal_LIHC_refProfiles.csv"), quote = TRUE)

write(toJSON(Ref_Normal$sigGenes, auto_unbox = FALSE, pretty = FALSE),
      file.path(outdir, "Ref_Normal_LIHC_sigGenes.json"))

write(toJSON(Ref_Normal$lineageTree, auto_unbox = FALSE, pretty = FALSE),
      file.path(outdir, "Ref_Normal_LIHC_lineageTree.json"))

cat("  refProfiles:", nrow(Ref_Normal$refProfiles), "genes x", ncol(Ref_Normal$refProfiles), "cell types\n")


cat("\n=== Exporting Visium_BC example data for validation ===\n")
valdir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/validation"

# Run full deconvolution and save intermediate results
library(SpaCET)
visiumPath <- file.path(spacet_extdata, "Visium_BC")
obj <- create.SpaCET.object.10X(visiumPath)
obj <- SpaCET.quality.control(obj, min.genes = 100)

# Save input data
write.csv(as.matrix(obj@input$counts), file.path(valdir, "visium_bc_counts.csv"), quote = TRUE)
write.csv(obj@input$spotCoordinates, file.path(valdir, "visium_bc_spotCoordinates.csv"), quote = TRUE)

# Run deconvolution with single core for reproducibility
obj <- SpaCET.deconvolution(obj, cancerType = "BRCA", coreNo = 1)

# Save results
write.csv(obj@results$deconvolution$propMat, file.path(valdir, "visium_bc_propMat.csv"), quote = TRUE)
write.csv(obj@results$deconvolution$malRes$malProp, file.path(valdir, "visium_bc_malProp.csv"), quote = TRUE)
write.csv(obj@results$deconvolution$malRes$malRef, file.path(valdir, "visium_bc_malRef.csv"), quote = TRUE)

cat("  Saved propMat:", nrow(obj@results$deconvolution$propMat), "x", ncol(obj@results$deconvolution$propMat), "\n")

cat("\nDone!\n")
