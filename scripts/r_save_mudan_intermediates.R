#!/usr/bin/env Rscript
# Save MUDAN normalization intermediates for Stage 1 validation.
# Saves: dfm, dfv, GAM residuals, overdispersed genes, gsf, PCA scores,
#         clustering, and final malProp/malRef for VST1.

library(SpaCET)
library(Matrix)
library(MUDAN)

vst_dir <- "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
out_dir <- "/Users/seongyongpark/project/psychemist/spatial-gpu/validation/mudan_intermediates"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

i <- 1
cat(sprintf("\n========== Dataset %d ==========\n", i))

# Load counts
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

# Create SpaCET object and run QC
obj <- new("SpaCET",
  input = list(counts = counts_sparse, spotCoordinates = spotCoordinates,
               platform = "Visium", image = list(path = NA)),
  results = list()
)
obj <- SpaCET.quality.control(obj, min.genes = 1)

st.matrix.data <- obj@input$counts
gene_names <- rownames(st.matrix.data)
spot_names <- colnames(st.matrix.data)

cat(sprintf("Genes: %d, Spots: %d\n", length(gene_names), length(spot_names)))

# ========== Stage 1 Intermediates ==========

# 1. CPM + log2 + center (this is for the correlation part, not for clustering)
st.matrix.data.diff <- sweep(st.matrix.data, 2, Matrix::colSums(st.matrix.data), "/") * 1e6
st.matrix.data.diff[is.na(st.matrix.data.diff)] <- 0
st.matrix.data.diff@x <- log2(st.matrix.data.diff@x + 1)
st.matrix.data.diff <- st.matrix.data.diff - Matrix::rowMeans(st.matrix.data.diff)

# 2. MUDAN normalizeVariance
set.seed(123)
matnorm.info <- normalizeVariance(methods::as(st.matrix.data, "dgCMatrix"),
                                  details = TRUE, verbose = TRUE)

# Save normalizeVariance outputs
df <- matnorm.info$df
write.csv(data.frame(gene = rownames(df), m = df$m, v = df$v, res = df$res,
                     lp = df$lp, lpa = df$lpa, qv = df$qv, gsf = df$gsf),
          file.path(out_dir, "normalizeVariance_df.csv"), row.names = FALSE)

ods <- matnorm.info$ods
ods_gene_names <- gene_names[ods]
write.csv(data.frame(gene = ods_gene_names, idx = as.integer(ods)),
          file.path(out_dir, "ods_genes.csv"), row.names = FALSE)
cat(sprintf("Overdispersed genes: %d\n", length(ods)))

# 3. log10 transform
matnorm <- log10(matnorm.info$mat + 1)

# 4. getPcs
pcs <- getPcs(matnorm[ods, ], nGenes = length(ods), nPcs = 30, verbose = TRUE)

# Save PCA scores (spots x PCs)
write.csv(pcs, file.path(out_dir, "pca_scores.csv"))
cat(sprintf("PCA scores: %d spots x %d PCs\n", nrow(pcs), ncol(pcs)))

# 5. Distance matrix and clustering
d <- as.dist(1 - cor(t(pcs)))
hc <- hclust(d, method = "ward.D")

# Save dendrogram merge matrix (n-1 rows) and order (n elements) separately
write.csv(data.frame(merge1 = hc$merge[, 1], merge2 = hc$merge[, 2],
                     height = hc$height),
          file.path(out_dir, "hclust_merge.csv"), row.names = FALSE)
write.csv(data.frame(order = hc$order),
          file.path(out_dir, "hclust_order.csv"), row.names = FALSE)

# 6. Silhouette analysis
suppressPackageStartupMessages({
  library(cluster)
})

cluster_numbers <- 2:9
sil_values <- c()
for (k in cluster_numbers) {
  clust_k <- cutree(hc, k = k)
  sil <- silhouette(clust_k, d, Fun = mean)
  sil_values <- c(sil_values, mean(sil[, 3]))
}

sil_diff <- sil_values[1:(length(sil_values) - 1)] - sil_values[2:length(sil_values)]
maxN <- which(sil_diff == max(sil_diff)) + 1

write.csv(data.frame(k = cluster_numbers, silhouette = sil_values),
          file.path(out_dir, "silhouette_scores.csv"), row.names = FALSE)
cat(sprintf("Optimal k (max sil diff): %d\n", cluster_numbers[maxN]))

clustering <- cutree(hc, k = cluster_numbers[maxN])
write.csv(data.frame(spot = names(clustering), cluster = as.integer(clustering)),
          file.path(out_dir, "clustering.csv"), row.names = FALSE)

# 7. CNA signature correlation
load(system.file("extdata", "cancerDictionary.rda", package = "SpaCET"))
idx <- grepl("BRCA", names(cancerDictionary$CNA))
sig_vec <- cancerDictionary$CNA[idx][[1]]
sig <- matrix(sig_vec, ncol = 1)
rownames(sig) <- names(sig_vec)
olp_genes_sig <- intersect(gene_names, rownames(sig))
write.csv(data.frame(gene = olp_genes_sig),
          file.path(out_dir, "cna_olp_genes.csv"), row.names = FALSE)

cor_sig <- SpaCET:::corMat(as.matrix(st.matrix.data.diff), sig)
write.csv(data.frame(spot = rownames(cor_sig), cor_r = cor_sig[, "cor_r"],
                     cor_p = cor_sig[, "cor_p"], cor_padj = cor_sig[, "cor_padj"]),
          file.path(out_dir, "cor_sig.csv"), row.names = FALSE)

# 8. Cluster statistics
stat.df <- data.frame()
seq_depth <- Matrix::colSums(st.matrix.data > 0)
for (ci in sort(unique(clustering))) {
  cor_sig_c <- cor_sig[clustering == ci, ]
  seq_depth_c <- seq_depth[clustering == ci]
  stat.df[ci, "cluster"] <- ci
  stat.df[ci, "spotNum"] <- nrow(cor_sig_c)
  stat.df[ci, "mean"] <- mean(cor_sig_c[, 1])
  stat.df[ci, "wilcoxTestG0"] <- suppressWarnings(
    wilcox.test(cor_sig_c[, 1], mu = 0, alternative = "greater")$p.value)
  stat.df[ci, "fraction_spot_padj"] <- sum(cor_sig_c[, "cor_r"] > 0 &
    cor_sig_c[, "cor_padj"] < 0.25) / nrow(cor_sig_c)
  stat.df[ci, "seq_depth_diff"] <- mean(seq_depth_c) - mean(seq_depth)
  stat.df[ci, "clusterMal"] <- stat.df[ci, "seq_depth_diff"] > 0 &
    stat.df[ci, "mean"] > 0 & stat.df[ci, "wilcoxTestG0"] < 0.05 &
    stat.df[ci, "fraction_spot_padj"] >= sum(cor_sig[, "cor_r"] > 0 &
      cor_sig[, "cor_padj"] < 0.25) / nrow(cor_sig)
}
write.csv(stat.df, file.path(out_dir, "cluster_stats.csv"), row.names = FALSE)
cat("Cluster stats:\n")
print(stat.df)

# 9. Malignant spots and malRef/malProp
top5p <- round(length(seq_depth) * 0.05)
spotMal <- names(clustering)[clustering %in% stat.df[stat.df[, "clusterMal"] == TRUE, "cluster"] &
                              cor_sig[, "cor_r"] > 0]
write.csv(data.frame(spot = spotMal),
          file.path(out_dir, "spotMal.csv"), row.names = FALSE)
cat(sprintf("Malignant spots: %d\n", length(spotMal)))

malRef <- Matrix::rowMeans(Matrix::t(Matrix::t(st.matrix.data[, spotMal]) * 1e6 /
                                       Matrix::colSums(st.matrix.data[, spotMal])))
write.csv(data.frame(gene = names(malRef), malRef = as.numeric(malRef)),
          file.path(out_dir, "malRef.csv"), row.names = FALSE)

# Compute final malProp
sig_mal <- apply(st.matrix.data.diff[, spotMal, drop = FALSE], 1, mean)
sig_mal <- matrix(sig_mal)
rownames(sig_mal) <- rownames(st.matrix.data.diff)
cor_sig_mal <- SpaCET:::corMat(as.matrix(st.matrix.data.diff), sig_mal)
malProp <- cor_sig_mal[, "cor_r"]
names(malProp) <- rownames(cor_sig_mal)

# Save raw malProp (before clipping)
write.csv(data.frame(spot = names(malProp), malProp_raw = as.numeric(malProp)),
          file.path(out_dir, "malProp_raw.csv"), row.names = FALSE)

malPropSorted <- sort(malProp)
p5 <- malPropSorted[top5p]
p95 <- malPropSorted[length(malPropSorted) - top5p + 1]
malProp[malProp <= p5] <- p5
malProp[malProp >= p95] <- p95
malProp <- (malProp - min(malProp)) / (max(malProp) - min(malProp))

write.csv(data.frame(spot = names(malProp), malProp = as.numeric(malProp)),
          file.path(out_dir, "malProp_final.csv"), row.names = FALSE)

cat(sprintf("\nAll intermediates saved to: %s\n", out_dir))
cat("========== DONE ==========\n")
