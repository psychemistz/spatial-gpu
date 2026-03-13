#!/usr/bin/env python
"""Step-by-step validation of Python MUDAN implementation against R intermediates.

Compares: normalizeVariance outputs, PCA scores, clustering, and final malProp.
R intermediates must be pre-saved by scripts/r_save_mudan_intermediates.R.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import sparse

# Paths
VST_DIR = "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
R_DIR = "validation/mudan_intermediates"
COUNTS_FILE = f"{VST_DIR}/1_counts.tsv"


def load_counts():
    """Load VST1 counts as genes x spots sparse matrix."""
    counts_df = pd.read_csv(COUNTS_FILE, sep="\t", index_col=0)
    gene_names = np.array(counts_df.index)
    spot_names = np.array(counts_df.columns)
    counts = sparse.csc_matrix(counts_df.values.astype(np.float64))
    return counts, gene_names, spot_names


def validate_step(name, py_val, r_val, rtol=1e-10, atol=1e-10):
    """Compare Python and R values."""
    if isinstance(py_val, set) and isinstance(r_val, set):
        overlap = len(py_val & r_val)
        total = len(py_val | r_val)
        print(f"  {name}: Python={len(py_val)}, R={len(r_val)}, "
              f"overlap={overlap}/{total} ({100*overlap/total:.1f}%)")
        return overlap == total

    py_arr = np.asarray(py_val, dtype=np.float64).ravel()
    r_arr = np.asarray(r_val, dtype=np.float64).ravel()

    if len(py_arr) != len(r_arr):
        print(f"  {name}: LENGTH MISMATCH Python={len(py_arr)} vs R={len(r_arr)}")
        return False

    abs_err = np.abs(py_arr - r_arr)
    max_err = np.max(abs_err)
    mean_err = np.mean(abs_err)
    med_err = np.median(abs_err)

    status = "PASS" if max_err < atol else "FAIL"
    print(f"  {name}: max={max_err:.6e}, mean={mean_err:.6e}, "
          f"median={med_err:.6e}  [{status}]")
    return max_err < atol


def main():
    print("Loading VST1 counts...")
    counts, gene_names, spot_names = load_counts()
    n_genes, n_spots = counts.shape
    print(f"  {n_genes} genes x {n_spots} spots")

    # QC: remove genes with 0 expression (matching SpaCET.quality.control)
    gene_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero = gene_sums > 0
    counts = counts[nonzero]
    gene_names = gene_names[nonzero]
    n_genes = counts.shape[0]
    print(f"  After QC: {n_genes} genes x {n_spots} spots")

    # ========== Step 1: normalizeVariance ==========
    print("\n=== Step 1: normalizeVariance ===")
    from spatialgpu.deconvolution.mudan import normalize_variance

    norm_mat, ods, gsf = normalize_variance(counts, gam_k=5, alpha=0.05)
    print(f"  Python: {len(ods)} overdispersed genes")

    # Load R's normalizeVariance output
    r_df = pd.read_csv(f"{R_DIR}/normalizeVariance_df.csv")
    r_ods = pd.read_csv(f"{R_DIR}/ods_genes.csv")

    # Compare gene means/variances
    r_m = r_df["m"].values
    r_v = r_df["v"].values
    r_gsf = r_df["gsf"].values
    r_ods_genes = set(r_ods["gene"].values)

    # Recompute Python gene means/variances for comparison
    mat_t = counts.T.tocsc()
    py_means = np.asarray(mat_t.mean(axis=0)).ravel()
    mat_sq = mat_t.copy()
    mat_sq.data = mat_sq.data ** 2
    mean_sq = np.asarray(mat_sq.mean(axis=0)).ravel()
    py_vars = (mean_sq - py_means**2) * n_spots / (n_spots - 1)

    py_m = np.log(py_means)
    py_v = np.log(py_vars)

    validate_step("gene log-means (dfm)", py_m, r_m, atol=1e-10)
    validate_step("gene log-vars (dfv)", py_v, r_v, atol=1e-10)

    # Compare overdispersed genes
    py_ods_genes = set(gene_names[ods])
    validate_step("overdispersed genes", py_ods_genes, r_ods_genes)

    # Compare gene scale factors
    validate_step("gene scale factors (gsf)", gsf, r_gsf, atol=1e-6)

    # ========== Step 2: log10 + PCA ==========
    print("\n=== Step 2: PCA ===")
    from spatialgpu.deconvolution.mudan import get_pcs

    if sparse.issparse(norm_mat):
        log_mat = norm_mat.copy().astype(np.float64)
        log_mat.data = np.log10(log_mat.data + 1)
    else:
        log_mat = np.log10(norm_mat + 1)

    ods_mat = log_mat[ods, :] if sparse.issparse(log_mat) else log_mat[ods, :]
    n_pcs = min(30, len(ods) - 1, n_spots - 1)
    pcs_py = get_pcs(ods_mat, n_pcs=n_pcs)

    # Load R's PCA scores
    r_pcs = pd.read_csv(f"{R_DIR}/pca_scores.csv", index_col=0)
    r_pcs_arr = r_pcs.values

    # PCA signs may be flipped — check both orientations
    print(f"  Python PCs shape: {pcs_py.shape}, R PCs shape: {r_pcs_arr.shape}")
    n_compare = min(pcs_py.shape[1], r_pcs_arr.shape[1])

    for pc_i in range(min(5, n_compare)):
        py_pc = pcs_py[:, pc_i]
        r_pc = r_pcs_arr[:, pc_i]

        err_same = np.max(np.abs(py_pc - r_pc))
        err_flip = np.max(np.abs(py_pc + r_pc))
        err = min(err_same, err_flip)
        sign = "same" if err_same < err_flip else "flipped"
        print(f"  PC{pc_i+1}: max_err={err:.6e} (sign {sign})")

    # ========== Step 3: Clustering ==========
    print("\n=== Step 3: Clustering ===")
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_samples

    corr_matrix = np.corrcoef(pcs_py)
    corr_matrix = np.clip(corr_matrix, -1, 1)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed ** 2, method="ward")

    # Silhouette analysis
    cluster_numbers = list(range(2, 10))
    sil_scores = []
    for k in cluster_numbers:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_samples(dist_matrix, labels, metric="precomputed")
        sil_scores.append(np.mean(sil))

    sil_diff = [sil_scores[i] - sil_scores[i + 1] for i in range(len(sil_scores) - 1)]
    max_n = cluster_numbers[np.argmax(sil_diff) + 1]
    print(f"  Python optimal k: {max_n}")

    # Load R's silhouette scores
    r_sil = pd.read_csv(f"{R_DIR}/silhouette_scores.csv")
    print(f"  R silhouette scores: {r_sil['silhouette'].values}")
    print(f"  Python silhouette scores: {np.array(sil_scores)}")

    r_sil_diff = r_sil['silhouette'].values[:-1] - r_sil['silhouette'].values[1:]
    r_max_n = cluster_numbers[np.argmax(r_sil_diff) + 1]
    print(f"  R optimal k: {r_max_n}")

    # Load R's clustering
    r_clust = pd.read_csv(f"{R_DIR}/clustering.csv")
    py_clustering = fcluster(Z, t=max_n, criterion="maxclust")

    # Compare cluster assignments
    if max_n == r_max_n:
        r_clust_arr = r_clust["cluster"].values
        # Cluster labels might be permuted — check agreement
        from scipy.optimize import linear_sum_assignment
        from collections import Counter

        # Build confusion matrix
        labels_py = py_clustering
        labels_r = r_clust_arr
        unique_py = sorted(set(labels_py))
        unique_r = sorted(set(labels_r))

        confusion = np.zeros((len(unique_py), len(unique_r)))
        for i, lpy in enumerate(unique_py):
            for j, lr in enumerate(unique_r):
                confusion[i, j] = np.sum((labels_py == lpy) & (labels_r == lr))

        # Find best label matching
        row_ind, col_ind = linear_sum_assignment(-confusion)
        matched = sum(confusion[row_ind, col_ind])
        total = len(labels_py)
        print(f"  Cluster agreement: {int(matched)}/{total} ({100*matched/total:.1f}%)")
    else:
        print(f"  Different optimal k: Python={max_n}, R={r_max_n}")

    # ========== Step 4: malProp ==========
    print("\n=== Step 4: Final malProp ===")
    r_malProp = pd.read_csv(f"{R_DIR}/malProp_final.csv")
    r_malProp_vals = r_malProp.set_index("spot")["malProp"]

    # We need to run the full pipeline to get malProp
    # For now, just compare R's raw malProp
    r_malProp_raw = pd.read_csv(f"{R_DIR}/malProp_raw.csv")
    print(f"  R malProp range: [{r_malProp_vals.min():.6f}, {r_malProp_vals.max():.6f}]")
    print(f"  R malProp raw range: [{r_malProp_raw['malProp_raw'].min():.6f}, {r_malProp_raw['malProp_raw'].max():.6f}]")

    # Load R's full malProp for comparison with current Python validation files
    try:
        r_malProp_full = pd.read_csv("validation/vst1_malProp.csv")
        print(f"  R full pipeline malProp range: [{r_malProp_full['malProp'].min():.6f}, {r_malProp_full['malProp'].max():.6f}]")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
