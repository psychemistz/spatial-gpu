#!/usr/bin/env python
"""Step-by-step validation of Stage 1 Steps 2-3 against R intermediates.

Compares: cor_sig, spotMal, malRef, malProp at each step.
R intermediates must be pre-saved by scripts/r_save_mudan_intermediates.R.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import sparse, stats

VST_DIR = "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
R_DIR = "validation/mudan_intermediates"


def main():
    # Load counts
    counts_df = pd.read_csv(f"{VST_DIR}/1_counts.tsv", sep="\t", index_col=0)
    gene_names = np.array(counts_df.index)
    spot_names = np.array(counts_df.columns)
    counts = sparse.csc_matrix(counts_df.values.astype(np.float64))

    # QC: remove zero genes
    gene_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero = gene_sums > 0
    counts = counts[nonzero]
    gene_names = gene_names[nonzero]
    n_genes, n_spots = counts.shape
    print(f"After QC: {n_genes} genes x {n_spots} spots")

    # === Step 1: centered matrix ===
    from spatialgpu.deconvolution.core import _cpm_log2_center
    centered = _cpm_log2_center(counts)
    print(f"\nCentered matrix shape: {centered.shape}")

    # === Step 2: cor_sig (CNA signature correlation) ===
    print("\n=== Step 2: CNA signature correlation ===")

    # Load R's overlap genes and cor_sig
    r_olp = pd.read_csv(f"{R_DIR}/cna_olp_genes.csv")
    r_olp_genes = r_olp["gene"].values
    r_cor_sig = pd.read_csv(f"{R_DIR}/cor_sig.csv")

    # Python: get CNA signature
    from spatialgpu.deconvolution.reference import get_cancer_signature
    _, sig = get_cancer_signature("BRCA", "CNA")

    # Intersect genes
    olp = np.intersect1d(gene_names, sig.index)
    print(f"  Python overlap genes: {len(olp)}")
    print(f"  R overlap genes: {len(r_olp_genes)}")

    # Check if overlap sets match
    py_set = set(olp)
    r_set = set(r_olp_genes)
    if py_set == r_set:
        print(f"  Overlap genes: MATCH ({len(olp)})")
    else:
        print(f"  MISMATCH: only in Python: {py_set - r_set}")
        print(f"  MISMATCH: only in R: {r_set - py_set}")

    # Compute cor_sig
    gene_idx = np.array([np.where(gene_names == g)[0][0] for g in olp])
    sig_vals = sig.reindex(olp).values.reshape(-1, 1)
    X_sub = centered[gene_idx, :]

    # Use Python's cormat
    from spatialgpu.deconvolution.core import cormat
    cor_sig_py = cormat(X_sub, sig_vals)
    cor_sig_py.index = spot_names

    # Compare cor_r
    r_cor_r = r_cor_sig.set_index("spot")["cor_r"].reindex(spot_names).values
    py_cor_r = cor_sig_py["cor_r"].values
    err_cor_r = np.max(np.abs(py_cor_r - r_cor_r))
    print(f"\n  cor_r max error: {err_cor_r:.6e}")
    print(f"  cor_r mean error: {np.mean(np.abs(py_cor_r - r_cor_r)):.6e}")

    # Show worst cases
    worst = np.argsort(np.abs(py_cor_r - r_cor_r))[-5:]
    for idx in worst:
        print(f"    spot {spot_names[idx]}: py={py_cor_r[idx]:.6f}, r={r_cor_r[idx]:.6f}, diff={py_cor_r[idx]-r_cor_r[idx]:.6e}")

    # Compare cor_p
    r_cor_p = r_cor_sig.set_index("spot")["cor_p"].reindex(spot_names).values
    py_cor_p = cor_sig_py["cor_p"].values
    err_cor_p = np.max(np.abs(py_cor_p - r_cor_p))
    print(f"\n  cor_p max error: {err_cor_p:.6e}")

    # Compare cor_padj
    r_cor_padj = r_cor_sig.set_index("spot")["cor_padj"].reindex(spot_names).values
    py_cor_padj = cor_sig_py["cor_padj"].values
    err_cor_padj = np.max(np.abs(py_cor_padj - r_cor_padj))
    print(f"  cor_padj max error: {err_cor_padj:.6e}")

    # === Step 2b: cluster stats ===
    print("\n=== Step 2b: Cluster statistics ===")

    # Load R clustering
    r_clust = pd.read_csv(f"{R_DIR}/clustering.csv")
    clustering = pd.Series(r_clust["cluster"].values, index=r_clust["spot"].values)

    # Seq depth
    seq_depth = pd.Series(
        np.asarray((counts > 0).sum(axis=0)).ravel(),
        index=spot_names,
    )

    # Load R cluster stats
    r_stats = pd.read_csv(f"{R_DIR}/cluster_stats.csv")
    print(f"  R cluster stats:")
    for _, row in r_stats.iterrows():
        print(f"    Cluster {int(row['cluster'])}: mean={row['mean']:.6f}, wilcox={row['wilcoxTestG0']:.6e}, "
              f"frac={row['fraction_spot_padj']:.6f}, seqDiff={row['seq_depth_diff']:.1f}, "
              f"mal={bool(row['clusterMal'])}")

    # Compute Python cluster stats
    from spatialgpu.deconvolution.core import _compute_cluster_stats
    stat_df_py = _compute_cluster_stats(cor_sig_py, clustering, seq_depth)
    print(f"\n  Python cluster stats:")
    for idx, row in stat_df_py.iterrows():
        print(f"    Cluster {idx}: mean={row['mean']:.6f}, wilcox={row['wilcoxTestG0']:.6e}, "
              f"frac={row['fraction_spot_padj']:.6f}, seqDiff={row['seq_depth_diff']:.1f}, "
              f"mal={bool(row['clusterMal'])}")

    # Compare which clusters are malignant
    r_mal_clusters = set(r_stats[r_stats["clusterMal"] == True]["cluster"].astype(int))
    py_mal_clusters = set(stat_df_py.index[stat_df_py["clusterMal"]])
    print(f"\n  R malignant clusters: {r_mal_clusters}")
    print(f"  Python malignant clusters: {py_mal_clusters}")

    # === Step 3: spotMal ===
    print("\n=== Step 3: Malignant spot selection ===")

    # R spotMal
    r_spotMal = pd.read_csv(f"{R_DIR}/spotMal.csv")
    r_spotMal_set = set(r_spotMal["spot"].values)

    # Python spotMal
    mal_clusters = stat_df_py.index[stat_df_py["clusterMal"]]
    spot_mal_mask = clustering.isin(mal_clusters) & (cor_sig_py["cor_r"].reindex(clustering.index).values > 0)
    spot_mal = set(spot_names[spot_mal_mask.values])

    print(f"  R spotMal: {len(r_spotMal_set)}")
    print(f"  Python spotMal: {len(spot_mal)}")
    print(f"  Overlap: {len(spot_mal & r_spotMal_set)}")
    print(f"  Only in Python: {len(spot_mal - r_spotMal_set)}")
    print(f"  Only in R: {len(r_spotMal_set - spot_mal)}")

    if spot_mal != r_spotMal_set:
        # Show borderline spots
        for s in list(spot_mal - r_spotMal_set)[:5]:
            idx = np.where(spot_names == s)[0][0]
            print(f"    Python-only {s}: cor_r={py_cor_r[idx]:.6f}, cluster={clustering[s]}")
        for s in list(r_spotMal_set - spot_mal)[:5]:
            idx = np.where(spot_names == s)[0][0]
            print(f"    R-only {s}: cor_r={py_cor_r[idx]:.6f}, cluster={clustering[s]}")

    # === Step 3b: malRef ===
    print("\n=== Step 3b: malRef comparison ===")

    # Use R's spotMal for comparison
    spot_mal_r_list = sorted(r_spotMal_set)
    spot_mal_idx_r = np.array([np.where(spot_names == s)[0][0] for s in spot_mal_r_list])

    # Python malRef computation
    mal_counts = counts[:, spot_mal_idx_r]
    mal_col_sums = np.asarray(mal_counts.sum(axis=0)).ravel()
    mal_cpm = mal_counts.toarray().astype(np.float64)
    mal_cpm = mal_cpm / mal_col_sums[np.newaxis, :] * 1e6
    py_mal_ref = np.nanmean(mal_cpm, axis=1)

    # R malRef
    r_malRef = pd.read_csv(f"{R_DIR}/malRef.csv")
    r_mal_ref_vals = r_malRef.set_index("gene")["malRef"].reindex(gene_names).values

    mask = np.isfinite(py_mal_ref) & np.isfinite(r_mal_ref_vals)
    err_malRef = np.max(np.abs(py_mal_ref[mask] - r_mal_ref_vals[mask]))
    print(f"  malRef max error (using R's spotMal): {err_malRef:.6e}")
    print(f"  malRef mean error: {np.mean(np.abs(py_mal_ref[mask] - r_mal_ref_vals[mask])):.6e}")

    if err_malRef > 1e-6:
        worst = np.argsort(np.abs(py_mal_ref[mask] - r_mal_ref_vals[mask]))[-5:]
        mask_genes = np.where(mask)[0]
        for idx in worst:
            gi = mask_genes[idx]
            print(f"    gene {gene_names[gi]}: py={py_mal_ref[gi]:.6f}, r={r_mal_ref_vals[gi]:.6f}")

    # === Step 3c: malProp ===
    print("\n=== Step 3c: malProp comparison ===")

    # sig_from_mal using R's spotMal
    sig_from_mal = centered[:, spot_mal_idx_r].mean(axis=1).reshape(-1, 1)

    # cor_sig_mal
    cor_sig_mal = cormat(centered, sig_from_mal)
    cor_sig_mal.index = spot_names
    mal_prop_raw = cor_sig_mal["cor_r"].values.copy()

    # R's raw malProp
    r_malProp_raw = pd.read_csv(f"{R_DIR}/malProp_raw.csv")
    r_raw = r_malProp_raw.set_index("spot")["malProp_raw"].reindex(spot_names).values

    err_raw = np.max(np.abs(mal_prop_raw - r_raw))
    print(f"  malProp raw max error: {err_raw:.6e}")
    print(f"  malProp raw mean error: {np.mean(np.abs(mal_prop_raw - r_raw)):.6e}")

    if err_raw > 1e-6:
        worst = np.argsort(np.abs(mal_prop_raw - r_raw))[-5:]
        for idx in worst:
            print(f"    spot {spot_names[idx]}: py={mal_prop_raw[idx]:.6f}, r={r_raw[idx]:.6f}")

    # Clipping and normalization
    top5p = max(1, round(n_spots * 0.05))
    sorted_prop = np.sort(mal_prop_raw)
    p5 = sorted_prop[top5p - 1]
    p95 = sorted_prop[len(sorted_prop) - top5p]

    mal_prop = np.clip(mal_prop_raw, p5, p95)
    mal_prop = (mal_prop - mal_prop.min()) / (mal_prop.max() - mal_prop.min())

    # R final malProp
    r_malProp = pd.read_csv(f"{R_DIR}/malProp_final.csv")
    r_final = r_malProp.set_index("spot")["malProp"].reindex(spot_names).values

    err_final = np.max(np.abs(mal_prop - r_final))
    print(f"\n  malProp final max error: {err_final:.6e}")
    print(f"  malProp final mean error: {np.mean(np.abs(mal_prop - r_final)):.6e}")

    print(f"\n  Python p5={p5:.6f}, p95={p95:.6f}")
    print(f"  Python raw range: [{mal_prop_raw.min():.6f}, {mal_prop_raw.max():.6f}]")
    print(f"  R raw range: [{r_raw.min():.6f}, {r_raw.max():.6f}]")


if __name__ == "__main__":
    main()
