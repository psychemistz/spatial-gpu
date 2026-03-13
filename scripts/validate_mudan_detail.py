#!/usr/bin/env python
"""Detailed step-by-step comparison of normalizeVariance internals."""

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
    n_genes = len(gene_names)
    counts = sparse.csc_matrix(counts_df.values.astype(np.float64))
    n_spots = counts.shape[1]

    # Load R values
    r_df = pd.read_csv(f"{R_DIR}/normalizeVariance_df.csv")
    r_m = r_df["m"].values
    r_v = r_df["v"].values
    r_res = r_df["res"].values
    r_lp = r_df["lp"].values
    r_lpa = r_df["lpa"].values
    r_qv = r_df["qv"].values
    r_gsf = r_df["gsf"].values

    # Compute Python values
    mat_t = counts.T.tocsc()
    gene_means = np.asarray(mat_t.mean(axis=0)).ravel()
    mat_sq = mat_t.copy()
    mat_sq.data = mat_sq.data ** 2
    mean_sq = np.asarray(mat_sq.mean(axis=0)).ravel()
    gene_vars = (mean_sq - gene_means**2) * n_spots / (n_spots - 1)

    dfm = np.log(gene_means)
    dfv = np.log(gene_vars)
    vi = np.where(np.isfinite(dfv))[0]

    print(f"Genes: {n_genes}, Spots: {n_spots}")
    print(f"Finite variance genes: {len(vi)}")

    # Step 1: dfm and dfv
    err_m = np.max(np.abs(dfm - r_m))
    err_v = np.max(np.abs(dfv[np.isfinite(dfv)] - r_v[np.isfinite(r_v)]))
    print(f"\ndfm max error: {err_m:.2e}")
    print(f"dfv max error: {err_v:.2e}")

    # Step 2: GAM residuals
    from spatialgpu.deconvolution.mudan import _fit_gam_via_r
    fitted = _fit_gam_via_r(dfm[vi], dfv[vi], dfm, vi, k=5)
    py_res = np.full(n_genes, -np.inf)
    py_res[vi] = dfv[vi] - fitted[vi]

    # Compare residuals (only for finite values)
    mask = np.isfinite(py_res) & np.isfinite(r_res)
    err_res = np.max(np.abs(py_res[mask] - r_res[mask]))
    print(f"\nGAM residuals max error: {err_res:.2e}")
    print(f"  (among {mask.sum()} genes with finite residuals)")

    # Step 3: F-distribution log p-values
    py_lp = np.full(n_genes, np.nan)
    exp_res_vi = np.exp(py_res[vi])
    py_lp[vi] = stats.f.logsf(exp_res_vi, dfn=n_spots, dfd=n_spots)

    mask_lp = np.isfinite(py_lp) & np.isfinite(r_lp)
    abs_lp_err = np.abs(py_lp[mask_lp] - r_lp[mask_lp])
    err_lp = np.max(abs_lp_err)
    # Also check relative error for very negative values
    rel_lp_err = abs_lp_err / np.maximum(np.abs(r_lp[mask_lp]), 1e-300)
    print(f"\nF log-pvalue max abs error: {err_lp:.2e}")
    print(f"  max relative error: {np.max(rel_lp_err):.2e}")
    print(f"  among {mask_lp.sum()} finite genes")

    # Show worst cases
    worst_idx = np.argsort(abs_lp_err)[-5:]
    print(f"  Worst 5 genes:")
    finite_genes = np.where(mask_lp)[0]
    for idx in worst_idx:
        gi = finite_genes[idx]
        print(f"    gene {gi}: py={py_lp[gi]:.10f}, r={r_lp[gi]:.10f}, "
              f"diff={py_lp[gi]-r_lp[gi]:.2e}, res={py_res[gi]:.6f}")

    # Step 4: BH adjustment
    from spatialgpu.deconvolution.mudan import _bh_adjust_log
    py_lpa = _bh_adjust_log(py_lp)

    mask_lpa = np.isfinite(py_lpa) & np.isfinite(r_lpa)
    abs_lpa_err = np.abs(py_lpa[mask_lpa] - r_lpa[mask_lpa])
    err_lpa = np.max(abs_lpa_err)
    print(f"\nBH adjusted log-p max error: {err_lpa:.2e}")

    # Step 5: ods comparison
    alpha = 0.05
    py_ods = np.where(py_lpa < np.log(alpha))[0]
    r_ods_csv = pd.read_csv(f"{R_DIR}/ods_genes.csv")
    r_ods_genes = set(r_ods_csv["gene"].values)
    py_ods_genes = set(gene_names[py_ods])

    only_in_py = py_ods_genes - r_ods_genes
    only_in_r = r_ods_genes - py_ods_genes
    print(f"\nods: Python={len(py_ods)}, R={len(r_ods_genes)}")
    print(f"  Only in Python ({len(only_in_py)}): {list(only_in_py)[:10]}")
    print(f"  Only in R ({len(only_in_r)}): {list(only_in_r)[:10]}")

    # For borderline genes, show their log-pvalues
    if only_in_py or only_in_r:
        print("\n  Borderline genes (near log(0.05) = -2.9957):")
        for g in list(only_in_py)[:5] + list(only_in_r)[:5]:
            gi = np.where(gene_names == g)[0][0]
            print(f"    {g}: py_lpa={py_lpa[gi]:.6f}, r_lpa={r_lpa[gi]:.6f}, "
                  f"threshold={np.log(alpha):.6f}")

    # Step 6: qv and gsf
    py_qv = np.full(n_genes, np.nan)
    finite_lp = np.isfinite(py_lp)
    p_vals = np.exp(py_lp[finite_lp])
    p_vals = np.clip(p_vals, 0, 1)
    py_qv[finite_lp] = stats.chi2.isf(p_vals, df=n_genes - 1) / n_genes

    mask_qv = np.isfinite(py_qv) & np.isfinite(r_qv)
    abs_qv_err = np.abs(py_qv[mask_qv] - r_qv[mask_qv])
    err_qv = np.max(abs_qv_err)
    rel_qv_err = abs_qv_err / np.maximum(np.abs(r_qv[mask_qv]), 1e-300)
    print(f"\nqv max abs error: {err_qv:.2e}")
    print(f"  max relative error: {np.max(rel_qv_err):.2e}")

    # Worst qv cases
    worst_qv = np.argsort(abs_qv_err)[-5:]
    finite_qv_genes = np.where(mask_qv)[0]
    print(f"  Worst 5 qv:")
    for idx in worst_qv:
        gi = finite_qv_genes[idx]
        print(f"    gene {gi}: py_qv={py_qv[gi]:.6f}, r_qv={r_qv[gi]:.6f}, "
              f"py_lp={py_lp[gi]:.6f}, r_lp={r_lp[gi]:.6f}")

    # gsf
    py_gsf = np.sqrt(np.clip(py_qv, 0.001, 1000) / np.exp(dfv))
    py_gsf[~np.isfinite(py_gsf)] = 0.0

    mask_gsf = np.isfinite(py_gsf) & np.isfinite(r_gsf)
    abs_gsf_err = np.abs(py_gsf[mask_gsf] - r_gsf[mask_gsf])
    err_gsf = np.max(abs_gsf_err)
    print(f"\ngsf max abs error: {err_gsf:.2e}")
    print(f"  mean abs error: {np.mean(abs_gsf_err):.2e}")

    # Worst gsf cases
    worst_gsf = np.argsort(abs_gsf_err)[-5:]
    finite_gsf_genes = np.where(mask_gsf)[0]
    print(f"  Worst 5 gsf:")
    for idx in worst_gsf:
        gi = finite_gsf_genes[idx]
        print(f"    gene {gene_names[gi]}: py_gsf={py_gsf[gi]:.6f}, r_gsf={r_gsf[gi]:.6f}, "
              f"py_qv={py_qv[gi]:.6f}, r_qv={r_qv[gi]:.6f}")


if __name__ == "__main__":
    main()
