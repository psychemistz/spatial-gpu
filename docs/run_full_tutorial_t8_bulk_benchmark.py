"""Run Tutorial 8 — Bulk Deconvolution Pseudobulk Benchmark.

Generates semi-synthetic scRNA-seq, creates pseudobulk with known
proportions, runs deconvolution_bulk, evaluates accuracy, and exports
data for comparison with MuSiC/CIBERSORTx.

Usage: python docs/run_full_tutorial_t8_bulk_benchmark.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def write_output(name, text):
    path = os.path.join(OUTPUTS_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"  Output saved: {path}")


def main():
    import numpy as np
    import pandas as pd

    import spatialgpu.deconvolution as spacet
    from spatialgpu.benchmarks.pseudobulk import (
        _LEVEL1_TYPES,
        _collapse_to_level1,
        evaluate_deconvolution,
        export_for_cibersortx,
        export_for_music,
        generate_pseudobulk_dirichlet,
        generate_pseudobulk_titration,
        generate_semi_synthetic_scrna,
    )

    print("=== Tutorial 8: Bulk Deconvolution Pseudobulk Benchmark ===\n")

    # ---- Step 1: Generate semi-synthetic scRNA-seq ----
    print("1. Generating semi-synthetic scRNA-seq...")
    scrna_brca = generate_semi_synthetic_scrna(
        n_cells_per_type=500, include_malignant=True, cancer_type="BRCA", seed=42
    )
    print(
        f"   BRCA: {scrna_brca.n_obs} cells, {scrna_brca.n_vars} genes, "
        f"{scrna_brca.obs['cell_type'].nunique()} types"
    )

    scrna_normal = generate_semi_synthetic_scrna(
        n_cells_per_type=500, include_malignant=False, seed=42
    )
    print(
        f"   Normal: {scrna_normal.n_obs} cells, {scrna_normal.n_vars} genes, "
        f"{scrna_normal.obs['cell_type'].nunique()} types"
    )

    # ---- Step 2: Generate pseudobulk — Dirichlet ----
    print("\n2. Generating pseudobulk (Dirichlet)...")
    bulk_brca, gt_brca = generate_pseudobulk_dirichlet(
        scrna_brca, n_samples=100, n_cells_per_sample=1000, alpha=1.0, seed=42
    )
    print(f"   BRCA: {bulk_brca.n_obs} samples, {bulk_brca.n_vars} genes")

    bulk_normal, gt_normal = generate_pseudobulk_dirichlet(
        scrna_normal, n_samples=100, n_cells_per_sample=1000, alpha=1.0, seed=42
    )
    print(f"   Normal: {bulk_normal.n_obs} samples, {bulk_normal.n_vars} genes")

    # ---- Step 3: Generate pseudobulk — Titration ----
    print("\n3. Generating pseudobulk (Titration)...")
    bulk_titr, gt_titr = generate_pseudobulk_titration(
        scrna_brca,
        target_type="Malignant_BRCA",
        fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        n_replicates=5,
        n_cells_per_sample=1000,
        seed=42,
    )
    print(f"   Titration: {bulk_titr.n_obs} samples")

    # ---- Step 4: Deconvolution — BRCA ----
    print("\n4. Deconvolution — BRCA (100 samples)...")
    spacet.deconvolution_bulk(bulk_brca, cancer_type="BRCA")
    pm_brca = bulk_brca.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_brca.shape[0]} types x {pm_brca.shape[1]} samples")

    # ---- Step 5: Deconvolution — Normal ----
    print("\n5. Deconvolution — Normal (100 samples)...")
    spacet.deconvolution_bulk(bulk_normal, cancer_type="normal")
    pm_normal = bulk_normal.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_normal.shape[0]} types x {pm_normal.shape[1]} samples")

    # ---- Step 6: Deconvolution — Titration ----
    print("\n6. Deconvolution — Titration (45 samples)...")
    spacet.deconvolution_bulk(bulk_titr, cancer_type="BRCA")
    pm_titr = bulk_titr.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_titr.shape[0]} types x {pm_titr.shape[1]} samples")

    # ---- Step 7: Evaluate accuracy ----
    print("\n7. Evaluating accuracy...")

    # Collapse propMat to Level 1, transpose to samples x types
    est_brca = _collapse_to_level1(pm_brca, _LEVEL1_TYPES + ["Malignant"]).T
    est_brca = est_brca.rename(columns={"Malignant": "Malignant_BRCA"})

    est_normal = _collapse_to_level1(pm_normal, _LEVEL1_TYPES).T

    est_titr = _collapse_to_level1(pm_titr, _LEVEL1_TYPES + ["Malignant"]).T
    est_titr = est_titr.rename(columns={"Malignant": "Malignant_BRCA"})

    gt_titr_eval = gt_titr.drop(columns=["target_fraction"], errors="ignore")

    metrics_brca = evaluate_deconvolution(est_brca, gt_brca)
    metrics_normal = evaluate_deconvolution(est_normal, gt_normal)
    metrics_titr = evaluate_deconvolution(est_titr, gt_titr_eval)

    for name, m in [
        ("BRCA", metrics_brca),
        ("Normal", metrics_normal),
        ("Titration", metrics_titr),
    ]:
        print(f"\n   {name}:")
        print(f"     Pearson r:    {m['overall']['pearson_r']:.4f}")
        print(f"     Spearman rho: {m['overall']['spearman_rho']:.4f}")
        print(f"     RMSE:         {m['overall']['rmse']:.4f}")
        print(f"     Rare MAE:     {m['rare_type_mae']:.4f}")

    for name, m in [
        ("brca", metrics_brca),
        ("normal", metrics_normal),
        ("titration", metrics_titr),
    ]:
        lines = [
            f"Overall Pearson r:    {m['overall']['pearson_r']:.4f}",
            f"Overall Spearman rho: {m['overall']['spearman_rho']:.4f}",
            f"Overall RMSE:         {m['overall']['rmse']:.4f}",
            f"Rare type MAE:        {m['rare_type_mae']:.4f}",
            "",
            "Per-cell-type:",
            m["per_type"].to_string(),
        ]
        write_output(f"t8_metrics_{name}.txt", "\n".join(lines))

    # ---- Step 8: Generate figures ----
    print("\n8. Generating figures...")

    # 8a. Scatter: BRCA
    fig, ax = plt.subplots(figsize=(7, 7))
    common_types = est_brca.columns.intersection(gt_brca.columns)
    colors = plt.cm.tab20(np.linspace(0, 1, len(common_types)))
    for i, ct in enumerate(common_types):
        ax.scatter(
            gt_brca[ct],
            est_brca.reindex(gt_brca.index)[ct],
            s=15,
            alpha=0.6,
            label=ct,
            color=colors[i],
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Estimated proportion")
    ax.set_title(f"BRCA (r={metrics_brca['overall']['pearson_r']:.3f})")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "benchmark_scatter_brca.png")

    # 8b. Scatter: Normal
    fig, ax = plt.subplots(figsize=(7, 7))
    common_types_n = est_normal.columns.intersection(gt_normal.columns)
    colors_n = plt.cm.tab20(np.linspace(0, 1, len(common_types_n)))
    for i, ct in enumerate(common_types_n):
        ax.scatter(
            gt_normal[ct],
            est_normal.reindex(gt_normal.index)[ct],
            s=15,
            alpha=0.6,
            label=ct,
            color=colors_n[i],
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Estimated proportion")
    ax.set_title(f"Normal tissue (r={metrics_normal['overall']['pearson_r']:.3f})")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "benchmark_scatter_normal.png")

    # 8c. Per-cell-type Pearson r
    fig, ax = plt.subplots(figsize=(10, 5))
    brca_pt = metrics_brca["per_type"]["pearson_r"].rename("BRCA")
    normal_pt = metrics_normal["per_type"]["pearson_r"].rename("Normal")
    combined = pd.DataFrame({"BRCA": brca_pt, "Normal": normal_pt}).fillna(0)
    combined.plot(kind="barh", ax=ax)
    ax.set_xlabel("Pearson r")
    ax.set_title("Per-cell-type accuracy")
    ax.axvline(x=0, color="k", linewidth=0.5)
    fig.tight_layout()
    save(fig, "benchmark_per_type_r.png")

    # 8d. Titration curve
    from scipy.stats import pearsonr

    titr_fracs = np.sort(gt_titr["target_fraction"].unique())
    titr_r_values = []
    for frac in titr_fracs:
        mask = gt_titr["target_fraction"] == frac
        sub_est = est_titr.loc[mask]
        sub_gt = gt_titr_eval.loc[mask]
        common = sub_est.columns.intersection(sub_gt.columns)
        e = sub_est[common].values.ravel()
        g = sub_gt[common].values.ravel()
        r, _ = pearsonr(e, g)
        titr_r_values.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(titr_fracs, titr_r_values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Malignant fraction")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("Deconvolution accuracy vs tumor purity")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "benchmark_titration.png")

    # 8e. Rare cell type errors
    fig, ax = plt.subplots(figsize=(8, 5))
    rare_errors = {}
    common_brca = est_brca.columns.intersection(gt_brca.columns)
    for ct in common_brca:
        mask = gt_brca[ct] < 0.05
        if mask.sum() > 2:
            errors = np.abs(
                est_brca.reindex(gt_brca.index).loc[mask, ct] - gt_brca.loc[mask, ct]
            )
            rare_errors[ct] = errors.values
    if rare_errors:
        ax.boxplot(rare_errors.values(), labels=rare_errors.keys(), vert=True)
        ax.set_ylabel("Absolute error")
        ax.set_title("Error distribution for rare cell types (true < 5%)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        ax.text(
            0.5, 0.5, "No rare cell type entries",
            ha="center", va="center", transform=ax.transAxes,
        )
    fig.tight_layout()
    save(fig, "benchmark_rare_types.png")

    # ---- Step 9: Export for external tools ----
    print("\n9. Exporting for external tools...")
    music_dir = os.path.join(OUTPUTS_DIR, "t8_export_music")
    cibersortx_dir = os.path.join(OUTPUTS_DIR, "t8_export_cibersortx")

    export_for_music(bulk_brca, scrna_brca, music_dir, gt_brca)
    print(f"   MuSiC export: {music_dir}")

    export_for_cibersortx(bulk_brca, scrna_brca, cibersortx_dir, gt_brca)
    print(f"   CIBERSORTx export: {cibersortx_dir}")

    # ---- Step 10: Session info ----
    print("\n10. Session info...")
    import spatialgpu

    session_lines = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"matplotlib: {matplotlib.__version__}",
    ]

    try:
        import anndata

        session_lines.append(f"anndata: {anndata.__version__}")
    except Exception:
        pass

    try:
        import scipy

        session_lines.append(f"scipy: {scipy.__version__}")
    except Exception:
        pass

    try:
        from spatialgpu.core.backend import get_backend

        backend = get_backend()
        session_lines.append(f"GPU available: {backend.is_gpu_available}")
        session_lines.append(f"GPU active: {backend.is_gpu_active}")
    except Exception:
        session_lines.append("GPU: unknown")

    write_output("t8_session_info.txt", "\n".join(session_lines))
    for line in session_lines:
        print(f"   {line}")

    print("\nDone.")


if __name__ == "__main__":
    main()
