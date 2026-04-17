"""Run Tutorial 8 — Bulk Deconvolution with Matched scRNA-seq Reference.

Generates semi-synthetic scRNA-seq, splits it into TRAIN (deconvolution
reference) and TEST (pseudobulk source), runs ``deconvolution_bulk`` — a
thin wrapper around ``deconvolution_matched_scrnaseq`` — on the pseudobulk,
evaluates accuracy, and exports data for MuSiC/CIBERSORTx comparison.

The train/test split is performed **by cell** with a cell-type-stratified
shuffle, so no single cell appears in both the reference and the mixtures
(mirrors the fair-evaluation design of the SpaCET BRCA tutorial).

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


def _stratified_cell_split(adata, label_col="cell_type", train_frac=0.5, seed=0):
    """Stratified train/test cell split preserving cell-type balance."""
    import numpy as np

    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    labels = adata.obs[label_col].values
    for ct in np.unique(labels):
        idx = np.where(labels == ct)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * train_frac)
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())
    return np.array(train_idx), np.array(test_idx)


def _adata_to_sc_inputs(adata, label_col="cell_type"):
    """Convert an scRNA-seq AnnData to (sc_counts_df, sc_annotation, lineage_tree)
    as expected by ``deconvolution_matched_scrnaseq``.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    counts_T = X.T.astype(np.float64)  # genes x cells
    cell_ids = np.array(adata.obs_names, dtype=object)
    gene_names = np.array(adata.var_names, dtype=object)
    sc_counts = pd.DataFrame(counts_T, index=gene_names, columns=cell_ids)
    sc_annotation = pd.DataFrame(
        {"cellID": cell_ids, "cellType": adata.obs[label_col].values},
        index=cell_ids,
    )
    cell_types = sorted(adata.obs[label_col].unique().tolist())
    lineage_tree = {ct: [ct] for ct in cell_types}
    return sc_counts, sc_annotation, lineage_tree


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

    print("=== Tutorial 8: Bulk Deconvolution with Matched scRNA-seq ===\n")

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

    # ---- Step 2: Stratified train/test split (reference vs pseudobulk source) ----
    print("\n2. Splitting scRNA-seq into TRAIN (reference) / TEST (pseudobulk)...")
    brca_train_idx, brca_test_idx = _stratified_cell_split(
        scrna_brca, train_frac=0.5, seed=0
    )
    scrna_brca_train = scrna_brca[brca_train_idx].copy()
    scrna_brca_test = scrna_brca[brca_test_idx].copy()

    normal_train_idx, normal_test_idx = _stratified_cell_split(
        scrna_normal, train_frac=0.5, seed=0
    )
    scrna_normal_train = scrna_normal[normal_train_idx].copy()
    scrna_normal_test = scrna_normal[normal_test_idx].copy()
    print(
        f"   BRCA train: {scrna_brca_train.n_obs} | test: {scrna_brca_test.n_obs}"
    )
    print(
        f"   Normal train: {scrna_normal_train.n_obs} | "
        f"test: {scrna_normal_test.n_obs}"
    )

    sc_brca = _adata_to_sc_inputs(scrna_brca_train)
    sc_normal = _adata_to_sc_inputs(scrna_normal_train)

    # ---- Step 3: Generate pseudobulk — Dirichlet from TEST cells ----
    print("\n3. Generating pseudobulk (Dirichlet) from TEST cells...")
    bulk_brca, gt_brca = generate_pseudobulk_dirichlet(
        scrna_brca_test, n_samples=100, n_cells_per_sample=1000, alpha=1.0, seed=42
    )
    print(f"   BRCA: {bulk_brca.n_obs} samples, {bulk_brca.n_vars} genes")

    bulk_normal, gt_normal = generate_pseudobulk_dirichlet(
        scrna_normal_test,
        n_samples=100,
        n_cells_per_sample=1000,
        alpha=1.0,
        seed=42,
    )
    print(f"   Normal: {bulk_normal.n_obs} samples, {bulk_normal.n_vars} genes")

    # ---- Step 4: Generate pseudobulk — Titration ----
    print("\n4. Generating pseudobulk (Titration) from TEST cells...")
    bulk_titr, gt_titr = generate_pseudobulk_titration(
        scrna_brca_test,
        target_type="Malignant_BRCA",
        fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        n_replicates=5,
        n_cells_per_sample=1000,
        seed=42,
    )
    print(f"   Titration: {bulk_titr.n_obs} samples")

    # ---- Step 5: Deconvolution — BRCA (matched scRNA-seq from TRAIN) ----
    print("\n5. Deconvolution — BRCA (100 samples, matched reference)...")
    spacet.deconvolution_bulk(
        bulk_brca,
        sc_counts=sc_brca[0],
        sc_annotation=sc_brca[1],
        sc_lineage_tree=sc_brca[2],
        sc_include_malignant=True,
    )
    pm_brca = bulk_brca.uns["spacet"]["deconvolution"]["propMat"]
    print(f"   propMat: {pm_brca.shape[0]} types x {pm_brca.shape[1]} samples")

    # ---- Step 6: Deconvolution — Normal ----
    print("\n6. Deconvolution — Normal (100 samples, matched reference)...")
    spacet.deconvolution_bulk(
        bulk_normal,
        sc_counts=sc_normal[0],
        sc_annotation=sc_normal[1],
        sc_lineage_tree=sc_normal[2],
        sc_include_malignant=False,
    )
    pm_normal = bulk_normal.uns["spacet"]["deconvolution"]["propMat"]
    print(f"   propMat: {pm_normal.shape[0]} types x {pm_normal.shape[1]} samples")

    # ---- Step 7: Deconvolution — Titration ----
    print("\n7. Deconvolution — Titration (45 samples, matched reference)...")
    spacet.deconvolution_bulk(
        bulk_titr,
        sc_counts=sc_brca[0],
        sc_annotation=sc_brca[1],
        sc_lineage_tree=sc_brca[2],
        sc_include_malignant=True,
    )
    pm_titr = bulk_titr.uns["spacet"]["deconvolution"]["propMat"]
    print(f"   propMat: {pm_titr.shape[0]} types x {pm_titr.shape[1]} samples")

    # ---- Step 8: Evaluate accuracy ----
    print("\n8. Evaluating accuracy...")

    est_brca = _collapse_to_level1(pm_brca, _LEVEL1_TYPES + ["Malignant_BRCA"]).T
    est_normal = _collapse_to_level1(pm_normal, _LEVEL1_TYPES).T
    est_titr = _collapse_to_level1(pm_titr, _LEVEL1_TYPES + ["Malignant_BRCA"]).T

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

    # ---- Step 9: Generate figures ----
    print("\n9. Generating figures...")

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

    # ---- Step 10: Export for external tools (reference = TRAIN split) ----
    print("\n10. Exporting for external tools...")
    music_dir = os.path.join(OUTPUTS_DIR, "t8_export_music")
    cibersortx_dir = os.path.join(OUTPUTS_DIR, "t8_export_cibersortx")

    export_for_music(bulk_brca, scrna_brca_train, music_dir, gt_brca)
    print(f"   MuSiC export: {music_dir}")

    export_for_cibersortx(bulk_brca, scrna_brca_train, cibersortx_dir, gt_brca)
    print(f"   CIBERSORTx export: {cibersortx_dir}")

    # ---- Step 11: Session info ----
    print("\n11. Session info...")
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
