"""Tutorial 8b — Fair Bulk Deconvolution Benchmark: SpaCET vs MuSiC.

Uses Wu et al. 2021 (Nature Genetics, GSE176078) real BRCA single-cell
RNA-seq data. Splits by SUBJECT into reference (train) and pseudobulk (test)
sets so neither method sees the cells that generated the bulk mixtures.

SpaCET uses deconvolution_matched_scrnaseq with the train-set reference.
MuSiC uses the same train-set scRNA-seq as its reference.
Pseudobulk is generated from test-set cells only.

Requires: data/BRCA_scRNA/BRCA_scRNA_full.h5ad (from slurm_download_brca_scrna.sh)

Usage: python docs/run_full_tutorial_t8_real_brca.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

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


# Wu et al. -> collapsed category mapping for evaluation
WU_TO_EVAL = {
    "Cancer Epithelial": "Malignant",
    "CAFs": "CAF",
    "Endothelial": "Endothelial",
    "T-cells": "T_cells",
    "B-cells": "B cell",
    "Plasmablasts": "Plasma",
    "Myeloid": "Myeloid",
    "PVL": "PVL",
    "Normal Epithelial": "Normal_Epithelial",
}

# SpaCET fine types that collapse to Wu broad categories
SPACET_TO_EVAL = {
    "T CD4": "T_cells",
    "T CD8": "T_cells",
    "NK": "T_cells",
    "Macrophage": "Myeloid",
    "cDC": "Myeloid",
    "pDC": "Myeloid",
    "Macrophage other": "Myeloid",
    "CAF": "CAF",
    "Endothelial": "Endothelial",
    "B cell": "B cell",
    "Plasma": "Plasma",
    "Mast": "Other",
    "Neutrophil": "Other",
    "Unidentifiable": "Other",
    "Malignant": "Malignant",
}


def collapse_spacet_propmat(prop_mat):
    """Collapse SpaCET propMat to evaluation categories."""
    result = {}
    for spacet_type in prop_mat.index:
        eval_type = SPACET_TO_EVAL.get(spacet_type, "Other")
        if eval_type not in result:
            result[eval_type] = prop_mat.loc[spacet_type].values.copy()
        else:
            result[eval_type] += prop_mat.loc[spacet_type].values
    df = pd.DataFrame(result, index=prop_mat.columns)  # samples x types
    return df


def generate_pseudobulk(adata, n_samples, n_cells_per_sample, alpha, seed=42):
    """Generate pseudobulk from scRNA-seq with Dirichlet proportions."""
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    n_types = len(cell_types)

    type_indices = {
        ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types
    }

    X_all = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    bulk_counts = np.zeros((n_samples, adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, n_types), dtype=np.float64)
    alpha_vec = np.full(n_types, alpha)

    for i in range(n_samples):
        props = rng.dirichlet(alpha_vec)
        proportions[i] = props
        cell_counts = rng.multinomial(n_cells_per_sample, props)

        sample_sum = np.zeros(adata.n_vars, dtype=np.float64)
        for j, ct in enumerate(cell_types):
            if cell_counts[j] == 0:
                continue
            idx = rng.choice(type_indices[ct], size=cell_counts[j], replace=True)
            sample_sum += X_all[idx].sum(axis=0)
        bulk_counts[i] = sample_sum

    import anndata as ad

    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=[f"Bulk_{i:04d}" for i in range(n_samples)]),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    ground_truth = pd.DataFrame(
        proportions, index=adata_bulk.obs_names, columns=cell_types
    )
    return adata_bulk, ground_truth


def generate_pseudobulk_tumor(
    adata, n_samples, n_cells_per_sample, tumor_fractions, seed=42
):
    """Generate pseudobulk with fixed malignant fraction, Dirichlet for rest."""
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    mal_type = "Cancer Epithelial"
    nonmal_types = [ct for ct in cell_types if ct != mal_type]
    all_types = [mal_type] + nonmal_types

    type_indices = {
        ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types
    }

    X_all = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    bulk_counts = np.zeros((n_samples, adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, len(all_types)), dtype=np.float64)

    for i in range(n_samples):
        mal_frac = rng.choice(tumor_fractions)
        nonmal_props = rng.dirichlet(np.ones(len(nonmal_types)))
        nonmal_props *= 1 - mal_frac

        props = np.zeros(len(all_types))
        props[0] = mal_frac
        props[1:] = nonmal_props
        proportions[i] = props

        cell_counts = rng.multinomial(n_cells_per_sample, props)
        sample_sum = np.zeros(adata.n_vars, dtype=np.float64)
        for j, ct in enumerate(all_types):
            if cell_counts[j] == 0:
                continue
            idx = rng.choice(type_indices[ct], size=cell_counts[j], replace=True)
            sample_sum += X_all[idx].sum(axis=0)
        bulk_counts[i] = sample_sum

    import anndata as ad

    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=[f"Bulk_{i:04d}" for i in range(n_samples)]),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    ground_truth = pd.DataFrame(
        proportions, index=adata_bulk.obs_names, columns=all_types
    )
    return adata_bulk, ground_truth


def evaluate(est_df, gt_df, label):
    """Evaluate deconvolution accuracy. Returns (r, rho, rmse)."""
    common = sorted(set(est_df.columns) & set(gt_df.columns))
    if not common:
        print(f"   WARNING [{label}]: No common types!")
        return np.nan, np.nan, np.nan

    gt_aligned = gt_df.reindex(est_df.index)[common]
    est = est_df[common].values.ravel()
    gt = gt_aligned.values.ravel()

    r, _ = pearsonr(est, gt)
    rho, _ = spearmanr(est, gt)
    rmse = np.sqrt(np.mean((est - gt) ** 2))

    lines = [
        f"Overall Pearson r:    {r:.4f}",
        f"Overall Spearman rho: {rho:.4f}",
        f"Overall RMSE:         {rmse:.4f}",
        "",
        "Per-cell-type:",
    ]
    for ct in common:
        ct_r, _ = pearsonr(est_df[ct].values, gt_aligned[ct].values)
        ct_rmse = np.sqrt(np.mean((est_df[ct].values - gt_aligned[ct].values) ** 2))
        lines.append(f"  {ct:25s}  r={ct_r:.4f}  RMSE={ct_rmse:.4f}")

    write_output(f"t8_real_{label}.txt", "\n".join(lines))
    print(f"   {label}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.4f}")
    return r, rho, rmse


def main():
    import spatialgpu.deconvolution as spacet

    print("=== Tutorial 8b: Fair BRCA Benchmark (Subject-Split) ===\n")

    # ---- Step 1: Load data ----
    print("1. Loading Wu et al. 2021 BRCA scRNA-seq...")
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "BRCA_scRNA",
        "BRCA_scRNA_full.h5ad",
    )
    adata_sc = sc.read_h5ad(data_path)
    if (
        "celltype_major" in adata_sc.obs.columns
        and "cell_type" not in adata_sc.obs.columns
    ):
        adata_sc.obs["cell_type"] = adata_sc.obs["celltype_major"]
    print(f"   {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    print(f"   Subjects: {adata_sc.obs['orig.ident'].nunique()}")

    # ---- Step 2: Subject-level train/test split ----
    print("\n2. Splitting by subject (50/50)...")
    subjects = sorted(adata_sc.obs["orig.ident"].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(subjects)
    mid = len(subjects) // 2
    train_subjects = subjects[:mid]
    test_subjects = subjects[mid:]

    train_mask = adata_sc.obs["orig.ident"].isin(train_subjects)
    test_mask = adata_sc.obs["orig.ident"].isin(test_subjects)

    adata_train = adata_sc[train_mask].copy()
    adata_test = adata_sc[test_mask].copy()

    print(f"   Train: {adata_train.n_obs} cells, {len(train_subjects)} subjects")
    print(f"     {adata_train.obs['cell_type'].value_counts().to_dict()}")
    print(f"   Test:  {adata_test.n_obs} cells, {len(test_subjects)} subjects")
    print(f"     {adata_test.obs['cell_type'].value_counts().to_dict()}")

    # ---- Step 3: Generate pseudobulk from TEST set only ----
    print("\n3. Generating pseudobulk from TEST subjects...")

    scenarios = {}

    bulk_a, gt_a = generate_pseudobulk(
        adata_test, n_samples=200, n_cells_per_sample=2000, alpha=1.0, seed=42
    )
    scenarios["uniform"] = (bulk_a, gt_a, "Uniform (alpha=1.0)")
    print(f"   Uniform: {bulk_a.n_obs} samples")

    bulk_b, gt_b = generate_pseudobulk(
        adata_test, n_samples=200, n_cells_per_sample=2000, alpha=0.3, seed=43
    )
    scenarios["sparse"] = (bulk_b, gt_b, "Sparse (alpha=0.3)")
    print(f"   Sparse: {bulk_b.n_obs} samples")

    bulk_c, gt_c = generate_pseudobulk_tumor(
        adata_test,
        n_samples=200,
        n_cells_per_sample=2000,
        tumor_fractions=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        seed=44,
    )
    scenarios["tumor_purity"] = (bulk_c, gt_c, "Tumor Purity (60-90%)")
    print(f"   Tumor purity: {bulk_c.n_obs} samples")

    bulk_d, gt_d = generate_pseudobulk_tumor(
        adata_test,
        n_samples=100,
        n_cells_per_sample=2000,
        tumor_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seed=45,
    )
    scenarios["titration"] = (bulk_d, gt_d, "Titration (0-90%)")
    print(f"   Titration: {bulk_d.n_obs} samples")

    # ---- Step 4: Export for MuSiC (uses TRAIN set as reference) ----
    print("\n4. Exporting TRAIN set for MuSiC (R)...")

    # Subsample train set for manageable export
    rng_export = np.random.RandomState(99)
    max_per_type = 500
    sub_idx = []
    for ct in adata_train.obs["cell_type"].unique():
        ct_idx = np.where(adata_train.obs["cell_type"].values == ct)[0]
        n = min(max_per_type, len(ct_idx))
        sub_idx.extend(rng_export.choice(ct_idx, n, replace=False))
    train_export = adata_train[sorted(sub_idx)].copy()

    # scRNA-seq counts (genes x cells)
    sc_dense = (
        train_export.X.toarray()
        if sparse.issparse(train_export.X)
        else np.asarray(train_export.X)
    )
    sc_df = pd.DataFrame(
        sc_dense.T, index=train_export.var_names, columns=train_export.obs_names
    )
    sc_df.to_csv(os.path.join(OUTPUTS_DIR, "t8_real_sc_counts.csv"))

    sc_meta = pd.DataFrame(
        {
            "cell_type": train_export.obs["cell_type"].values,
            "subject_id": train_export.obs["orig.ident"].values,
        },
        index=train_export.obs_names,
    )
    sc_meta.to_csv(os.path.join(OUTPUTS_DIR, "t8_real_sc_meta.csv"))
    print(
        f"   Exported: {train_export.n_obs} cells, {train_export.obs['orig.ident'].nunique()} subjects"
    )

    # Export pseudobulk + ground truth
    for label, (bulk, gt, _) in scenarios.items():
        bulk_dense = bulk.X.toarray() if sparse.issparse(bulk.X) else np.asarray(bulk.X)
        pd.DataFrame(bulk_dense, index=bulk.obs_names, columns=bulk.var_names).to_csv(
            os.path.join(OUTPUTS_DIR, f"t8_real_bulk_{label}.csv")
        )
        gt.to_csv(os.path.join(OUTPUTS_DIR, f"t8_real_gt_{label}.csv"))
    print("   Exported 4 scenarios")

    # ---- Step 5: SpaCET deconvolution with matched scRNA-seq (TRAIN set) ----
    print("\n5. SpaCET deconvolution (matched scRNA-seq from TRAIN set)...")

    # Build reference inputs for deconvolution_matched_scrnaseq
    train_counts_df = pd.DataFrame(
        (
            adata_train.X.toarray()
            if sparse.issparse(adata_train.X)
            else np.asarray(adata_train.X)
        ),
        index=adata_train.obs_names,
        columns=adata_train.var_names,
    ).T  # genes x cells

    train_annotation = pd.DataFrame(
        {
            "cellID": adata_train.obs_names,
            "cellType": adata_train.obs["cell_type"].values,
        }
    )

    # Build lineage tree from Wu cell types
    lineage_tree = {
        "Cancer Epithelial": ["Cancer Epithelial"],
        "CAFs": ["CAFs"],
        "Endothelial": ["Endothelial"],
        "T-cells": ["T-cells"],
        "B-cells": ["B-cells"],
        "Plasmablasts": ["Plasmablasts"],
        "Myeloid": ["Myeloid"],
        "PVL": ["PVL"],
        "Normal Epithelial": ["Normal Epithelial"],
    }

    # Add subject_id to annotation for cross-subject weighting
    train_annotation_with_subj = pd.DataFrame(
        {
            "cellID": adata_train.obs_names,
            "cellType": adata_train.obs["cell_type"].values,
            "subject_id": adata_train.obs["orig.ident"].values,
        }
    )

    # --- 5a: SpaCET without weighting ---
    print("   --- Without cross-subject weighting ---")
    spacet_results = {}
    for label, (bulk, _gt, desc) in scenarios.items():
        print(f"   {desc}...")
        try:
            # Need a fresh copy since deconvolution modifies adata in-place
            bulk_copy = bulk.copy()
            spacet.deconvolution_matched_scrnaseq(
                bulk_copy,
                sc_counts=train_counts_df.copy(),
                sc_annotation=train_annotation,
                sc_lineage_tree=lineage_tree,
                sc_include_malignant=True,
                sc_downsampling=True,
                sc_n_cell_each_lineage=200,
            )
            pm = bulk_copy.uns["spacet"]["deconvolution"]["propMat"]
            est = pm.T.rename(columns=WU_TO_EVAL)
            est = est.T.groupby(level=0).sum().T
            spacet_results[label] = est
        except Exception as e:
            print(f"   ERROR in {label}: {e}")
            spacet_results[label] = None

    # --- 5b: SpaCET WITH cross-subject weighting ---
    print("\n   --- With cross-subject weighting (MuSiC-style) ---")
    spacet_weighted_results = {}
    for label, (bulk, _gt, desc) in scenarios.items():
        print(f"   {desc}...")
        try:
            bulk_copy = bulk.copy()
            spacet.deconvolution_matched_scrnaseq(
                bulk_copy,
                sc_counts=train_counts_df.copy(),
                sc_annotation=train_annotation_with_subj,
                sc_lineage_tree=lineage_tree,
                sc_include_malignant=True,
                sc_downsampling=True,
                sc_n_cell_each_lineage=200,
                cross_subject_weighting=True,
                subject_col="subject_id",
            )
            pm = bulk_copy.uns["spacet"]["deconvolution"]["propMat"]
            est = pm.T.rename(columns=WU_TO_EVAL)
            est = est.T.groupby(level=0).sum().T
            spacet_weighted_results[label] = est
        except Exception as e:
            print(f"   ERROR in {label}: {e}")
            spacet_weighted_results[label] = None

    # ---- Step 6: Evaluate SpaCET (both variants) ----
    print("\n6. Evaluating SpaCET...")
    spacet_metrics = {}
    for label, (_, gt, _) in scenarios.items():
        est = spacet_results.get(label)
        if est is None:
            continue
        gt_eval = gt.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T
        r, rho, rmse = evaluate(est, gt_eval, f"spacet_{label}")
        spacet_metrics[label] = {"r": r, "rho": rho, "rmse": rmse}

    print("\n   Evaluating SpaCET + cross-subject weighting...")
    spacet_w_metrics = {}
    for label, (_, gt, _) in scenarios.items():
        est = spacet_weighted_results.get(label)
        if est is None:
            continue
        gt_eval = gt.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T
        r, rho, rmse = evaluate(est, gt_eval, f"spacet_weighted_{label}")
        spacet_w_metrics[label] = {"r": r, "rho": rho, "rmse": rmse}

    # ---- Step 7: Figures ----
    print("\n7. Generating SpaCET figures...")

    for label, (_, gt, desc) in scenarios.items():
        est = spacet_results.get(label)
        if est is None:
            continue
        gt_eval = gt.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T
        common = sorted(set(est.columns) & set(gt_eval.columns))

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.tab10(range(len(common)))
        for i, ct in enumerate(common):
            ax.scatter(gt_eval[ct], est[ct], alpha=0.5, s=15, label=ct, color=colors[i])
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Ground Truth Fraction")
        ax.set_ylabel("Predicted Fraction")
        ax.set_title(f"SpaCET — {desc}")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
        r_val = spacet_metrics.get(label, {}).get("r", 0)
        ax.text(
            0.95,
            0.05,
            f"r = {r_val:.3f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=12,
        )
        save(fig, f"benchmark_real_brca_scatter_{label}.png")

    # ---- Step 8: Check MuSiC results ----
    print("\n8. Checking MuSiC results...")
    music_metrics = {}
    has_music = False
    for label, (_, gt, _desc) in scenarios.items():
        music_file = os.path.join(OUTPUTS_DIR, f"t8_music_{label}.csv")
        if not os.path.exists(music_file):
            continue
        has_music = True
        music_props = pd.read_csv(music_file, index_col=0)
        # MuSiC uses Wu cell type names directly
        gt_eval = gt.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T
        music_eval = music_props.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T
        common = sorted(set(music_eval.columns) & set(gt_eval.columns))
        if common:
            r, _ = pearsonr(
                music_eval[common].values.ravel(),
                gt_eval.reindex(music_eval.index)[common].values.ravel(),
            )
            rho, _ = spearmanr(
                music_eval[common].values.ravel(),
                gt_eval.reindex(music_eval.index)[common].values.ravel(),
            )
            rmse = np.sqrt(
                np.mean(
                    (
                        music_eval[common].values.ravel()
                        - gt_eval.reindex(music_eval.index)[common].values.ravel()
                    )
                    ** 2
                )
            )
            music_metrics[label] = {"r": r, "rho": rho, "rmse": rmse}
            print(f"   MuSiC {label}: r={r:.4f}")

    if not has_music:
        print(
            "   MuSiC results not found. Run: sbatch scripts/slurm_music_benchmark.sh"
        )

    # ---- Step 9: Comparison figure (3 methods) ----
    print("\n9. Generating comparison figure...")
    common_labels = sorted(set(spacet_metrics) & set(spacet_w_metrics))
    if common_labels:
        methods = [
            ("SpaCET", spacet_metrics, "#3b82f6"),
            ("SpaCET + weighting", spacet_w_metrics, "#10b981"),
        ]
        if has_music:
            common_labels = sorted(set(common_labels) & set(music_metrics))
            methods.append(("MuSiC", music_metrics, "#f97316"))

        n_methods = len(methods)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(common_labels))
        w = 0.8 / n_methods

        for j, (method_name, metrics, color) in enumerate(methods):
            offset = (j - n_methods / 2 + 0.5) * w
            rs = [metrics.get(lab, {}).get("r", 0) for lab in common_labels]
            ax.bar(x + offset, rs, w, label=method_name, color=color)
            for i, r in enumerate(rs):
                ax.text(x[i] + offset, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([scenarios[lab][2] for lab in common_labels], fontsize=8)
        ax.set_ylabel("Pearson r (overall)")
        ax.set_title("Deconvolution Benchmark — Fair Subject-Split (Wu et al. BRCA)")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.tight_layout()
        save(fig, "benchmark_spacet_vs_music.png")

        # Summary table
        rows = []
        for lab in common_labels:
            row = {"Scenario": scenarios[lab][2]}
            for method_name, metrics, _ in methods:
                row[f"{method_name}_r"] = metrics.get(lab, {}).get("r", np.nan)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(
            os.path.join(OUTPUTS_DIR, "t8_comparison_summary.csv"), index=False
        )
        print("\n" + summary.to_string(index=False))

    # ---- Session info ----
    print("\n10. Session info...")
    import spatialgpu

    session = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        "Data: Wu et al. 2021 (GSE176078)",
        f"Total cells: {adata_sc.n_obs}",
        f"Train subjects: {train_subjects}",
        f"Test subjects: {test_subjects}",
        f"Train cells: {adata_train.n_obs}",
        f"Test cells: {adata_test.n_obs}",
    ]
    try:
        import cupy

        session.append(f"cupy: {cupy.__version__} (GPU)")
    except ImportError:
        session.append("cupy: not installed (CPU)")
    write_output("t8_real_session_info.txt", "\n".join(session))

    print("\n=== Tutorial 8b complete! ===")


if __name__ == "__main__":
    main()
