"""Tutorial 8b — Bulk Deconvolution Benchmark with Real BRCA scRNA-seq.

Uses Wu et al. 2021 (Nature Genetics, GSE176078) real BRCA single-cell
RNA-seq data to generate pseudobulk with known proportions. Tests
deconvolution accuracy with real malignant cells carrying actual CNA
expression patterns.

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


# Wu et al. -> SpaCET Level 1 mapping (many-to-one for evaluation)
# SpaCET deconvolution outputs fine-grained types; we collapse both
# ground truth and predictions to these matched categories.
WU_TO_SPACET = {
    "Cancer Epithelial": "Malignant",
    "CAFs": "CAF",
    "Endothelial": "Endothelial",
    "T-cells": "T_cells",  # collapsed: T CD4 + T CD8
    "B-cells": "B cell",
    "Plasmablasts": "Plasma",
    "Myeloid": "Myeloid",  # collapsed: Macrophage + cDC + pDC
    "PVL": "PVL",
    "Normal Epithelial": "Normal_Epithelial",
}

# SpaCET Level 1 types that map to collapsed Wu categories
SPACET_COLLAPSE = {
    "T_cells": ["T CD4", "T CD8", "NK"],  # NK often grouped with T in broad
    "Myeloid": ["Macrophage", "cDC", "pDC"],
}


def collapse_spacet_to_wu(prop_mat):
    """Collapse SpaCET propMat to match Wu et al. broad categories."""
    result = {}
    used = set()

    for wu_type, spacet_types in SPACET_COLLAPSE.items():
        cols = [c for c in spacet_types if c in prop_mat.index]
        if cols:
            result[wu_type] = prop_mat.loc[cols].sum(axis=0)
            used.update(cols)

    # Direct mappings
    for wu_name, spacet_name in [
        ("Malignant", "Malignant"),
        ("CAF", "CAF"),
        ("Endothelial", "Endothelial"),
        ("B cell", "B cell"),
        ("Plasma", "Plasma"),
    ]:
        if spacet_name in prop_mat.index:
            result[wu_name] = prop_mat.loc[spacet_name]
            used.add(spacet_name)

    # Remaining unmapped types go into "Other"
    unmapped = [c for c in prop_mat.index if c not in used]
    if unmapped:
        result["Other"] = prop_mat.loc[unmapped].sum(axis=0)

    return pd.DataFrame(result).T


def generate_pseudobulk_real(
    adata, n_samples, n_cells_per_sample, alpha, seed=42
):
    """Generate pseudobulk from real scRNA-seq with Dirichlet proportions."""
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    n_types = len(cell_types)

    type_indices = {
        ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types
    }

    if sparse.issparse(adata.X):
        X_all = adata.X.toarray()
    else:
        X_all = np.asarray(adata.X)

    bulk_counts = np.zeros((n_samples, adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, n_types), dtype=np.float64)

    alpha_vec = np.full(n_types, alpha) if isinstance(alpha, (int, float)) else np.array(alpha)

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


def generate_pseudobulk_tumor_realistic(
    adata, n_samples, n_cells_per_sample, tumor_fractions, seed=42
):
    """Generate pseudobulk with realistic tumor purity (60-90% malignant).

    For each sample, the malignant fraction is drawn from tumor_fractions,
    and the remaining fraction is distributed among non-malignant types
    via Dirichlet.
    """
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    mal_type = "Cancer Epithelial"
    nonmal_types = [ct for ct in cell_types if ct != mal_type]
    n_nonmal = len(nonmal_types)
    all_types = [mal_type] + nonmal_types

    type_indices = {
        ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types
    }

    if sparse.issparse(adata.X):
        X_all = adata.X.toarray()
    else:
        X_all = np.asarray(adata.X)

    bulk_counts = np.zeros((n_samples, adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, len(all_types)), dtype=np.float64)

    for i in range(n_samples):
        mal_frac = rng.choice(tumor_fractions)
        nonmal_props = rng.dirichlet(np.ones(n_nonmal))
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
    """Evaluate deconvolution accuracy."""
    from scipy.stats import pearsonr, spearmanr

    # Align columns
    common = sorted(set(est_df.columns) & set(gt_df.columns))
    if not common:
        print(f"   WARNING: No common types between estimation and ground truth!")
        print(f"   Est types: {list(est_df.columns)}")
        print(f"   GT types: {list(gt_df.columns)}")
        return ""

    est = est_df[common].values.ravel()
    gt = gt_df[common].values.ravel()

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

    per_type = []
    for ct in common:
        e = est_df[ct].values
        g = gt_df[ct].values
        ct_r, _ = pearsonr(e, g)
        ct_rmse = np.sqrt(np.mean((e - g) ** 2))
        per_type.append({"cell_type": ct, "pearson_r": ct_r, "rmse": ct_rmse, "n": len(e)})

    per_type_df = pd.DataFrame(per_type).set_index("cell_type")
    lines.append(per_type_df.to_string())

    result = "\n".join(lines)
    write_output(f"t8_real_{label}.txt", result)
    print(f"   {label}: r={r:.4f}, rho={rho:.4f}, RMSE={rmse:.4f}")
    return result


def main():
    import spatialgpu.deconvolution as spacet
    from spatialgpu.benchmarks.pseudobulk import _LEVEL1_TYPES, _collapse_to_level1

    print("=== Tutorial 8b: Real BRCA Pseudobulk Benchmark ===\n")

    # ---- Step 1: Load real BRCA scRNA-seq ----
    print("1. Loading Wu et al. 2021 BRCA scRNA-seq...")
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "BRCA_scRNA", "BRCA_scRNA_full.h5ad",
    )
    adata_sc = sc.read_h5ad(data_path)
    # Standardize cell type column (full h5ad uses 'celltype_major')
    if "celltype_major" in adata_sc.obs.columns and "cell_type" not in adata_sc.obs.columns:
        adata_sc.obs["cell_type"] = adata_sc.obs["celltype_major"]
    print(f"   {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    print(f"   Cell types:")
    for ct, n in adata_sc.obs["cell_type"].value_counts().items():
        print(f"     {ct}: {n}")

    # ---- Step 2: Scenario A — Uniform Dirichlet (diverse ratios) ----
    print("\n2. Scenario A: Uniform Dirichlet (alpha=1.0, 200 samples)...")
    bulk_a, gt_a = generate_pseudobulk_real(
        adata_sc, n_samples=200, n_cells_per_sample=2000, alpha=1.0, seed=42
    )
    print(f"   Pseudobulk: {bulk_a.n_obs} samples x {bulk_a.n_vars} genes")

    # ---- Step 3: Scenario B — Sparse Dirichlet (concentrated ratios) ----
    print("\n3. Scenario B: Sparse Dirichlet (alpha=0.3, 200 samples)...")
    bulk_b, gt_b = generate_pseudobulk_real(
        adata_sc, n_samples=200, n_cells_per_sample=2000, alpha=0.3, seed=43
    )

    # ---- Step 4: Scenario C — Realistic tumor purity (60-90%) ----
    print("\n4. Scenario C: Realistic tumor purity (60-90%, 200 samples)...")
    tumor_fracs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    bulk_c, gt_c = generate_pseudobulk_tumor_realistic(
        adata_sc, n_samples=200, n_cells_per_sample=2000,
        tumor_fractions=tumor_fracs, seed=44
    )

    # ---- Step 5: Scenario D — Titration of malignant cells ----
    print("\n5. Scenario D: Malignant titration (0-90%, 10 replicates)...")
    titration_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bulk_d, gt_d = generate_pseudobulk_tumor_realistic(
        adata_sc, n_samples=len(titration_fracs) * 10, n_cells_per_sample=2000,
        tumor_fractions=titration_fracs, seed=45
    )

    # ---- Step 5b: Export for MuSiC (R) ----
    print("\n5b. Exporting data for MuSiC (R)...")

    # Export scRNA-seq reference (subsample to keep file size manageable)
    rng_export = np.random.RandomState(99)
    max_cells = 3000  # 3K cells max for MuSiC reference
    if adata_sc.n_obs > max_cells:
        sub_idx = []
        for ct in adata_sc.obs["cell_type"].unique():
            ct_idx = np.where(adata_sc.obs["cell_type"].values == ct)[0]
            n = min(max_cells // adata_sc.obs["cell_type"].nunique(), len(ct_idx))
            sub_idx.extend(rng_export.choice(ct_idx, n, replace=False))
        sc_export = adata_sc[sorted(sub_idx)].copy()
    else:
        sc_export = adata_sc

    # scRNA-seq counts (genes x cells CSV)
    if sparse.issparse(sc_export.X):
        sc_dense = sc_export.X.toarray()
    else:
        sc_dense = np.asarray(sc_export.X)
    sc_df = pd.DataFrame(
        sc_dense.T,  # genes x cells
        index=sc_export.var_names,
        columns=sc_export.obs_names,
    )
    sc_df.to_csv(os.path.join(OUTPUTS_DIR, "t8_real_sc_counts.csv"))

    # scRNA-seq metadata
    sc_meta = pd.DataFrame({
        "cell_type": sc_export.obs["cell_type"].values,
        "subject_id": sc_export.obs["orig.ident"].values,
    }, index=sc_export.obs_names)
    sc_meta.to_csv(os.path.join(OUTPUTS_DIR, "t8_real_sc_meta.csv"))
    print(f"   Exported scRNA-seq: {sc_export.n_obs} cells, {sc_export.obs['orig.ident'].nunique()} subjects")

    # Export pseudobulk counts (samples x genes CSV) + ground truth
    for label, bulk, gt in [
        ("uniform", bulk_a, gt_a),
        ("sparse", bulk_b, gt_b),
        ("tumor_purity", bulk_c, gt_c),
        ("titration", bulk_d, gt_d),
    ]:
        if sparse.issparse(bulk.X):
            bulk_dense = bulk.X.toarray()
        else:
            bulk_dense = np.asarray(bulk.X)
        bulk_df = pd.DataFrame(bulk_dense, index=bulk.obs_names, columns=bulk.var_names)
        bulk_df.to_csv(os.path.join(OUTPUTS_DIR, f"t8_real_bulk_{label}.csv"))
        gt.to_csv(os.path.join(OUTPUTS_DIR, f"t8_real_gt_{label}.csv"))

    print("   Exported 4 scenarios (bulk counts + ground truth)")
    print("   Run: Rscript scripts/run_music_benchmark.R")

    # ---- Step 6: Run deconvolution ----
    print("\n6. Running deconvolution...")
    for label, bulk in [("A", bulk_a), ("B", bulk_b), ("C", bulk_c), ("D", bulk_d)]:
        print(f"   Scenario {label}...")
        spacet.deconvolution_bulk(bulk, cancer_type="BRCA")

    # ---- Step 7: Evaluate ----
    print("\n7. Evaluating accuracy...")

    for label, bulk, gt in [
        ("uniform", bulk_a, gt_a),
        ("sparse", bulk_b, gt_b),
        ("tumor_purity", bulk_c, gt_c),
        ("titration", bulk_d, gt_d),
    ]:
        pm = bulk.uns["deconv"]["propMat"]
        # Collapse SpaCET output to match Wu broad categories
        est_collapsed = collapse_spacet_to_wu(pm).T  # samples x types

        # Collapse ground truth Wu types using same mapping
        gt_renamed = gt.rename(columns=WU_TO_SPACET)
        # Aggregate columns with same target name
        gt_collapsed = gt_renamed.T.groupby(level=0).sum().T

        evaluate(est_collapsed, gt_collapsed, label)

    # ---- Step 8: Figures ----
    print("\n8. Generating figures...")

    # --- Scatter: Scenario A (uniform) ---
    pm_a = bulk_a.uns["deconv"]["propMat"]
    est_a = collapse_spacet_to_wu(pm_a).T
    gt_a_r = gt_a.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
    common_a = sorted(set(est_a.columns) & set(gt_a_r.columns))

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(range(len(common_a)))
    for i, ct in enumerate(common_a):
        ax.scatter(gt_a_r[ct], est_a[ct], alpha=0.5, s=15, label=ct, color=colors[i])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Ground Truth Fraction")
    ax.set_ylabel("Predicted Fraction")
    ax.set_title("Scenario A: Uniform Dirichlet (Real BRCA)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    from scipy.stats import pearsonr
    r_val, _ = pearsonr(est_a[common_a].values.ravel(), gt_a_r[common_a].values.ravel())
    ax.text(0.95, 0.05, f"r = {r_val:.3f}", transform=ax.transAxes, ha="right", fontsize=12)
    save(fig, "benchmark_real_brca_scatter_uniform.png")

    # --- Scatter: Scenario C (tumor purity) ---
    pm_c = bulk_c.uns["deconv"]["propMat"]
    est_c = collapse_spacet_to_wu(pm_c).T
    gt_c_r = gt_c.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
    common_c = sorted(set(est_c.columns) & set(gt_c_r.columns))

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, ct in enumerate(common_c):
        ax.scatter(gt_c_r[ct], est_c[ct], alpha=0.5, s=15, label=ct, color=colors[i % len(colors)])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Ground Truth Fraction")
    ax.set_ylabel("Predicted Fraction")
    ax.set_title("Scenario C: Realistic Tumor Purity (60-90%)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    r_val_c, _ = pearsonr(est_c[common_c].values.ravel(), gt_c_r[common_c].values.ravel())
    ax.text(0.95, 0.05, f"r = {r_val_c:.3f}", transform=ax.transAxes, ha="right", fontsize=12)
    save(fig, "benchmark_real_brca_scatter_tumor.png")

    # --- Titration curve ---
    pm_d = bulk_d.uns["deconv"]["propMat"]
    est_d = collapse_spacet_to_wu(pm_d).T
    gt_d_r = gt_d.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T

    if "Malignant" in est_d.columns and "Malignant" in gt_d_r.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(gt_d_r["Malignant"], est_d["Malignant"], alpha=0.6, s=20, c="#e74c3c")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("True Malignant Fraction")
        ax.set_ylabel("Predicted Malignant Fraction")
        ax.set_title("Malignant Cell Titration (Real BRCA)")
        r_titr, _ = pearsonr(gt_d_r["Malignant"], est_d["Malignant"])
        ax.text(0.95, 0.05, f"r = {r_titr:.3f}", transform=ax.transAxes, ha="right", fontsize=12)
        save(fig, "benchmark_real_brca_titration.png")

    # --- Per-type r across scenarios ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, bulk, gt) in zip(axes, [
        ("Uniform", bulk_a, gt_a),
        ("Sparse", bulk_b, gt_b),
        ("Tumor Purity", bulk_c, gt_c),
    ]):
        pm = bulk.uns["deconv"]["propMat"]
        est = collapse_spacet_to_wu(pm).T
        gt_r = gt.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
        common = sorted(set(est.columns) & set(gt_r.columns))

        rs = []
        for ct in common:
            r, _ = pearsonr(est[ct], gt_r[ct])
            rs.append(r)

        bars = ax.barh(common, rs, color="#3b82f6", alpha=0.8)
        ax.set_xlim(-0.1, 1.05)
        ax.set_xlabel("Pearson r")
        ax.set_title(label)
        ax.axvline(0, color="k", lw=0.5)
        for bar, r in zip(bars, rs):
            ax.text(max(r + 0.02, 0.05), bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    save(fig, "benchmark_real_brca_per_type.png")

    # ---- Step 9: Compare with MuSiC (if results available) ----
    print("\n9. Comparing with MuSiC (if available)...")
    from scipy.stats import pearsonr

    has_music = False
    for label, gt in [
        ("uniform", gt_a),
        ("sparse", gt_b),
        ("tumor_purity", gt_c),
        ("titration", gt_d),
    ]:
        music_file = os.path.join(OUTPUTS_DIR, f"t8_music_{label}.csv")
        if not os.path.exists(music_file):
            continue
        has_music = True

        music_props = pd.read_csv(music_file, index_col=0)
        gt_renamed = gt.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
        common = sorted(set(music_props.columns) & set(gt_renamed.columns))

        if common:
            r_music, _ = pearsonr(
                music_props[common].values.ravel(),
                gt_renamed.reindex(music_props.index)[common].values.ravel(),
            )
            print(f"   MuSiC {label}: r={r_music:.4f}")

    if not has_music:
        print("   MuSiC results not found. Run: Rscript scripts/run_music_benchmark.R")

    # ---- Step 10: Combined comparison figure ----
    if has_music:
        print("\n10. Generating SpaCET vs MuSiC comparison figure...")
        spacet_rs = []
        music_rs = []
        scenario_labels = []

        for label, bulk, gt in [
            ("uniform", bulk_a, gt_a),
            ("sparse", bulk_b, gt_b),
            ("tumor_purity", bulk_c, gt_c),
            ("titration", bulk_d, gt_d),
        ]:
            music_file = os.path.join(OUTPUTS_DIR, f"t8_music_{label}.csv")
            if not os.path.exists(music_file):
                continue

            # SpaCET
            pm = bulk.uns["deconv"]["propMat"]
            est_sp = collapse_spacet_to_wu(pm).T
            gt_r = gt.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
            common_sp = sorted(set(est_sp.columns) & set(gt_r.columns))
            r_sp, _ = pearsonr(est_sp[common_sp].values.ravel(), gt_r[common_sp].values.ravel())

            # MuSiC
            music_props = pd.read_csv(music_file, index_col=0)
            common_mu = sorted(set(music_props.columns) & set(gt_r.columns))
            r_mu, _ = pearsonr(
                music_props[common_mu].values.ravel(),
                gt_r.reindex(music_props.index)[common_mu].values.ravel(),
            )

            spacet_rs.append(r_sp)
            music_rs.append(r_mu)
            scenario_labels.append(label.replace("_", "\n"))

        if scenario_labels:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(scenario_labels))
            w = 0.35
            ax.bar(x - w / 2, spacet_rs, w, label="SpaCET (spatial-gpu)", color="#3b82f6")
            ax.bar(x + w / 2, music_rs, w, label="MuSiC", color="#f97316")
            ax.set_xticks(x)
            ax.set_xticklabels(scenario_labels)
            ax.set_ylabel("Pearson r (overall)")
            ax.set_title("SpaCET vs MuSiC — Real BRCA Pseudobulk")
            ax.set_ylim(0, 1.05)
            ax.legend()
            for i, (s, m) in enumerate(zip(spacet_rs, music_rs)):
                ax.text(i - w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
                ax.text(i + w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
            save(fig, "benchmark_real_brca_spacet_vs_music.png")

    # Session info
    print("\n11. Session info...")
    import spatialgpu
    session = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        f"Data: Wu et al. 2021 (GSE176078)",
        f"Cells: {adata_sc.n_obs}",
        f"Genes: {adata_sc.n_vars}",
        f"Cell types: {adata_sc.obs['cell_type'].nunique()}",
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
