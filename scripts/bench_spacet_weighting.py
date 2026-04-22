"""Benchmark SpaCET cross-subject weighting variants.

Runs the SpaCET matched-scRNAseq deconvolution with one `weighting_method`
across the four T8 pseudobulk scenarios, saves prediction CSVs tagged by
variant, and writes per-scenario overall + per-cell-type metrics.

Variants (set via --method):
  v0_none      : no weighting (equivalent to SpaCET default)
  v1_ratio     : w = 1 / (1 + var_between/var_within) — SpaCET's current default
  v3_irwls     : IRWLS-lite (one residual-reweighting pass; production default
                 when cross_subject_weighting=True is the recommended choice)

(Exploratory variants dropped after benchmarking on T8 BRCA:
  V2 absolute   — hurts r by ~−0.20 across all scenarios; over-suppresses
                  discriminative high-variance markers.
  V4 per-type   — averaged-per-type degenerates to V2 numerically; proper
                  per-lineage weighting deferred.
  V3 multi-iter — additional IRWLS iterations hurt tumor-dominated scenarios
                  (the under-prediction is a hierarchical-constraint artifact
                  that residual reweighting cannot fix).)

Outputs: docs/outputs/t8_spacet_{variant}_{scenario}.csv
         docs/outputs/t8_spacet_{variant}_{scenario}.txt

Usage: python scripts/bench_spacet_weighting.py --method v2_absolute
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from _t8_common import compute_method_r, remap_and_collapse  # noqa: E402
from scipy import sparse  # noqa: E402

OUTPUTS_DIR = os.path.join(_REPO_ROOT, "docs", "outputs")

# Wu -> SpaCET fine type collapser (same as tutorial)
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


def generate_pseudobulk(adata, n_samples, n_cells_per_sample, alpha, seed=42):
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
    gt = pd.DataFrame(proportions, index=adata_bulk.obs_names, columns=cell_types)
    return adata_bulk, gt


def generate_pseudobulk_tumor(adata, n_samples, n_cells_per_sample, tumor_fractions, seed=42):
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
    gt = pd.DataFrame(proportions, index=adata_bulk.obs_names, columns=all_types)
    return adata_bulk, gt


def collapse_spacet_propmat(prop_mat):
    result = {}
    for spacet_type in prop_mat.index:
        eval_type = SPACET_TO_EVAL.get(spacet_type, "Other")
        if eval_type not in result:
            result[eval_type] = prop_mat.loc[spacet_type].values.copy()
        else:
            result[eval_type] += prop_mat.loc[spacet_type].values
    return pd.DataFrame(result, index=prop_mat.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=["v0_none", "v1_ratio", "v3_irwls"],
        help="Weighting variant to benchmark.",
    )
    args = ap.parse_args()

    # Map CLI label -> (cross_subject_weighting, weighting_method)
    _METHOD_MAP = {
        "v0_none":  (False, "ratio"),
        "v1_ratio": (True,  "ratio"),
        "v3_irwls": (True,  "irwls"),
    }
    csw, wm = _METHOD_MAP[args.method]

    print(f"=== SpaCET weighting benchmark — variant={args.method} ===")
    print(f"  cross_subject_weighting={csw}  weighting_method={wm}\n")

    import spatialgpu.deconvolution as spacet

    data_path = os.path.join(_REPO_ROOT, "data", "BRCA_scRNA", "BRCA_scRNA_full.h5ad")
    print(f"1. Loading scRNA-seq: {data_path}")
    adata_sc = sc.read_h5ad(data_path)
    if (
        "celltype_major" in adata_sc.obs.columns
        and "cell_type" not in adata_sc.obs.columns
    ):
        adata_sc.obs["cell_type"] = adata_sc.obs["celltype_major"]
    print(f"   {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")

    # Subject split (same seed as tutorial)
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
    print(f"   Train: {adata_train.n_obs} cells / {len(train_subjects)} subjects")
    print(f"   Test:  {adata_test.n_obs} cells / {len(test_subjects)} subjects")

    # Option A (fair): 500 cells/type subset — same reference size as MuSiC/DWLS
    # in the committed benchmark. Isolates the weighting-formula effect on an
    # equal-footing SpaCET-vs-MuSiC-vs-DWLS comparison.
    rng_export = np.random.RandomState(99)
    sub_idx = []
    for ct in adata_train.obs["cell_type"].unique():
        ct_idx = np.where(adata_train.obs["cell_type"].values == ct)[0]
        n = min(500, len(ct_idx))
        sub_idx.extend(rng_export.choice(ct_idx, n, replace=False))
    train_ref = adata_train[sorted(sub_idx)].copy()
    print(f"   SpaCET reference: {train_ref.n_obs} cells (500/type subset)")

    sc_counts_df = pd.DataFrame(
        train_ref.X.toarray() if sparse.issparse(train_ref.X) else np.asarray(train_ref.X),
        index=train_ref.obs_names,
        columns=train_ref.var_names,
    ).T  # genes x cells

    sc_annotation = pd.DataFrame(
        {
            "cellID": train_ref.obs_names,
            "cellType": train_ref.obs["cell_type"].values,
            "subject_id": train_ref.obs["orig.ident"].values,
        },
        index=train_ref.obs_names,
    )

    lineage_tree = {ct: [ct] for ct in sorted(train_ref.obs["cell_type"].unique())}

    # Generate 4 scenarios (same seeds as tutorial)
    print("\n2. Generating pseudobulk scenarios...")
    scenarios = {}
    bulk, gt = generate_pseudobulk(adata_test, 200, 2000, alpha=1.0, seed=42)
    scenarios["uniform"] = (bulk, gt, "Uniform (alpha=1.0)")
    bulk, gt = generate_pseudobulk(adata_test, 200, 2000, alpha=0.3, seed=43)
    scenarios["sparse"] = (bulk, gt, "Sparse (alpha=0.3)")
    bulk, gt = generate_pseudobulk_tumor(
        adata_test, 200, 2000, tumor_fractions=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], seed=44,
    )
    scenarios["tumor_purity"] = (bulk, gt, "Tumor Purity (60-90%)")
    bulk, gt = generate_pseudobulk_tumor(
        adata_test, 100, 2000,
        tumor_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seed=45,
    )
    scenarios["titration"] = (bulk, gt, "Titration (0-90%)")
    for lbl, (b, _, _) in scenarios.items():
        print(f"   {lbl}: {b.n_obs} samples")

    # Run SpaCET per scenario
    print(f"\n3. SpaCET deconvolution (variant={args.method})...")
    out_rows = []
    for label, (bulk, gt_raw, desc) in scenarios.items():
        print(f"\n   --- {desc} ---")
        bulk_copy = bulk.copy()
        kwargs = {
            "sc_counts": sc_counts_df.copy(),
            "sc_annotation": sc_annotation,
            "sc_lineage_tree": lineage_tree,
            "sc_include_malignant": True,
            "sc_downsampling": True,
            "sc_n_cell_each_lineage": 200,
        }
        if csw:
            kwargs["cross_subject_weighting"] = True
            kwargs["subject_col"] = "subject_id"
            kwargs["weighting_method"] = wm
        spacet.deconvolution_matched_scrnaseq(bulk_copy, **kwargs)
        pm = bulk_copy.uns["spacet"]["deconvolution"]["propMat"]
        est = remap_and_collapse(pm.T)

        # Save prediction CSV tagged by variant
        out_csv = os.path.join(OUTPUTS_DIR, f"t8_spacet_{args.method}_{label}.csv")
        est.to_csv(out_csv)

        # Evaluate (overall + per-cell-type)
        gt_eval = remap_and_collapse(gt_raw)
        common = sorted(set(est.columns) & set(gt_eval.columns))
        gt_aligned = gt_eval.reindex(est.index)[common]
        metrics = compute_method_r(est, gt_eval)

        lines = [
            f"Variant: {args.method}",
            f"Scenario: {desc}",
            f"Overall Pearson r:    {metrics['r']:.4f}",
            f"Overall Spearman rho: {metrics['rho']:.4f}",
            f"Overall RMSE:         {metrics['rmse']:.4f}",
            "",
            "Per-cell-type:",
        ]
        from scipy.stats import pearsonr
        for ct in common:
            e = est[ct].values
            g = gt_aligned[ct].values
            ct_r, _ = pearsonr(e, g)
            bias = (e - g).mean()
            vr = e.var() / g.var() if g.var() > 0 else np.nan
            ct_rmse = np.sqrt(np.mean((e - g) ** 2))
            lines.append(
                f"  {ct:20s}  r={ct_r:.4f}  bias={bias:+.4f}  varR={vr:.3f}  RMSE={ct_rmse:.4f}"
            )
            out_rows.append(
                {
                    "variant": args.method,
                    "scenario": label,
                    "cell_type": ct,
                    "r": ct_r,
                    "bias": bias,
                    "var_ratio": vr,
                    "rmse": ct_rmse,
                }
            )

        out_txt = os.path.join(OUTPUTS_DIR, f"t8_spacet_{args.method}_{label}.txt")
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))
        print(
            f"   r={metrics['r']:.4f}  rho={metrics['rho']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}  -> {out_csv}"
        )

    # Aggregate long-form table across scenarios
    long_df = pd.DataFrame(out_rows)
    long_path = os.path.join(OUTPUTS_DIR, f"t8_spacet_{args.method}_per_type.csv")
    long_df.to_csv(long_path, index=False)
    print(f"\nSaved per-type table: {long_path}")


if __name__ == "__main__":
    main()
