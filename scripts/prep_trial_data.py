"""Generate per-trial pseudobulk + reference data for replicate-trial benchmarks.

For trial T (0..N-1):
  subject_split_seed = 42 + T          # different cohort split
  pseudobulk_uniform = 100 + T*4 + 0   # different bulk draws
  pseudobulk_sparse  = 100 + T*4 + 1
  pseudobulk_tumor   = 100 + T*4 + 2
  pseudobulk_titrate = 100 + T*4 + 3
  reference_subsamp  = 200 + T          # different 500/type subsample

Writes to docs/outputs/trials/T{NN}/:
  sc_counts.csv, sc_meta.csv          (500/type reference for that trial)
  bulk_<scenario>.csv                  (pseudobulk for that trial × scenario)
  gt_<scenario>.csv                    (ground truth proportions)
  sc_meta_minor.csv                    (minor-type meta for DWLS minor)
  minor_to_major.json                  (minor->major map)

Usage: python scripts/prep_trial_data.py --trial T [--n 10]
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from scipy import sparse  # noqa: E402

OUTPUTS_DIR = os.path.join(_REPO_ROOT, "docs", "outputs")
TRIALS_DIR = os.path.join(OUTPUTS_DIR, "trials")
DATA_PATH = os.path.join(_REPO_ROOT, "data", "BRCA_scRNA", "BRCA_scRNA_full.h5ad")

MIN_CELLS_PER_MINOR = 30


def trial_seeds(trial: int):
    """Map trial index -> all RandomState seeds used for that trial."""
    return {
        "subject_split": 42 + trial,
        "pseudobulk_uniform": 100 + trial * 4 + 0,
        "pseudobulk_sparse": 100 + trial * 4 + 1,
        "pseudobulk_tumor": 100 + trial * 4 + 2,
        "pseudobulk_titration": 100 + trial * 4 + 3,
        "reference_subsamp": 200 + trial,
    }


def generate_pseudobulk(adata, n_samples, n_cells_per_sample, alpha, seed):
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    n_types = len(cell_types)
    type_indices = {ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types}
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
    obs_idx = [f"Bulk_{i:04d}" for i in range(n_samples)]
    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=obs_idx),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    gt = pd.DataFrame(proportions, index=obs_idx, columns=cell_types)
    return adata_bulk, gt


def generate_pseudobulk_tumor(adata, n_samples, n_cells_per_sample, tumor_fractions, seed):
    rng = np.random.RandomState(seed)
    cell_types = sorted(adata.obs["cell_type"].unique())
    mal_type = "Cancer Epithelial"
    nonmal_types = [ct for ct in cell_types if ct != mal_type]
    all_types = [mal_type] + nonmal_types
    type_indices = {ct: np.where(adata.obs["cell_type"].values == ct)[0] for ct in cell_types}
    X_all = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    bulk_counts = np.zeros((n_samples, adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, len(all_types)), dtype=np.float64)
    for i in range(n_samples):
        mal_frac = rng.choice(tumor_fractions)
        nonmal_props = rng.dirichlet(np.ones(len(nonmal_types))) * (1 - mal_frac)
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
    obs_idx = [f"Bulk_{i:04d}" for i in range(n_samples)]
    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=obs_idx),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    gt = pd.DataFrame(proportions, index=obs_idx, columns=all_types)
    return adata_bulk, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True, help="Trial index (0-based).")
    args = ap.parse_args()
    seeds = trial_seeds(args.trial)
    trial_dir = os.path.join(TRIALS_DIR, f"T{args.trial:02d}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"=== Trial {args.trial} ===")
    print(f"  seeds: {seeds}")
    print(f"  out:   {trial_dir}")

    print(f"\n1. Loading {DATA_PATH}")
    adata_sc = sc.read_h5ad(DATA_PATH)
    if (
        "celltype_major" in adata_sc.obs.columns
        and "cell_type" not in adata_sc.obs.columns
    ):
        adata_sc.obs["cell_type"] = adata_sc.obs["celltype_major"]
    print(f"   {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")

    # Subject split
    subjects = sorted(adata_sc.obs["orig.ident"].unique())
    rng = np.random.RandomState(seeds["subject_split"])
    rng.shuffle(subjects)
    mid = len(subjects) // 2
    train_subjects = subjects[:mid]
    test_subjects = subjects[mid:]
    train_mask = adata_sc.obs["orig.ident"].isin(train_subjects)
    test_mask = adata_sc.obs["orig.ident"].isin(test_subjects)
    adata_train = adata_sc[train_mask].copy()
    adata_test = adata_sc[test_mask].copy()
    print(f"   Train: {adata_train.n_obs} cells / {len(train_subjects)} subj")
    print(f"   Test:  {adata_test.n_obs} cells / {len(test_subjects)} subj")

    # Reference subsample (500/type)
    rng_ref = np.random.RandomState(seeds["reference_subsamp"])
    sub_idx = []
    for ct in adata_train.obs["cell_type"].unique():
        ct_idx = np.where(adata_train.obs["cell_type"].values == ct)[0]
        n = min(500, len(ct_idx))
        sub_idx.extend(rng_ref.choice(ct_idx, n, replace=False))
    train_ref = adata_train[sorted(sub_idx)].copy()
    print(f"   Reference: {train_ref.n_obs} cells (500/type)")

    # Save sc_counts.csv (genes x cells) + sc_meta.csv
    counts = (
        train_ref.X.toarray() if sparse.issparse(train_ref.X) else np.asarray(train_ref.X)
    )
    sc_counts_df = pd.DataFrame(counts.T, index=train_ref.var_names, columns=train_ref.obs_names)
    sc_counts_df.to_csv(os.path.join(trial_dir, "sc_counts.csv"))
    sc_meta_df = pd.DataFrame(
        {
            "cell_type": train_ref.obs["cell_type"].values,
            "subject_id": train_ref.obs["orig.ident"].values,
        },
        index=train_ref.obs_names,
    )
    sc_meta_df.to_csv(os.path.join(trial_dir, "sc_meta.csv"))

    # Save sc_meta_minor.csv + minor_to_major.json (for DWLS minor)
    raw = ad.read_h5ad(DATA_PATH, backed="r")
    raw_obs = raw.obs[["celltype_major", "celltype_minor", "orig.ident"]].copy()
    raw_obs.columns = ["celltype_major", "celltype_minor", "subject_id"]
    minor_meta = raw_obs.loc[train_ref.obs_names].copy()
    minor_meta["celltype_minor"] = minor_meta["celltype_minor"].astype(str)
    minor_meta["celltype_major"] = minor_meta["celltype_major"].astype(str)
    counts_per_minor = minor_meta["celltype_minor"].value_counts()
    keep_minors = counts_per_minor[counts_per_minor >= MIN_CELLS_PER_MINOR].index
    minor_meta = minor_meta[minor_meta["celltype_minor"].isin(keep_minors)].copy()
    minor_meta.index.name = ""
    minor_meta.to_csv(os.path.join(trial_dir, "sc_meta_minor.csv"))
    minor_to_major = (
        minor_meta.groupby("celltype_minor")["celltype_major"]
        .agg(lambda s: s.mode().iat[0])
        .to_dict()
    )
    with open(os.path.join(trial_dir, "minor_to_major.json"), "w") as f:
        json.dump(minor_to_major, f, indent=2, sort_keys=True)
    print(f"   Minor types kept (>={MIN_CELLS_PER_MINOR}): {len(keep_minors)}")

    # Pseudobulk × 4 scenarios
    print("\n2. Generating pseudobulk:")
    bulk_a, gt_a = generate_pseudobulk(
        adata_test, n_samples=200, n_cells_per_sample=2000, alpha=1.0,
        seed=seeds["pseudobulk_uniform"],
    )
    bulk_b, gt_b = generate_pseudobulk(
        adata_test, n_samples=200, n_cells_per_sample=2000, alpha=0.3,
        seed=seeds["pseudobulk_sparse"],
    )
    bulk_c, gt_c = generate_pseudobulk_tumor(
        adata_test, n_samples=200, n_cells_per_sample=2000,
        tumor_fractions=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        seed=seeds["pseudobulk_tumor"],
    )
    bulk_d, gt_d = generate_pseudobulk_tumor(
        adata_test, n_samples=100, n_cells_per_sample=2000,
        tumor_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seed=seeds["pseudobulk_titration"],
    )
    for label, (b, g) in zip(
        ["uniform", "sparse", "tumor_purity", "titration"],
        [(bulk_a, gt_a), (bulk_b, gt_b), (bulk_c, gt_c), (bulk_d, gt_d)],
    ):
        bulk_dense = b.X.toarray() if sparse.issparse(b.X) else np.asarray(b.X)
        pd.DataFrame(bulk_dense, index=b.obs_names, columns=b.var_names).to_csv(
            os.path.join(trial_dir, f"bulk_{label}.csv")
        )
        g.to_csv(os.path.join(trial_dir, f"gt_{label}.csv"))
        print(f"   {label}: {b.n_obs} samples")

    # Stash the seeds for traceability
    with open(os.path.join(trial_dir, "seeds.json"), "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"\nDone. Trial files in {trial_dir}")


if __name__ == "__main__":
    main()
