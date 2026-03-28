#!/usr/bin/env python3
"""Validate sublineage deconvolution concordance between R SpaCET and spatial-gpu.

Loads R SpaCET deconvolution outputs (malProp, counts) and runs the Python
_spatial_deconv_python() with the same inputs. Compares Level 1 (major lineage)
and Level 2 (sublineage) proportions.

Usage:
    python scripts/validate_sublineage_concordance.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/sublineage_concordance")


def load_r_outputs():
    """Load all R SpaCET intermediate outputs."""
    logger.info("Loading R outputs from %s", DATA_DIR)

    # Full propMat
    propMat_full = pd.read_csv(DATA_DIR / "r_propMat_full.csv", index_col=0)
    logger.info("R propMat: %s", propMat_full.shape)

    # Level 1 and Level 2 propMat
    propMat_L1 = pd.read_csv(DATA_DIR / "r_propMat_L1.csv", index_col=0)
    propMat_L2 = pd.read_csv(DATA_DIR / "r_propMat_L2.csv", index_col=0)
    logger.info("R L1 types (%d): %s", len(propMat_L1), list(propMat_L1.index))
    logger.info("R L2 types (%d): %s", len(propMat_L2), list(propMat_L2.index))

    # malProp
    mal_df = pd.read_csv(DATA_DIR / "r_malProp.csv")
    # Handle various column name formats
    spot_col = [c for c in mal_df.columns if c.lower() in ("spot", "x", "")][0] if any(c.lower() in ("spot", "x", "") for c in mal_df.columns) else mal_df.columns[0]
    val_col = [c for c in mal_df.columns if "malprop" in c.lower()][0] if any("malprop" in c.lower() for c in mal_df.columns) else mal_df.columns[-1]
    mal_prop = pd.Series(mal_df[val_col].values, index=mal_df[spot_col].values)
    logger.info("malProp: %d spots, range [%.4f, %.4f]", len(mal_prop), mal_prop.min(), mal_prop.max())

    # malRef
    mal_ref_path = DATA_DIR / "r_malRef.csv"
    if mal_ref_path.exists():
        mal_ref_df = pd.read_csv(mal_ref_path)
        if "gene" in mal_ref_df.columns:
            mal_ref = pd.Series(mal_ref_df["value"].values, index=mal_ref_df["gene"].values)
        else:
            mal_ref = pd.Series(mal_ref_df.iloc[:, 1].values, index=mal_ref_df.iloc[:, 0].values)
        logger.info("malRef: %d genes", len(mal_ref))
    else:
        mal_ref = None
        logger.info("malRef: not available (NULL in R)")

    # Counts (sparse triplet)
    triplet = pd.read_csv(DATA_DIR / "r_counts_triplet.csv")
    gene_names = pd.read_csv(DATA_DIR / "r_gene_names.csv")["gene"].values
    spot_names = pd.read_csv(DATA_DIR / "r_spot_names.csv")["spot"].values
    counts = sparse.csc_matrix(
        (triplet["val"].values, (triplet["row"].values, triplet["col"].values)),
        shape=(len(gene_names), len(spot_names)),
    )
    logger.info("Counts: %d genes x %d spots, %d nnz", *counts.shape, counts.nnz)

    # Lineage tree
    tree_keys = pd.read_csv(DATA_DIR / "r_tree_keys.csv")["lineage"].values
    tree = {}
    for k in tree_keys:
        fname = DATA_DIR / f"r_tree_{k.replace(' ', '_')}.csv"
        if fname.exists():
            tree[k] = pd.read_csv(fname)["subtype"].tolist()
    logger.info("Tree: %s", {k: v for k, v in tree.items()})

    return {
        "propMat_full": propMat_full,
        "propMat_L1": propMat_L1,
        "propMat_L2": propMat_L2,
        "mal_prop": mal_prop,
        "mal_ref": mal_ref,
        "counts": counts,
        "gene_names": gene_names,
        "spot_names": spot_names,
        "tree": tree,
    }


def run_python_deconvolution(r_data):
    """Run Python full deconvolution pipeline independently.

    Uses the same raw counts as R but runs the complete Stage 1 + Stage 2
    pipeline in Python for a head-to-head comparison.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from spatialgpu.deconvolution.core import _spatial_deconv_python, _deconvolution_python
    from spatialgpu.deconvolution.reference import load_comb_ref

    ref = load_comb_ref()

    # Filter zero-sum genes (same as R)
    counts = r_data["counts"]
    gene_names = r_data["gene_names"]
    spot_names = r_data["spot_names"]

    gene_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero_mask = gene_sums > 0
    counts = counts[nonzero_mask]
    gene_names = gene_names[nonzero_mask]

    logger.info("After zero-gene filter: %d genes x %d spots", *counts.shape)

    # Build a minimal AnnData for the full pipeline
    import anndata as ad

    adata = ad.AnnData(
        X=counts.T.tocsr(),
        obs=pd.DataFrame(index=pd.Index(spot_names)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )

    logger.info("Running full Python deconvolution pipeline (Stage 1 + Stage 2, solver=r_compat)...")
    py_propMat, mal_res = _deconvolution_python(
        adata,
        cancer_type="BRCA",
        signature_type=None,
        adjacent_normal=False,
        n_jobs=4,
        solver="auto",
    )

    logger.info("Python propMat: %s", py_propMat.shape)
    logger.info("Python cell types: %s", list(py_propMat.index))
    return py_propMat


def compare_results(r_data, py_propMat):
    """Compare R vs Python sublineage deconvolution results."""
    r_full = r_data["propMat_full"]
    tree = r_data["tree"]

    # Align columns (spots)
    common_spots = r_full.columns.intersection(py_propMat.columns)
    logger.info("Common spots: %d", len(common_spots))

    # Align rows (cell types)
    common_types = r_full.index.intersection(py_propMat.index)
    logger.info("Common cell types (%d): %s", len(common_types), list(common_types))

    r_aligned = r_full.loc[common_types, common_spots].values.astype(np.float64)
    py_aligned = py_propMat.loc[common_types, common_spots].values.astype(np.float64)

    # Overall correlation
    r_flat = r_aligned.ravel()
    py_flat = py_aligned.ravel()
    pearson_r, pearson_p = stats.pearsonr(r_flat, py_flat)
    spearman_rho, spearman_p = stats.spearmanr(r_flat, py_flat)
    rmse = float(np.sqrt(np.mean((r_flat - py_flat) ** 2)))
    mae = float(np.mean(np.abs(r_flat - py_flat)))

    print("\n" + "=" * 70)
    print("OVERALL CONCORDANCE (all cell types, all spots)")
    print("=" * 70)
    print(f"  Pearson r  = {pearson_r:.6f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_rho:.6f} (p={spearman_p:.2e})")
    print(f"  RMSE       = {rmse:.6f}")
    print(f"  MAE        = {mae:.6f}")
    print(f"  Max diff   = {np.max(np.abs(r_flat - py_flat)):.6f}")

    # Identify Level 1 and Level 2 types
    l1_types = [t for t in list(tree.keys()) + ["Malignant", "Unidentifiable"] if t in common_types]
    l2_types = [t for t in common_types if t not in l1_types]

    # Level 1 concordance
    if l1_types:
        r_l1 = r_full.loc[[t for t in l1_types if t in r_full.index], common_spots].values.ravel()
        py_l1 = py_propMat.loc[[t for t in l1_types if t in py_propMat.index], common_spots].values.ravel()
        l1_r, _ = stats.pearsonr(r_l1, py_l1)
        l1_rmse = float(np.sqrt(np.mean((r_l1 - py_l1) ** 2)))
        print(f"\n{'=' * 70}")
        print(f"LEVEL 1 (Major Lineages): {l1_types}")
        print(f"{'=' * 70}")
        print(f"  Pearson r = {l1_r:.6f}")
        print(f"  RMSE      = {l1_rmse:.6f}")

    # Level 2 concordance
    if l2_types:
        r_l2 = r_full.loc[[t for t in l2_types if t in r_full.index], common_spots].values.ravel()
        py_l2 = py_propMat.loc[[t for t in l2_types if t in py_propMat.index], common_spots].values.ravel()
        l2_r, _ = stats.pearsonr(r_l2, py_l2)
        l2_rmse = float(np.sqrt(np.mean((r_l2 - py_l2) ** 2)))
        print(f"\n{'=' * 70}")
        print(f"LEVEL 2 (Sublineages): {l2_types}")
        print(f"{'=' * 70}")
        print(f"  Pearson r = {l2_r:.6f}")
        print(f"  RMSE      = {l2_rmse:.6f}")

    # Per-cell-type breakdown
    print(f"\n{'=' * 70}")
    print("PER-CELL-TYPE CONCORDANCE")
    print(f"{'=' * 70}")
    print(f"{'Cell Type':<25} {'Level':<8} {'Pearson r':>10} {'RMSE':>10} {'MAE':>10} {'MaxDiff':>10}")
    print("-" * 75)

    results_rows = []
    for ct in common_types:
        r_ct = r_full.loc[ct, common_spots].values.astype(np.float64)
        py_ct = py_propMat.loc[ct, common_spots].values.astype(np.float64)

        if np.std(r_ct) < 1e-15 and np.std(py_ct) < 1e-15:
            ct_r = 1.0  # both constant
        elif np.std(r_ct) < 1e-15 or np.std(py_ct) < 1e-15:
            ct_r = 0.0
        else:
            ct_r, _ = stats.pearsonr(r_ct, py_ct)

        ct_rmse = float(np.sqrt(np.mean((r_ct - py_ct) ** 2)))
        ct_mae = float(np.mean(np.abs(r_ct - py_ct)))
        ct_max = float(np.max(np.abs(r_ct - py_ct)))
        level = "L1" if ct in l1_types else "L2"

        print(f"{ct:<25} {level:<8} {ct_r:>10.6f} {ct_rmse:>10.6f} {ct_mae:>10.6f} {ct_max:>10.6f}")
        results_rows.append({
            "cell_type": ct, "level": level,
            "pearson_r": ct_r, "rmse": ct_rmse, "mae": ct_mae, "max_diff": ct_max,
        })

    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(DATA_DIR / "concordance_results.csv", index=False)
    logger.info("Results saved to %s", DATA_DIR / "concordance_results.csv")

    # Save Python propMat for further analysis
    py_propMat.to_csv(DATA_DIR / "py_propMat_full.csv")

    return results_df


def main():
    r_data = load_r_outputs()
    py_propMat = run_python_deconvolution(r_data)
    results = compare_results(r_data, py_propMat)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    n_high = (results["pearson_r"] > 0.99).sum()
    n_good = ((results["pearson_r"] > 0.95) & (results["pearson_r"] <= 0.99)).sum()
    n_low = (results["pearson_r"] <= 0.95).sum()
    print(f"  r > 0.99: {n_high} cell types")
    print(f"  r > 0.95: {n_good} cell types")
    print(f"  r <= 0.95: {n_low} cell types")

    if n_low > 0:
        low_types = results[results["pearson_r"] <= 0.95]
        print(f"\n  Low concordance types:")
        for _, row in low_types.iterrows():
            print(f"    {row['cell_type']} ({row['level']}): r={row['pearson_r']:.4f}, RMSE={row['rmse']:.6f}")


if __name__ == "__main__":
    main()
