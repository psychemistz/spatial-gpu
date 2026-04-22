"""Run SpaCET (one weighting variant) on a single trial's data.

Reads docs/outputs/trials/T{NN}/{sc_counts,sc_meta,bulk_*,gt_*}.csv
Writes docs/outputs/trials/T{NN}/spacet_{method}_{scenario}.csv
                                  spacet_{method}_per_type.csv

Usage: python scripts/bench_spacet_trial.py --method v3_irwls --trial 0
"""

import argparse
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
from _t8_common import compute_method_r, remap_and_collapse  # noqa: E402

OUTPUTS_DIR = os.path.join(_REPO_ROOT, "docs", "outputs")
TRIALS_DIR = os.path.join(OUTPUTS_DIR, "trials")

_METHOD_MAP = {
    "v0_none":  (False, "ratio"),
    "v1_ratio": (True,  "ratio"),
    "v3_irwls": (True,  "irwls"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(_METHOD_MAP))
    ap.add_argument("--trial", type=int, required=True)
    args = ap.parse_args()

    csw, wm = _METHOD_MAP[args.method]
    trial_dir = os.path.join(TRIALS_DIR, f"T{args.trial:02d}")
    print(f"=== SpaCET trial T{args.trial:02d} variant={args.method} ===")
    print(f"  trial_dir: {trial_dir}")

    import spatialgpu.deconvolution as spacet

    # Load reference
    sc_counts_df = pd.read_csv(
        os.path.join(trial_dir, "sc_counts.csv"), index_col=0
    )  # genes x cells
    sc_meta_df = pd.read_csv(os.path.join(trial_dir, "sc_meta.csv"), index_col=0)
    print(f"  Reference: {sc_counts_df.shape[1]} cells, {sc_counts_df.shape[0]} genes")

    sc_annotation = pd.DataFrame(
        {
            "cellID": sc_meta_df.index,
            "cellType": sc_meta_df["cell_type"].values,
            "subject_id": sc_meta_df["subject_id"].values,
        },
        index=sc_meta_df.index,
    )
    lineage_tree = {ct: [ct] for ct in sorted(sc_annotation["cellType"].unique())}

    # Load 4 scenarios
    scenarios = {}
    for label in ["uniform", "sparse", "tumor_purity", "titration"]:
        bulk_df = pd.read_csv(
            os.path.join(trial_dir, f"bulk_{label}.csv"), index_col=0
        )
        bulk_adata = ad.AnnData(
            X=bulk_df.values,
            obs=pd.DataFrame(index=bulk_df.index),
            var=pd.DataFrame(index=bulk_df.columns),
        )
        gt_df = pd.read_csv(os.path.join(trial_dir, f"gt_{label}.csv"), index_col=0)
        scenarios[label] = (bulk_adata, gt_df)

    out_rows = []
    for label, (bulk, gt_raw) in scenarios.items():
        print(f"\n  --- {label} ---")
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
        bulk_copy = bulk.copy()
        spacet.deconvolution_matched_scrnaseq(bulk_copy, **kwargs)
        pm = bulk_copy.uns["spacet"]["deconvolution"]["propMat"]
        est = remap_and_collapse(pm.T)

        # Save predictions
        out_csv = os.path.join(trial_dir, f"spacet_{args.method}_{label}.csv")
        est.to_csv(out_csv)

        # Metrics
        gt_eval = remap_and_collapse(gt_raw)
        m = compute_method_r(est, gt_eval)
        common = sorted(set(est.columns) & set(gt_eval.columns))
        gt_aligned = gt_eval.reindex(est.index)[common]
        from scipy.stats import pearsonr
        for ct in common:
            e = est[ct].values
            g = gt_aligned[ct].values
            r, _ = pearsonr(e, g)
            out_rows.append(
                {
                    "trial": args.trial,
                    "method": args.method,
                    "scenario": label,
                    "cell_type": ct,
                    "r": r,
                    "bias": float((e - g).mean()),
                    "var_ratio": float(e.var() / g.var()) if g.var() > 0 else np.nan,
                    "rmse": float(np.sqrt(np.mean((e - g) ** 2))),
                    "overall_r": m["r"],
                    "overall_rmse": m["rmse"],
                }
            )
        print(f"     overall r={m['r']:.4f}  rmse={m['rmse']:.4f}")

    out_long = os.path.join(trial_dir, f"spacet_{args.method}_per_type.csv")
    pd.DataFrame(out_rows).to_csv(out_long, index=False)
    print(f"\nSaved: {out_long}")


if __name__ == "__main__":
    main()
