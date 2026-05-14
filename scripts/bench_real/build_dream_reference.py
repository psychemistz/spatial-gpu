"""Build a SpaCET-compatible scRNA reference from DREAM training pure profiles.

`spatialgpu.deconvolution.deconvolution_bulk` expects a single-cell reference:
  - sc_counts: genes x cells (raw counts)
  - sc_annotation: per-cell DataFrame with columns ('cellID', 'cellType', 'subject_id')
  - sc_lineage_tree: dict cellType -> list[sub-types]  (we use flat: ct -> [ct])

DREAM training data ships FACS-sorted purified expression profiles. Each purified
profile becomes one synthetic "cell"; donor ID = subject_id (enables IRWLS
cross-subject weighting). This matches how several top DREAM participants
constructed their references.

Usage:
    python scripts/bench_real/build_dream_reference.py \
        --pure /vf/users/parks34/projects/0sigdiscov/data/dream/pure_profiles \
        --out /vf/users/parks34/projects/0sigdiscov/data/dream/reference \
        --resolution coarse
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", required=True,
                    help="directory of DREAM training pure-profile expression matrices")
    ap.add_argument("--meta", required=True,
                    help="DREAM training metadata TSV (sample -> cell_type, donor)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--resolution", choices=["coarse", "fine"], default="coarse")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = pd.read_csv(args.meta, sep="\t")
    needed_cols = {"sample", "cell_type", "donor"}
    if missing := needed_cols - set(meta.columns):
        sys.exit(f"metadata is missing required columns: {missing}")

    expr_path = os.path.join(args.pure, f"pure_{args.resolution}_counts.tsv")
    if not os.path.exists(expr_path):
        sys.exit(f"expected pure profiles at {expr_path}")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    common = sorted(set(expr.columns) & set(meta["sample"]))
    expr = expr[common]
    meta = meta.set_index("sample").loc[common].reset_index()
    print(f"reference: {expr.shape[0]} genes x {expr.shape[1]} pure profiles "
          f"({meta['cell_type'].nunique()} cell types, {meta['donor'].nunique()} donors)")

    annotation = pd.DataFrame({
        "cellID": meta["sample"].values,
        "cellType": meta["cell_type"].values,
        "subject_id": meta["donor"].values,
    }, index=meta["sample"].values)

    lineage_tree = {ct: [ct] for ct in sorted(meta["cell_type"].unique())}

    expr.to_parquet(os.path.join(args.out, f"sc_counts_{args.resolution}.parquet"))
    annotation.to_csv(os.path.join(args.out, f"sc_annotation_{args.resolution}.csv"))
    with open(os.path.join(args.out, f"sc_lineage_tree_{args.resolution}.json"), "w") as f:
        json.dump(lineage_tree, f, indent=2)
    # Pickle for one-shot loading downstream
    with open(os.path.join(args.out, f"reference_{args.resolution}.pkl"), "wb") as f:
        pickle.dump({
            "sc_counts": expr,
            "sc_annotation": annotation,
            "sc_lineage_tree": lineage_tree,
        }, f)
    print(f"wrote reference bundle to {args.out}/reference_{args.resolution}.pkl")


if __name__ == "__main__":
    main()
