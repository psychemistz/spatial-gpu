"""Score SpaCET predictions against DREAM Challenge ground truth.

Aggregates per-method, per-dataset:
  - DREAM primary score: cross-sample within-cell-type Pearson r, mean across
    cell types, bootstrap n=1000.
  - Per-cell-type r (no bootstrap) for the leaderboard heat-map.
  - Cross-cell-type within-sample r/rho/RMSE for the auxiliary panel.

Writes:
  <out>/scores_<resolution>.csv     -- one row per (method, dataset, cell_type)
  <out>/aggregate_<resolution>.csv  -- bootstrap mean + 95% CI per (method, dataset)
  <out>/leaderboard_<resolution>.png (optional --plot)

Usage:
    python scripts/bench_real/score_dream.py \
        --predictions /vf/.../bench_outputs \
        --gt /vf/.../dream/gt_coarse/coarse.csv \
        --sample-map /vf/.../dream/sample_map/in-vitro-admixture-sample-datasets.tsv \
        --resolution coarse \
        --out /vf/.../bench_outputs/scores \
        --plot
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _common import (  # noqa: E402
    align_cell_types,
    bootstrap_aggregate_r,
    cross_cell_type_within_sample,
    pearson_per_cell_type,
)

_PRED_FILENAME = re.compile(r"(?P<dataset>[A-Za-z0-9]+)_(?P<method>v\d+_\w+)\.csv$")


def _parse_pred_files(predictions_dir: str) -> list[dict]:
    out = []
    for p in glob.glob(os.path.join(predictions_dir, "*.csv")):
        m = _PRED_FILENAME.search(os.path.basename(p))
        if not m:
            continue
        out.append({"path": p, **m.groupdict()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="directory of <dataset>_<method>.csv")
    ap.add_argument("--gt", required=True, help="ground-truth proportions CSV (DREAM)")
    ap.add_argument("--resolution", choices=["coarse", "fine"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = pd.read_csv(args.gt, index_col=0)
    print(f"GT: {gt.shape[0]} samples x {gt.shape[1]} cell types")

    pred_files = _parse_pred_files(args.predictions)
    if not pred_files:
        sys.exit(f"no prediction CSVs found in {args.predictions}")

    rows_perct, rows_agg = [], []
    for entry in pred_files:
        pred = pd.read_csv(entry["path"], index_col=0)
        # Limit GT to samples we predicted
        common = pred.index.intersection(gt.index)
        if not len(common):
            print(f"  [skip] {entry['path']}: no overlapping samples with GT")
            continue
        pred, gt_sub = pred.loc[common], gt.loc[common]
        pred, gt_sub = align_cell_types(pred, gt_sub)
        if not pred.shape[1]:
            print(f"  [skip] {entry['path']}: no overlapping cell types after align")
            continue
        rs = pearson_per_cell_type(pred, gt_sub)
        for ct, r in rs.items():
            rows_perct.append({
                "method": entry["method"], "dataset": entry["dataset"],
                "cell_type": ct, "pearson_r": r,
            })
        agg = bootstrap_aggregate_r(pred, gt_sub, n_boot=args.n_boot)
        rows_agg.append({
            "method": entry["method"], "dataset": entry["dataset"],
            "n_samples": len(pred), "n_cell_types": pred.shape[1], **agg,
        })
        within = cross_cell_type_within_sample(pred, gt_sub)
        within.to_csv(os.path.join(
            args.out, f"within_sample_{entry['method']}_{entry['dataset']}.csv"
        ))
        print(f"  {entry['method']:10s} {entry['dataset']:10s}  agg r = {agg['mean']:.3f} "
              f"[{agg['lo95']:.3f}, {agg['hi95']:.3f}]  (n={len(pred)})")

    perct = pd.DataFrame(rows_perct)
    agg = pd.DataFrame(rows_agg)
    perct.to_csv(os.path.join(args.out, f"scores_{args.resolution}.csv"), index=False)
    agg.to_csv(os.path.join(args.out, f"aggregate_{args.resolution}.csv"), index=False)
    print(f"wrote scores to {args.out}")

    if args.plot:
        _make_leaderboard_plot(agg, perct, args.out, args.resolution)


def _make_leaderboard_plot(agg, perct, out_dir, resolution):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: aggregate r per (method, dataset)
    pivot = agg.pivot_table(index="dataset", columns="method", values="mean")
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("Aggregate Pearson r")
    axes[0].set_title(f"DREAM {resolution} — aggregate r (mean across cell types, bootstrap)")
    axes[0].legend(title="method", fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].axhline(0, color="grey", lw=0.5)

    # Panel B: per-cell-type r averaged across datasets, by method
    perct_pivot = (
        perct.groupby(["method", "cell_type"])["pearson_r"].mean().unstack("method")
    )
    perct_pivot.plot(kind="barh", ax=axes[1])
    axes[1].set_xlabel("Pearson r (mean across datasets)")
    axes[1].set_title(f"DREAM {resolution} — per cell type")
    axes[1].axvline(0, color="grey", lw=0.5)
    axes[1].legend(title="method", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"leaderboard_{resolution}.png"), dpi=120)
    plt.close(fig)
    print(f"plotted leaderboard_{resolution}.png")


if __name__ == "__main__":
    main()
