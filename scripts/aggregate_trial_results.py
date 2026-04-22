"""Aggregate per-trial benchmark outputs and compute paired statistics.

Reads docs/outputs/trials/T{NN}/{spacet,music,dwls_minor}_{...}.csv across
all trials, computes per-(method × scenario) overall Pearson r per trial,
runs paired t-tests on per-trial differences between method pairs, and
generates a comparison figure with error bars.

Outputs:
  - docs/outputs/t8_trials_overall_r.csv     : long table (trial, method, scenario, r, rmse)
  - docs/outputs/t8_trials_summary.csv       : wide table (scenario × method, mean ± std)
  - docs/outputs/t8_trials_paired_tests.csv  : paired t-tests for selected method pairs
  - docs/figures/benchmark_trials.png        : grouped bars with error bars

Usage: python scripts/aggregate_trial_results.py
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _t8_common import SCENARIOS, compute_method_r, remap_and_collapse  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUTPUTS_DIR = os.path.join(_REPO_ROOT, "docs", "outputs")
TRIALS_DIR = os.path.join(OUTPUTS_DIR, "trials")
FIGURES_DIR = os.path.join(_REPO_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

METHODS = [
    ("spacet_v0_none", "SpaCET (no weighting)", "#6b7280"),
    ("spacet_v1_ratio", "SpaCET + ratio (V1)", "#3b82f6"),
    ("spacet_v3_irwls", "SpaCET + IRWLS (V3)", "#f59e0b"),
    ("music", "MuSiC", "#f97316"),
    ("dwls_minor", "DWLS (minor)", "#ec4899"),
]

# Method pairs to test for paired-difference significance
PAIRS_TO_TEST = [
    ("spacet_v3_irwls", "spacet_v1_ratio"),  # IRWLS vs ratio (within SpaCET)
    ("spacet_v3_irwls", "spacet_v0_none"),   # IRWLS vs no weighting
    ("spacet_v3_irwls", "music"),             # SpaCET-best vs MuSiC
    ("spacet_v3_irwls", "dwls_minor"),        # SpaCET-best vs DWLS
    ("music", "dwls_minor"),                  # MuSiC vs DWLS
]


def load_pred(method, trial, scenario):
    """Locate per-method per-trial prediction CSV."""
    trial_dir = os.path.join(TRIALS_DIR, f"T{trial:02d}")
    if method.startswith("spacet_"):
        variant = method[len("spacet_"):]
        path = os.path.join(trial_dir, f"spacet_{variant}_{scenario}.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path, index_col=0)  # already collapsed
    if method == "music":
        path = os.path.join(trial_dir, f"music_{scenario}.csv")
        if not os.path.exists(path):
            return None
        return remap_and_collapse(pd.read_csv(path, index_col=0))
    if method == "dwls_minor":
        path = os.path.join(trial_dir, f"dwls_minor_{scenario}.csv")
        map_path = os.path.join(trial_dir, "minor_to_major.json")
        if not (os.path.exists(path) and os.path.exists(map_path)):
            return None
        with open(map_path) as f:
            minor_to_major = json.load(f)
        minor_pred = pd.read_csv(path, index_col=0)
        major_pred = (
            minor_pred.rename(columns=minor_to_major).T.groupby(level=0).sum().T
        )
        return remap_and_collapse(major_pred)
    return None


def main():
    rows = []
    # Discover trials by scanning trial directories (handles N=100+).
    trial_ids = sorted(
        int(d[1:]) for d in os.listdir(TRIALS_DIR)
        if d.startswith("T") and d[1:].isdigit() and os.path.isdir(os.path.join(TRIALS_DIR, d))
    )
    n_trials = len(trial_ids)
    for trial in trial_ids:
        trial_dir = os.path.join(TRIALS_DIR, f"T{trial:02d}")
        for method, _disp, _col in METHODS:
            for scenario, desc in SCENARIOS:
                pred = load_pred(method, trial, scenario)
                gt_path = os.path.join(trial_dir, f"gt_{scenario}.csv")
                if pred is None or not os.path.exists(gt_path):
                    continue
                gt = remap_and_collapse(pd.read_csv(gt_path, index_col=0))
                m = compute_method_r(pred, gt)
                rows.append(
                    {
                        "trial": trial,
                        "method": method,
                        "scenario": scenario,
                        "scenario_desc": desc,
                        "r": m["r"],
                        "rho": m["rho"],
                        "rmse": m["rmse"],
                    }
                )

    if not rows:
        print(
            "No trial outputs found. Run trial array jobs first "
            "(scripts/slurm_trial_array.sh)."
        )
        return

    long_df = pd.DataFrame(rows)
    long_path = os.path.join(OUTPUTS_DIR, "t8_trials_overall_r.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Saved long table ({len(long_df)} rows): {long_path}")

    # Per-method per-scenario mean and std across trials
    summary = (
        long_df.groupby(["scenario_desc", "method"])["r"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    desc_order = [d for _, d in SCENARIOS if d in summary["scenario_desc"].values]
    method_order = [m for m, _, _ in METHODS if m in long_df["method"].values]
    wide_mean = summary.pivot(index="scenario_desc", columns="method", values="mean")
    wide_std = summary.pivot(index="scenario_desc", columns="method", values="std")
    wide_n = summary.pivot(index="scenario_desc", columns="method", values="count")
    wide_mean = wide_mean.loc[desc_order, method_order]
    wide_std = wide_std.loc[desc_order, method_order]
    wide_n = wide_n.loc[desc_order, method_order]

    print(f"\nPer-method per-scenario mean r (over {n_trials} trials):")
    print(wide_mean.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nStd:")
    print(wide_std.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nN:")
    print(wide_n.fillna(0).astype(int).to_string())

    summary_out = os.path.join(OUTPUTS_DIR, "t8_trials_summary.csv")
    pd.concat(
        [
            wide_mean.add_suffix("_mean"),
            wide_std.add_suffix("_std"),
            wide_n.fillna(0).astype(int).add_suffix("_n"),
        ],
        axis=1,
    ).to_csv(summary_out)
    print(f"\nSaved summary: {summary_out}")

    # Paired Wilcoxon signed-rank tests — non-parametric, robust to outliers
    # (we observed r=-0.10 in V0 trial 5, normality assumption is violated).
    # Pair the per-trial r values within (method_A, method_B, scenario).
    print("\nPaired Wilcoxon signed-rank tests (per-trial Δr between methods):")
    test_rows = []
    for m1, m2 in PAIRS_TO_TEST:
        for scenario, desc in SCENARIOS:
            sub1 = long_df[
                (long_df["method"] == m1) & (long_df["scenario"] == scenario)
            ].set_index("trial")["r"]
            sub2 = long_df[
                (long_df["method"] == m2) & (long_df["scenario"] == scenario)
            ].set_index("trial")["r"]
            paired = pd.concat([sub1, sub2], axis=1, join="inner").dropna()
            if len(paired) < 2:
                continue
            r1 = paired.iloc[:, 0].values
            r2 = paired.iloc[:, 1].values
            delta = r1 - r2
            # Wilcoxon: test H0 that median of paired differences = 0.
            # zsplit handles ties by splitting their contribution; pratt drops
            # zero-difference pairs. Use "wilcox" (drops zeros) as the standard
            # signed-rank choice. zero_method="wilcox" is the default.
            try:
                if np.allclose(delta, 0):
                    w_stat, p_val = np.nan, 1.0
                else:
                    w_stat, p_val = wilcoxon(r1, r2, zero_method="wilcox")
            except ValueError as e:
                w_stat, p_val = np.nan, np.nan
                print(f"   warning {desc} {m1} vs {m2}: {e}")
            row = {
                "method_A": m1,
                "method_B": m2,
                "scenario": desc,
                "n_trials": len(paired),
                "mean_r_A": r1.mean(),
                "mean_r_B": r2.mean(),
                "median_delta": float(np.median(delta)),
                "delta_mean": delta.mean(),
                "delta_std": delta.std(ddof=1),
                "wilcoxon_stat": w_stat,
                "p_value": p_val,
                "significant_005": (p_val < 0.05) if not np.isnan(p_val) else False,
            }
            test_rows.append(row)
            sig = "*" if p_val < 0.05 else " "
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "  NaN"
            print(
                f"  {sig} {desc:24s} {m1[:18]:>18s} − {m2[:18]:<18s} "
                f"median Δ={np.median(delta):+.4f}  "
                f"mean Δ={delta.mean():+.4f}  "
                f"n={len(paired):2d}  p={p_str}"
            )
    test_out = os.path.join(OUTPUTS_DIR, "t8_trials_paired_tests.csv")
    pd.DataFrame(test_rows).to_csv(test_out, index=False)
    print(f"\nSaved paired-test results: {test_out}")

    # Bar chart with error bars
    fig, ax = plt.subplots(figsize=(14, 5))
    method_disp = {m: d for m, d, _ in METHODS}
    method_col = {m: c for m, _, c in METHODS}
    n_methods = len(method_order)
    x = np.arange(len(desc_order))
    w = 0.8 / n_methods
    for j, method in enumerate(method_order):
        offset = (j - n_methods / 2 + 0.5) * w
        means = wide_mean[method].values
        stds = wide_std[method].fillna(0).values
        ax.bar(
            x + offset, means, w,
            yerr=stds, capsize=2,
            label=method_disp[method], color=method_col[method],
        )
        for i, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                ax.text(
                    x[i] + offset, m + s + 0.005, f"{m:.2f}",
                    ha="center", fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(desc_order, fontsize=9)
    ax.set_ylabel(f"Mean Pearson r (n={n_trials} trials, ±1 SD)")
    ax.set_title(
        f"BRCA bulk deconvolution — N={n_trials} trial replicates "
        "(varied subject split + pseudobulk seed)"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "benchmark_trials.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
