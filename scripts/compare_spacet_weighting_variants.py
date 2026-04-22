"""Compare SpaCET cross-subject weighting variants.

Reads per-variant per-scenario prediction CSVs written by
scripts/bench_spacet_weighting.py and produces:
  - docs/outputs/t8_spacet_variants_summary.csv : wide table (scenarios x variants x metric)
  - docs/outputs/t8_spacet_variants_per_type.csv : long table (variant, scenario, cell_type, r/bias/var_ratio/rmse)
  - docs/figures/benchmark_spacet_variants.png  : overall Pearson r per (variant x scenario) bar chart
  - docs/figures/benchmark_spacet_variants_per_type_uniform.png : per-type heatmap for uniform scenario

Usage: python scripts/compare_spacet_weighting_variants.py
"""

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

OUTPUTS_DIR = os.path.join(_REPO_ROOT, "docs", "outputs")
FIGURES_DIR = os.path.join(_REPO_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

VARIANTS = [
    ("v0_none", "SpaCET (no weighting)", "#6b7280"),
    ("v1_ratio", "SpaCET + ratio weighting (V1)", "#3b82f6"),
    ("v3_irwls", "SpaCET + IRWLS-lite (V3, default)", "#f59e0b"),
]

# Cross-method baselines (committed at 500/type ref): MuSiC and DWLS (minor).
# DWLS (broad) intentionally dropped per Beibei's request — DWLS (minor) is
# the canonical fair-resolution comparison.
EXTRA_METHODS = [
    ("music", "MuSiC", "#f97316"),
    ("dwls_minor", "DWLS (minor)", "#ec4899"),
]


def _load_pred(method, scenario):
    """Locate per-method prediction CSV. Returns DataFrame or None.

    SpaCET variants:    docs/outputs/t8_spacet_{method}_{scenario}.csv  (already collapsed)
    MuSiC:              docs/outputs/t8_music_{scenario}.csv             (Wu names; needs collapse)
    DWLS (minor):       docs/outputs/t8_dwls_minor_{scenario}.csv        (minor names; needs minor->major map)
    """
    if method.startswith("v"):
        path = os.path.join(OUTPUTS_DIR, f"t8_spacet_{method}_{scenario}.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path, index_col=0)  # already collapsed by bench script
    if method == "music":
        path = os.path.join(OUTPUTS_DIR, f"t8_music_{scenario}.csv")
        if not os.path.exists(path):
            return None
        return remap_and_collapse(pd.read_csv(path, index_col=0))
    if method == "dwls_minor":
        path = os.path.join(OUTPUTS_DIR, f"t8_dwls_minor_{scenario}.csv")
        map_path = os.path.join(OUTPUTS_DIR, "t8_minor_to_major.json")
        if not (os.path.exists(path) and os.path.exists(map_path)):
            return None
        import json
        with open(map_path) as f:
            minor_to_major = json.load(f)
        minor_pred = pd.read_csv(path, index_col=0)
        # Collapse minor -> major, then apply Wu->eval rename
        major_pred = minor_pred.rename(columns=minor_to_major).T.groupby(level=0).sum().T
        return remap_and_collapse(major_pred)
    return None


def main():
    rows = []
    long_rows = []
    found_methods = []
    all_methods = VARIANTS + EXTRA_METHODS
    for method, _display, _color in all_methods:
        any_scenario = False
        for scenario, desc in SCENARIOS:
            pred_e = _load_pred(method, scenario)
            gt_path = os.path.join(OUTPUTS_DIR, f"t8_real_gt_{scenario}.csv")
            if pred_e is None or not os.path.exists(gt_path):
                continue
            any_scenario = True
            gt = pd.read_csv(gt_path, index_col=0)
            gt_e = remap_and_collapse(gt)
            m = compute_method_r(pred_e, gt_e)
            rows.append(
                {
                    "variant": method,
                    "scenario": scenario,
                    "desc": desc,
                    "r": m["r"],
                    "rho": m["rho"],
                    "rmse": m["rmse"],
                }
            )
            # per-type
            from scipy.stats import pearsonr
            common = sorted(set(pred_e.columns) & set(gt_e.columns))
            gt_aligned = gt_e.reindex(pred_e.index)[common]
            for ct in common:
                e = pred_e[ct].values
                g = gt_aligned[ct].values
                r, _ = pearsonr(e, g)
                long_rows.append(
                    {
                        "variant": method,
                        "scenario": scenario,
                        "cell_type": ct,
                        "r": r,
                        "bias": float((e - g).mean()),
                        "var_ratio": float(e.var() / g.var()) if g.var() > 0 else np.nan,
                        "rmse": float(np.sqrt(np.mean((e - g) ** 2))),
                    }
                )
        if any_scenario:
            found_methods.append(method)

    if not rows:
        print("No variant predictions found — run scripts/bench_spacet_weighting.py first.")
        return

    summary = pd.DataFrame(rows)
    wide_r = summary.pivot(index="desc", columns="variant", values="r")
    # Preserve canonical SCENARIOS order (tumor-dominated first) and CLI order
    desc_order = [d for _, d in SCENARIOS if d in wide_r.index]
    wide_r = wide_r.loc[desc_order, found_methods]
    print("Overall Pearson r (rows = scenario, cols = method):")
    print(wide_r.to_string(float_format=lambda x: f"{x:.4f}"))
    summary_out = os.path.join(OUTPUTS_DIR, "t8_spacet_variants_summary.csv")
    wide_r.to_csv(summary_out)
    print(f"\nSaved summary table: {summary_out}")

    long_df = pd.DataFrame(long_rows)
    long_out = os.path.join(OUTPUTS_DIR, "t8_spacet_variants_per_type.csv")
    long_df.to_csv(long_out, index=False)
    print(f"Saved per-type table: {long_out}")

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(14, 5))
    display_labels = {v[0]: v[1] for v in VARIANTS + EXTRA_METHODS}
    colors = {v[0]: v[2] for v in VARIANTS + EXTRA_METHODS}
    scenarios_in_order = [desc for _, desc in SCENARIOS]
    x = np.arange(len(scenarios_in_order))
    n_methods = len(found_methods)
    w = 0.8 / n_methods
    for j, method in enumerate(found_methods):
        offset = (j - n_methods / 2 + 0.5) * w
        rs = [wide_r.at[s, method] if s in wide_r.index else np.nan for s in scenarios_in_order]
        ax.bar(x + offset, rs, w, label=display_labels[method], color=colors[method])
        for i, r in enumerate(rs):
            if not np.isnan(r):
                ax.text(x[i] + offset, r + 0.01, f"{r:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_in_order, fontsize=9)
    ax.set_ylabel("Pearson r (overall)")
    ax.set_title("SpaCET cross-subject weighting variants — T8 BRCA benchmark")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "benchmark_spacet_variants.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {bar_path}")

    # ---- Per-type heatmap (uniform scenario only) ----
    uniform = long_df[long_df["scenario"] == "uniform"]
    if not uniform.empty:
        hm = uniform.pivot(index="cell_type", columns="variant", values="r")
        hm = hm[[v for v in found_methods if v in hm.columns]]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(hm.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(hm.columns)))
        ax.set_xticklabels([display_labels[v] for v in hm.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(hm.index)))
        ax.set_yticklabels(hm.index)
        for i in range(len(hm.index)):
            for j in range(len(hm.columns)):
                val = hm.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if val > 0.75 else "white")
        ax.set_title("Per-cell-type r (Uniform Dirichlet scenario)")
        fig.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        hm_path = os.path.join(FIGURES_DIR, "benchmark_spacet_variants_per_type_uniform.png")
        fig.savefig(hm_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {hm_path}")


if __name__ == "__main__":
    main()
