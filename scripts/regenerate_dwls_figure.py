"""Regenerate the T8 bulk benchmark comparison figure + summary table.

Reads the existing SpaCET/MuSiC results from t8_comparison_summary.csv, computes
DWLS (broad, 9 major types) metrics from t8_dwls_*.csv vs t8_real_gt_*.csv, and
— if minor-resolution DWLS outputs exist — collapses t8_dwls_minor_*.csv from
22 minor types down to 9 major types and computes a fair-DWLS r.

Writes:
  - docs/outputs/t8_comparison_summary.csv  (DWLS_r, DWLS_minor_r columns)
  - docs/figures/benchmark_spacet_vs_music.png  (3- or 4-method bar chart)

Standalone — does not rerun SpaCET (no GPU needed).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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

SCENARIO_DESC = {
    "uniform": "Uniform (alpha=1.0)",
    "sparse": "Sparse (alpha=0.3)",
    "tumor_purity": "Tumor Purity (60-90%)",
    "titration": "Titration (0-90%)",
}

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
OUTPUTS_DIR = os.path.join(DOCS_DIR, "outputs")
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")
MINOR_TO_MAJOR_PATH = os.path.join(OUTPUTS_DIR, "t8_minor_to_major.json")
os.makedirs(FIGURES_DIR, exist_ok=True)


def remap_and_collapse(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T


def compute_r_against_gt(pred: pd.DataFrame, gt: pd.DataFrame) -> float:
    pred_e = remap_and_collapse(pred)
    gt_e = remap_and_collapse(gt)
    common = sorted(set(pred_e.columns) & set(gt_e.columns))
    est = pred_e[common].values.ravel()
    truth = gt_e.reindex(pred_e.index)[common].values.ravel()
    r, _ = pearsonr(est, truth)
    return r


def compute_dwls_broad_r() -> dict[str, float]:
    out = {}
    for label in SCENARIO_DESC:
        dwls_file = os.path.join(OUTPUTS_DIR, f"t8_dwls_{label}.csv")
        gt_file = os.path.join(OUTPUTS_DIR, f"t8_real_gt_{label}.csv")
        if not (os.path.exists(dwls_file) and os.path.exists(gt_file)):
            print(f"   skip {label} (broad): missing file")
            continue
        r = compute_r_against_gt(
            pd.read_csv(dwls_file, index_col=0),
            pd.read_csv(gt_file, index_col=0),
        )
        out[label] = r
        print(f"   DWLS (broad) {label}: r={r:.4f}")
    return out


def compute_dwls_minor_r() -> dict[str, float]:
    if not os.path.exists(MINOR_TO_MAJOR_PATH):
        print("   minor-to-major map not found; skipping DWLS (minor) block")
        return {}
    with open(MINOR_TO_MAJOR_PATH) as f:
        minor_to_major = json.load(f)

    out = {}
    for label in SCENARIO_DESC:
        minor_file = os.path.join(OUTPUTS_DIR, f"t8_dwls_minor_{label}.csv")
        gt_file = os.path.join(OUTPUTS_DIR, f"t8_real_gt_{label}.csv")
        if not (os.path.exists(minor_file) and os.path.exists(gt_file)):
            print(f"   skip {label} (minor): missing file")
            continue
        minor_pred = pd.read_csv(minor_file, index_col=0)
        unknown = set(minor_pred.columns) - set(minor_to_major)
        if unknown:
            raise RuntimeError(f"minor columns not in map for {label}: {sorted(unknown)}")
        major_pred = minor_pred.rename(columns=minor_to_major).T.groupby(level=0).sum().T
        r = compute_r_against_gt(major_pred, pd.read_csv(gt_file, index_col=0))
        out[label] = r
        print(f"   DWLS (minor -> major) {label}: r={r:.4f}")
    return out


def main():
    summary_path = os.path.join(OUTPUTS_DIR, "t8_comparison_summary.csv")
    summary = pd.read_csv(summary_path)
    print(f"Loaded existing summary: {summary_path}")
    print(summary)

    print("\nComputing DWLS (broad, 9 major) metrics...")
    dwls_r = compute_dwls_broad_r()

    print("\nComputing DWLS (minor -> major, 22 -> 9) metrics...")
    dwls_minor_r = compute_dwls_minor_r()

    desc_to_label = {v: k for k, v in SCENARIO_DESC.items()}
    summary["DWLS_r"] = summary["Scenario"].map(
        lambda s: dwls_r.get(desc_to_label[s], np.nan)
    )
    summary["DWLS_minor_r"] = summary["Scenario"].map(
        lambda s: dwls_minor_r.get(desc_to_label[s], np.nan)
    )

    summary.to_csv(summary_path, index=False)
    print(f"\nUpdated summary: {summary_path}")
    print(summary.to_string(index=False))

    methods = [
        ("SpaCET", "SpaCET_r", "#3b82f6"),
        ("MuSiC", "MuSiC_r", "#f97316"),
        ("DWLS (broad)", "DWLS_r", "#a855f7"),
    ]
    if summary["DWLS_minor_r"].notna().any():
        methods.append(("DWLS (minor)", "DWLS_minor_r", "#ec4899"))

    fig, ax = plt.subplots(figsize=(11, 5))
    scenarios = summary["Scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.8 / len(methods)

    for j, (name, col, color) in enumerate(methods):
        offset = (j - len(methods) / 2 + 0.5) * w
        rs = summary[col].tolist()
        ax.bar(x + offset, rs, w, label=name, color=color)
        for i, r in enumerate(rs):
            if not np.isnan(r):
                ax.text(x[i] + offset, r + 0.02, f"{r:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Pearson r (overall)")
    ax.set_title("Deconvolution Benchmark — Fair Subject-Split (Wu et al. BRCA)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", ncol=2)
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "benchmark_spacet_vs_music.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
