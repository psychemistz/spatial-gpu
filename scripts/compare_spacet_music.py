"""Compare SpaCET vs MuSiC deconvolution results.

Reads pre-computed results from docs/outputs/ and generates comparison figures.
Run after both SpaCET (t8_real_brca.py) and MuSiC (run_music_benchmark.R) complete.

Usage: python scripts/compare_spacet_music.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUTPUTS_DIR = "docs/outputs"
FIGURES_DIR = "docs/figures"

# Wu et al. -> collapsed category mapping
WU_TO_SPACET = {
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

SCENARIOS = [
    ("uniform", "Uniform (alpha=1.0)"),
    ("sparse", "Sparse (alpha=0.3)"),
    ("tumor_purity", "Tumor Purity (60-90%)"),
    ("titration", "Titration (0-90%)"),
]


def load_gt(scenario):
    gt = pd.read_csv(os.path.join(OUTPUTS_DIR, f"t8_real_gt_{scenario}.csv"), index_col=0)
    gt_renamed = gt.rename(columns=WU_TO_SPACET).T.groupby(level=0).sum().T
    return gt_renamed


def eval_method(est_df, gt_df):
    common = sorted(set(est_df.columns) & set(gt_df.columns))
    if not common:
        return {"r": np.nan, "rho": np.nan, "rmse": np.nan, "per_type": {}}
    gt_aligned = gt_df.reindex(est_df.index)[common]
    est = est_df[common].values.ravel()
    gt = gt_aligned.values.ravel()
    r, _ = pearsonr(est, gt)
    rho, _ = spearmanr(est, gt)
    rmse = np.sqrt(np.mean((est - gt) ** 2))
    per_type = {}
    for ct in common:
        ct_r, _ = pearsonr(est_df[ct].values, gt_aligned[ct].values)
        per_type[ct] = ct_r
    return {"r": r, "rho": rho, "rmse": rmse, "per_type": per_type}


def main():
    print("=== SpaCET vs MuSiC Comparison ===\n")

    results = []

    for scenario, label in SCENARIOS:
        gt = load_gt(scenario)

        # SpaCET results (try fair-split file first, then original)
        spacet_file = os.path.join(OUTPUTS_DIR, f"t8_real_spacet_{scenario}.txt")
        if not os.path.exists(spacet_file):
            spacet_file = os.path.join(OUTPUTS_DIR, f"t8_real_{scenario}.txt")
        # MuSiC results
        music_file = os.path.join(OUTPUTS_DIR, f"t8_music_{scenario}.csv")

        if not os.path.exists(music_file):
            print(f"   {scenario}: MuSiC results not found, skipping.")
            continue

        music_props = pd.read_csv(music_file, index_col=0)
        music_eval = eval_method(music_props, gt)

        # Load SpaCET from ground truth comparison
        # We need the SpaCET propMat — read from the exported bulk CSV + re-evaluate
        # Actually, we saved the metrics in t8_real_*.txt — parse the overall r
        spacet_r = np.nan
        if os.path.exists(spacet_file):
            with open(spacet_file) as f:
                for line in f:
                    if line.startswith("Overall Pearson r:"):
                        spacet_r = float(line.split(":")[1].strip())
                        break

        print(f"   {label}:")
        print(f"     SpaCET r={spacet_r:.4f}")
        print(f"     MuSiC  r={music_eval['r']:.4f}, rho={music_eval['rho']:.4f}, RMSE={music_eval['rmse']:.4f}")

        results.append({
            "scenario": scenario,
            "label": label,
            "spacet_r": spacet_r,
            "music_r": music_eval["r"],
            "music_rho": music_eval["rho"],
            "music_rmse": music_eval["rmse"],
            "music_per_type": music_eval["per_type"],
        })

    if not results:
        print("\nNo MuSiC results found. Run: sbatch scripts/slurm_music_benchmark.sh")
        return

    # ---- Figure 1: Overall r comparison bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in results]
    spacet_rs = [r["spacet_r"] for r in results]
    music_rs = [r["music_r"] for r in results]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, spacet_rs, w, label="SpaCET (spatial-gpu)", color="#3b82f6")
    ax.bar(x + w / 2, music_rs, w, label="MuSiC", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson r (overall)")
    ax.set_title("SpaCET vs MuSiC — Real BRCA Pseudobulk Benchmark")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (s, m) in enumerate(zip(spacet_rs, music_rs)):
        if not np.isnan(s):
            ax.text(i - w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "benchmark_spacet_vs_music.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n   Saved {path}")

    # ---- Figure 2: Per-type comparison for uniform scenario ----
    uniform = [r for r in results if r["scenario"] == "uniform"]
    if uniform and uniform[0]["music_per_type"]:
        r0 = uniform[0]
        types = sorted(r0["music_per_type"].keys())
        music_type_rs = [r0["music_per_type"][t] for t in types]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(types))
        ax.barh(x, music_type_rs, color="#f97316", alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(types)
        ax.set_xlabel("Pearson r")
        ax.set_title("MuSiC Per-Cell-Type Accuracy (Uniform Dirichlet)")
        ax.set_xlim(-0.1, 1.05)
        for i, r in enumerate(music_type_rs):
            ax.text(max(r + 0.02, 0.05), i, f"{r:.2f}", va="center", fontsize=8)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "benchmark_music_per_type.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"   Saved {path}")

    # ---- Summary table ----
    summary = pd.DataFrame([{
        "Scenario": r["label"],
        "SpaCET_r": r["spacet_r"],
        "MuSiC_r": r["music_r"],
        "MuSiC_rho": r["music_rho"],
        "MuSiC_RMSE": r["music_rmse"],
    } for r in results])
    summary_path = os.path.join(OUTPUTS_DIR, "t8_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"   Saved {summary_path}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
