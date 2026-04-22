"""Generate per-cell-type scatter plots of SpaCET (IRWLS) predictions vs
ground truth on the uniform and tumor_purity scenarios, using a representative
trial's outputs.

Outputs:
  docs/figures/benchmark_spacet_irwls_scatter_uniform.png
  docs/figures/benchmark_spacet_irwls_scatter_tumor_purity.png
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from _t8_common import remap_and_collapse  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

TRIAL = "T00"  # representative trial
SCENARIOS = [
    ("uniform", "Uniform Dirichlet (α=1.0)"),
    ("tumor_purity", "Tumor Purity (60–90%)"),
]
TRIAL_DIR = os.path.join(_REPO_ROOT, "docs", "outputs", "trials", TRIAL)
FIGURES_DIR = os.path.join(_REPO_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    for scenario, title in SCENARIOS:
        pred_path = os.path.join(TRIAL_DIR, f"spacet_v3_irwls_{scenario}.csv")
        gt_path = os.path.join(TRIAL_DIR, f"gt_{scenario}.csv")
        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f"skip {scenario}: missing files")
            continue
        pred = pd.read_csv(pred_path, index_col=0)  # already collapsed
        gt = remap_and_collapse(pd.read_csv(gt_path, index_col=0))
        common = sorted(set(pred.columns) & set(gt.columns))
        gt_aligned = gt.reindex(pred.index)[common]

        overall_r, _ = pearsonr(
            pred[common].values.ravel(),
            gt_aligned.values.ravel(),
        )

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        colors = plt.cm.tab10(range(len(common)))
        for i, ct in enumerate(common):
            ct_r, _ = pearsonr(pred[ct].values, gt_aligned[ct].values)
            ax.scatter(
                gt_aligned[ct],
                pred[ct],
                alpha=0.55,
                s=18,
                color=colors[i],
                label=f"{ct} (r={ct_r:.2f})",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Ground truth fraction")
        ax.set_ylabel("SpaCET (IRWLS) predicted fraction")
        ax.set_title(
            f"SpaCET (IRWLS) — {title}\n"
            f"overall r = {overall_r:.3f}  (trial {TRIAL})"
        )
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        plt.tight_layout()
        out = os.path.join(
            FIGURES_DIR, f"benchmark_spacet_irwls_scatter_{scenario}.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out}  (overall r={overall_r:.3f})")


if __name__ == "__main__":
    main()
