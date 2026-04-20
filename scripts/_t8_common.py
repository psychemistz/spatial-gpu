"""Shared constants and helpers for the T8 BRCA benchmark scripts.

Single source of truth for:
    - WU_TO_EVAL       : Wu et al. cell-type names -> evaluation categories
    - SCENARIOS        : ordered list of (label, description) for the 4 scenarios
    - remap_and_collapse(df) : rename Wu columns + sum duplicates
    - compute_method_r(pred, gt) : Pearson r / Spearman rho / RMSE vs collapsed GT

Imported by docs/run_full_tutorial_t8_real_brca.py,
scripts/regenerate_dwls_figure.py, and scripts/compare_spacet_music.py.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr

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

SCENARIOS = [
    ("uniform", "Uniform (alpha=1.0)"),
    ("sparse", "Sparse (alpha=0.3)"),
    ("tumor_purity", "Tumor Purity (60-90%)"),
    ("titration", "Titration (0-90%)"),
]


def remap_and_collapse(df):
    """Rename Wu cell types to eval categories and sum duplicate columns."""
    return df.rename(columns=WU_TO_EVAL).T.groupby(level=0).sum().T


def compute_method_r(pred, gt):
    """Compute Pearson r, Spearman rho, RMSE of `pred` vs `gt` after remap+collapse.

    `pred` and `gt` are DataFrames with matching row index (samples) and Wu-style
    cell-type column names. Returns {"r", "rho", "rmse"} — NaN if no common types.
    """
    pred_eval = remap_and_collapse(pred)
    gt_eval = remap_and_collapse(gt)
    common = sorted(set(pred_eval.columns) & set(gt_eval.columns))
    if not common:
        return {"r": np.nan, "rho": np.nan, "rmse": np.nan}
    est = pred_eval[common].values.ravel()
    truth = gt_eval.reindex(pred_eval.index)[common].values.ravel()
    r, _ = pearsonr(est, truth)
    rho, _ = spearmanr(est, truth)
    rmse = np.sqrt(np.mean((est - truth) ** 2))
    return {"r": r, "rho": rho, "rmse": rmse}
