"""Shared metrics + cell-type harmonization for the real-bulk benchmark (T8).

Implements the DREAM Challenge primary scoring rule (cross-sample, within-cell-type
Pearson r, bootstrapped n=1000) plus auxiliary metrics: spillover, LoD, RMSE.

Reference: White et al., Nat Commun 2024 (DREAM Tumor Deconvolution Challenge),
Methods §"Deconvolution method scoring".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DREAM_COARSE = [
    "B.cells", "CD4.T.cells", "CD8.T.cells", "NK.cells",
    "neutrophils", "monocytic.lineage", "endothelial.cells", "fibroblasts",
]
DREAM_FINE = [
    "memory.B.cells", "naive.B.cells",
    "memory.CD4.T.cells", "naive.CD4.T.cells", "regulatory.T.cells",
    "memory.CD8.T.cells", "naive.CD8.T.cells",
    "NK.cells", "neutrophils", "monocytes", "macrophages",
    "myeloid.dendritic.cells", "endothelial.cells", "fibroblasts",
]


def align_cell_types(pred: pd.DataFrame, gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect cell-type columns and reorder so pred/gt share schema.

    Drops any cell type present in one and not the other (warns).
    """
    common = sorted(set(pred.columns) & set(gt.columns))
    missing_pred = sorted(set(gt.columns) - set(pred.columns))
    missing_gt = sorted(set(pred.columns) - set(gt.columns))
    if missing_pred:
        print(f"  [align] in GT, not predicted: {missing_pred}")
    if missing_gt:
        print(f"  [align] predicted, not in GT (dropped from scoring): {missing_gt}")
    return pred[common], gt[common]


def pearson_per_cell_type(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.Series:
    """Cross-sample Pearson r per cell type (DREAM primary axis).

    Returns one r per cell type; constant columns -> NaN.
    """
    r = {}
    for ct in pred.columns:
        p, g = pred[ct].values, gt[ct].values
        mask = ~(np.isnan(p) | np.isnan(g))
        if mask.sum() < 3 or np.std(p[mask]) == 0 or np.std(g[mask]) == 0:
            r[ct] = np.nan
        else:
            r[ct] = pearsonr(p[mask], g[mask])[0]
    return pd.Series(r, name="pearson_r")


def bootstrap_aggregate_r(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    n_boot: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """DREAM aggregate score: bootstrap over samples, average cell-type r.

    Returns {"mean", "lo95", "hi95"} of the across-cell-type mean per bootstrap.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(pred)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rs = pearson_per_cell_type(pred.iloc[idx], gt.iloc[idx])
        scores[b] = np.nanmean(rs.values)
    return {
        "mean": float(np.nanmean(scores)),
        "lo95": float(np.nanpercentile(scores, 2.5)),
        "hi95": float(np.nanpercentile(scores, 97.5)),
    }


def cross_cell_type_within_sample(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Per-sample r/rho/RMSE across cell types — DREAM Fig. 4.

    Only meaningful if pred sums to a fraction-like quantity.
    """
    rows = []
    for s in pred.index:
        p, g = pred.loc[s].values, gt.loc[s].values
        mask = ~(np.isnan(p) | np.isnan(g))
        if mask.sum() < 3:
            rows.append((s, np.nan, np.nan, np.nan))
            continue
        rows.append((
            s,
            pearsonr(p[mask], g[mask])[0],
            spearmanr(p[mask], g[mask])[0],
            float(np.sqrt(np.mean((p[mask] - g[mask]) ** 2))),
        ))
    return pd.DataFrame(rows, columns=["sample", "r", "rho", "rmse"]).set_index("sample")


def spillover_matrix(pred_on_pure: pd.DataFrame, true_label: pd.Series) -> pd.DataFrame:
    """Spillover: prediction on purified-Y samples should be 100% Y, 0% else.

    `pred_on_pure` rows = pure samples, cols = predicted fractions.
    `true_label` maps each row -> true cell type.
    Returns mean predicted fraction per (true, predicted) pair.
    """
    df = pred_on_pure.copy()
    df["__true__"] = true_label.loc[df.index].values
    return df.groupby("__true__").mean()


def limit_of_detection(
    pred: pd.DataFrame, spike_in: pd.DataFrame, cell_type: str, threshold: float = 0.05
) -> float:
    """LoD: minimum true fraction at which predicted ≥ threshold for `cell_type`.

    `spike_in` is the design — rows match `pred.index`, column `cell_type` holds
    the true spike-in fraction. Returns the minimum spike that the model recovers.
    """
    s = spike_in[cell_type].values
    p = pred[cell_type].values
    detected = s[p >= threshold]
    return float(detected.min()) if len(detected) else np.nan
