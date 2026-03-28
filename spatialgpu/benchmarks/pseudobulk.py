"""Pseudobulk benchmark utilities for evaluating deconvolution accuracy.

Generates semi-synthetic scRNA-seq data, creates pseudobulk mixtures with
known cell type proportions, evaluates deconvolution results, and exports
data for comparison with external tools (MuSiC, CIBERSORTx).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# Level 1 cell types used for pseudobulk mixing
_LEVEL1_TYPES = [
    "CAF",
    "Endothelial",
    "Plasma",
    "B cell",
    "T CD4",
    "T CD8",
    "NK",
    "cDC",
    "pDC",
    "Macrophage",
    "Mast",
    "Neutrophil",
]


def generate_semi_synthetic_scrna(
    n_cells_per_type: int = 500,
    include_malignant: bool = True,
    cancer_type: str = "BRCA",
    seed: int = 42,
) -> ad.AnnData:
    """Generate semi-synthetic scRNA-seq from reference profiles.

    Uses negative binomial noise and logistic dropout to create realistic
    single-cell counts from the bundled SpaCET reference profiles.

    Parameters
    ----------
    n_cells_per_type
        Number of cells to generate per cell type.
    include_malignant
        If True, adds a Malignant_{cancer_type} cell type from the cancer
        signature overlaid on mean expression.
    cancer_type
        Cancer type code for malignant signature (e.g., 'BRCA').
    seed
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Synthetic scRNA-seq with obs['cell_type'] labels and integer counts.
    """
    import anndata as ad

    from spatialgpu.deconvolution.reference import (
        get_cancer_signature,
        load_comb_ref,
    )

    rng = np.random.RandomState(seed)
    ref = load_comb_ref()
    ref_profiles = ref["refProfiles"]

    # Use Level 1 types only
    type_names = [t for t in _LEVEL1_TYPES if t in ref_profiles.columns]
    profiles = ref_profiles[type_names].copy()

    # Add malignant type from cancer signature
    if include_malignant:
        _, sig = get_cancer_signature(cancer_type)
        if sig is not None and len(sig) > 0:
            mal_name = f"Malignant_{cancer_type}"
            mean_expr = ref_profiles[type_names].mean(axis=1)
            mal_profile = mean_expr.copy()
            olp = mal_profile.index.intersection(sig.index)
            mal_profile.loc[olp] = mal_profile.loc[olp] + sig.loc[olp] * mean_expr.loc[
                olp
            ].clip(lower=1)
            mal_profile = mal_profile.clip(lower=0)
            profiles[mal_name] = mal_profile
            type_names.append(mal_name)

    gene_names = np.array(profiles.index)
    n_genes = len(gene_names)

    all_counts = []
    all_labels = []

    for ct in type_names:
        profile = profiles[ct].values.astype(np.float64)
        profile_prob = profile / (profile.sum() + 1e-10)

        for _ in range(n_cells_per_type):
            total_umi = int(rng.lognormal(mean=8.5, sigma=0.5))
            total_umi = max(total_umi, 100)

            mu = profile_prob * total_umi
            dispersion = np.maximum(0.5, mu / 2)
            p = dispersion / (dispersion + mu + 1e-10)
            counts = rng.negative_binomial(
                n=np.maximum(dispersion, 0.01).astype(np.float64), p=p
            )

            log_mu = np.log(mu + 1)
            drop_prob = 1.0 / (1.0 + np.exp(1.5 * log_mu - 2.0))
            dropout_mask = rng.random(n_genes) < drop_prob
            counts[dropout_mask] = 0

            all_counts.append(counts)
            all_labels.append(ct)

    X = np.vstack(all_counts).astype(np.float64)
    X_sparse = sparse.csr_matrix(X)

    adata = ad.AnnData(
        X=X_sparse,
        obs=pd.DataFrame({"cell_type": all_labels}),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obs_names = [f"Cell_{i:06d}" for i in range(adata.n_obs)]

    return adata


def generate_pseudobulk_dirichlet(
    scrna_adata: ad.AnnData,
    n_samples: int = 100,
    n_cells_per_sample: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Generate pseudobulk by mixing single cells with Dirichlet proportions.

    Parameters
    ----------
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    n_samples
        Number of pseudobulk samples to generate.
    n_cells_per_sample
        Total cells to sample per pseudobulk mixture.
    alpha
        Dirichlet concentration. 1.0 = uniform; < 1.0 = sparser mixtures.
    seed
        Random seed.

    Returns
    -------
    tuple of (AnnData, DataFrame)
        AnnData: pseudobulk (samples x genes, raw counts).
        DataFrame: ground truth proportions (samples x cell_types, sums to 1.0).
    """
    import anndata as ad

    rng = np.random.RandomState(seed)
    cell_types = sorted(scrna_adata.obs["cell_type"].unique())
    n_types = len(cell_types)

    type_indices = {}
    for ct in cell_types:
        type_indices[ct] = np.where(scrna_adata.obs["cell_type"].values == ct)[0]

    X_all = scrna_adata.X
    if sparse.issparse(X_all):
        X_all = X_all.toarray()

    bulk_counts = np.zeros((n_samples, scrna_adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, n_types), dtype=np.float64)
    alpha_vec = np.full(n_types, alpha)

    for i in range(n_samples):
        props = rng.dirichlet(alpha_vec)
        proportions[i] = props
        cell_counts = rng.multinomial(n_cells_per_sample, props)

        sample_sum = np.zeros(scrna_adata.n_vars, dtype=np.float64)
        for j, ct in enumerate(cell_types):
            if cell_counts[j] == 0:
                continue
            idx = rng.choice(type_indices[ct], size=cell_counts[j], replace=True)
            sample_sum += X_all[idx].sum(axis=0)

        bulk_counts[i] = sample_sum

    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=[f"Bulk_{i:04d}" for i in range(n_samples)]),
        var=pd.DataFrame(index=scrna_adata.var_names.copy()),
    )

    ground_truth = pd.DataFrame(
        proportions,
        index=adata_bulk.obs_names,
        columns=cell_types,
    )

    return adata_bulk, ground_truth


def generate_pseudobulk_titration(
    scrna_adata: ad.AnnData,
    target_type: str = "Malignant_BRCA",
    fractions: list[float] | None = None,
    n_replicates: int = 5,
    n_cells_per_sample: int = 1000,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Generate pseudobulk with systematic titration of one cell type.

    Fixes the target type at each specified fraction and distributes
    the remainder across other types via Dirichlet(1.0).

    Parameters
    ----------
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    target_type
        Cell type to titrate (e.g., 'Malignant_BRCA').
    fractions
        Target fractions to sweep. Default: [0.0, 0.1, ..., 0.8].
    n_replicates
        Number of replicates per fraction.
    n_cells_per_sample
        Total cells per pseudobulk sample.
    seed
        Random seed.

    Returns
    -------
    tuple of (AnnData, DataFrame)
        AnnData: pseudobulk (samples x genes).
        DataFrame: ground truth with extra 'target_fraction' column.
    """
    import anndata as ad

    if fractions is None:
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rng = np.random.RandomState(seed)
    cell_types = sorted(scrna_adata.obs["cell_type"].unique())

    if target_type not in cell_types:
        raise ValueError(
            f"target_type '{target_type}' not in scrna_adata cell types: {cell_types}"
        )

    other_types = [ct for ct in cell_types if ct != target_type]
    n_other = len(other_types)

    type_indices = {}
    for ct in cell_types:
        type_indices[ct] = np.where(scrna_adata.obs["cell_type"].values == ct)[0]

    X_all = scrna_adata.X
    if sparse.issparse(X_all):
        X_all = X_all.toarray()

    all_counts = []
    all_proportions = []
    all_target_fracs = []

    for frac in fractions:
        for _rep in range(n_replicates):
            props = {}
            props[target_type] = frac

            if frac < 1.0 - 1e-10:
                remainder_props = rng.dirichlet(np.ones(n_other))
                for j, ct in enumerate(other_types):
                    props[ct] = remainder_props[j] * (1.0 - frac)
            else:
                for ct in other_types:
                    props[ct] = 0.0

            sample_sum = np.zeros(scrna_adata.n_vars, dtype=np.float64)
            for ct in cell_types:
                n_cells = int(round(props[ct] * n_cells_per_sample))
                if n_cells == 0:
                    continue
                idx = rng.choice(type_indices[ct], size=n_cells, replace=True)
                sample_sum += X_all[idx].sum(axis=0)

            all_counts.append(sample_sum)
            all_proportions.append([props[ct] for ct in cell_types])
            all_target_fracs.append(frac)

    n_total = len(all_counts)
    adata_bulk = ad.AnnData(
        X=np.vstack(all_counts),
        obs=pd.DataFrame(index=[f"Titr_{i:04d}" for i in range(n_total)]),
        var=pd.DataFrame(index=scrna_adata.var_names.copy()),
    )

    ground_truth = pd.DataFrame(
        np.array(all_proportions),
        index=adata_bulk.obs_names,
        columns=cell_types,
    )
    ground_truth["target_fraction"] = all_target_fracs

    return adata_bulk, ground_truth


def _collapse_to_level1(prop_mat: pd.DataFrame, level1_types: list) -> pd.DataFrame:
    """Collapse hierarchical propMat to Level 1 types.

    The propMat from deconvolution_bulk has both Level 1 and Level 2 rows.
    For evaluation against pseudobulk mixed at Level 1 granularity, we
    keep only Level 1 rows.
    """
    available = [t for t in level1_types if t in prop_mat.index]
    return prop_mat.loc[available]


def evaluate_deconvolution(
    estimated: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """Evaluate deconvolution accuracy against known proportions.

    Parameters
    ----------
    estimated
        Estimated proportions (samples x cell_types) or (cell_types x samples).
    ground_truth
        True proportions (samples x cell_types).

    Returns
    -------
    dict with keys: overall, per_type, rare_type_mae
    """
    from scipy import stats

    est = estimated.copy()
    gt = ground_truth.copy()

    # Auto-transpose if needed
    if est.shape[0] != gt.shape[0] and est.shape[1] == gt.shape[0]:
        est = est.T
    if est.shape[1] != gt.shape[1] and est.shape[0] == gt.shape[1]:
        est = est.T

    common_types = est.columns.intersection(gt.columns)
    common_samples = est.index.intersection(gt.index)

    if len(common_types) == 0:
        raise ValueError("No common cell types between estimated and ground truth.")

    n_gt_types = len(gt.columns)
    if len(common_types) < 0.8 * n_gt_types:
        logger.warning(
            "evaluate_deconvolution: only %d/%d cell types overlap (%.0f%%).",
            len(common_types),
            n_gt_types,
            100 * len(common_types) / n_gt_types,
        )

    est_aligned = est.loc[common_samples, common_types].values.astype(np.float64)
    gt_aligned = gt.loc[common_samples, common_types].values.astype(np.float64)

    est_flat = est_aligned.ravel()
    gt_flat = gt_aligned.ravel()

    pearson_r, _ = stats.pearsonr(est_flat, gt_flat)
    spearman_rho, _ = stats.spearmanr(est_flat, gt_flat)
    rmse = float(np.sqrt(np.mean((est_flat - gt_flat) ** 2)))

    per_type_rows = []
    for i, ct in enumerate(common_types):
        e = est_aligned[:, i]
        g = gt_aligned[:, i]
        if np.std(g) < 1e-15 or np.std(e) < 1e-15:
            r = np.nan
        else:
            r, _ = stats.pearsonr(e, g)
        ct_rmse = float(np.sqrt(np.mean((e - g) ** 2)))
        per_type_rows.append(
            {
                "cell_type": ct,
                "pearson_r": r,
                "rmse": ct_rmse,
                "n_samples": len(common_samples),
            }
        )

    per_type = pd.DataFrame(per_type_rows).set_index("cell_type")

    rare_mask = gt_aligned < 0.05
    if rare_mask.sum() > 0:
        rare_mae = float(np.mean(np.abs(est_aligned[rare_mask] - gt_aligned[rare_mask])))
    else:
        rare_mae = 0.0

    return {
        "overall": {
            "pearson_r": float(pearson_r),
            "spearman_rho": float(spearman_rho),
            "rmse": rmse,
        },
        "per_type": per_type,
        "rare_type_mae": rare_mae,
    }
