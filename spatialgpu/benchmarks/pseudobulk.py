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
    "CAF", "Endothelial", "Plasma", "B cell", "T CD4", "T CD8",
    "NK", "cDC", "pDC", "Macrophage", "Mast", "Neutrophil",
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
            mal_profile.loc[olp] = mal_profile.loc[olp] + sig.loc[olp] * mean_expr.loc[olp].clip(lower=1)
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
            counts = rng.negative_binomial(n=np.maximum(dispersion, 0.01).astype(np.float64), p=p)

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
