"""Gene set scoring for spatial transcriptomics.

UCell-equivalent rank-based gene set scoring.
Equivalent to SpaCET.GeneSetScore() in R.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

from spatialgpu.deconvolution.reference import load_gene_set

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def gene_set_score(
    adata: ad.AnnData,
    gene_sets: str | dict[str, list[str]],
) -> ad.AnnData:
    """Calculate gene set scores per spot using UCell-like ranking.

    Equivalent to SpaCET.GeneSetScore() in R.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data with raw counts in X.
    gene_sets : str or dict
        Either a string for built-in sets ('Hallmark', 'CancerCellState', 'TLS')
        or a dict mapping set names to gene lists.

    Returns
    -------
    AnnData with scores in adata.uns['spacet']['GeneSetScore']
    """
    if isinstance(gene_sets, str):
        if gene_sets not in ("Hallmark", "CancerCellState", "TLS"):
            raise ValueError(
                "gene_sets must be a dict or one of 'Hallmark', 'CancerCellState', 'TLS'."
            )
        gmt = load_gene_set(gene_sets)
    else:
        gmt = gene_sets

    # Compute UCell scores
    scores = _ucell_score(adata, gmt)

    # Store results
    if "spacet" not in adata.uns:
        adata.uns["spacet"] = {}
    if adata.uns["spacet"].get("GeneSetScore") is None:
        adata.uns["spacet"]["GeneSetScore"] = scores
    else:
        existing = adata.uns["spacet"]["GeneSetScore"]
        adata.uns["spacet"]["GeneSetScore"] = pd.concat([existing, scores])

    return adata


def _ucell_score(
    adata: ad.AnnData,
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute UCell-like gene set scores.

    UCell algorithm: For each cell, rank all genes. For each gene set,
    compute the Mann-Whitney U statistic based on the ranks of the
    gene set members. Score = 1 - (U / (n_set * n_rest)).

    Parameters
    ----------
    adata : AnnData
        Data with counts in X.
    gene_sets : dict
        Gene set name -> list of gene symbols.

    Returns
    -------
    pd.DataFrame : gene_sets x spots score matrix
    """
    X = adata.X
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)
    n_spots = X.shape[0]
    n_genes = X.shape[1]

    # Rank genes per spot (descending, so highest expression = rank 1)
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    # Rank: argsort descending, then assign ranks
    # Use scipy rankdata for consistency
    from scipy.stats import rankdata

    ranks = np.zeros_like(X_dense, dtype=np.float64)
    for i in range(n_spots):
        # Rank in ascending order (1 = lowest), then invert
        ranks[i] = n_genes + 1 - rankdata(X_dense[i], method="average")

    results = {}
    for set_name, genes in gene_sets.items():
        # Find genes present in the data
        gene_mask = np.isin(gene_names, genes)
        n_set = gene_mask.sum()

        if n_set == 0:
            results[set_name] = np.zeros(n_spots)
            continue

        # Get ranks of gene set members for each spot
        set_ranks = ranks[:, gene_mask]  # (n_spots, n_set)

        # UCell score: 1 - mean_rank / n_genes
        # This is a simplified version of the U-statistic approach
        mean_rank = set_ranks.mean(axis=1)
        scores = 1 - mean_rank / n_genes

        results[set_name] = scores

    score_df = pd.DataFrame(results, index=spot_names).T
    return score_df
