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

    # Compute UCell scores: try R first, fall back to Python
    try:
        scores = _ucell_score_via_r(adata, gmt)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"R/UCell not available ({e}), using Python approximation. "
            "Gene set scores may differ from SpaCET.",
            stacklevel=2,
        )
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


def _ucell_score_via_r(
    adata: ad.AnnData,
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Run UCell::ScoreSignatures_UCell in R for exact SpaCET match."""
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    # Get counts as genes x spots sparse matrix
    X = adata.X
    if sparse.issparse(X):
        counts = X.T.tocsc().astype(np.float64)
    else:
        counts = sparse.csc_matrix(X.T, dtype=np.float64)

    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    tmpdir = tempfile.mkdtemp()
    input_mtx = os.path.join(tmpdir, "counts.mtx")
    input_genes = os.path.join(tmpdir, "genes.csv")
    input_spots = os.path.join(tmpdir, "spots.csv")
    input_gmt = os.path.join(tmpdir, "gene_sets.json")
    output_scores = os.path.join(tmpdir, "scores.csv")

    mmwrite(input_mtx, counts)
    pd.DataFrame({"gene": gene_names}).to_csv(input_genes, index=False)
    pd.DataFrame({"spot": spot_names}).to_csv(input_spots, index=False)

    # Convert gene sets to JSON for R
    with open(input_gmt, "w") as f:
        json.dump(gene_sets, f)

    r_code = f"""
    suppressPackageStartupMessages({{
        library(Matrix)
        library(UCell)
        library(jsonlite)
    }})
    suppressWarnings({{
        counts <- readMM("{input_mtx}")
        counts <- as(counts, "CsparseMatrix")
    }})
    genes <- read.csv("{input_genes}")$gene
    spots <- read.csv("{input_spots}")$spot
    rownames(counts) <- genes
    colnames(counts) <- spots

    gmt <- fromJSON("{input_gmt}")

    res <- t(UCell::ScoreSignatures_UCell(counts, gmt, name=""))
    write.csv(as.matrix(res), "{output_scores}", row.names=TRUE)
    """

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R UCell failed: {result.stderr}")

        scores_df = pd.read_csv(output_scores, index_col=0)
        scores_df.columns = spot_names
        return scores_df.astype(np.float64)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
