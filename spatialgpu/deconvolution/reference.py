"""Reference data loading for SpaCET deconvolution.

Loads cell type reference profiles, cancer signatures, and ligand-receptor
databases from bundled data files. Uses lazy loading with caching.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anndata as ad

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_comb_ref() -> dict[str, Any]:
    """Load the combined cell type reference (combRef_0.5).

    Returns
    -------
    dict with keys:
        refProfiles : pd.DataFrame
            Gene expression reference profiles (genes x cell_types).
        sigGenes : dict[str, list[str]]
            Signature genes per cell type.
        lineageTree : dict[str, list[str]]
            Hierarchical lineage tree (major -> [sub1, sub2, ...]).
    """
    ref_profiles = pd.read_csv(_DATA_DIR / "combRef_refProfiles.csv", index_col=0)
    with open(_DATA_DIR / "combRef_sigGenes.json") as f:
        sig_genes = json.load(f)
    with open(_DATA_DIR / "combRef_lineageTree.json") as f:
        lineage_tree = json.load(f)

    return {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": lineage_tree,
    }


@lru_cache(maxsize=1)
def load_cancer_dictionary() -> dict[str, dict[str, pd.Series]]:
    """Load cancer-specific CNA and expression signatures.

    Returns
    -------
    dict with keys 'CNA' and 'expr', each mapping cancer type names
    to pd.Series of gene-level signature values.
    """
    with open(_DATA_DIR / "cancerDictionary_index.json") as f:
        index = json.load(f)

    result: dict[str, dict[str, pd.Series]] = {"CNA": {}, "expr": {}}

    for sig_type in ["CNA", "expr"]:
        for name in index.get(sig_type, []):
            safe_name = name.replace(" ", "_").replace("-", "_")
            # Handle any special characters in filename
            import re

            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            fname = f"cancerDict_{sig_type}_{safe_name}.csv"
            fpath = _DATA_DIR / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                series = pd.Series(
                    df["value"].values, index=df["gene"].values, name=name
                )
                result[sig_type][name] = series

    return result


@lru_cache(maxsize=1)
def load_ref_normal_lihc() -> dict[str, Any]:
    """Load liver-specific normal tissue reference (hepatocytes, cholangiocytes).

    Returns
    -------
    dict with keys: refProfiles, sigGenes, lineageTree
    """
    ref_profiles = pd.read_csv(
        _DATA_DIR / "Ref_Normal_LIHC_refProfiles.csv", index_col=0
    )
    with open(_DATA_DIR / "Ref_Normal_LIHC_sigGenes.json") as f:
        sig_genes = json.load(f)
    with open(_DATA_DIR / "Ref_Normal_LIHC_lineageTree.json") as f:
        lineage_tree = json.load(f)

    return {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": lineage_tree,
    }


@lru_cache(maxsize=1)
def load_lr_database() -> pd.DataFrame:
    """Load the Ramilowski 2015 ligand-receptor database.

    Returns
    -------
    pd.DataFrame with columns including ligand and receptor gene names.
    """
    return pd.read_csv(_DATA_DIR / "Ramilowski2015.txt", sep="\t")


def load_gene_set(name: str) -> dict[str, list[str]]:
    """Load a built-in gene set (Hallmark, CancerCellState, or TLS).

    Parameters
    ----------
    name : str
        One of 'Hallmark', 'CancerCellState', 'TLS'.

    Returns
    -------
    dict mapping gene set names to lists of gene symbols.
    """
    return read_gmt(_DATA_DIR / "GeneSets" / f"{name}.gmt")


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Read a GMT gene set file.

    Parameters
    ----------
    path : str or Path
        Path to GMT file.

    Returns
    -------
    dict mapping gene set names to lists of gene symbols.
    """
    gmt: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gmt[parts[0]] = parts[2:]
    return gmt


def write_gmt(gmt: dict[str, list[str]], path: str | Path) -> None:
    """Write a gene set dict as a GMT file.

    Parameters
    ----------
    gmt : dict
        Gene set name -> list of gene symbols.
    path : str or Path
        Output path.
    """
    lines = []
    for name, genes in gmt.items():
        lines.append(f"{name}\t\t{chr(9).join(genes)}")
    Path(path).write_text("\n".join(lines) + "\n")


def get_cancer_signature(
    cancer_type: str, sig_type: str | None = None
) -> tuple[str, pd.Series]:
    """Get the cancer signature for a given cancer type.

    Tries CNA first, then expr, then PANCAN expr (matching R behavior).

    Parameters
    ----------
    cancer_type : str
        Cancer type code (e.g., 'BRCA', 'LIHC', 'CRC').
    sig_type : str or None
        Force signature type: 'CNA', 'expr', or 'seq_depth'. None for auto.

    Returns
    -------
    tuple of (sig_type_used, signature_series)
    """
    cancer_dict = load_cancer_dictionary()

    if sig_type is not None:
        if sig_type == "seq_depth":
            return ("seq_depth", pd.Series(dtype=np.float64))
        for name, series in cancer_dict[sig_type].items():
            if cancer_type in name:
                return (sig_type, series)
        raise ValueError(f"SpaCET does not include {cancer_type} {sig_type} signature.")

    if cancer_type == "PANCAN":
        comb_list = [("expr", "PANCAN")]
    else:
        comb_list = [
            ("CNA", cancer_type),
            ("expr", cancer_type),
            ("expr", "PANCAN"),
        ]

    for st, ct in comb_list:
        for name, series in cancer_dict[st].items():
            if ct in name:
                return (st, series)

    return ("seq_depth", pd.Series(dtype=np.float64))


@lru_cache(maxsize=1)
def _load_mouse2human_map() -> pd.DataFrame:
    """Load the mouse-to-human gene mapping table (cached)."""
    return pd.read_csv(_DATA_DIR / "Mouse2Human_filter.csv", index_col=0)


@lru_cache(maxsize=1)
def _mouse2human_dict() -> dict[str, str]:
    """Build cached mouse-to-human gene name lookup dict."""
    m2h = _load_mouse2human_map()[["mouse", "human"]]
    return dict(zip(m2h["mouse"], m2h["human"]))


_VALID_ORGANISMS = {"human", "mouse"}


def mouse2human_mat(
    counts: np.ndarray,
    gene_names: np.ndarray,
) -> tuple:
    """Convert mouse gene expression matrix to human orthologs.

    Rows (genes) with mouse gene symbols are mapped to human orthologs.
    When multiple mouse genes map to the same human gene, their expression
    values are summed via sparse matrix aggregation.

    Parameters
    ----------
    counts : sparse matrix or ndarray
        Gene expression matrix, shape (genes x spots). The number of rows
        must equal ``len(gene_names)``.
    gene_names : ndarray of str
        Mouse gene symbols corresponding to rows of *counts*.

    Returns
    -------
    tuple of (converted_counts, human_gene_names)
        Expression matrix with human gene names and summed duplicates.
    """
    from scipy import sparse as sp

    if counts.shape[0] != len(gene_names):
        raise ValueError(
            f"counts has {counts.shape[0]} rows but gene_names has "
            f"{len(gene_names)} entries. Counts must be genes x spots."
        )

    mouse_to_human = _mouse2human_dict()

    # Vectorized gene matching
    gene_series = pd.Series(gene_names)
    mapped = gene_series.map(mouse_to_human)
    mask = mapped.notna()
    keep_idx = np.where(mask.values)[0]
    human_names = mapped[mask].values

    if len(keep_idx) == 0:
        raise ValueError("No mouse genes matched the mapping table.")

    # Subset to matched genes
    counts_sub = counts[keep_idx]

    # Group by human gene and sum (sparse aggregation matrix)
    unique_human, group_labels = np.unique(human_names, return_inverse=True)
    n_human = len(unique_human)
    agg = sp.csr_matrix(
        (np.ones(len(group_labels)), (group_labels, np.arange(len(group_labels)))),
        shape=(n_human, len(group_labels)),
    )

    out = agg @ counts_sub

    logger.info(
        "Mapped %d mouse genes to %d unique human orthologs.",
        len(keep_idx),
        n_human,
    )

    return out, unique_human


def ensure_human_genes(
    adata: ad.AnnData,
    counts: np.ndarray,
    gene_names: np.ndarray,
) -> tuple:
    """Convert mouse genes to human orthologs if organism is mouse.

    Parameters
    ----------
    adata : AnnData
        Used to check ``adata.uns.get('spacet_organism')``.
    counts : sparse matrix or ndarray
        Gene expression matrix (genes x spots).
    gene_names : ndarray of str
        Gene symbols corresponding to rows of *counts*.

    Returns
    -------
    tuple of (counts, gene_names) — unchanged if human, converted if mouse.
    """
    organism = adata.uns.get("spacet_organism", "human")
    if organism == "mouse":
        logger.info("Converting mouse genes to human orthologs.")
        return mouse2human_mat(counts, gene_names)
    return counts, gene_names
