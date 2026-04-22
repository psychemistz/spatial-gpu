"""SpaCET extension functions for spatial transcriptomics deconvolution.

Functions for:
  - Malignant cell state discovery (deconvolution_malignant)
  - Deconvolution with matched scRNA-seq (deconvolution_matched_scrnaseq)
  - Malignant deconvolution with custom scRNA-seq (deconvolution_malignant_custom_scrnaseq)
  - Reference generation from scRNA-seq (generate_ref)

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from statsmodels.stats.multitest import multipletests

from spatialgpu.core.array_utils import filter_zero_genes, to_dense_float64
from spatialgpu.deconvolution._keys import (
    COL_CELLTYPE,
    KEY_DECONV,
    KEY_MALPROP,
    KEY_MALREF,
    KEY_PROPMAT,
    KEY_PROPMAT_COLS,
    KEY_REF,
    LABEL_MALIGNANT,
    LABEL_UNIDENTIFIABLE,
    OBSM_SPACET_PROPMAT,
    UNS_SPACET,
)
from spatialgpu.deconvolution.core import (
    _get_counts_genes_by_spots,
    _intersect_and_normalize,
    _spatial_deconv,
)
from spatialgpu.deconvolution.reference import ensure_human_genes

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deconvolution_malignant(
    adata: ad.AnnData,
    malignant: str = LABEL_MALIGNANT,
    malignant_cutoff: float = 0.7,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Explore different malignant cell states in tumor ST data.

    Equivalent to SpaCET.deconvolution.malignant() in R.

    Clusters spots with high malignant fraction to identify distinct
    malignant cell states, then re-deconvolves the malignant fraction
    into those states.

    Parameters
    ----------
    adata : AnnData
        Must already have deconvolution results in ``adata.uns['spacet']``.
    malignant : str
        Name of the malignant cell type in the existing deconvolution.
    malignant_cutoff : float
        Fraction cutoff (0-1) for selecting spots with high malignant content.
    n_jobs : int
        Number of parallel jobs for deconvolution.

    Returns
    -------
    AnnData with updated ``adata.uns['spacet']['deconvolution']['propMat']``
    including malignant cell state sub-fractions.
    """
    # --- Validation ---
    deconv, res_deconv, lineage_tree = _validate_malignant_inputs(
        adata, malignant, malignant_cutoff
    )

    # --- Get counts (genes x spots) ---
    counts, gene_names, spot_names = _prepare_counts(adata)

    # --- Select malignant spots and CPM-normalize ---
    mal_spots, cpm_mal, log_mal, counts_mal, mal_spot_idx = (
        _select_and_normalize_malignant_spots(
            res_deconv, malignant, malignant_cutoff, counts, spot_names, gene_names
        )
    )

    # --- Cluster malignant spots into states ---
    states, content = _cluster_malignant_spots(
        counts_mal, gene_names, mal_spots, cpm_mal
    )

    # --- Build new reference from clustered states ---
    ref_new = _build_malignant_reference(
        gene_names, cpm_mal, log_mal, states, content, malignant
    )

    # --- Re-deconvolve malignant fraction ---
    mal_prop_known, mal_ref_known = _prepare_known_fractions(
        deconv, res_deconv, lineage_tree, malignant
    )

    logger.info("Re-deconvolving malignant cell states.")
    prop_mat_new = _spatial_deconv(
        ST=counts,
        gene_names=gene_names,
        spot_names=spot_names,
        ref=ref_new,
        mal_prop=mal_prop_known,
        mal_ref=mal_ref_known,
        mode="deconvMal",
        n_jobs=n_jobs,
    )

    # --- Merge results and store ---
    _merge_and_store_results(adata, deconv, res_deconv, prop_mat_new, malignant)

    return adata


def deconvolution_matched_scrnaseq(
    adata: ad.AnnData,
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
    sc_include_malignant: bool = True,
    cancer_type: str | None = None,
    sc_downsampling: bool = True,
    sc_n_cell_each_lineage: int = 100,
    cross_subject_weighting: bool = False,
    subject_col: str | None = None,
    weighting_method: str = "ratio",
    n_jobs: int = 1,
) -> ad.AnnData:
    """Deconvolve ST data with matched scRNA-seq reference.

    Equivalent to SpaCET.deconvolution.matched.scRNAseq() in R.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data with raw counts.
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells). If DataFrame, index = gene
        names and columns = cell IDs.
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Hierarchical lineage tree. Keys = major lineages, values = lists of
        sub-lineages. If a major lineage has no sub-lineages, value = [itself].
    sc_include_malignant : bool
        Whether the scRNA-seq data includes malignant cells. If False,
        ``cancer_type`` must be provided to infer malignant fraction.
    cancer_type : str or None
        Cancer type code. Required when ``sc_include_malignant=False``.
    sc_downsampling : bool
        Whether to downsample cells per type.
    sc_n_cell_each_lineage : int
        Max cells per lineage for downsampling (seed=123).
    cross_subject_weighting : bool
        If True, weight genes by inverse cross-subject variance (MuSiC-style).
        Genes stable across subjects get higher weight, noisy genes get
        downweighted. Requires ``subject_col``.
    subject_col : str or None
        Column name in ``sc_annotation`` containing subject/donor IDs.
        Required when ``cross_subject_weighting=True``.
    weighting_method : str
        Active when ``cross_subject_weighting=True``. One of:

        - ``"ratio"`` (default): SpaCET's scale-free form,
          ``w = 1 / (1 + var_between/var_within)``.
        - ``"irwls"``: One residual-informed reweighting pass on top of
          absolute-variance initial weights ``w_0 = 1 / (var_between +
          var_within)``: solve, compute per-gene residual variance from
          the first solve, update ``w_1 = 1 / (var_between + var_within +
          residual_var)``, then re-solve. Strict Pareto improvement over
          ``"ratio"`` on the T8 BRCA benchmark. (Multi-iteration IRWLS was
          tested and dropped — additional passes hurt tumor-dominated
          scenarios because the malignant under-prediction is a
          hierarchical-constraint artifact that residual reweighting
          cannot fix.)
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with deconvolution results in ``adata.uns['spacet']``.
    """
    from spatialgpu.deconvolution.core import _infer_mal_cor

    # --- Validate inputs ---
    sc_counts, sc_annotation = _validate_sc_inputs(
        sc_counts, sc_annotation, sc_lineage_tree
    )

    # --- Downsampling ---
    if sc_downsampling:
        logger.info(f"Down-sampling: True, n={sc_n_cell_each_lineage}")
        sc_counts, sc_annotation = _downsample_cells(
            sc_counts, sc_annotation, sc_n_cell_each_lineage
        )
    else:
        logger.info("Down-sampling: False")

    # Filter zero-sum genes
    if isinstance(sc_counts, pd.DataFrame):
        row_sums = sc_counts.sum(axis=1)
        sc_counts = sc_counts.loc[row_sums > 0]
    else:
        row_sums = sc_counts.sum(axis=1)
        keep = row_sums > 0
        sc_counts = sc_counts[keep]

    # --- Generate reference ---
    logger.info("1. Generate the cell type reference from the matched scRNAseq data.")
    ref = generate_ref(sc_counts, sc_annotation, sc_lineage_tree, n_jobs=n_jobs)

    # --- Cross-subject gene weighting (MuSiC-style) ---
    gene_weights = None
    if cross_subject_weighting:
        if subject_col is None:
            raise ValueError(
                "subject_col is required when cross_subject_weighting=True."
            )
        if subject_col not in sc_annotation.columns:
            raise ValueError(
                f"subject_col='{subject_col}' not found in sc_annotation. "
                f"Available columns: {list(sc_annotation.columns)}"
            )
        valid_methods = {"ratio", "irwls"}
        if weighting_method not in valid_methods:
            raise ValueError(
                f"weighting_method must be one of {sorted(valid_methods)}, "
                f"got {weighting_method!r}."
            )
        logger.info(f"   Computing cross-subject gene weights ({weighting_method})...")
        if weighting_method == "ratio":
            gene_weights = _compute_cross_subject_weights(
                sc_counts, sc_annotation, subject_col
            )
        else:  # "irwls"
            gene_weights = _compute_cross_subject_weights_absolute(
                sc_counts, sc_annotation, subject_col
            )
            # Stash variance components for the residual-reweighting passes.
            vc_genes, vc = _subject_variance_components(
                sc_counts, sc_annotation, subject_col
            )
            ref["_irwls_variance"] = (vc_genes, vc)
        logger.info(
            "   Gene weights: %d genes, median=%.4f, mean=%.4f",
            len(gene_weights),
            np.median(gene_weights.values),
            np.mean(gene_weights.values),
        )
        ref["gene_weights"] = gene_weights
        ref["weighting_method"] = weighting_method

    # --- Get ST counts ---
    logger.info("2. Hierarchically deconvolve the Spatial Transcriptomics dataset.")
    counts = _get_counts_genes_by_spots(adata)
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    # Filter zero-sum genes
    counts, gene_names = filter_zero_genes(counts, gene_names)

    def _run_deconv(ref_arg):
        if sc_include_malignant:
            mal_prop = pd.Series(0.0, index=spot_names)
            return _spatial_deconv(
                ST=counts,
                gene_names=gene_names,
                spot_names=spot_names,
                ref=ref_arg,
                mal_prop=mal_prop,
                mal_ref=None,
                mode="deconvWithSC",
                unidentifiable=True,
                macrophage_other=False,
                n_jobs=n_jobs,
            )
        if cancer_type is None:
            raise ValueError("cancer_type is required when sc_include_malignant=False.")
        logger.info("Stage 1. Infer malignant cell fraction.")
        mal_res = _infer_mal_cor(
            counts, gene_names, spot_names, cancer_type, signature_type=None
        )
        logger.info("Stage 2. Deconvolve non-malignant cell fraction.")
        return _spatial_deconv(
            ST=counts,
            gene_names=gene_names,
            spot_names=spot_names,
            ref=ref_arg,
            mal_prop=mal_res[KEY_MALPROP],
            mal_ref=mal_res[KEY_MALREF],
            mode="deconvWithSC_alt",
            unidentifiable=True,
            macrophage_other=False,
            n_jobs=n_jobs,
        )

    prop_mat = _run_deconv(ref)

    # IRWLS-lite: one residual-informed reweighting pass.
    if cross_subject_weighting and weighting_method == "irwls":
        updated = _irwls_lite_updated_weights(ref, counts, gene_names, prop_mat)
        ref["gene_weights"] = updated
        logger.info(
            "IRWLS-lite: updated weights (median=%.4f mean=%.4f)",
            float(np.median(updated.values)),
            float(np.mean(updated.values)),
        )
        prop_mat = _run_deconv(ref)

    # Store results
    adata.uns[UNS_SPACET] = {
        KEY_DECONV: {
            KEY_PROPMAT: prop_mat,
            KEY_REF: ref,
        },
        KEY_PROPMAT_COLS: list(prop_mat.index),
    }
    adata.obsm[OBSM_SPACET_PROPMAT] = prop_mat.T.reindex(adata.obs_names).values

    return adata


def deconvolution_bulk(
    adata: ad.AnnData,
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
    sc_include_malignant: bool = True,
    cancer_type: str | None = None,
    sc_downsampling: bool = True,
    sc_n_cell_each_lineage: int = 100,
    cross_subject_weighting: bool = False,
    subject_col: str | None = None,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Cell-type deconvolution for bulk RNA-seq using a matched scRNA-seq reference.

    Thin wrapper around :func:`deconvolution_matched_scrnaseq`. Each sample in
    ``adata`` is treated as an independent mixture; cell-type fractions are
    solved jointly via hierarchical constrained NNLS against the user-supplied
    scRNA-seq reference. No malignant-first cascade, no within-cohort purity
    rescaling — when ``sc_include_malignant=True``, the malignant cluster is
    deconvolved jointly with non-malignant cell types.

    Parameters
    ----------
    adata : AnnData
        Bulk RNA-seq data. ``X`` = raw counts, ``obs`` = samples, ``var`` = genes.
    sc_counts, sc_annotation, sc_lineage_tree
        Matched scRNA-seq reference. See
        :func:`deconvolution_matched_scrnaseq` for format.
    sc_include_malignant : bool
        Whether the scRNA-seq reference already contains malignant cells.
        Default True. If False, ``cancer_type`` must be provided.
    cancer_type : str or None
        Required only when ``sc_include_malignant=False``.
    sc_downsampling, sc_n_cell_each_lineage
        Reference downsampling controls.
    cross_subject_weighting, subject_col
        MuSiC-style cross-subject gene weighting (optional).
    n_jobs : int
        Parallel jobs.

    Returns
    -------
    AnnData with results in ``adata.uns['spacet']`` and ``adata.obsm['spacet_propMat']``.
    """
    return deconvolution_matched_scrnaseq(
        adata,
        sc_counts=sc_counts,
        sc_annotation=sc_annotation,
        sc_lineage_tree=sc_lineage_tree,
        sc_include_malignant=sc_include_malignant,
        cancer_type=cancer_type,
        sc_downsampling=sc_downsampling,
        sc_n_cell_each_lineage=sc_n_cell_each_lineage,
        cross_subject_weighting=cross_subject_weighting,
        subject_col=subject_col,
        n_jobs=n_jobs,
    )


def deconvolution_malignant_custom_scrnaseq(
    adata: ad.AnnData,
    malignant: str = LABEL_MALIGNANT,
    sc_counts: pd.DataFrame | np.ndarray | None = None,
    sc_annotation: pd.DataFrame | None = None,
    sc_lineage_tree: dict[str, list[str]] | None = None,
    sc_n_cell_each_lineage: int = 100,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Deconvolve malignant fraction using custom scRNA-seq reference.

    Equivalent to SpaCET.deconvolution.malignant.customized.scRNAseq() in R.

    Uses user-provided scRNA-seq data to build a malignant-specific reference,
    then re-deconvolves the malignant fraction into sub-states.

    Parameters
    ----------
    adata : AnnData
        Must already have deconvolution results in ``adata.uns['spacet']``.
    malignant : str
        Name of the malignant cell type in existing results.
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Lineage tree with exactly one entry for the malignant lineage.
    sc_n_cell_each_lineage : int
        Max cells per lineage for downsampling.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    AnnData with updated deconvolution results.
    """
    # --- Validation ---
    if UNS_SPACET not in adata.uns or KEY_DECONV not in adata.uns[UNS_SPACET]:
        raise ValueError(
            "Please run deconvolution first using spatialgpu.deconvolution.core.deconvolution."
        )

    deconv = adata.uns[UNS_SPACET][KEY_DECONV]
    res_deconv: pd.DataFrame = deconv[KEY_PROPMAT]

    if malignant not in res_deconv.index:
        raise ValueError(
            f"Malignant cell type '{malignant}' not found in deconvolution results."
        )

    lineage_tree_orig = deconv[KEY_REF]["lineageTree"]
    if malignant in lineage_tree_orig and len(lineage_tree_orig[malignant]) > 1:
        raise ValueError(
            "Deconvolution results already include multiple malignant cell states."
        )

    if sc_counts is None or sc_annotation is None or sc_lineage_tree is None:
        raise ValueError(
            "sc_counts, sc_annotation, and sc_lineage_tree are all required."
        )

    if len(sc_lineage_tree) != 1:
        raise ValueError(
            "sc_lineage_tree must have exactly one entry for the malignant lineage."
        )

    sc_counts, sc_annotation = _validate_sc_inputs(
        sc_counts, sc_annotation, sc_lineage_tree
    )

    # --- Downsampling ---
    sc_counts, sc_annotation = _downsample_cells(
        sc_counts, sc_annotation, sc_n_cell_each_lineage
    )

    # Filter zero-sum genes
    if isinstance(sc_counts, pd.DataFrame):
        row_sums = sc_counts.sum(axis=1)
        sc_counts = sc_counts.loc[row_sums > 0]
    else:
        row_sums = sc_counts.sum(axis=1)
        keep = row_sums > 0
        sc_counts = sc_counts[keep]

    # --- Generate reference ---
    logger.info("1. Generate the reference from the input scRNAseq data.")
    ref_new = generate_ref(sc_counts, sc_annotation, sc_lineage_tree, n_jobs=n_jobs)

    # --- Get ST counts ---
    logger.info("2. Deconvolve malignant cells.")
    counts = _get_counts_genes_by_spots(adata)
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    # Filter zero-sum genes
    counts, gene_names = filter_zero_genes(counts, gene_names)

    # --- Known cell fractions (non-malignant) ---
    known_cell_types = [k for k in lineage_tree_orig.keys() if k != malignant]
    known_fractions = list(known_cell_types)
    if LABEL_UNIDENTIFIABLE in res_deconv.index:
        known_fractions.append(LABEL_UNIDENTIFIABLE)

    mal_prop_known = res_deconv.loc[known_fractions]

    # Known cell reference
    orig_ref = deconv[KEY_REF]["refProfiles"]
    if isinstance(orig_ref, pd.DataFrame):
        known_cols = [c for c in known_cell_types if c in orig_ref.columns]
        mal_ref_known = orig_ref[known_cols]
    else:
        mal_ref_known = None

    # --- Deconvolve ---
    prop_mat_new = _spatial_deconv(
        ST=counts,
        gene_names=gene_names,
        spot_names=spot_names,
        ref=ref_new,
        mal_prop=mal_prop_known,
        mal_ref=mal_ref_known,
        mode="deconvMal",
        n_jobs=n_jobs,
    )

    # Merge: keep existing + add new rows (exclude the parent lineage name)
    lineage_parent = list(sc_lineage_tree.keys())[0]
    new_rows = prop_mat_new.loc[~prop_mat_new.index.isin([lineage_parent])]
    prop_mat_merged = pd.concat([res_deconv, new_rows])

    # Update adata
    deconv[KEY_PROPMAT] = prop_mat_merged
    deconv[KEY_MALREF] = ref_new
    adata.uns[UNS_SPACET][KEY_DECONV] = deconv
    adata.uns[UNS_SPACET][KEY_PROPMAT_COLS] = list(prop_mat_merged.index)
    adata.obsm[OBSM_SPACET_PROPMAT] = prop_mat_merged.T.reindex(adata.obs_names).values

    return adata


def generate_ref(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Generate cell type reference from scRNA-seq data.

    Equivalent to SpaCET's generateRef() in R.

    For each major lineage and sub-lineage:
      1. CPM normalize (1e5, not 1e6).
      2. Compute mean expression as refProfile.
      3. DE analysis via t-test (limma equivalent): top 500 by t-stat,
         filter logFC > 0.25 and FDR < 0.01.
      4. Signature genes = significant in >= (n_lineages - 1) comparisons.

    Parameters
    ----------
    sc_counts : pd.DataFrame or np.ndarray
        scRNA-seq count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Two-column DataFrame with 'cellID' and 'cellType'.
    sc_lineage_tree : dict
        Keys = major lineages, values = lists of sub-lineages.
    n_jobs : int
        Number of parallel jobs (reserved for future use).

    Returns
    -------
    dict with keys:
        refProfiles : pd.DataFrame (genes x cell_types)
        sigGenes : dict[str, list[str]]
        lineageTree : dict[str, list[str]]
    """
    # Ensure sc_counts is a DataFrame with gene names as index
    if not isinstance(sc_counts, pd.DataFrame):
        raise TypeError("sc_counts must be a pd.DataFrame with gene names as index.")

    # Build cell ID -> cell type mapping
    if (
        "cellID" not in sc_annotation.columns
        or COL_CELLTYPE not in sc_annotation.columns
    ):
        raise ValueError("sc_annotation must have 'cellID' and 'cellType' columns.")
    sc_annotation = sc_annotation.copy()
    sc_annotation.index = sc_annotation["cellID"].astype(str).values

    # Ensure consistent ordering
    cell_types = sc_annotation[COL_CELLTYPE].values.astype(str)

    gene_names = np.array(sc_counts.index)

    # --- CPM normalization (1e5) ---
    counts_dense = to_dense_float64(sc_counts.values)

    col_sums = counts_dense.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    sc_norm = counts_dense / col_sums[np.newaxis, :] * 1e5
    sc_log2 = np.log2(sc_norm + 1)

    # --- Build reference ---
    cell_types_level_1 = list(sc_lineage_tree.keys())
    cell_types_to_be_split = [
        ct
        for ct in cell_types_level_1
        if len(sc_lineage_tree[ct]) != 1 or sc_lineage_tree[ct][0] != ct
    ]

    ref_profiles = pd.DataFrame(index=gene_names, dtype=np.float64)
    sig_genes: dict[str, list[str]] = {}

    for cell_type in cell_types_level_1:
        logger.info(f"  {cell_type}")

        # Cells belonging to this lineage (all subtypes)
        subtypes_of_ct = sc_lineage_tree[cell_type]
        ct_mask = np.isin(cell_types, subtypes_of_ct)
        ct_col_idx = np.where(ct_mask)[0]

        # Reference profile: mean CPM across all cells in this lineage
        ref_profiles[cell_type] = sc_norm[:, ct_col_idx].mean(axis=1)

        # --- DE analysis against each other major lineage ---
        if len(cell_types_level_1) > 1:
            all_markers: list[list[str]] = []

            for other_ct in cell_types_level_1:
                if other_ct == cell_type:
                    continue

                other_subtypes = sc_lineage_tree[other_ct]
                other_mask = np.isin(cell_types, other_subtypes)

                markers = _de_ttest(sc_log2, gene_names, ct_mask, other_mask, n_top=500)
                all_markers.append(markers)

            # Signature genes: present in >= (n_lineages - 1) comparisons
            if all_markers:
                flat_markers = [m for sublist in all_markers for m in sublist]
                marker_counts = pd.Series(flat_markers).value_counts()
                threshold = len(cell_types_level_1) - 1
                sig_genes[cell_type] = list(
                    marker_counts[marker_counts >= threshold].index
                )

        # --- Sub-lineage level ---
        if cell_type in cell_types_to_be_split:
            subtypes = sc_lineage_tree[cell_type]

            for subtype in subtypes:
                sub_mask = cell_types == subtype
                sub_col_idx = np.where(sub_mask)[0]

                ref_profiles[subtype] = sc_norm[:, sub_col_idx].mean(axis=1)

                # DE: subtype vs rest of the same lineage
                other_subtypes_in_lineage = [s for s in subtypes if s != subtype]
                other_sub_mask = np.isin(cell_types, other_subtypes_in_lineage)

                if other_sub_mask.sum() > 0:
                    markers = _de_ttest(
                        sc_log2, gene_names, sub_mask, other_sub_mask, n_top=500
                    )
                    sig_genes[subtype] = markers
                else:
                    sig_genes[subtype] = []

    return {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": sc_lineage_tree,
    }


# ---------------------------------------------------------------------------
# Internal helpers — deconvolution_malignant
# ---------------------------------------------------------------------------


def _validate_malignant_inputs(
    adata: ad.AnnData,
    malignant: str,
    malignant_cutoff: float,
) -> tuple[dict, pd.DataFrame, dict]:
    """Validate prerequisites for malignant deconvolution.

    Returns
    -------
    tuple of (deconv dict, propMat DataFrame, lineageTree dict).
    """
    if UNS_SPACET not in adata.uns or KEY_DECONV not in adata.uns[UNS_SPACET]:
        raise ValueError(
            "Please run deconvolution first using spatialgpu.deconvolution.core.deconvolution."
        )

    deconv = adata.uns[UNS_SPACET][KEY_DECONV]
    res_deconv: pd.DataFrame = deconv[KEY_PROPMAT]  # cell_types x spots

    if malignant not in res_deconv.index:
        raise ValueError(
            f"Malignant cell type '{malignant}' not found in deconvolution results. "
            f"Available types: {list(res_deconv.index)}"
        )

    lineage_tree = deconv[KEY_REF]["lineageTree"]
    if malignant in lineage_tree and len(lineage_tree[malignant]) > 1:
        raise ValueError(
            "Deconvolution results already include multiple malignant cell states. "
            "Further deconvolution is not recommended."
        )

    if not 0 <= malignant_cutoff <= 1:
        raise ValueError("malignant_cutoff must be between 0 and 1.")

    return deconv, res_deconv, lineage_tree


def _prepare_counts(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get counts (genes x spots), filter zero-sum genes, convert mouse genes.

    Returns
    -------
    tuple of (counts, gene_names, spot_names).
    """
    counts = _get_counts_genes_by_spots(adata)
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)

    # Filter zero-sum genes
    counts, gene_names = filter_zero_genes(counts, gene_names)

    # Mouse-to-human gene conversion
    counts, gene_names = ensure_human_genes(adata, counts, gene_names)

    return counts, gene_names, spot_names


def _select_and_normalize_malignant_spots(
    res_deconv: pd.DataFrame,
    malignant: str,
    malignant_cutoff: float,
    counts: np.ndarray,
    spot_names: np.ndarray,
    gene_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select spots with high malignant fraction and CPM-normalize them.

    Returns
    -------
    tuple of (mal_spots, cpm_mal, log_mal, counts_mal, mal_spot_idx).
    """
    mal_fractions = res_deconv.loc[malignant]
    mal_spot_mask = mal_fractions >= malignant_cutoff
    mal_spots = mal_fractions.index[mal_spot_mask].values

    if len(mal_spots) < 3:
        raise ValueError(
            f"Only {len(mal_spots)} spots have malignant fraction >= {malignant_cutoff}. "
            "Consider lowering the cutoff."
        )

    mal_spot_idx = np.array([np.where(spot_names == s)[0][0] for s in mal_spots])

    # CPM normalize malignant spots (1e5, matching R)
    counts_mal = counts[:, mal_spot_idx]
    counts_mal_dense = to_dense_float64(counts_mal)

    col_sums_mal = counts_mal_dense.sum(axis=0)
    col_sums_mal[col_sums_mal == 0] = 1.0
    cpm_mal = counts_mal_dense / col_sums_mal[np.newaxis, :] * 1e5
    log_mal = np.log2(cpm_mal + 1)

    return mal_spots, cpm_mal, log_mal, counts_mal, mal_spot_idx


def _cluster_malignant_spots(
    counts_mal: np.ndarray,
    gene_names: np.ndarray,
    mal_spots: np.ndarray,
    cpm_mal: np.ndarray,
) -> tuple[list[str], pd.Series]:
    """Cluster malignant spots via HVG + PCA + hierarchical clustering.

    Uses scanpy for preprocessing, Ward's linkage on correlation distance,
    and silhouette analysis to pick the optimal number of clusters (k=2..9).

    Returns
    -------
    tuple of (states sorted list, content Series mapping spots to state letters).
    """
    import scanpy as sc

    np.random.seed(123)
    logger.info("Clustering malignant spots.")

    # Create temporary AnnData for scanpy processing (spots x genes)
    if sparse.issparse(counts_mal):
        adata_tmp = sc.AnnData(X=counts_mal.T.tocsr())
    else:
        adata_tmp = sc.AnnData(X=counts_mal.T.copy())
    adata_tmp.var_names = pd.Index(gene_names)
    adata_tmp.obs_names = pd.Index(mal_spots)

    # Variance normalization + HVG + PCA (MUDAN equivalent)
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    n_hvg = min(3000, len(gene_names))
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=n_hvg)
    adata_tmp = adata_tmp[:, adata_tmp.var.highly_variable].copy()
    sc.pp.scale(adata_tmp, max_value=10)
    n_comps = min(30, adata_tmp.shape[1] - 1)
    sc.tl.pca(adata_tmp, n_comps=n_comps)
    pcs = adata_tmp.obsm["X_pca"]

    # Hierarchical clustering with Ward's method on correlation distance
    corr_matrix = np.corrcoef(pcs)
    corr_matrix = np.clip(corr_matrix, -1, 1)
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    # Silhouette analysis for k=2:9 — use MAX silhouette (not max decrease)
    cluster_numbers = list(range(2, 10))
    sil_scores: list[float] = []
    for k in cluster_numbers:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_samples(dist_matrix, labels, metric="precomputed")
        sil_scores.append(float(np.mean(sil)))

    max_n = cluster_numbers[int(np.argmax(sil_scores))]

    clustering_raw = fcluster(Z, t=max_n, criterion="maxclust")
    # Convert numeric cluster labels to letters (A, B, C, ...)
    clustering_letters = np.array(
        [string.ascii_uppercase[c - 1] for c in clustering_raw]
    )
    content = pd.Series(clustering_letters, index=mal_spots)

    states = sorted(content.unique())
    logger.info(f"Identified {len(states)} malignant cell states.")

    return states, content


def _build_malignant_reference(
    gene_names: np.ndarray,
    cpm_mal: np.ndarray,
    log_mal: np.ndarray,
    states: list[str],
    content: pd.Series,
    malignant: str,
) -> dict[str, Any]:
    """Build reference profiles and signature genes for malignant states.

    Returns
    -------
    dict with keys ``refProfiles``, ``sigGenes``, ``lineageTree``.
    """
    ref_profiles = pd.DataFrame(index=gene_names, dtype=np.float64)
    sig_genes: dict[str, list[str]] = {}

    # Overall malignant reference profile
    ref_profiles[LABEL_MALIGNANT] = cpm_mal.mean(axis=1)

    for state in states:
        state_name = f"Malignant cell state {state}"
        state_mask = (content == state).values
        ref_profiles[state_name] = cpm_mal[:, state_mask].mean(axis=1)

        # DE analysis: find marker genes for this state
        temp_markers: list[str] = []
        for other_state in states:
            if other_state == state:
                continue

            other_mask = (content == other_state).values
            markers = _de_ttest(log_mal, gene_names, state_mask, other_mask, n_top=500)
            temp_markers.extend(markers)

        # Signature genes: appear in exactly 1 comparison
        # (R code: tempMarkers==1, which means unique to one comparison)
        marker_counts = pd.Series(temp_markers).value_counts()
        sig_genes[state_name] = list(marker_counts[marker_counts == 1].index)

    lineage_tree_new: dict[str, list[str]] = {
        malignant: [f"Malignant cell state {s}" for s in states]
    }
    return {
        "refProfiles": ref_profiles,
        "sigGenes": sig_genes,
        "lineageTree": lineage_tree_new,
    }


def _prepare_known_fractions(
    deconv: dict,
    res_deconv: pd.DataFrame,
    lineage_tree: dict,
    malignant: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Extract known (non-malignant) fractions and reference for re-deconvolution.

    Returns
    -------
    tuple of (mal_prop_known DataFrame, mal_ref_known DataFrame or None).
    """
    known_cell_types = [k for k in lineage_tree.keys() if k != malignant]

    known_fractions = list(known_cell_types)
    if LABEL_UNIDENTIFIABLE in res_deconv.index:
        known_fractions.append(LABEL_UNIDENTIFIABLE)

    mal_prop_known = res_deconv.loc[known_fractions]

    # Known cell reference profiles
    orig_ref = deconv[KEY_REF]["refProfiles"]
    if isinstance(orig_ref, pd.DataFrame):
        known_cols = [c for c in known_cell_types if c in orig_ref.columns]
        mal_ref_known = orig_ref[known_cols]
    else:
        mal_ref_known = None

    return mal_prop_known, mal_ref_known


def _merge_and_store_results(
    adata: ad.AnnData,
    deconv: dict,
    res_deconv: pd.DataFrame,
    prop_mat_new: pd.DataFrame,
    malignant: str,
) -> None:
    """Merge new malignant state rows into existing results and update adata."""
    new_rows = prop_mat_new.loc[~prop_mat_new.index.isin([malignant])]
    prop_mat_merged = pd.concat([res_deconv, new_rows])

    deconv[KEY_PROPMAT] = prop_mat_merged
    adata.uns[UNS_SPACET][KEY_DECONV] = deconv
    adata.uns[UNS_SPACET][KEY_PROPMAT_COLS] = list(prop_mat_merged.index)
    adata.obsm[OBSM_SPACET_PROPMAT] = prop_mat_merged.T.reindex(adata.obs_names).values


# ---------------------------------------------------------------------------
# Cross-subject gene weighting (MuSiC-style)
# ---------------------------------------------------------------------------


def _subject_variance_components(
    sc_counts: pd.DataFrame,
    sc_annotation: pd.DataFrame,
    subject_col: str,
):
    """Shared core: compute per-(gene, cell-type) between/within-subject variances.

    Returns a dict mapping cell_type -> (var_between, var_within) arrays.
    Cell types with <2 subjects of data are skipped.
    """
    gene_names = np.array(sc_counts.index)
    counts_dense = to_dense_float64(sc_counts.values)
    col_sums = counts_dense.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    expr = counts_dense / col_sums[np.newaxis, :] * 1e5

    ann = sc_annotation.set_index("cellID")
    cell_ids = np.array(sc_counts.columns)
    cell_types = ann.loc[cell_ids, COL_CELLTYPE].values.astype(str)
    subjects = ann.loc[cell_ids, subject_col].values.astype(str)

    out = {}
    for ct in np.unique(cell_types):
        ct_mask = cell_types == ct
        ct_expr = expr[:, ct_mask]
        ct_subjects = subjects[ct_mask]
        unique_subjects = np.unique(ct_subjects)
        if len(unique_subjects) < 2:
            continue

        n_genes = len(gene_names)
        subject_means = np.zeros((n_genes, len(unique_subjects)), dtype=np.float64)
        for s_idx, subj in enumerate(unique_subjects):
            s_mask = ct_subjects == subj
            if s_mask.sum() == 0:
                continue
            subject_means[:, s_idx] = ct_expr[:, s_mask].mean(axis=1)

        var_between = np.var(subject_means, axis=1, ddof=1)

        var_within = np.zeros(n_genes, dtype=np.float64)
        n_subj_with_data = 0
        for subj in unique_subjects:
            s_mask = ct_subjects == subj
            if s_mask.sum() < 2:
                continue
            var_within += np.var(ct_expr[:, s_mask], axis=1, ddof=1)
            n_subj_with_data += 1
        if n_subj_with_data > 0:
            var_within /= n_subj_with_data

        out[ct] = (var_between, var_within)
    return gene_names, out


def _compute_cross_subject_weights(
    sc_counts: pd.DataFrame,
    sc_annotation: pd.DataFrame,
    subject_col: str,
) -> pd.Series:
    """Cross-subject gene weights — V1 (ratio form, SpaCET default).

    Weight_g = 1 / (1 + var_between_g / (var_within_g + eps))

    Averaged across cell types. Scale-free per gene — two genes with the same
    between/within ratio get the same weight regardless of absolute variance.
    """
    gene_names, vc = _subject_variance_components(sc_counts, sc_annotation, subject_col)
    n_genes = len(gene_names)
    eps = 1e-10
    weight_sum = np.zeros(n_genes, dtype=np.float64)
    n_contrib = 0
    for var_between, var_within in vc.values():
        weight_sum += 1.0 / (1.0 + var_between / (var_within + eps))
        n_contrib += 1
    gw = weight_sum / max(n_contrib, 1)
    if gw.max() > 0:
        gw = gw / gw.max()
    return pd.Series(gw, index=gene_names)


def _compute_cross_subject_weights_absolute(
    sc_counts: pd.DataFrame,
    sc_annotation: pd.DataFrame,
    subject_col: str,
) -> pd.Series:
    """Cross-subject gene weights — V2 (MuSiC absolute-variance inverse).

    Weight_g = 1 / (var_between_g + var_within_g + eps)

    Downweights high-absolute-variance genes hard. Matches the published MuSiC
    formula (Wang et al. 2019) and tends to help in mixed scenarios (Dirichlet
    mixtures) where no cell type dominates the bulk.
    """
    gene_names, vc = _subject_variance_components(sc_counts, sc_annotation, subject_col)
    n_genes = len(gene_names)
    eps = 1e-10
    weight_sum = np.zeros(n_genes, dtype=np.float64)
    n_contrib = 0
    for var_between, var_within in vc.values():
        weight_sum += 1.0 / (var_between + var_within + eps)
        n_contrib += 1
    gw = weight_sum / max(n_contrib, 1)
    if gw.max() > 0:
        gw = gw / gw.max()
    return pd.Series(gw, index=gene_names)


def _irwls_lite_updated_weights(
    ref: dict,
    counts,
    gene_names: np.ndarray,
    prop_mat: pd.DataFrame,
) -> pd.Series:
    """Compute IRWLS-lite updated weights from first-pass residuals.

    w_new = 1 / (var_between + var_within + residual_var + eps)

    Residuals are computed in CPM space: pred_cpm = ref_cpm @ prop_mat,
    residual = bulk_cpm - pred_cpm, per-gene variance across samples.
    """
    reference = ref["refProfiles"]
    vc_genes, vc = ref["_irwls_variance"]

    ST_cpm, ref_cpm, olp_genes, _ = _intersect_and_normalize(
        counts, gene_names, reference.copy()
    )

    # Align prop_mat rows with reference columns; sub-lineages not in
    # `reference.columns` are skipped (they shouldn't appear at the output
    # level, but guard anyway).
    ref_cols = list(reference.columns)
    present = [t for t in prop_mat.index if t in ref_cols]
    if not present:
        logger.warning("IRWLS-lite: no overlapping cell types; skipping update.")
        return ref["gene_weights"]

    col_idx = [ref_cols.index(t) for t in present]
    ref_used = ref_cpm[:, col_idx]                       # (n_genes, n_types)
    prop_used = prop_mat.loc[present].values             # (n_types, n_samples)
    pred_cpm = ref_used @ prop_used                      # (n_genes, n_samples)

    residuals = ST_cpm - pred_cpm
    # Drop samples with NaNs (carried through from _remove_nan_spots usage)
    valid = ~np.isnan(residuals).any(axis=0)
    if valid.sum() < 2:
        logger.warning("IRWLS-lite: too few valid samples; skipping update.")
        return ref["gene_weights"]
    residuals = residuals[:, valid]
    resid_var = residuals.var(axis=1, ddof=1)            # (n_genes,)

    # Average var_between + var_within per gene across all cell types
    vc_idx = {g: i for i, g in enumerate(vc_genes)}
    inv_w_per_gene = np.zeros(len(olp_genes), dtype=np.float64)
    for i, g in enumerate(olp_genes):
        if g not in vc_idx:
            inv_w_per_gene[i] = np.nan
            continue
        gi = vc_idx[g]
        s, n = 0.0, 0
        for vb, vw in vc.values():
            s += vb[gi] + vw[gi]
            n += 1
        inv_w_per_gene[i] = s / n if n > 0 else np.nan

    eps = 1e-10
    new_w = 1.0 / (np.nan_to_num(inv_w_per_gene, nan=1e10) + resid_var + eps)
    if new_w.max() > 0:
        new_w = new_w / new_w.max()

    updated = ref["gene_weights"].copy()
    updated.loc[olp_genes] = new_w
    return updated


# ---------------------------------------------------------------------------
# Internal helpers — DE analysis and scRNA-seq utilities
# ---------------------------------------------------------------------------


def _de_ttest(
    log_expr: np.ndarray,
    gene_names: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    n_top: int = 500,
    logfc_cutoff: float = 0.25,
    fdr_cutoff: float = 0.01,
) -> list[str]:
    """Differential expression via Welch's t-test.

    Parameters
    ----------
    log_expr : np.ndarray
        Log2-transformed expression matrix (genes x cells).
    gene_names : np.ndarray
        Gene names corresponding to rows of log_expr.
    group1_mask : np.ndarray
        Boolean mask for group 1 (treatment).
    group2_mask : np.ndarray
        Boolean mask for group 2 (control).
    n_top : int
        Number of top genes by t-statistic to consider.
    logfc_cutoff : float
        Minimum log-fold-change threshold.
    fdr_cutoff : float
        Maximum FDR threshold.

    Returns
    -------
    list[str]
        Gene names passing both logFC and FDR filters among the top n_top.
    """
    return _de_ttest_python(
        log_expr,
        gene_names,
        group1_mask,
        group2_mask,
        n_top,
        logfc_cutoff,
        fdr_cutoff,
    )


def _de_ttest_python(
    log_expr: np.ndarray,
    gene_names: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    n_top: int = 500,
    logfc_cutoff: float = 0.25,
    fdr_cutoff: float = 0.01,
) -> list[str]:
    """Python fallback DE analysis using Welch's t-test."""
    group1 = log_expr[:, group1_mask]
    group2 = log_expr[:, group2_mask]

    n1 = group1.shape[1]
    n2 = group2.shape[1]

    if n1 < 2 or n2 < 2:
        return []

    mean1 = group1.mean(axis=1)
    mean2 = group2.mean(axis=1)
    var1 = group1.var(axis=1, ddof=1)
    var2 = group2.var(axis=1, ddof=1)

    se = np.sqrt(var1 / n1 + var2 / n2)
    se[se == 0] = np.inf

    t_stat = (mean1 - mean2) / se

    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    denom[denom == 0] = 1.0
    df = num / denom
    df = np.clip(df, 1, np.inf)

    p_values = 2 * stats.t.sf(np.abs(t_stat), df)
    logfc = mean1 - mean2

    sorted_idx = np.argsort(-t_stat)
    top_idx = sorted_idx[: min(n_top, len(sorted_idx))]

    top_genes = gene_names[top_idx]
    top_logfc = logfc[top_idx]
    top_pvals = p_values[top_idx]

    if len(top_pvals) > 0:
        top_pvals = np.where(np.isnan(top_pvals), 1.0, top_pvals)
        _, top_fdr, _, _ = multipletests(top_pvals, method="fdr_bh")
    else:
        return []

    pass_mask = (top_logfc > logfc_cutoff) & (top_fdr < fdr_cutoff)
    return list(top_genes[pass_mask])


def _validate_sc_inputs(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    sc_lineage_tree: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate scRNA-seq inputs and set annotation index.

    Returns
    -------
    tuple of (sc_counts, sc_annotation) with consistent formatting.
    """
    # Ensure annotation has cellID as index
    sc_annotation = sc_annotation.copy()
    if "cellID" in sc_annotation.columns:
        sc_annotation.index = sc_annotation["cellID"].astype(str).values
    else:
        raise ValueError("sc_annotation must have a 'cellID' column.")

    if COL_CELLTYPE not in sc_annotation.columns:
        raise ValueError("sc_annotation must have a 'cellType' column.")

    # Check dimensions
    if isinstance(sc_counts, pd.DataFrame):
        n_cells_counts = sc_counts.shape[1]
    else:
        n_cells_counts = sc_counts.shape[1]

    n_cells_anno = len(sc_annotation)

    if n_cells_counts != n_cells_anno:
        raise ValueError(
            f"Cell count mismatch: sc_counts has {n_cells_counts} cells, "
            f"sc_annotation has {n_cells_anno} cells."
        )

    # Check cell ID matching
    if isinstance(sc_counts, pd.DataFrame):
        count_ids = {str(c) for c in sc_counts.columns}
        anno_ids = {str(c) for c in sc_annotation.index}
        if count_ids != anno_ids:
            raise ValueError("Cell IDs in sc_counts and sc_annotation do not match.")

    # Validate lineage tree
    if len(sc_lineage_tree) == 0:
        raise ValueError("Lineage tree is empty.")

    all_cell_types = []
    for subtypes in sc_lineage_tree.values():
        if isinstance(subtypes, str):
            all_cell_types.append(subtypes)
        else:
            all_cell_types.extend(subtypes)

    unique_anno_types = set(sc_annotation[COL_CELLTYPE].astype(str).unique())
    missing = [ct for ct in all_cell_types if ct not in unique_anno_types]
    if missing:
        raise ValueError(
            f"Cell types in lineage tree not found in annotation: {missing}"
        )

    return sc_counts, sc_annotation


def _downsample_cells(
    sc_counts: pd.DataFrame | np.ndarray,
    sc_annotation: pd.DataFrame,
    n_cell_each_lineage: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downsample cells to n_cell_each_lineage per cell type.

    Uses seed=123 for reproducibility, matching R behavior.

    Parameters
    ----------
    sc_counts : pd.DataFrame
        Count matrix (genes x cells).
    sc_annotation : pd.DataFrame
        Annotation with 'cellID' and 'cellType' columns.
    n_cell_each_lineage : int
        Maximum number of cells per cell type.

    Returns
    -------
    tuple of downsampled (sc_counts, sc_annotation).
    """
    np.random.seed(123)

    cell_ids = sc_annotation["cellID"].astype(str).values
    cell_types = sc_annotation[COL_CELLTYPE].astype(str).values

    # Group cell IDs by cell type
    type_to_ids: dict[str, list[str]] = {}
    for cid, ctype in zip(cell_ids, cell_types):
        type_to_ids.setdefault(ctype, []).append(cid)

    # Downsample each type
    keep_ids: list[str] = []
    for ctype in sorted(type_to_ids.keys()):
        ids = type_to_ids[ctype]
        n = len(ids)
        if n > n_cell_each_lineage:
            n = n_cell_each_lineage
        sampled = list(np.random.choice(ids, size=n, replace=False))
        keep_ids.extend(sampled)

    # Subset
    if isinstance(sc_counts, pd.DataFrame):
        sc_counts = sc_counts[keep_ids]
    else:
        # If numpy array, need to find column indices
        all_col_ids = np.array(sc_annotation["cellID"].astype(str).values)
        keep_mask = np.isin(all_col_ids, keep_ids)
        sc_counts = sc_counts[:, keep_mask]

    sc_annotation = sc_annotation.loc[keep_ids]

    return sc_counts, sc_annotation
