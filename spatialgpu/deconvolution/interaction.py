"""Cell-cell interaction analysis for SpaCET deconvolution.

Implements the SpaCET CCI (Cell-Cell Interaction) pipeline:
  1. Colocalization analysis via Spearman correlation
  2. Ligand-receptor network scoring with permutation testing
  3. Cell-type pair interaction analysis (Cohen's d + Wilcoxon)
  4. Tumor-stroma interface identification (Visium hex neighbors)
  5. Interface overlay with interaction spots
  6. Distance-to-interface permutation test

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cdist

from spatialgpu.deconvolution.reference import load_lr_database

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: ensure adata.uns['spacet']['CCI'] exists
# ---------------------------------------------------------------------------


def _ensure_cci(adata: ad.AnnData) -> dict:
    """Return adata.uns['spacet']['CCI'], creating it if absent."""
    if "spacet" not in adata.uns:
        raise ValueError(
            "No SpaCET results found. Run deconvolution() first."
        )
    spacet = adata.uns["spacet"]
    if "CCI" not in spacet:
        spacet["CCI"] = {}
    return spacet["CCI"]


# ---------------------------------------------------------------------------
# 1. Colocalization
# ---------------------------------------------------------------------------


def cci_colocalization(adata: ad.AnnData) -> ad.AnnData:
    """Cell-type pair colocalization via Spearman correlation.

    Equivalent to SpaCET.CCI.colocalization() in R.

    Computes pairwise Spearman correlations of deconvolution cell-type
    fractions across spots, and also correlates reference profiles
    (centered by row means).

    Parameters
    ----------
    adata : AnnData
        Must have adata.uns['spacet']['deconvolution']['propMat'] (a
        DataFrame with cell_types as rows and spots as columns) and
        adata.uns['spacet']['deconvolution']['Ref'].

    Returns
    -------
    AnnData with results stored in
        adata.uns['spacet']['CCI']['colocalization'] : pd.DataFrame
    """
    spacet = adata.uns["spacet"]
    res_deconv: pd.DataFrame = spacet["deconvolution"]["propMat"].copy()

    # Remove helper rows
    exclude = {"Unidentifiable", "Macrophage other"}
    res_deconv = res_deconv.loc[
        ~res_deconv.index.isin(exclude)
    ]

    # Round to 2 decimals
    res_deconv = res_deconv.round(2)

    # Overall fraction per cell type (mean across spots)
    overall_fraction = res_deconv.mean(axis=1)

    # Spearman correlation of fraction profiles (cell_types x spots -> transpose)
    # Each cell type is a variable, each spot is an observation
    frac_mat = res_deconv.values  # (n_types, n_spots)
    n_types = frac_mat.shape[0]
    cell_types = list(res_deconv.index)

    rho_frac = np.zeros((n_types, n_types), dtype=np.float64)
    pval_frac = np.zeros((n_types, n_types), dtype=np.float64)

    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                rho_frac[i, j] = 1.0
                pval_frac[i, j] = 0.0
            elif j > i:
                r, p = stats.spearmanr(frac_mat[i], frac_mat[j])
                rho_frac[i, j] = r if not np.isnan(r) else 0.0
                rho_frac[j, i] = rho_frac[i, j]
                pval_frac[i, j] = p if not np.isnan(p) else 1.0
                pval_frac[j, i] = pval_frac[i, j]

    # Build summary dataframe (melt-style, matching R output)
    rows = []
    for i in range(n_types):
        for j in range(n_types):
            ct1 = cell_types[i]
            ct2 = cell_types[j]
            rows.append({
                "cell_type_1": ct1,
                "cell_type_2": ct2,
                "fraction_product": float(
                    overall_fraction[ct1] * overall_fraction[ct2]
                ),
                "fraction_rho": round(float(rho_frac[i, j]), 3),
                "fraction_pv": float(pval_frac[i, j]),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.index = [
        f"{r['cell_type_1']}_{r['cell_type_2']}" for r in rows
    ]

    # Replace NaN rho with 0, NaN pv with 1
    summary_df["fraction_rho"] = summary_df["fraction_rho"].fillna(0.0)
    summary_df["fraction_pv"] = summary_df["fraction_pv"].fillna(1.0)

    # --- Reference profile correlation ---
    ref = spacet["deconvolution"]["Ref"]
    ref_profiles: pd.DataFrame = ref["refProfiles"]
    sig_genes: dict = ref["sigGenes"]
    lineage_tree: dict = ref["lineageTree"]

    # Collect signature genes for lineage tree types + T cell
    sig_keys = set(lineage_tree.keys())
    if "T cell" in sig_genes:
        sig_keys.add("T cell")
    all_sig_genes: list[str] = []
    for k in sig_keys:
        if k in sig_genes:
            all_sig_genes.extend(sig_genes[k])
    all_sig_genes = list(set(all_sig_genes))

    # Filter to genes present in refProfiles
    all_sig_genes = [g for g in all_sig_genes if g in ref_profiles.index]
    reff = ref_profiles.loc[all_sig_genes].copy()

    # Center by row means
    reff = reff.subtract(reff.mean(axis=1), axis=0)

    ref_types = list(reff.columns)
    n_ref = len(ref_types)
    ref_mat = reff.values  # (n_genes, n_ref_types)

    rho_ref = np.zeros((n_ref, n_ref), dtype=np.float64)
    pval_ref = np.zeros((n_ref, n_ref), dtype=np.float64)

    for i in range(n_ref):
        for j in range(n_ref):
            if i == j:
                rho_ref[i, j] = 1.0
                pval_ref[i, j] = 0.0
            elif j > i:
                r, p = stats.spearmanr(ref_mat[:, i], ref_mat[:, j])
                rho_ref[i, j] = r if not np.isnan(r) else 0.0
                rho_ref[j, i] = rho_ref[i, j]
                pval_ref[i, j] = p if not np.isnan(p) else 1.0
                pval_ref[j, i] = pval_ref[i, j]

    # Build reference summary and merge into main summary
    for i in range(n_ref):
        for j in range(n_ref):
            ct1 = ref_types[i]
            ct2 = ref_types[j]
            key = f"{ct1}_{ct2}"
            if key in summary_df.index:
                summary_df.loc[key, "reference_rho"] = round(
                    float(rho_ref[i, j]), 3
                )
                summary_df.loc[key, "reference_pv"] = float(pval_ref[i, j])

    # Remove same cell-type pairs
    summary_df = summary_df.loc[
        summary_df["cell_type_1"] != summary_df["cell_type_2"]
    ]

    cci = _ensure_cci(adata)
    cci["colocalization"] = summary_df

    return adata


# ---------------------------------------------------------------------------
# 2. Ligand-Receptor Network Score
# ---------------------------------------------------------------------------


def cci_lr_network_score(
    adata: ad.AnnData,
    n_jobs: int = 1,
) -> ad.AnnData:
    """Ligand-receptor network score with permutation testing.

    Equivalent to SpaCET.CCI.LRNetworkScore() in R.

    For each spot, computes the mean product of ligand-receptor expression
    pairs, then compares against 1000 network permutations (bipartite
    degree-preserving edge swaps) to derive a network score and p-value.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics AnnData with raw counts.
    n_jobs : int
        Number of parallel jobs (currently used for future extension).

    Returns
    -------
    AnnData with results in
        adata.uns['spacet']['CCI']['LRNetworkScore'] : np.ndarray
            (3, n_spots) matrix with rows: Raw_expr, Network_Score,
            Network_Score_pv.
    """
    # Get counts (spots x genes) and transpose to genes x spots
    X = adata.X
    if sparse.issparse(X):
        counts = X.T.tocsc().astype(np.float64)
    else:
        counts = X.T.astype(np.float64)

    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)
    n_spots = counts.shape[1]

    # CPM + log2 normalization
    if sparse.issparse(counts):
        col_sums = np.asarray(counts.sum(axis=0)).ravel()
        csc = sparse.csc_matrix(counts, dtype=np.float64, copy=True)
        for j in range(csc.shape[1]):
            start, end = csc.indptr[j], csc.indptr[j + 1]
            if col_sums[j] > 0:
                csc.data[start:end] = csc.data[start:end] / col_sums[j] * 1e6
        csc.data = np.log2(csc.data + 1)
        st_mat = csc
    else:
        col_sums = counts.sum(axis=0)
        st_mat = counts / col_sums[np.newaxis, :] * 1e6
        st_mat = np.log2(st_mat + 1)

    # Build gene name -> index mapping
    gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # Load LR database
    lr_db = load_lr_database()
    lr_pairs = pd.DataFrame({
        "L": lr_db.iloc[:, 1].values,
        "R": lr_db.iloc[:, 3].values,
    })

    # Filter DLK2 from receptors
    lr_pairs = lr_pairs.loc[lr_pairs["R"] != "DLK2"]

    # Filter to genes present in expression data
    lr_pairs = lr_pairs.loc[
        lr_pairs["L"].isin(gene_to_idx) & lr_pairs["R"].isin(gene_to_idx)
    ].copy()
    lr_pairs = lr_pairs.drop_duplicates()
    lr_pairs.index = lr_pairs["L"] + "_" + lr_pairs["R"]

    n_pairs = len(lr_pairs)
    logger.info(
        f"Step 1. Permute Ligand-Receptor network. ({n_pairs} pairs)"
    )

    # Build bipartite adjacency matrix
    ligands = sorted(lr_pairs["L"].unique())
    receptors = sorted(lr_pairs["R"].unique())
    lig_to_idx = {g: i for i, g in enumerate(ligands)}
    rec_to_idx = {g: i for i, g in enumerate(receptors)}

    lr_mat = np.zeros((len(ligands), len(receptors)), dtype=np.int32)
    for _, row in lr_pairs.iterrows():
        lr_mat[lig_to_idx[row["L"]], rec_to_idx[row["R"]]] = 1

    # Get indices for original LR pairs in expression matrix
    l_indices = np.array([gene_to_idx[g] for g in lr_pairs["L"]])
    r_indices = np.array([gene_to_idx[g] for g in lr_pairs["R"]])

    # Permute LR network 1000 times using bipartite edge swap
    rng = np.random.RandomState(123456)

    # Collect all permuted LR pair indices
    logger.info("  Generating 1000 permuted networks...")
    all_perm_l_indices = []
    all_perm_r_indices = []

    for _perm_i in range(1000):
        perm_mat = _bipartite_edge_swap(lr_mat.copy(), rng)

        # Extract edges from permuted matrix
        perm_l_idx_list = []
        perm_r_idx_list = []
        li_arr, ri_arr = np.where(perm_mat == 1)
        for li, ri in zip(li_arr, ri_arr):
            perm_l_idx_list.append(gene_to_idx[ligands[li]])
            perm_r_idx_list.append(gene_to_idx[receptors[ri]])

        all_perm_l_indices.append(np.array(perm_l_idx_list))
        all_perm_r_indices.append(np.array(perm_r_idx_list))

    # Calculate LR network score per spot
    logger.info("Step 2. Calculate L-R network score.")

    lr_result = np.zeros((3, n_spots), dtype=np.float64)

    for s in range(n_spots):
        # Get spot expression vector
        if sparse.issparse(st_mat):
            spot = np.asarray(st_mat[:, s].todense()).ravel()
        else:
            spot = st_mat[:, s]

        # Raw LR co-expression score
        lr_raw = np.mean(spot[l_indices] * spot[r_indices])

        # Permuted scores
        perm_scores = np.zeros(1000, dtype=np.float64)
        for p in range(1000):
            perm_scores[p] = np.mean(
                spot[all_perm_l_indices[p]] * spot[all_perm_r_indices[p]]
            )

        perm_mean = np.mean(perm_scores)
        score = lr_raw / perm_mean if perm_mean != 0 else 0.0
        pv = (np.sum(perm_scores >= lr_raw) + 1) / 1001

        lr_result[0, s] = lr_raw
        lr_result[1, s] = score
        lr_result[2, s] = pv

    # Store results as labeled matrix
    lr_score_mat = pd.DataFrame(
        lr_result,
        index=["Raw_expr", "Network_Score", "Network_Score_pv"],
        columns=spot_names,
    )

    cci = _ensure_cci(adata)
    cci["LRNetworkScore"] = lr_score_mat.values
    cci["LRNetworkScore_columns"] = list(spot_names)
    cci["LRNetworkScore_index"] = [
        "Raw_expr",
        "Network_Score",
        "Network_Score_pv",
    ]

    return adata


def _bipartite_edge_swap(
    mat: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Degree-preserving bipartite edge swap (Curveball-style).

    For each swap attempt, randomly pick two edges (L1,R1) and (L2,R2).
    If L1!=L2 and R1!=R2 and the swap edges (L1,R2) and (L2,R1) don't
    already exist, perform the swap. Repeat for a number of iterations
    proportional to the number of edges.

    Parameters
    ----------
    mat : np.ndarray
        Binary bipartite adjacency matrix (ligands x receptors).
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    np.ndarray : Rewired adjacency matrix with preserved degrees.
    """
    edges_l, edges_r = np.where(mat == 1)
    n_edges = len(edges_l)

    if n_edges < 2:
        return mat

    # Number of swap attempts: ~5 * n_edges (matching BiRewire default)
    n_swaps = 5 * n_edges

    for _ in range(n_swaps):
        # Pick two random edge indices
        idx1, idx2 = rng.randint(0, n_edges, size=2)
        if idx1 == idx2:
            continue

        l1, r1 = edges_l[idx1], edges_r[idx1]
        l2, r2 = edges_l[idx2], edges_r[idx2]

        # Skip if same ligand or same receptor
        if l1 == l2 or r1 == r2:
            continue

        # Check that swapped edges don't already exist
        if mat[l1, r2] == 1 or mat[l2, r1] == 1:
            continue

        # Perform swap
        mat[l1, r1] = 0
        mat[l2, r2] = 0
        mat[l1, r2] = 1
        mat[l2, r1] = 1

        # Update edge list
        edges_r[idx1] = r2
        edges_r[idx2] = r1

    return mat


# ---------------------------------------------------------------------------
# 3. Cell-type pair interaction analysis
# ---------------------------------------------------------------------------


def cci_cell_type_pair(
    adata: ad.AnnData,
    cell_type_pair: list[str] | tuple[str, str],
) -> ad.AnnData:
    """Test LR enrichment for a colocalized cell-type pair.

    Equivalent to SpaCET.CCI.cellTypePair() in R.

    Classifies spots based on deconvolution fractions into "Both" (high in
    both types), single-type (high in one, low in other), or NA. Then tests
    whether "Both" spots have higher LR network scores than "Single" spots
    using Cohen's d and Wilcoxon test.

    Parameters
    ----------
    adata : AnnData
        Must have CCI colocalization and LRNetworkScore results.
    cell_type_pair : list or tuple of 2 strings
        The two cell types to test for interaction.

    Returns
    -------
    AnnData with results in
        adata.uns['spacet']['CCI']['interaction']['groupMat'] : pd.DataFrame
        adata.uns['spacet']['CCI']['interaction']['testRes'] : pd.DataFrame
    """
    if len(cell_type_pair) != 2:
        raise ValueError("Please input a pair of cell-types.")

    spacet = adata.uns["spacet"]
    res_deconv: pd.DataFrame = spacet["deconvolution"]["propMat"]

    missing = [
        ct for ct in cell_type_pair if ct not in res_deconv.index
    ]
    if missing:
        raise ValueError(
            f"Cell type(s) not found in deconvolution results: {missing}"
        )

    # Sort alphabetically (matching R behavior)
    cell_type_pair = sorted(cell_type_pair)
    ct1, ct2 = cell_type_pair
    pair_key = f"{ct1}_{ct2}"

    cci = _ensure_cci(adata)
    lr_score_mat = cci["LRNetworkScore"]  # (3, n_spots) array
    spot_names = list(res_deconv.columns)

    # Initialize or retrieve interaction results
    if "interaction" not in cci:
        cci["interaction"] = {}
    interaction = cci["interaction"]

    if "groupMat" in interaction and len(interaction["groupMat"]) > 0:
        group_mat = interaction["groupMat"]
    else:
        group_mat = pd.DataFrame()

    if "testRes" in interaction and len(interaction["testRes"]) > 0:
        test_res = interaction["testRes"]
    else:
        test_res = pd.DataFrame()

    # Get colocalization rho and pv
    coloc = cci["colocalization"]
    mask = (coloc["cell_type_1"] == ct1) & (coloc["cell_type_2"] == ct2)
    rho = float(coloc.loc[mask, "fraction_rho"].values[0])
    pv1 = float(coloc.loc[mask, "fraction_pv"].values[0])

    test_res.loc[pair_key, "colocalization_rho"] = rho
    test_res.loc[pair_key, "colocalization_pv"] = pv1

    # Spot classification
    frac1 = res_deconv.loc[ct1].values.astype(np.float64)
    frac2 = res_deconv.loc[ct2].values.astype(np.float64)

    # 75th percentile (R summary()[5] = 3rd quartile)
    cutoff1 = np.percentile(frac1, 75)
    cutoff2 = np.percentile(frac2, 75)

    # 85th percentile for "Both" classification
    cutoff11 = np.percentile(frac1, 85)
    cutoff22 = np.percentile(frac2, 85)

    content = pd.Series(index=spot_names, dtype=object)
    for i, spot in enumerate(spot_names):
        f1 = frac1[i]
        f2 = frac2[i]
        if f1 > cutoff11 and f2 > cutoff22:
            content[spot] = "Both"
        elif f1 > cutoff11 and f2 < cutoff2:
            content[spot] = ct1
        elif f2 > cutoff22 and f1 < cutoff1:
            content[spot] = ct2
        else:
            content[spot] = np.nan

    group_mat.loc[pair_key, spot_names] = content.values

    if content.eq("Both").sum() > 5:
        # Build comparison dataframe
        network_score = lr_score_mat[1, :]  # Network_Score row
        fg_df = pd.DataFrame({
            "group": content.values,
            "value": network_score,
        })
        fg_df = fg_df.dropna(subset=["group"])
        fg_df.loc[fg_df["group"].isin(cell_type_pair), "group"] = "Single"

        # Cohen's d: Both vs Single
        both_vals = fg_df.loc[fg_df["group"] == "Both", "value"].values
        single_vals = fg_df.loc[fg_df["group"] == "Single", "value"].values

        cd1 = _cohens_d(both_vals, single_vals)
        cd1 = float(f"{cd1:.2g}")

        # Wilcoxon rank-sum test
        _, pv2 = stats.mannwhitneyu(
            both_vals, single_vals, alternative="two-sided"
        )
        pv2 = float(f"{pv2:.2g}")

        test_res.loc[pair_key, "groupCompare_cohen.d"] = cd1
        test_res.loc[pair_key, "groupCompare_pv"] = pv2

        # Interaction criteria
        if rho > 0 and pv1 < 0.05 and cd1 < 0 and pv2 < 0.05:
            logger.info(
                f"Based on colocalization analysis and L-R enrichment "
                f"analysis, {ct1} and {ct2} have potential intercellular "
                f"interaction in the current tissue."
            )
            test_res.loc[pair_key, "Interaction"] = True
        else:
            logger.info(
                "Based on colocalization analysis and L-R enrichment "
                "analysis, the intercellular interaction is not significant "
                "for the current cell-type pair."
            )
            test_res.loc[pair_key, "Interaction"] = False
    else:
        logger.info(
            "The colocalization analysis is not significant for the "
            "current cell-type pair."
        )
        test_res.loc[pair_key, "Interaction"] = False

    interaction["groupMat"] = group_mat
    interaction["testRes"] = test_res
    cci["interaction"] = interaction

    return adata


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation).

    Computes d = (mean1 - mean2) / s_pooled, matching psych::cohen.d in R.
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if s_pooled == 0:
        return 0.0

    return (mean1 - mean2) / s_pooled


# ---------------------------------------------------------------------------
# 4. Identify tumor-stroma interface
# ---------------------------------------------------------------------------


def identify_interface(
    adata: ad.AnnData,
    malignant: str = "Malignant",
    malignant_cutoff: float = 0.5,
) -> ad.AnnData:
    """Identify spots at the tumor-stroma interface (Visium only).

    Equivalent to SpaCET.identify.interface() in R.

    Classifies spots as Tumor (malignant fraction >= cutoff), then refines
    non-tumor spots into Stroma (all hex neighbors are Stroma) or Interface
    (at least one neighbor is Tumor).

    Spot IDs must be in "{row}x{col}" format. Hexagonal neighbors for Visium
    are computed from array coordinates.

    Parameters
    ----------
    adata : AnnData
        Must have deconvolution results with Visium data.
    malignant : str
        Name of the malignant cell type row in propMat.
    malignant_cutoff : float
        Fraction threshold for classifying a spot as Tumor.

    Returns
    -------
    AnnData with results in
        adata.uns['spacet']['CCI']['interface'] : pd.DataFrame
            1-row DataFrame ("Interface") with values Tumor/Stroma/Interface.
    """
    spacet = adata.uns["spacet"]
    res_deconv: pd.DataFrame = spacet["deconvolution"]["propMat"]

    if malignant not in res_deconv.index:
        raise ValueError(
            f"Malignant cell type '{malignant}' not found in deconvolution "
            f"results. Available types: {list(res_deconv.index)}"
        )

    if not (0 <= malignant_cutoff <= 1):
        raise ValueError(
            "malignant_cutoff must be between 0 and 1."
        )

    spot_names = list(res_deconv.columns)
    mal_fracs = res_deconv.loc[malignant].values.astype(np.float64)

    # Initial classification: Tumor vs Stroma
    content = pd.Series(index=spot_names, dtype=object)
    for i, spot in enumerate(spot_names):
        content[spot] = "Tumor" if mal_fracs[i] >= malignant_cutoff else "Stroma"

    # Refine Stroma spots: check hex neighbors
    stroma_spots = [s for s in spot_names if content[s] == "Stroma"]
    content_new = content.copy()

    for spot in stroma_spots:
        parts = spot.split("x")
        row = int(parts[0])
        col = int(parts[1])

        # 6 hexagonal neighbors for Visium array coordinates
        neighbors = [
            f"{row}x{col - 2}",
            f"{row}x{col + 2}",
            f"{row - 1}x{col - 1}",
            f"{row + 1}x{col + 1}",
            f"{row - 1}x{col + 1}",
            f"{row + 1}x{col - 1}",
        ]

        # Get classifications of existing neighbors
        neighbor_types = [
            content[n] for n in neighbors if n in content.index
        ]

        if len(neighbor_types) == 0:
            # No neighbors found, keep as Stroma
            content_new[spot] = "Stroma"
        elif len(set(neighbor_types)) == 1 and neighbor_types[0] == "Stroma":
            # All neighbors are Stroma
            content_new[spot] = "Stroma"
        else:
            # At least one Tumor neighbor (or mixed)
            content_new[spot] = "Interface"

    # Store as (1, n_spots) matrix
    interface_df = pd.DataFrame(
        [content_new.values],
        index=["Interface"],
        columns=spot_names,
    )

    cci = _ensure_cci(adata)
    cci["interface"] = interface_df

    return adata


# ---------------------------------------------------------------------------
# 5. Combine interface with interaction spots
# ---------------------------------------------------------------------------


def combine_interface(
    adata: ad.AnnData,
    cell_type_pair: list[str] | tuple[str, str],
) -> ad.AnnData:
    """Overlay interaction spots onto the interface map.

    Equivalent to SpaCET.combine.interface() in R.

    Spots classified as "Both" in the cell-type pair interaction analysis
    that fall within "Stroma" regions on the interface map are reclassified
    as "Interaction".

    Parameters
    ----------
    adata : AnnData
        Must have interface and cell-type pair interaction results.
    cell_type_pair : list or tuple of 2 strings
        The two cell types.

    Returns
    -------
    AnnData with an additional row in
        adata.uns['spacet']['CCI']['interface'] named
        "Interface&{ct1}_{ct2}".
    """
    if len(cell_type_pair) != 2:
        raise ValueError("Please input a pair of cell-types.")

    spacet = adata.uns["spacet"]
    res_deconv: pd.DataFrame = spacet["deconvolution"]["propMat"]
    cci = _ensure_cci(adata)

    missing = [
        ct for ct in cell_type_pair if ct not in res_deconv.index
    ]
    if missing:
        raise ValueError(
            f"Cell type(s) not found in deconvolution results: {missing}"
        )

    cell_type_pair = sorted(cell_type_pair)
    ct1, ct2 = cell_type_pair
    pair_key = f"{ct1}_{ct2}"

    if "interface" not in cci:
        raise ValueError("Run identify_interface() first.")

    interface_df: pd.DataFrame = cci["interface"]
    if "Interface" not in interface_df.index:
        raise ValueError("Run identify_interface() first.")

    if "interaction" not in cci or pair_key not in cci["interaction"].get(
        "testRes", pd.DataFrame()
    ).index:
        raise ValueError(
            f"Run cci_cell_type_pair() for {cell_type_pair} first."
        )

    group_mat = cci["interaction"]["groupMat"]
    spot_names = list(res_deconv.columns)

    # Get "Both" spots
    pair_groups = group_mat.loc[pair_key, spot_names]
    both_spots = [s for s in spot_names if pair_groups[s] == "Both"]

    # Filter to Stroma spots on interface
    stroma_spots = [
        s for s in spot_names
        if interface_df.loc["Interface", s] == "Stroma"
    ]
    interaction_spots = [s for s in both_spots if s in stroma_spots]

    # Create new interface row with "Interaction" overlay
    content_new = interface_df.loc["Interface"].copy()
    for s in interaction_spots:
        content_new[s] = "Interaction"

    gname = f"Interface&{pair_key}"
    if gname in interface_df.index:
        interface_df.loc[gname] = content_new.values
    else:
        new_row = pd.DataFrame(
            [content_new.values],
            index=[gname],
            columns=interface_df.columns,
        )
        interface_df = pd.concat([interface_df, new_row])

    cci["interface"] = interface_df

    return adata


# ---------------------------------------------------------------------------
# 6. Distance to interface
# ---------------------------------------------------------------------------


def distance_to_interface(
    adata: ad.AnnData,
    cell_type_pair: list[str] | tuple[str, str],
    n_permutation: int = 1000,
) -> ad.AnnData:
    """Distance from interaction spots to nearest tumor border.

    Equivalent to SpaCET.distance.to.interface() in R.

    Computes the mean Euclidean distance from "Both" interaction spots
    (in Stroma) to the nearest Interface spot, then tests significance
    via permutation (sampling single-type spots from Stroma).

    Parameters
    ----------
    adata : AnnData
        Must have interface and cell-type pair interaction results.
    cell_type_pair : list or tuple of 2 strings
        The two cell types.
    n_permutation : int
        Number of permutations for significance testing.

    Returns
    -------
    AnnData with results in
        adata.uns['spacet']['CCI']['distance_to_interface'][pair_key] : dict
            Keys: 'd_observed', 'd_permuted', 'pvalue'.
    """
    if len(cell_type_pair) != 2:
        raise ValueError("Please input a pair of cell-types.")

    spacet = adata.uns["spacet"]
    res_deconv: pd.DataFrame = spacet["deconvolution"]["propMat"]
    cci = _ensure_cci(adata)

    cell_type_pair = sorted(cell_type_pair)
    ct1, ct2 = cell_type_pair
    pair_key = f"{ct1}_{ct2}"

    test_res = cci["interaction"]["testRes"]
    if not test_res.loc[pair_key, "Interaction"]:
        if pd.isna(test_res.loc[pair_key].get("groupCompare_pv", np.nan)):
            logger.info(
                "The colocalization analysis is not significant for the "
                "current cell-type pair."
            )
        else:
            logger.info(
                "Based on colocalization analysis and L-R enrichment "
                "analysis, the intercellular interaction is not significant "
                "for the current cell-type pair."
            )
        return adata

    interface_df = cci["interface"]
    spot_names = list(res_deconv.columns)

    # Interface (border) spots and Stroma core spots
    spot_border = [
        s for s in spot_names
        if interface_df.loc["Interface", s] == "Interface"
    ]
    spot_stroma = [
        s for s in spot_names
        if interface_df.loc["Interface", s] == "Stroma"
    ]

    # "Both" spots restricted to Stroma
    group_mat = cci["interaction"]["groupMat"]
    pair_groups = group_mat.loc[pair_key, spot_names]

    both_spots = [
        s for s in spot_names
        if pair_groups[s] == "Both" and s in spot_stroma
    ]

    # Single-type spots restricted to Stroma
    single_spots = [
        s for s in spot_names
        if pair_groups[s] in cell_type_pair and s in spot_stroma
    ]

    if len(both_spots) == 0 or len(spot_border) == 0:
        logger.info("No interaction spots or border spots found.")
        return adata

    # Compute distances using array coordinates from spot IDs
    def _spot_to_coords(spot: str) -> tuple[float, float]:
        parts = spot.split("x")
        return (float(parts[0]), float(parts[1]))

    border_coords = np.array([_spot_to_coords(s) for s in spot_border])

    def _mean_min_distance(spots: list[str]) -> float:
        """Mean minimum distance from each spot to nearest border spot."""
        if len(spots) == 0:
            return np.inf
        coords = np.array([_spot_to_coords(s) for s in spots])
        dists = cdist(coords, border_coords)
        min_dists = dists.min(axis=1)
        return float(np.mean(min_dists))

    # Observed distance
    d_observed = _mean_min_distance(both_spots)

    # Pre-compute distances from all single-type spots to border
    if len(single_spots) == 0:
        logger.info("No single-type spots found for permutation test.")
        return adata

    single_coords = np.array([_spot_to_coords(s) for s in single_spots])
    single_dists = cdist(single_coords, border_coords)
    single_min_dists = single_dists.min(axis=1)

    # Permutation test
    n_both = len(both_spots)
    d_permuted = np.zeros(n_permutation, dtype=np.float64)

    for i in range(n_permutation):
        rng = np.random.RandomState(i + 1)
        sampled_idx = rng.choice(len(single_spots), size=n_both, replace=False)
        d_permuted[i] = np.mean(single_min_dists[sampled_idx])

    pvalue = (np.sum(d_permuted <= d_observed) + 1) / (n_permutation + 1)

    logger.info(
        f"Distance to interface: d={d_observed:.3f}, "
        f"p={pvalue:.4g} (n_perm={n_permutation})"
    )

    if "distance_to_interface" not in cci:
        cci["distance_to_interface"] = {}

    cci["distance_to_interface"][pair_key] = {
        "d_observed": d_observed,
        "d_permuted": d_permuted,
        "pvalue": pvalue,
    }

    return adata
