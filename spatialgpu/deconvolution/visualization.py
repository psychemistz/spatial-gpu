"""SpaCET spatial feature visualization.

Matplotlib-based spatial scatter plots for deconvolution results,
gene expression, LR scores, interface, gene set scores, and more.
Equivalent to SpaCET.visualize.spatialFeature() and related R functions.

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Wedge
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color schemes matching R SpaCET exactly
# ---------------------------------------------------------------------------

_COLORMAPS = {
    "QualityControl": ["lightblue", "blue", "darkblue"],
    "GeneExpression": ["#4d9221", "yellow", "#c51b7d"],
    "CellFraction": ["blue", "yellow", "red"],
    "LRNetworkScore": [
        "blue",
        "blue",
        "blue",
        "blue",
        "cyan",
        "cyan",
        "yellow",
    ],
    "GeneSetScore": ["#91bfdb", "#fee090", "#d73027"],
    "SecretedProteinActivity": [
        "#b8e186",
        "#b8e186",
        "#b8e186",
        "#de77ae",
        "#c51b7d",
    ],
    "SignalingPattern": [
        "#000004",
        "#1A1042",
        "#4A1079",
        "#D9466B",
        "#FCFDBF",
    ],
}

_DISCRETE_COLORS = {
    "Interface": {"Tumor": "black", "Stroma": "darkgrey", "Interface": "#f3c300"},
    "Interface_interaction": {
        "Interaction": "green",
        "Tumor": "black",
        "Stroma": "darkgrey",
        "Interface": "#f3c300",
    },
}

# Tab20-style palette for cell types (used by MostAbundantCellType)
_CELLTYPE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


# ===========================================================================
# Main spatial feature visualization
# ===========================================================================


def visualize_spatial_feature(
    adata: ad.AnnData,
    spatial_type: str,
    spatial_features: list[str] | None = None,
    scale_type_gene: str = "LogTPM",
    same_scale_fraction: bool = False,
    colors: list[str] | None = None,
    point_size: float = 5.0,
    point_alpha: float = 1.0,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Visualize spatial features as scatter plots.

    Equivalent to SpaCET.visualize.spatialFeature() in R.

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data with SpaCET results.
    spatial_type : str
        One of: 'QualityControl', 'GeneExpression', 'CellFraction',
        'MostAbundantCellType', 'CellTypeComposition', 'LRNetworkScore',
        'Interface', 'GeneSetScore', 'SecretedProteinActivity',
        'SignalingPattern', 'metaData'.
    spatial_features : list of str or None
        Features to plot. If None, uses defaults for the type.
    scale_type_gene : str
        Scale for gene expression: 'RawCounts', 'LogRawCounts',
        'LogTPM/10', 'LogTPM'.
    same_scale_fraction : bool
        Use [0, 1] scale for all cell fractions.
    colors : list of str or None
        Custom color gradient. None uses defaults.
    point_size : float
        Size of scatter points.
    point_alpha : float
        Alpha transparency (0-1).
    ncols : int
        Number of columns in multi-panel layout.
    figsize : tuple or None
        Figure size (width, height). None auto-sizes.
    ax : Axes or None
        Matplotlib axes for single feature. Ignored for multi-feature.

    Returns
    -------
    matplotlib Figure, or None if ax was provided.
    """
    valid_types = (
        "QualityControl",
        "GeneExpression",
        "CellFraction",
        "MostAbundantCellType",
        "CellTypeComposition",
        "LRNetworkScore",
        "Interface",
        "GeneSetScore",
        "SecretedProteinActivity",
        "SignalingPattern",
        "metaData",
    )
    if spatial_type not in valid_types:
        raise ValueError(
            f"spatial_type must be one of {valid_types}, got '{spatial_type}'."
        )

    # Get coordinates
    coords = _get_spot_coordinates(adata)

    # CellTypeComposition uses pie charts — special path
    if spatial_type == "CellTypeComposition":
        return _plot_cell_type_composition(
            adata,
            spatial_features,
            colors,
            point_size,
            figsize,
            coords,
        )

    # Prepare data and metadata
    values_dict, legend_name, is_discrete = _prepare_data(
        adata, spatial_type, spatial_features, scale_type_gene
    )

    if not values_dict:
        raise ValueError("No valid features to plot.")

    features = list(values_dict.keys())
    n_features = len(features)

    if colors is None:
        colors = _COLORMAPS.get(spatial_type)

    # Per-feature legend name override for LRNetworkScore
    legend_overrides = {}
    if spatial_type == "LRNetworkScore":
        for feat in features:
            legend_overrides[feat] = "Score" if feat == "Network_Score" else "-log10pv"

    # Single feature on provided axes
    if n_features == 1 and ax is not None:
        _plot_single(
            ax,
            coords,
            values_dict[features[0]],
            features[0],
            legend_overrides.get(features[0], legend_name),
            is_discrete,
            colors,
            point_size,
            point_alpha,
            same_scale_fraction and spatial_type == "CellFraction",
            spatial_type,
        )
        return None

    # Multi-panel layout
    nrows = int(np.ceil(n_features / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, feature in enumerate(features):
        r, c = divmod(idx, ncols)
        _plot_single(
            axes[r, c],
            coords,
            values_dict[feature],
            feature,
            legend_overrides.get(feature, legend_name),
            is_discrete,
            colors,
            point_size,
            point_alpha,
            same_scale_fraction and spatial_type == "CellFraction",
            spatial_type,
        )

    # Hide unused axes
    for idx in range(n_features, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig


# ===========================================================================
# Colocalization visualization (2-panel)
# ===========================================================================


def visualize_colocalization(
    adata: ad.AnnData,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Visualize cell-type pair colocalization.

    Equivalent to SpaCET.visualize.colocalization() in R.
    Panel 1: Dot plot (fraction_rho color, fraction_product size).
    Panel 2: Scatter of fraction_rho vs reference_rho with weighted LM.

    Parameters
    ----------
    adata : AnnData
        Must have adata.uns['spacet']['CCI']['colocalization'].

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    cci = spacet.get("CCI", {})
    summary_df = cci.get("colocalization")
    if summary_df is None:
        raise ValueError("Run cci_colocalization() first.")

    summary_df = summary_df.copy()

    # Cap fraction_product at 0.02 for display
    summary_df.loc[summary_df["fraction_product"] > 0.02, "fraction_product"] = 0.02

    # Determine cell type order
    deconv = spacet.get("deconvolution", {})
    prop_mat = deconv.get("propMat")
    ref = deconv.get("Ref", {})
    lineage_tree = ref.get("lineageTree", {})

    if prop_mat is not None and "Malignant cell state A" in prop_mat.index:
        states = [ct for ct in prop_mat.index if ct.startswith("Malignant cell state")]
        ct_order = states + _flatten_lineage_tree(lineage_tree)
    elif prop_mat is not None and "Malignant" in prop_mat.index:
        ct_order = ["Malignant"] + _flatten_lineage_tree(lineage_tree)
    else:
        ct_order = _flatten_lineage_tree(lineage_tree)
    ct_order = list(dict.fromkeys(ct_order))  # unique, preserve order

    # Filter to known cell types
    mask = summary_df["cell_type_1"].isin(ct_order) & summary_df["cell_type_2"].isin(
        ct_order
    )
    summary_df = summary_df[mask].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Panel 1: Dot plot ---
    ct1_cat = pd.Categorical(
        summary_df["cell_type_1"],
        categories=list(reversed(ct_order)),
        ordered=True,
    )
    ct2_cat = pd.Categorical(
        summary_df["cell_type_2"],
        categories=ct_order,
        ordered=True,
    )

    x_pos = ct2_cat.codes.astype(float)
    y_pos = ct1_cat.codes.astype(float)
    rho = summary_df["fraction_rho"].values.astype(float)
    fp = summary_df["fraction_product"].values.astype(float)

    norm = Normalize(vmin=-0.6, vmax=0.6)
    cmap = LinearSegmentedColormap.from_list("rho_cmap", ["blue", "white", "red"])
    sizes = fp / fp.max() * 200 if fp.max() > 0 else fp * 0 + 20

    sc = ax1.scatter(
        x_pos, y_pos, c=rho, cmap=cmap, norm=norm, s=sizes, edgecolors="none"
    )
    ax1.set_xticks(range(len(ct_order)))
    ax1.set_xticklabels(ct_order, rotation=45, ha="right", fontsize=9)
    ax1.set_yticks(range(len(ct_order)))
    ax1.set_yticklabels(list(reversed(ct_order)), fontsize=9)
    ax1.set_title("Cell-cell colocalization", fontsize=11)
    ax1.set_xlabel("Cell lineages")
    ax1.set_ylabel("Cell lineages")
    plt.colorbar(sc, ax=ax1, label="Rho", shrink=0.7)

    # --- Panel 2: Scatter (fraction_rho vs reference_rho) ---
    sub_lineage = _flatten_lineage_tree(lineage_tree)
    upper = summary_df[
        (summary_df["cell_type_1"].astype(str) < summary_df["cell_type_2"].astype(str))
        & summary_df["cell_type_1"].isin(sub_lineage)
        & summary_df["cell_type_2"].isin(sub_lineage)
    ].copy()

    if "reference_rho" in upper.columns and not upper.empty:
        ref_rho = upper["reference_rho"].values.astype(float)
        frac_rho = upper["fraction_rho"].values.astype(float)
        fp2 = upper["fraction_product"].values.astype(float)
        sizes2 = fp2 / fp2.max() * 200 if fp2.max() > 0 else fp2 * 0 + 20

        ax2.scatter(ref_rho, frac_rho, s=sizes2, color="#856aad", alpha=0.7)

        # Weighted linear regression
        valid = np.isfinite(ref_rho) & np.isfinite(frac_rho)
        if valid.sum() > 2:
            coeffs = np.polyfit(
                ref_rho[valid],
                frac_rho[valid],
                1,
                w=fp2[valid],
            )
            x_line = np.linspace(
                ref_rho[valid].min(),
                ref_rho[valid].max(),
                100,
            )
            ax2.plot(
                x_line,
                np.polyval(coeffs, x_line),
                color="darkgrey",
                linewidth=1.5,
            )

        # Labels for prominent points
        labels = (
            upper["cell_type_1"].astype(str) + "_" + upper["cell_type_2"].astype(str)
        ).values
        label_mask = (fp2 >= 0.0005) & (np.abs(frac_rho) >= 0.1)
        for i in np.where(label_mask)[0]:
            ax2.annotate(
                labels[i],
                (ref_rho[i], frac_rho[i]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax2.set_title(
        "Correlation of cell fractions and\n" "cell reference profiles",
        fontsize=11,
    )
    ax2.set_xlabel("Cor (Reference profiles)")
    ax2.set_ylabel("Cor (Cell fractions)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ===========================================================================
# Cell-type pair visualization (3-panel)
# ===========================================================================


def visualize_cell_type_pair(
    adata: ad.AnnData,
    cell_type_pair: tuple[str, str],
    figsize: tuple[float, float] = (18, 5),
) -> plt.Figure:
    """Visualize interaction analysis of a co-localized cell-type pair.

    Equivalent to SpaCET.visualize.cellTypePair() in R.
    Panel 1: Spatial map (Both=green, ct1=red, ct2=blue).
    Panel 2: Scatter (fraction ct1 vs ct2, color by group, LM line).
    Panel 3: Boxplot (Both vs Single LR network scores).

    Parameters
    ----------
    adata : AnnData
        Must have CCI interaction results.
    cell_type_pair : tuple of two cell type names

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    cci = spacet.get("CCI", {})
    interaction = cci.get("interaction", {})
    test_res = interaction.get("testRes")
    group_mat = interaction.get("groupMat")

    if test_res is None or group_mat is None:
        raise ValueError("Run cci_cell_type_pair() first.")

    ct_pair = sorted(cell_type_pair)
    pair_key = f"{ct_pair[0]}_{ct_pair[1]}"

    if pair_key not in test_res.index:
        raise ValueError(
            f"Cell-type pair '{pair_key}' not found " "in interaction results."
        )

    deconv = spacet.get("deconvolution", {})
    prop_mat = deconv.get("propMat")
    coords = _get_spot_coordinates(adata)

    rho = test_res.loc[pair_key, "colocalization_rho"]
    pv1 = test_res.loc[pair_key, "colocalization_pv"]
    pv1_str = " < 2.2e-16" if float(pv1) == 0 else f" = {float(pv1):.3g}"

    cd1 = test_res.loc[pair_key, "groupCompare_cohen.d"]
    pv2 = test_res.loc[pair_key, "groupCompare_pv"]

    # Group assignments
    groups = group_mat.loc[pair_key, prop_mat.columns].values

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # --- Panel 1: Spatial map ---
    group_colors = {
        "Both": "green",
        ct_pair[0]: "red",
        ct_pair[1]: "blue",
        "Other": "grey",
    }
    for group_name, color in group_colors.items():
        mask = groups == group_name
        if mask.any():
            ax1.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                s=3,
                alpha=1.0,
                label=group_name,
            )
    ax1.set_title("Spatial distribution of two cell-types", fontsize=10)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # --- Panel 2: Scatter (fraction vs fraction) ---
    frac1 = prop_mat.loc[ct_pair[0]].values.astype(float)
    frac2 = prop_mat.loc[ct_pair[1]].values.astype(float)

    for group_name, color in [
        ("Both", "green"),
        (ct_pair[0], "red"),
        (ct_pair[1], "blue"),
        ("Other", "grey"),
    ]:
        mask = groups == group_name
        if mask.any():
            ax2.scatter(
                frac1[mask],
                frac2[mask],
                c=color,
                s=2,
                alpha=0.7,
                label=group_name,
            )

    # LM line
    valid = np.isfinite(frac1) & np.isfinite(frac2)
    if valid.sum() > 2:
        coeffs = np.polyfit(frac1[valid], frac2[valid], 1)
        x_line = np.linspace(frac1[valid].min(), frac1[valid].max(), 100)
        ax2.plot(x_line, np.polyval(coeffs, x_line), color="orange", linewidth=1.5)

    ax2.set_title(f"Spearman Rho = {rho}, P{pv1_str}", fontsize=10)
    ax2.set_xlabel(f"Cell fraction ({ct_pair[0]})")
    ax2.set_ylabel(f"Cell fraction ({ct_pair[1]})")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --- Panel 3: Boxplot (Both vs Single) ---
    lr_score = cci.get("LRNetworkScore")
    if lr_score is not None:
        lr_vals = lr_score[1, :].astype(float).copy()
        lr_vals = np.clip(lr_vals, None, 2.0)

        both_mask = groups == "Both"
        single_mask = np.isin(groups, ct_pair)

        both_vals = lr_vals[both_mask]
        single_vals = lr_vals[single_mask]

        box_data = []
        box_labels = []
        box_colors = []
        if len(both_vals) > 0:
            box_data.append(both_vals)
            box_labels.append("Both")
            box_colors.append("green")
        if len(single_vals) > 0:
            box_data.append(single_vals)
            box_labels.append("Single")
            box_colors.append("purple")

        if box_data:
            bp = ax3.boxplot(
                box_data,
                tick_labels=box_labels,
                patch_artist=True,
                showfliers=False,
                widths=0.5,
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
            for i, (data, color) in enumerate(zip(box_data, box_colors)):
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(data))
                ax3.scatter(
                    np.full(len(data), i + 1) + jitter,
                    data,
                    c=color,
                    s=1,
                    alpha=0.5,
                )

        ax3.set_title(f"Cohen's d={cd1}, P={pv2}", fontsize=10)
        ax3.set_xlabel(f"{ct_pair[0]} - {ct_pair[1]}")
        ax3.set_ylabel("LR network score")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
    else:
        ax3.text(
            0.5,
            0.5,
            "LRNetworkScore not available",
            transform=ax3.transAxes,
            ha="center",
        )

    fig.tight_layout()
    return fig


# ===========================================================================
# Distance to interface visualization
# ===========================================================================


def visualize_distance_to_interface(
    adata: ad.AnnData,
    cell_type_pair: tuple[str, str],
    n_permutation: int = 1000,
    figsize: tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Visualize distance of cell-cell interactions to tumor border.

    Equivalent to the visualization part of SpaCET.distance.to.interface() in R.
    Histogram with density overlay and dashed vertical line at observed distance.

    Parameters
    ----------
    adata : AnnData
        Must have CCI interface and interaction results.
    cell_type_pair : tuple of two cell type names
    n_permutation : int
        Number of permutations for p-value.

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    cci = spacet.get("CCI", {})
    interface = cci.get("interface")
    interaction = cci.get("interaction", {})
    test_res = interaction.get("testRes")
    group_mat = interaction.get("groupMat")

    if interface is None:
        raise ValueError("Run identify_interface() first.")
    if test_res is None or group_mat is None:
        raise ValueError("Run cci_cell_type_pair() first.")

    ct_pair = sorted(cell_type_pair)
    pair_key = f"{ct_pair[0]}_{ct_pair[1]}"

    if test_res.loc[pair_key, "Interaction"] is False:
        raise ValueError(f"Interaction not significant for {pair_key}.")

    # Get interface spots
    if isinstance(interface, pd.DataFrame):
        interface_row = interface.iloc[0]
    else:
        interface_row = interface
    spot_border = [s for s in interface.columns if str(interface_row[s]) == "Interface"]
    spot_stroma = [s for s in interface.columns if str(interface_row[s]) == "Stroma"]

    # Get group assignments
    group_row = group_mat.loc[pair_key]
    spot_both = [
        s for s in group_row.index if str(group_row[s]) == "Both" and s in spot_stroma
    ]
    spot_single = [
        s for s in group_row.index if str(group_row[s]) in ct_pair and s in spot_stroma
    ]

    def _spot_distance(s1: str, s2: str) -> float:
        xy1 = [float(x) for x in s1.split("x")]
        xy2 = [float(x) for x in s2.split("x")]
        return np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

    def _min_distance_to_border(spot: str) -> float:
        if spot_border:
            return min(_spot_distance(spot, b) for b in spot_border)
        return np.inf

    # Observed distance
    if not spot_both:
        raise ValueError("No 'Both' spots found in stroma region.")

    d_both = [_min_distance_to_border(s) for s in spot_both]
    d0 = np.mean(d_both)

    # Permutation distribution
    d_single_all = {s: _min_distance_to_border(s) for s in spot_single}
    d_vals = np.array(list(d_single_all.values()))

    d_perm = np.zeros(n_permutation)
    for i in range(n_permutation):
        rng_i = np.random.default_rng(i + 1)
        if len(d_vals) >= len(spot_both):
            sampled = rng_i.choice(
                d_vals,
                size=len(spot_both),
                replace=False,
            )
        else:
            sampled = d_vals
        d_perm[i] = np.mean(sampled)

    pv = (np.sum(d_perm <= d0) + 1) / (n_permutation + 1)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        d_perm,
        bins=30,
        density=True,
        color="gainsboro",
        edgecolor="grey",
        alpha=0.5,
    )

    # KDE density overlay
    from scipy.stats import gaussian_kde

    if len(d_perm) > 1:
        kde = gaussian_kde(d_perm)
        x_kde = np.linspace(d_perm.min(), d_perm.max(), 200)
        ax.plot(x_kde, kde(x_kde), color="grey", linewidth=1.2)

    ax.axvline(d0, color="green", linestyle="--", linewidth=2)
    ax.set_title(f"Permutation\n P = {pv:.3g}", fontsize=12)
    ax.set_xlabel("Distance to Tumor-Stroma Interface", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ===========================================================================
# Internal helpers
# ===========================================================================


def _get_spot_coordinates(adata: ad.AnnData) -> np.ndarray:
    """Extract (x, y) coordinates from spot IDs or obsm['spatial'].

    For Visium data with 'rowxcol' format spot IDs, applies the same
    coordinate transformation as R's visualSpatial (coord_flip equivalent).
    """
    if "spatial" in adata.obsm:
        return adata.obsm["spatial"]

    # Parse from "row x col" spot IDs
    coords = []
    for sid in adata.obs_names:
        parts = sid.split("x")
        if len(parts) == 2:
            row, col = float(parts[0]), float(parts[1])
            coords.append((col, row))
        else:
            coords.append((0.0, 0.0))

    coords = np.array(coords)

    # Apply Visium-like coord_flip: R does x=xDiml-coordi[,1], y=coordi[,2]
    # This flips the x-axis to match tissue orientation
    platform = adata.uns.get("spacet", {}).get("platform", "")
    if "visium" in str(platform).lower():
        x_max = coords[:, 0].max()
        coords[:, 0] = x_max - coords[:, 0]

    return coords


def _prepare_data(
    adata: ad.AnnData,
    spatial_type: str,
    spatial_features: list[str] | None,
    scale_type_gene: str,
) -> tuple[dict[str, np.ndarray], str, bool]:
    """Prepare values dict, legend name, and discrete flag."""
    spacet = adata.uns.get("spacet", {})

    if spatial_type == "QualityControl":
        if "UMI" not in adata.obs.columns:
            raise ValueError("Run quality_control() first.")
        mat = {
            "UMI": adata.obs["UMI"].values.astype(float),
            "Gene": adata.obs["Gene"].values.astype(float),
        }
        if spatial_features:
            mat = {k: v for k, v in mat.items() if k in spatial_features}
        return mat, "Count", False

    elif spatial_type == "GeneExpression":
        return _prepare_gene_expression(adata, spatial_features, scale_type_gene)

    elif spatial_type == "CellFraction":
        return _prepare_cell_fraction(adata, spatial_features, spacet)

    elif spatial_type == "MostAbundantCellType":
        return _prepare_most_abundant(adata, spatial_features, spacet)

    elif spatial_type == "LRNetworkScore":
        return _prepare_lr_network_score(spatial_features, spacet)

    elif spatial_type == "Interface":
        return _prepare_interface(spatial_features, spacet)

    elif spatial_type == "GeneSetScore":
        return _prepare_gene_set_score(spatial_features, spacet)

    elif spatial_type == "SecretedProteinActivity":
        return _prepare_secreted_protein_activity(spatial_features, spacet)

    elif spatial_type == "SignalingPattern":
        return _prepare_signaling_pattern(spatial_features, spacet)

    elif spatial_type == "metaData":
        return _prepare_metadata(adata, spatial_features)

    return {}, "", False


def _prepare_gene_expression(
    adata: ad.AnnData,
    spatial_features: list[str] | None,
    scale_type_gene: str,
) -> tuple[dict[str, np.ndarray], str, bool]:
    if spatial_features is None:
        raise ValueError("spatial_features required for GeneExpression.")

    X = adata.X
    gene_names = np.array(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    mat = {}
    for gene in spatial_features:
        if gene not in gene_to_idx:
            logger.warning("Gene '%s' not found, skipping.", gene)
            continue
        idx = gene_to_idx[gene]
        if sparse.issparse(X):
            vals = np.asarray(X[:, idx].todense()).ravel().astype(np.float64)
        else:
            vals = X[:, idx].astype(np.float64)

        if scale_type_gene == "LogRawCounts":
            vals = np.log2(vals + 1)
        elif scale_type_gene == "LogTPM/10":
            if sparse.issparse(X):
                row_sums = np.asarray(X.sum(axis=1)).ravel()
            else:
                row_sums = X.sum(axis=1)
            vals = vals / row_sums * 1e5
            vals = np.log2(vals + 1)
        elif scale_type_gene == "LogTPM":
            if sparse.issparse(X):
                row_sums = np.asarray(X.sum(axis=1)).ravel()
            else:
                row_sums = X.sum(axis=1)
            vals = vals / row_sums * 1e6
            vals = np.log2(vals + 1)
        mat[gene] = vals

    legend = {
        "RawCounts": "Counts",
        "LogRawCounts": "LogCounts",
        "LogTPM/10": "LogTPM/10",
        "LogTPM": "LogTPM",
    }.get(scale_type_gene, "Expression")
    return mat, legend, False


def _prepare_cell_fraction(
    adata: ad.AnnData,
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    deconv = spacet.get("deconvolution", {})
    prop_mat = deconv.get("propMat")
    if prop_mat is None:
        raise ValueError("Run deconvolution() first.")

    if spatial_features is None or "All" in (spatial_features or []):
        spatial_features = list(prop_mat.index)

    mat = {}
    for ct in spatial_features:
        if ct in prop_mat.index:
            mat[ct] = prop_mat.loc[ct].values.astype(float)
    return mat, "Fraction", False


def _prepare_most_abundant(
    adata: ad.AnnData,
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    deconv = spacet.get("deconvolution", {})
    prop_mat = deconv.get("propMat")
    if prop_mat is None:
        raise ValueError("Run deconvolution() first.")

    if spatial_features is None:
        spatial_features = ["MajorLineage"]

    ref = deconv.get("Ref", {})
    lineage_tree = ref.get("lineageTree", {})

    mat = {}
    for feature in spatial_features:
        if feature not in ("MajorLineage", "SubLineage"):
            raise ValueError("spatial_features must be 'MajorLineage' or 'SubLineage'.")

        if feature == "MajorLineage":
            all_cell_types = list(lineage_tree.keys())
        else:
            all_cell_types = _flatten_lineage_tree(lineage_tree)

        # Add Malignant if present
        if "Malignant" not in all_cell_types and "Malignant" in prop_mat.index:
            all_cell_types = ["Malignant"] + all_cell_types

        # Filter to available cell types
        available = [ct for ct in all_cell_types if ct in prop_mat.index]
        sub_mat = prop_mat.loc[available]

        # For each spot, find the most abundant cell type
        most_abundant = sub_mat.idxmax(axis=0).values
        mat[feature] = most_abundant

    return mat, "Cell Type", True


def _prepare_lr_network_score(
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    cci = spacet.get("CCI", {})
    lr_score = cci.get("LRNetworkScore")
    if lr_score is None:
        raise ValueError("Run cci_lr_network_score() first.")

    if spatial_features is None:
        spatial_features = ["Network_Score", "Network_Score_pv"]

    lr_index = cci.get(
        "LRNetworkScore_index",
        [
            "Raw_expr",
            "Network_Score",
            "Network_Score_pv",
        ],
    )
    idx_map = {name: i for i, name in enumerate(lr_index)}

    mat = {}
    for feat in spatial_features:
        if feat not in idx_map:
            continue
        vals = lr_score[idx_map[feat], :].astype(float).copy()
        if feat == "Network_Score":
            vals = np.clip(vals, 0.5, 1.5)
        elif feat == "Network_Score_pv":
            vals = -np.log10(np.clip(vals, 1e-300, 1.0))
        mat[feat] = vals

    return mat, "Score", False


def _prepare_interface(
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    cci = spacet.get("CCI", {})
    interface_df = cci.get("interface")
    if interface_df is None:
        raise ValueError("Run identify_interface() first.")

    if spatial_features is None:
        spatial_features = list(interface_df.index)

    mat = {}
    for feat in spatial_features:
        if feat in interface_df.index:
            mat[feat] = interface_df.loc[feat].values
    return mat, "Spot", True


def _prepare_gene_set_score(
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    gs_scores = spacet.get("GeneSetScore")
    if gs_scores is None:
        raise ValueError("Run gene_set_score() first.")

    if spatial_features is None:
        spatial_features = list(gs_scores.index)

    mat = {}
    for feat in spatial_features:
        if feat in gs_scores.index:
            mat[feat] = gs_scores.loc[feat].values.astype(float)
    return mat, "Score", False


def _prepare_secreted_protein_activity(
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    sec_act = spacet.get("SecAct_output", {})
    spa = sec_act.get("SecretedProteinActivity", {})
    zscore = spa.get("zscore")
    if zscore is None:
        raise ValueError("Run SecAct signaling inference first.")

    if spatial_features is None:
        spatial_features = list(zscore.index) if hasattr(zscore, "index") else []

    mat = {}
    if isinstance(zscore, pd.DataFrame):
        for feat in spatial_features:
            if feat in zscore.index:
                mat[feat] = zscore.loc[feat].values.astype(float)
    return mat, "Activity", False


def _prepare_signaling_pattern(
    spatial_features: list[str] | None,
    spacet: dict,
) -> tuple[dict[str, np.ndarray], str, bool]:
    sec_act = spacet.get("SecAct_output", {})
    pattern = sec_act.get("pattern", {})
    signal_h = pattern.get("signal.H")
    if signal_h is None:
        raise ValueError("Run SecAct signaling pattern first.")

    if spatial_features is None or "All" in (spatial_features or []):
        spatial_features = list(signal_h.index) if hasattr(signal_h, "index") else []

    mat = {}
    if isinstance(signal_h, pd.DataFrame):
        for feat in spatial_features:
            if feat in signal_h.index:
                mat[feat] = signal_h.loc[feat].values.astype(float)
    return mat, "Signal", False


def _prepare_metadata(
    adata: ad.AnnData,
    spatial_features: list[str] | None,
) -> tuple[dict[str, np.ndarray], str, bool]:
    spacet = adata.uns.get("spacet", {})
    meta = spacet.get("metaData")

    if spatial_features is None:
        raise ValueError("spatial_features required for metaData.")
    if len(spatial_features) != 1:
        raise ValueError("metaData supports only one feature at a time.")

    feat = spatial_features[0]
    if meta is not None and isinstance(meta, pd.DataFrame) and feat in meta.index:
        mat = {feat: meta.loc[feat].values}
    elif feat in adata.obs.columns:
        mat = {feat: adata.obs[feat].values}
    else:
        raise ValueError(f"Feature '{feat}' not found in metaData or obs.")

    return mat, feat, True


def _plot_single(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    legend_name: str,
    is_discrete: bool,
    colors: list[str] | None,
    point_size: float,
    point_alpha: float,
    fixed_scale: bool,
    spatial_type: str = "",
) -> None:
    """Plot a single spatial feature on the given axes."""
    x = coords[:, 0]
    y = coords[:, 1]

    if is_discrete:
        categories = sorted({str(v) for v in values if pd.notna(v)})

        if colors is None:
            if spatial_type == "Interface":
                color_map = _DISCRETE_COLORS.get("Interface", {})
            elif spatial_type == "MostAbundantCellType":
                color_map = {
                    cat: _CELLTYPE_PALETTE[i % len(_CELLTYPE_PALETTE)]
                    for i, cat in enumerate(categories)
                }
            else:
                color_map = {cat: f"C{i}" for i, cat in enumerate(categories)}
        elif isinstance(colors, dict):
            color_map = colors
        elif isinstance(colors, list):
            color_map = {
                cat: colors[i % len(colors)] for i, cat in enumerate(categories)
            }
        else:
            color_map = {}

        for cat in categories:
            mask = np.array([str(v) == cat for v in values])
            c = color_map.get(cat, f"C{categories.index(cat)}")
            ax.scatter(
                x[mask],
                y[mask],
                c=c,
                s=point_size,
                alpha=point_alpha,
                label=cat,
            )
        ax.legend(
            title=legend_name,
            fontsize=7,
            title_fontsize=8,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        vals = np.asarray(values, dtype=float)
        if colors:
            cmap = LinearSegmentedColormap.from_list("custom", colors)
        else:
            cmap = "viridis"

        vmin = 0.0 if fixed_scale else None
        vmax = 1.0 if fixed_scale else None

        # Sort by value so high values plot on top
        order = np.argsort(vals)
        sc = ax.scatter(
            x[order],
            y[order],
            c=vals[order],
            cmap=cmap,
            s=point_size,
            alpha=point_alpha,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label=legend_name, shrink=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


# ===========================================================================
# CellTypeComposition (pie chart per spot)
# ===========================================================================


def _plot_cell_type_composition(
    adata: ad.AnnData,
    spatial_features: list[str] | None,
    colors: list[str] | None,
    point_size: float,
    figsize: tuple[float, float] | None,
    coords: np.ndarray,
) -> plt.Figure:
    """Plot cell type composition as pie charts at each spot.

    Equivalent to R's scatterpie::geom_scatterpie().
    """
    spacet = adata.uns.get("spacet", {})
    deconv = spacet.get("deconvolution", {})
    prop_mat = deconv.get("propMat")
    if prop_mat is None:
        raise ValueError("Run deconvolution() first.")

    ref = deconv.get("Ref", {})
    lineage_tree = ref.get("lineageTree", {})

    if spatial_features is None:
        spatial_features = ["MajorLineage"]

    feature = spatial_features[0] if spatial_features else "MajorLineage"
    if feature not in ("MajorLineage", "SubLineage"):
        raise ValueError("spatial_features must be 'MajorLineage' or 'SubLineage'.")

    if feature == "MajorLineage":
        all_cell_types = list(lineage_tree.keys())
    else:
        all_cell_types = _flatten_lineage_tree(lineage_tree)

    if "Malignant" not in all_cell_types and "Malignant" in prop_mat.index:
        all_cell_types = ["Malignant"] + all_cell_types

    available = [ct for ct in all_cell_types if ct in prop_mat.index]

    # Add Unidentifiable if present
    if "Unidentifiable" in prop_mat.index:
        available.append("Unidentifiable")

    sub_mat = prop_mat.loc[available].values.T  # (n_spots, n_cell_types)

    if colors is None:
        colors = _CELLTYPE_PALETTE[: len(available)]

    if figsize is None:
        figsize = (8, 7)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute radius from point_size
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    radius = min(x_range, y_range) / 80 * point_size

    for i in range(len(coords)):
        fracs = sub_mat[i]
        total = fracs.sum()
        if total <= 0:
            continue

        fracs_norm = fracs / total
        cx, cy = coords[i]

        # Draw pie wedges
        theta1 = 0
        for j, frac in enumerate(fracs_norm):
            if frac <= 0:
                continue
            theta2 = theta1 + frac * 360
            wedge = Wedge(
                (cx, cy),
                radius,
                theta1,
                theta2,
                facecolor=colors[j % len(colors)],
                edgecolor="none",
                linewidth=0,
            )
            ax.add_patch(wedge)
            theta1 = theta2

    ax.set_xlim(coords[:, 0].min() - radius * 3, coords[:, 0].max() + radius * 3)
    ax.set_ylim(coords[:, 1].min() - radius * 3, coords[:, 1].max() + radius * 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(feature, fontsize=11)

    # Legend
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=colors[i % len(colors)], label=ct)
        for i, ct in enumerate(available)
    ]
    ax.legend(
        handles=handles,
        title="Cell Type",
        fontsize=7,
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    fig.tight_layout()
    return fig


def _flatten_lineage_tree(lineage_tree: dict) -> list[str]:
    """Flatten lineage tree dict to ordered list of all cell types."""
    result = []
    for _major, subs in lineage_tree.items():
        if isinstance(subs, list):
            result.extend(subs)
        else:
            result.append(subs)
    return result
