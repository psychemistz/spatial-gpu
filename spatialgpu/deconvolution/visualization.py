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
        "#1A9850",
        "#1A9850",
        "#1A9850",
        "#1A9850",
        "#FDAE61",
        "#FC8D59",
        "#D7191C",
    ],
    "SignalingPattern": [
        "#1A9850",
        "#1A9850",
        "#1A9850",
        "#1A9850",
        "#FDAE61",
        "#FC8D59",
        "#D7191C",
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
    yflip = _needs_yflip(adata)

    # CellTypeComposition uses pie charts — special path
    if spatial_type == "CellTypeComposition":
        return _plot_cell_type_composition(
            adata,
            spatial_features,
            colors,
            point_size,
            figsize,
            coords,
            yflip,
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
            yflip,
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
            yflip,
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
                edgecolors="none",
            )
    ax1.set_title("Spatial distribution of two cell-types", fontsize=10)
    ax1.set_aspect("equal")
    if _needs_yflip(adata):
        ax1.invert_yaxis()
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

    # Draw histogram regardless of significance

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


def _needs_yflip(adata: ad.AnnData) -> bool:
    """Check if Y-axis should be inverted (Visium pixel coordinates)."""
    platform = adata.uns.get("spacet_platform", "")
    # Visium/VisiumHD use pixel coordinates where row increases downward
    return platform in ("Visium", "VisiumHD")


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
    signal_h = pattern.get("signal_H")
    if signal_h is None:
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
    yflip: bool = False,
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
                edgecolors="none",
            )
        # Fixed legend marker size (R uses override.aes size=3)
        legend = ax.legend(
            title=legend_name,
            fontsize=7,
            title_fontsize=8,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
        for h in legend.legend_handles:
            h.set_sizes([20])
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
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=legend_name, shrink=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    if yflip:
        ax.invert_yaxis()
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
    yflip: bool = False,
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

    # Fallback: if lineageTree doesn't match propMat, use all propMat types
    if len(available) <= 1:
        exclude = {"Unidentifiable", "Macrophage other"}
        available = [ct for ct in prop_mat.index if ct not in exclude]

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
    if yflip:
        ax.invert_yaxis()
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


# ---------------------------------------------------------------------------
# SecAct Visualization Functions
# ---------------------------------------------------------------------------

# Default color palettes matching R
_SECACT_HEATMAP_COLORS = ["#03c383", "#aad962", "#fbbf45", "#ef6a32"]
_SECACT_BAR_COLORS = ["#91bfdb", "#fc8d59"]
_SECACT_LOLLIPOP_COLOR = "#619CFF"
_SECACT_DOT_CMAP_COLORS = ["#fbbf45", "#ed0345"]
_SECACT_VELOCITY_COLORS = ["#b8e186", "#de77ae", "#c51b7d"]
_SECACT_VELOCITY_CONTOUR_COLORS = [
    "#f0fff0",
    "#b2e2b2",
    "#66cc66",
    "#ffcc99",
    "#ff9966",
    "#ff6633",
    "#cc3300",
]


def visualize_secact_heatmap(
    adata: ad.AnnData,
    colors_cell_type: dict[str, str] | None = None,
    row_sorted: bool = False,
    column_sorted: bool = False,
    figsize: tuple[float, float] = (10, 8),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """CCC heatmap: sender × receiver count matrix with marginal bar plots.

    Equivalent to ``SecAct.CCC.heatmap()`` in R.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct CCC results.
    colors_cell_type : dict, optional
        Cell type → color mapping.
    row_sorted : bool
        Sort rows by total count (descending).
    column_sorted : bool
        Sort columns by total count (descending).
    figsize : tuple
        Figure size.
    save : str, optional
        Path to save figure.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    from matplotlib.gridspec import GridSpec

    spacet = adata.uns.get("spacet", {})
    secact_out = spacet.get("SecAct_output", {})
    ccc = secact_out.get("SecretedProteinCCC")
    if ccc is None or len(ccc) == 0:
        raise ValueError("No CCC results. Run secact_spatial_ccc() first.")

    # Build sender × receiver count matrix
    all_types = sorted(set(ccc["sender"].tolist() + ccc["receiver"].tolist()))
    mat = pd.DataFrame(0, index=all_types, columns=all_types, dtype=float)
    for _, row in ccc.iterrows():
        s, r = row["sender"], row["receiver"]
        mat.loc[s, r] += 1

    # Set diagonal to NaN
    for ct in all_types:
        if ct in mat.index and ct in mat.columns:
            mat.loc[ct, ct] = np.nan

    if row_sorted:
        mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    if column_sorted:
        mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]

    if colors_cell_type is None:
        tab20 = plt.cm.tab20(np.linspace(0, 1, max(20, len(all_types))))
        colors_cell_type = {ct: tab20[i] for i, ct in enumerate(all_types)}

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.05, wspace=0.05
    )

    # Top bar (column sums)
    ax_top = fig.add_subplot(gs[0, 0])
    col_sums = mat.sum(axis=0, skipna=True)
    bar_colors = [colors_cell_type.get(c, "gray") for c in mat.columns]
    ax_top.bar(range(len(mat.columns)), col_sums.values, color=bar_colors)
    ax_top.set_xlim(-0.5, len(mat.columns) - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("Count")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right bar (row sums)
    ax_right = fig.add_subplot(gs[1, 1])
    row_sums = mat.sum(axis=1, skipna=True)
    bar_colors_r = [colors_cell_type.get(r, "gray") for r in mat.index]
    ax_right.barh(range(len(mat.index)), row_sums.values, color=bar_colors_r)
    ax_right.set_ylim(-0.5, len(mat.index) - 0.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Count")
    ax_right.invert_yaxis()
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 0])
    cmap = LinearSegmentedColormap.from_list("gwr", ["green", "white", "red"])
    vals = mat.values.copy()
    ax_main.imshow(vals, cmap=cmap, aspect="auto", interpolation="nearest")

    # Cell text
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax_main.text(j, i, str(int(v)), ha="center", va="center", fontsize=8)

    ax_main.set_xticks(range(len(mat.columns)))
    ax_main.set_xticklabels(mat.columns, rotation=90, fontsize=8)
    ax_main.set_yticks(range(len(mat.index)))
    ax_main.set_yticklabels(mat.index, fontsize=8)
    ax_main.set_ylabel("Sender")
    ax_main.set_xlabel("Receiver")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_circle(
    adata: ad.AnnData,
    colors_cell_type: dict[str, str] | None = None,
    sender: list[str] | None = None,
    receiver: list[str] | None = None,
    figsize: tuple[float, float] = (8, 8),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """CCC chord/circle diagram: directional links between cell types.

    Equivalent to ``SecAct.CCC.circle()`` in R.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct CCC results.
    colors_cell_type : dict, optional
        Cell type → color mapping.
    sender : list[str], optional
        Filter to these senders.
    receiver : list[str], optional
        Filter to these receivers.
    figsize : tuple
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    secact_out = spacet.get("SecAct_output", {})
    ccc = secact_out.get("SecretedProteinCCC")
    if ccc is None or len(ccc) == 0:
        raise ValueError("No CCC results. Run secact_spatial_ccc() first.")

    # Build count matrix
    all_types = sorted(set(ccc["sender"].tolist() + ccc["receiver"].tolist()))
    mat = pd.DataFrame(0, index=all_types, columns=all_types, dtype=float)
    for _, row in ccc.iterrows():
        s, r = row["sender"], row["receiver"]
        mat.loc[s, r] += 1

    for ct in all_types:
        if ct in mat.index and ct in mat.columns:
            mat.loc[ct, ct] = 0

    if colors_cell_type is None:
        tab20 = plt.cm.tab20(np.linspace(0, 1, max(20, len(all_types))))
        colors_cell_type = {ct: tab20[i] for i, ct in enumerate(all_types)}

    # Chord diagram via pycirclize (matches R circlize::chordDiagram)
    from pycirclize import Circos

    # Filter by sender/receiver if specified
    if sender is not None:
        for s in list(mat.index):
            if s not in sender:
                mat.loc[s, :] = 0
    if receiver is not None:
        for r in list(mat.columns):
            if r not in receiver:
                mat.loc[:, r] = 0

    sector_colors = {
        ct: colors_cell_type.get(ct, "gray") for ct in all_types
    }

    circos = Circos.initialize_from_matrix(
        mat,
        space=3,
        cmap=sector_colors,
        label_kws=dict(fontsize=8),
        link_kws=dict(direction=1, ec="white", lw=0.3),
    )

    fig = circos.plotfig(figsize=figsize)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_sankey(
    adata: ad.AnnData,
    sender: list[str],
    secreted_protein: list[str],
    receiver: list[str],
    colors_cell_type: dict[str, str] | None = None,
    figsize: tuple[float, float] = (10, 8),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """CCC Sankey/alluvial diagram: sender → protein → receiver flows.

    Equivalent to ``SecAct.CCC.sankey()`` in R.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct CCC results.
    sender : list[str]
        Sender cell types to include.
    secreted_protein : list[str]
        Secreted proteins to include.
    receiver : list[str]
        Receiver cell types to include.
    colors_cell_type : dict, optional
        Cell type / protein → color mapping.
    figsize : tuple
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    secact_out = spacet.get("SecAct_output", {})
    ccc = secact_out.get("SecretedProteinCCC")
    if ccc is None or len(ccc) == 0:
        raise ValueError("No CCC results. Run secact_spatial_ccc() first.")

    # Filter
    mask = (
        ccc["sender"].isin(sender)
        & ccc["secretedProtein"].isin(secreted_protein)
        & ccc["receiver"].isin(receiver)
    )
    ccc_sub = ccc[mask].copy()

    if len(ccc_sub) == 0:
        raise ValueError(
            "No CCC entries match the given sender/protein/receiver filters."
        )

    # All unique labels
    all_labels = sorted(
        set(
            ccc_sub["sender"].tolist()
            + ccc_sub["secretedProtein"].tolist()
            + ccc_sub["receiver"].tolist()
        )
    )

    if colors_cell_type is None:
        tab20 = plt.cm.tab20(np.linspace(0, 1, max(20, len(all_labels))))
        colors_cell_type = {lb: tab20[i] for i, lb in enumerate(all_labels)}

    fig, ax = plt.subplots(figsize=figsize)

    # Three columns: sender (x=0), protein (x=1), receiver (x=2)
    # Count flows
    s_to_p = (
        ccc_sub.groupby(["sender", "secretedProtein"]).size().reset_index(name="count")
    )
    p_to_r = (
        ccc_sub.groupby(["secretedProtein", "receiver"])
        .size()
        .reset_index(name="count")
    )

    # Compute node positions
    s_counts = ccc_sub["sender"].value_counts().sort_values(ascending=False)
    p_counts_l = (
        s_to_p.groupby("secretedProtein")["count"].sum().sort_values(ascending=False)
    )
    r_counts = ccc_sub["receiver"].value_counts().sort_values(ascending=False)

    def _node_positions(counts, x_pos):
        total = counts.sum()
        y = 0
        positions = {}
        for name, cnt in counts.items():
            h = cnt / total
            positions[name] = {"x": x_pos, "y_center": y + h / 2, "height": h}
            y += h + 0.02
        return positions

    s_pos = _node_positions(s_counts, 0)
    p_pos = _node_positions(p_counts_l, 1)
    r_pos = _node_positions(r_counts, 2)

    # Draw nodes as rectangles
    node_width = 0.08
    for positions in [s_pos, p_pos, r_pos]:
        for name, pos in positions.items():
            color = colors_cell_type.get(name, "gray")
            rect = plt.Rectangle(
                (pos["x"] - node_width / 2, pos["y_center"] - pos["height"] / 2),
                node_width,
                pos["height"],
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                pos["x"],
                pos["y_center"],
                name,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

    # Draw flows (sender → protein)
    for _, row in s_to_p.iterrows():
        s, p, cnt = row["sender"], row["secretedProtein"], row["count"]
        if s not in s_pos or p not in p_pos:
            continue
        color = colors_cell_type.get(s, "gray")
        y_s = s_pos[s]["y_center"]
        y_p = p_pos[p]["y_center"]
        flow_h = cnt / s_counts.sum() * 0.8
        ax.fill_between(
            [0 + node_width / 2, 1 - node_width / 2],
            [y_s - flow_h / 2, y_p - flow_h / 2],
            [y_s + flow_h / 2, y_p + flow_h / 2],
            alpha=0.3,
            color=color,
        )

    # Draw flows (protein → receiver)
    for _, row in p_to_r.iterrows():
        p, r, cnt = row["secretedProtein"], row["receiver"], row["count"]
        if p not in p_pos or r not in r_pos:
            continue
        color = colors_cell_type.get(p, "gray")
        y_p = p_pos[p]["y_center"]
        y_r = r_pos[r]["y_center"]
        flow_h = cnt / p_counts_l.sum() * 0.8
        ax.fill_between(
            [1 + node_width / 2, 2 - node_width / 2],
            [y_p - flow_h / 2, y_r - flow_h / 2],
            [y_p + flow_h / 2, y_r + flow_h / 2],
            alpha=0.3,
            color=color,
        )

    ax.set_xlim(-0.3, 2.3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sender", "Secreted Protein", "Receiver"])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_dotplot(
    adata: ad.AnnData,
    sender: list[str],
    secreted_protein: list[str],
    receiver: list[str],
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """CCC dot plot: secreted protein × sender→receiver pairs.

    Equivalent to ``SecAct.CCC.dot()`` in R.

    Parameters
    ----------
    adata : AnnData
        Must have SecAct CCC results.
    sender : list[str]
        Sender cell types to include.
    secreted_protein : list[str]
        Secreted proteins to include.
    receiver : list[str]
        Receiver cell types to include.
    figsize : tuple, optional
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    spacet = adata.uns.get("spacet", {})
    secact_out = spacet.get("SecAct_output", {})
    ccc = secact_out.get("SecretedProteinCCC")
    if ccc is None or len(ccc) == 0:
        raise ValueError("No CCC results. Run secact_spatial_ccc() first.")

    mask = (
        ccc["sender"].isin(sender)
        & ccc["secretedProtein"].isin(secreted_protein)
        & ccc["receiver"].isin(receiver)
    )
    ccc_sub = ccc[mask].copy()

    if len(ccc_sub) == 0:
        raise ValueError("No CCC entries match the given filters.")

    # Create s2r label
    ccc_sub["s2r"] = ccc_sub["sender"] + "->" + ccc_sub["receiver"]

    # Score and -log10(pv)
    if "ratio" in ccc_sub.columns:
        ccc_sub["score"] = ccc_sub["ratio"]
        pv_col = "pv"
    else:
        ccc_sub["score"] = ccc_sub.get("overall_strength", 0)
        pv_col = "overall_pv" if "overall_pv" in ccc_sub.columns else "pv"

    ccc_sub["logpv"] = -np.log10(ccc_sub[pv_col].clip(lower=1e-300))

    x_labels = sorted(ccc_sub["s2r"].unique())
    y_labels = list(reversed(secreted_protein))

    if figsize is None:
        figsize = (max(4, len(x_labels) * 0.8 + 2), max(4, len(y_labels) * 0.5 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    x_map = {v: i for i, v in enumerate(x_labels)}
    y_map = {v: i for i, v in enumerate(y_labels)}

    cmap = LinearSegmentedColormap.from_list("dot_cmap", _SECACT_DOT_CMAP_COLORS)

    scores = ccc_sub["score"].values
    norm = (
        Normalize(vmin=scores.min(), vmax=scores.max())
        if len(scores) > 0
        else Normalize(0, 1)
    )

    for _, row in ccc_sub.iterrows():
        sp = row["secretedProtein"]
        s2r = row["s2r"]
        if sp not in y_map or s2r not in x_map:
            continue
        x = x_map[s2r]
        y = y_map[sp]
        size = row["logpv"] * 20 + 10
        color = cmap(norm(row["score"]))
        ax.scatter(x, y, s=size, c=[color], edgecolors="none", zorder=3)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.grid(True, alpha=0.3)

    # Colorbar for score
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Score")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_heatmap_activity(
    data: pd.DataFrame | np.ndarray,
    title: str | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Generic activity heatmap.

    Equivalent to ``SecAct.heatmap.plot()`` in R.

    Parameters
    ----------
    data : DataFrame or ndarray
        Matrix of activity values (proteins × samples/cell types).
    title : str, optional
        Plot title.
    colors : list[str], optional
        Colormap colors. Default: green→yellow→orange→red.
    figsize : tuple, optional
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    if colors is None:
        colors = _SECACT_HEATMAP_COLORS

    if isinstance(data, pd.DataFrame):
        mat = data.values
        row_labels = list(reversed(data.index.tolist()))
        col_labels = data.columns.tolist()
        mat = mat[::-1]
    else:
        mat = data[::-1]
        row_labels = [str(i) for i in range(mat.shape[0] - 1, -1, -1)]
        col_labels = [str(j) for j in range(mat.shape[1])]

    if figsize is None:
        figsize = (max(4, mat.shape[1] * 0.6 + 2), max(4, mat.shape[0] * 0.4 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list("secact_hm", colors)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Activity")

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_bar(
    data: pd.Series | dict[str, float],
    title: str | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Diverging bar plot for secreted protein values.

    Equivalent to ``SecAct.bar.plot()`` in R.

    Parameters
    ----------
    data : Series or dict
        Named values (e.g., risk scores, z-scores).
    title : str, optional
        Plot title.
    colors : list[str], optional
        Two colors: [negative_color, positive_color].
    figsize : tuple, optional
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    if colors is None:
        colors = _SECACT_BAR_COLORS

    if isinstance(data, dict):
        data = pd.Series(data)

    # Sort by value
    data = data.sort_values()

    if figsize is None:
        figsize = (6, max(4, len(data) * 0.35 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    bar_colors = [colors[0] if v < 0 else colors[1] for v in data.values]
    ax.barh(
        range(len(data)), data.values, color=bar_colors, edgecolor="white", height=0.88
    )

    # Gene labels inside bars
    space_text = max(abs(data.values)) * 0.015
    for i, (gene, val) in enumerate(data.items()):
        ha = "left" if val < 0 else "right"
        y_offset = space_text if val < 0 else -space_text
        ax.text(y_offset, i, gene, ha=ha, va="center", fontsize=8)

    ax.axhline(y=-0.5, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_lollipop(
    data: pd.Series | dict[str, float],
    title: str | None = None,
    point_color: str | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Lollipop plot for secreted protein values.

    Equivalent to ``SecAct.lollipop.plot()`` in R.

    Parameters
    ----------
    data : Series or dict
        Named values.
    title : str, optional
        Plot title.
    point_color : str, optional
        Point color. Default: "#619CFF".
    figsize : tuple, optional
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    if point_color is None:
        point_color = _SECACT_LOLLIPOP_COLOR

    if isinstance(data, dict):
        data = pd.Series(data)

    data = data.sort_values()

    if figsize is None:
        figsize = (6, max(4, len(data) * 0.35 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Segments from 0 to value
    for i, (_gene, val) in enumerate(data.items()):
        ax.plot([0, val], [i, i], color="grey", linewidth=1)

    ax.scatter(data.values, range(len(data)), color=point_color, s=30, zorder=3)

    # Gene labels
    for i, (gene, val) in enumerate(data.items()):
        ha = "left" if val < 0 else "right"
        offset = 0.1 if val < 0 else -0.1
        ax.text(offset, i, gene, ha=ha, va="center", fontsize=8)

    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def visualize_secact_velocity(
    adata: ad.AnnData,
    gene: str,
    signal_mode: str = "receiving",
    contour_map: bool = False,
    animated: bool = False,
    arrow_color: str = "black",
    figsize: tuple[float, float] = (8, 8),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Signaling velocity plot with arrows overlaid on spatial coordinates.

    Equivalent to ``SecAct.signaling.velocity.spotST()`` in R (plot only).
    Requires velocity data computed by ``secact_signaling_velocity()``.

    Parameters
    ----------
    adata : AnnData
        Must have velocity results for the given gene.
    gene : str
        Gene symbol.
    signal_mode : str
        "receiving" or "sending".
    contour_map : bool
        If True, display as a smoothed contour flow field instead of
        individual arrows. Default: False.
    animated : bool
        If True, return a matplotlib FuncAnimation showing velocity
        arrows growing over time. Default: False.
    arrow_color : str
        Arrow color. Default: "black".
    figsize : tuple
        Figure size.
    save : str, optional
        Path to save. For animated=True, use a .gif path.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure (or FuncAnimation if animated=True)
    """
    spacet = adata.uns.get("spacet", {})
    secact_out = spacet.get("SecAct_output", {})
    vel = secact_out.get("velocity", {}).get(gene)

    if vel is None:
        raise ValueError(
            f"No velocity data for gene '{gene}'. "
            "Run secact_signaling_velocity() first."
        )

    arrow_df = vel["arrows"]
    points_df = vel["points"]

    if animated:
        return _velocity_animated(
            arrow_df,
            points_df,
            gene,
            signal_mode,
            arrow_color,
            figsize,
            save,
            dpi,
        )

    fig, ax = plt.subplots(figsize=figsize)

    if contour_map and len(arrow_df) > 0:
        cf = _velocity_contour(ax, arrow_df, points_df)
        fig.colorbar(cf, ax=ax, shrink=0.6, label="level")
        # Overlay direction arrows on contour, scaled by field intensity
        intensities = _get_arrow_intensities(arrow_df, points_df)
        _draw_velocity_arrows(
            ax, arrow_df, points_df, arrow_color, intensities=intensities
        )
    else:
        cmap = LinearSegmentedColormap.from_list("vel_cmap", _SECACT_VELOCITY_COLORS)
        sc = ax.scatter(
            points_df["x"],
            points_df["y"],
            c=points_df["value"],
            cmap=cmap,
            s=10,
            zorder=1,
        )
        fig.colorbar(sc, ax=ax, shrink=0.6)
        if len(arrow_df) > 0:
            intensities = _get_arrow_intensities(arrow_df, points_df)
            _draw_velocity_arrows(
                ax, arrow_df, points_df, arrow_color, intensities=intensities
            )

    ax.set_title(f"{gene} ({signal_mode})", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def _get_arrow_intensities(
    arrow_df: pd.DataFrame, points_df: pd.DataFrame
) -> np.ndarray:
    """Look up the field intensity at each arrow's start position."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([points_df["x"].values, points_df["y"].values]))
    arrow_pts = np.column_stack(
        [arrow_df["x_start"].values, arrow_df["y_start"].values]
    )
    _, idx = tree.query(arrow_pts)
    return points_df["value"].values[idx]


def _draw_velocity_arrows(
    ax: plt.Axes,
    arrow_df: pd.DataFrame,
    points_df: pd.DataFrame,
    color: str = "black",
    intensities: np.ndarray | None = None,
) -> None:
    """Draw arrowhead-only direction markers at each spot.

    Renders small filled triangles rotated to match velocity direction.

    When *intensities* is provided, each arrow is sized and alpha-scaled
    continuously by field intensity: high-intensity spots get large, fully
    opaque black arrows while low-intensity spots get small,
    semi-transparent black arrows.
    """
    from matplotlib.markers import MarkerStyle

    dx = arrow_df["x_change"].values
    dy = arrow_df["y_change"].values
    vec_len = arrow_df["vec_len"].values
    angles = np.degrees(np.arctan2(dy, dx))

    if intensities is not None:
        # Normalise intensities to [0, 1]
        imin, imax = float(intensities.min()), float(intensities.max())
        if imax > imin:
            norm_int = (intensities - imin) / (imax - imin)
        else:
            norm_int = np.full(len(intensities), 0.5)

        x_arr = arrow_df["x_start"].values
        y_arr = arrow_df["y_start"].values
        for i in range(len(arrow_df)):
            t = norm_int[i]
            base = 55 if vec_len[i] >= 0.1 else 20
            size = base * (0.6 + 1.4 * t)  # 0.6× – 2× base size
            alpha = 0.45 + 0.45 * t  # 0.45 – 0.9

            m = MarkerStyle("^")
            m._transform = m.get_transform().rotate_deg(angles[i] - 90)
            ax.scatter(
                x_arr[i],
                y_arr[i],
                marker=m,
                s=size,
                c="black",
                alpha=alpha,
                zorder=4,
                edgecolors="none",
            )
    else:
        strong = vec_len >= 0.1
        for mask, size in [(strong, 40), (~strong, 12)]:
            if not mask.any():
                continue
            sub_x = arrow_df["x_start"].values[mask]
            sub_y = arrow_df["y_start"].values[mask]
            sub_ang = angles[mask]
            for xi, yi, ang in zip(sub_x, sub_y, sub_ang):
                m = MarkerStyle("^")
                m._transform = m.get_transform().rotate_deg(ang - 90)
                ax.scatter(
                    xi,
                    yi,
                    marker=m,
                    s=size,
                    c=color,
                    alpha=0.7,
                    zorder=4,
                    edgecolors="none",
                )


def _velocity_contour(
    ax: plt.Axes,
    arrow_df: pd.DataFrame,
    points_df: pd.DataFrame,
    n_levels: int = 11,
) -> plt.cm.ScalarMappable:
    """Filled contour of velocity magnitude with spot overlay.

    Matches R's SecAct.signaling.velocity.spotST(contourMap=TRUE):
    green sequential filled contour with black dots for spot positions.

    Returns the contourf mappable for colorbar creation by caller.
    """
    from scipy.interpolate import RBFInterpolator

    # Use ALL spots and their expression x activity product values
    # (matches R's contourMap which interpolates the product field)
    x = points_df["x"].values
    y = points_df["y"].values
    values = points_df["value"].values

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Regular grid for contour interpolation
    n_grid = 80
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    XI, YI = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([XI.ravel(), YI.ravel()])

    # Interpolate expression x activity product onto grid
    pts = np.column_stack([x, y])
    domain_scale = max(x_max - x_min, y_max - y_min)
    rbf = RBFInterpolator(
        pts, values, kernel="thin_plate_spline", smoothing=domain_scale * 0.05
    )
    Z = rbf(grid_pts).reshape(n_grid, n_grid)
    Z = np.clip(Z, 0, None)

    greens = LinearSegmentedColormap.from_list(
        "vel_greens", _SECACT_VELOCITY_CONTOUR_COLORS
    )

    # Filled contour
    levels = np.linspace(0, Z.max(), n_levels)
    cf = ax.contourf(XI, YI, Z, levels=levels, cmap=greens, zorder=1)
    ax.contour(XI, YI, Z, levels=levels, colors="white", linewidths=0.3, zorder=2)

    # Spot positions as black dots
    ax.scatter(
        points_df["x"].values,
        points_df["y"].values,
        s=3,
        c="black",
        alpha=0.5,
        zorder=3,
        edgecolors="none",
    )
    return cf


def _velocity_animated(
    arrow_df: pd.DataFrame,
    points_df: pd.DataFrame,
    gene: str,
    signal_mode: str,
    arrow_color: str,
    figsize: tuple[float, float],
    save: str | None,
    dpi: int,
):
    """Animated velocity plot where all arrows grow simultaneously.

    Every arrow starts at frame 0.  High-intensity spots have larger final
    markers and therefore take more frames to reach full size, making them
    appear to "grow longer".  Low-intensity arrows finish early and stay at
    their (smaller) final size for the remaining frames.

    Arrows are coloured by field intensity (grey → orange-red), matching the
    static intensity-based styling.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.markers import MarkerStyle

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list("vel_cmap", _SECACT_VELOCITY_COLORS)
    ax.scatter(
        points_df["x"],
        points_df["y"],
        c=points_df["value"],
        cmap=cmap,
        s=10,
        zorder=1,
    )

    ax.set_title(f"{gene} ({signal_mode})", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    if len(arrow_df) == 0:
        return fig

    # -- precompute per-arrow properties --
    dx_all = arrow_df["x_change"].values
    dy_all = arrow_df["y_change"].values
    vec_len = arrow_df["vec_len"].values
    angles = np.degrees(np.arctan2(dy_all, dx_all))
    x_all = arrow_df["x_start"].values
    y_all = arrow_df["y_start"].values

    # Intensity-based sizing (same as static plots, black arrows)
    intensities = _get_arrow_intensities(arrow_df, points_df)
    imin, imax = float(intensities.min()), float(intensities.max())
    if imax > imin:
        norm_int = (intensities - imin) / (imax - imin)
    else:
        norm_int = np.full(len(intensities), 0.5)

    n_arrows = len(arrow_df)
    n_frames = 30

    # Final marker sizes (same formula as static)
    base_sizes = np.where(vec_len >= 0.1, 55.0, 20.0)
    final_sizes = base_sizes * (0.6 + 1.4 * norm_int)
    final_alphas = 0.45 + 0.45 * norm_int  # 0.45 – 0.9

    # Frame at which each arrow reaches full size:
    # low intensity → finishes at 30% of frames, high → 100%
    target_frames = (0.3 + 0.7 * norm_int) * n_frames
    target_frames = np.clip(target_frames, 1, n_frames)

    arrow_artists: list = []

    def update(frame):
        for a in arrow_artists:
            a.remove()
        arrow_artists.clear()

        frac = frame + 1  # 1-based frame count
        for i in range(n_arrows):
            progress = min(1.0, frac / target_frames[i])
            cur_size = final_sizes[i] * progress
            cur_alpha = 0.15 + (final_alphas[i] - 0.15) * progress

            m = MarkerStyle("^")
            m._transform = m.get_transform().rotate_deg(angles[i] - 90)
            sc = ax.scatter(
                x_all[i],
                y_all[i],
                marker=m,
                s=cur_size,
                c="black",
                alpha=cur_alpha,
                zorder=2,
                edgecolors="none",
            )
            arrow_artists.append(sc)
        return arrow_artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)

    if save:
        anim.save(save, writer="pillow", dpi=dpi)

    return anim


def visualize_secact_velocity_scst(
    velocity_result: dict,
    *,
    customized_area: list[float] | None = None,
    show_coordinates: bool = True,
    colors: dict[str, str] | None = None,
    point_size: float = 1.0,
    point_alpha: float = 1.0,
    arrow_color: str = "#ff0099",
    arrow_size: float = 0.3,
    arrow_width: float = 1.0,
    legend_position: str = "right",
    legend_size: float = 3.0,
    interactive: bool = False,
    figsize: tuple[float, float] = (10, 8),
    save: str | None = None,
    dpi: int = 300,
):
    """Single-cell resolution signaling velocity plot.

    Equivalent to ``SecAct.signaling.velocity.scST()`` in R.  Draws
    cell-type scatter coloured by annotation with arrows from sender to
    receiver cells overlaid.

    Parameters
    ----------
    velocity_result : dict
        Output of ``secact_signaling_velocity_scst()``.
    customized_area : list, optional
        ``[x_left, x_right, y_bottom, y_top]`` to zoom into a subregion.
    show_coordinates : bool
        If True, show axis ticks / frame. Default: True.
    colors : dict, optional
        ``{cell_type: color}`` mapping.
    point_size, point_alpha : float
        Scatter aesthetics.
    arrow_color : str
        Arrow colour. Default: ``"#ff0099"``.
    arrow_size : float
        Arrow scale (shaft width + head). Default: 0.3.
    arrow_width : float
        Arrow line width in points. Default: 1.0.
    legend_position : str
        "right", "left", or "none".
    legend_size : float
        Legend marker size.
    interactive : bool
        If True, return a plotly Figure (zoomable/pannable HTML).
        If False (default), return a matplotlib Figure.
    figsize, save, dpi : standard plotting params.

    Returns
    -------
    matplotlib Figure or plotly Figure (if interactive=True)
    """
    arrows = velocity_result["arrows"]
    cell_df = velocity_result["cell_types"]

    # Optional zoom
    if customized_area is not None:
        x_left, x_right, y_bottom, y_top = customized_area
        mask = (
            (cell_df["x"] > x_left)
            & (cell_df["x"] < x_right)
            & (cell_df["y"] > y_bottom)
            & (cell_df["y"] < y_top)
        )
        cell_df = cell_df[mask].copy()
        if len(arrows) > 0:
            a_mask = (
                (arrows["x_start"] > x_left)
                & (arrows["x_start"] < x_right)
                & (arrows["x_end"] > x_left)
                & (arrows["x_end"] < x_right)
                & (arrows["y_start"] > y_bottom)
                & (arrows["y_start"] < y_top)
                & (arrows["y_end"] > y_bottom)
                & (arrows["y_end"] < y_top)
            )
            arrows = arrows[a_mask].copy()

    if interactive:
        return _velocity_scst_plotly(
            cell_df, arrows, colors, point_size, point_alpha,
            arrow_color, arrow_size, arrow_width, figsize, save,
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Draw cells coloured by type
    unique_types = cell_df["cell_type"].unique()
    for ct in unique_types:
        sub = cell_df[cell_df["cell_type"] == ct]
        c = colors.get(ct, "#cccccc") if colors else None
        ax.scatter(
            sub["x"],
            sub["y"],
            s=point_size,
            c=c,
            alpha=point_alpha,
            label=ct,
            edgecolors="none",
            zorder=1,
        )

    # Draw arrows — matches R geom_segment + arrow()
    # arrow_size: arrow scale (shaft width + head)
    # arrow_width: shaft line weight (points)
    if len(arrows) > 0:
        x0 = arrows["x_start"].values
        y0 = arrows["y_start"].values
        x1 = arrows["x_end"].values
        y1 = arrows["y_end"].values
        dx = x1 - x0
        dy = y1 - y0

        ax.quiver(
            x0, y0, dx, dy,
            color=arrow_color,
            angles="xy", scale_units="xy", scale=1,
            width=arrow_size * 0.01,
            headwidth=4,
            headlength=5,
            headaxislength=4.5,
            linewidth=arrow_width,
            alpha=0.7,
            zorder=2,
        )

    # Don't force equal aspect for zoomed subregions (avoids stretched figures)
    if customized_area is None:
        ax.set_aspect("equal")

    if not show_coordinates:
        ax.axis("off")
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend with fixed marker size (R uses guide_legend override.aes size)
    if legend_position != "none":
        loc = "center left" if legend_position == "right" else "center right"
        bbox = (1.02, 0.5) if legend_position == "right" else (-0.02, 0.5)
        legend = ax.legend(
            loc=loc,
            bbox_to_anchor=bbox,
            frameon=False,
            fontsize=8,
        )
        target_size = max(20, point_size * legend_size)
        for h in legend.legend_handles:
            h.set_sizes([target_size])
    else:
        ax.legend().remove()

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def _velocity_scst_plotly(
    cell_df: pd.DataFrame,
    arrows: pd.DataFrame,
    colors: dict[str, str] | None,
    point_size: float,
    point_alpha: float,
    arrow_color: str,
    arrow_size: float,
    arrow_width: float,
    figsize: tuple[float, float],
    save: str | None,
):
    """Interactive plotly version of scST velocity plot."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Cell type scatter traces
    for ct in cell_df["cell_type"].unique():
        sub = cell_df[cell_df["cell_type"] == ct]
        c = colors.get(ct, "#cccccc") if colors else None
        fig.add_trace(
            go.Scattergl(
                x=sub["x"].values,
                y=sub["y"].values,
                mode="markers",
                marker=dict(
                    size=max(2, point_size),
                    color=c,
                    opacity=point_alpha,
                ),
                name=ct,
                hoverinfo="text",
                text=[f"{ct}" for _ in range(len(sub))],
            )
        )

    # Arrows drawn in data coordinates so they scale with zoom
    if len(arrows) > 0:
        x0 = arrows["x_start"].values
        y0 = arrows["y_start"].values
        x1_raw = arrows["x_end"].values
        y1_raw = arrows["y_end"].values
        dx_raw = x1_raw - x0
        dy_raw = y1_raw - y0

        # Scale arrows by arrow_size so they're visible at full zoom
        # arrow_size=1.0 → arrows are ~1% of data range
        x_range = cell_df["x"].max() - cell_df["x"].min()
        target_len = x_range * arrow_size * 0.01
        mag = np.sqrt(dx_raw**2 + dy_raw**2)
        mag[mag == 0] = 1
        dx = dx_raw / mag * target_len
        dy = dy_raw / mag * target_len
        x1 = x0 + dx
        y1 = y0 + dy

        # Shaft: line segments (None-separated)
        x_lines, y_lines = [], []
        for i in range(len(x0)):
            x_lines.extend([x0[i], x1[i], None])
            y_lines.extend([y0[i], y1[i], None])

        fig.add_trace(
            go.Scattergl(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=arrow_color, width=max(1, arrow_width)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Arrowheads: triangles drawn as filled polygons in data coords
        # Each head is 3 vertices forming a triangle at the endpoint
        mag = np.sqrt(dx**2 + dy**2)
        mag[mag == 0] = 1
        ux, uy = dx / mag, dy / mag  # unit direction
        px, py = -uy, ux  # perpendicular

        head_len = mag * 0.25  # head = 25% of arrow length
        head_w = head_len * 0.4

        # Triangle vertices: tip, left base, right base
        tip_x, tip_y = x1, y1
        base_x = x1 - ux * head_len
        base_y = y1 - uy * head_len
        left_x = base_x + px * head_w
        left_y = base_y + py * head_w
        right_x = base_x - px * head_w
        right_y = base_y - py * head_w

        x_heads, y_heads = [], []
        for i in range(len(tip_x)):
            x_heads.extend([tip_x[i], left_x[i], right_x[i], tip_x[i], None])
            y_heads.extend([tip_y[i], left_y[i], right_y[i], tip_y[i], None])

        fig.add_trace(
            go.Scatter(
                x=x_heads,
                y=y_heads,
                mode="lines",
                fill="toself",
                fillcolor=arrow_color,
                line=dict(color=arrow_color, width=0.5),
                opacity=0.7,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    if save:
        if save.endswith(".html"):
            fig.write_html(save)
        else:
            fig.write_image(save)

    return fig


def visualize_secact_survival(
    survival_result: dict,
    x_title: str = "Time",
    figsize: tuple[float, float] = (8, 6),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Kaplan-Meier survival plot for secreted protein stratification.

    Equivalent to ``SecAct.survival.plot()`` in R.

    Parameters
    ----------
    survival_result : dict
        Output of ``secact_survival_data()``, with keys 'high', 'low',
        'logrank_p', 'protein'.
    x_title : str
        X-axis label. Default: "Time".
    figsize : tuple
        Figure size.
    save : str, optional
        Path to save.
    dpi : int
        Resolution.

    Returns
    -------
    matplotlib Figure
    """
    from lifelines import KaplanMeierFitter

    high = survival_result["high"]
    low = survival_result["low"]
    p_val = survival_result["logrank_p"]
    protein = survival_result["protein"]

    fig, ax = plt.subplots(figsize=figsize)

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(low["Time"], low["Event"], label=f"Low (n={len(low)})")
    kmf_low.plot_survival_function(ax=ax, color="blue")

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(high["Time"], high["Event"], label=f"High (n={len(high)})")
    kmf_high.plot_survival_function(ax=ax, color="red")

    ax.set_xlabel(x_title)
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{protein} (p = {p_val:.2e})")
    ax.legend(loc="best")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig
