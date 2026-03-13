"""SpaCET spatial feature visualization.

Matplotlib-based spatial scatter plots for deconvolution results,
gene expression, LR scores, interface, and gene set scores.
Equivalent to SpaCET.visualize.spatialFeature() in R.

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# Default color maps matching R SpaCET
_COLORMAPS = {
    "QualityControl": ["lightblue", "blue", "darkblue"],
    "GeneExpression": ["#4d9221", "yellow", "#c51b7d"],
    "CellFraction": ["blue", "yellow", "red"],
    "LRNetworkScore": ["blue", "cyan", "yellow"],
    "GeneSetScore": ["#91bfdb", "#fee090", "#d73027"],
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
        'LRNetworkScore', 'Interface', 'GeneSetScore'.
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
        "LRNetworkScore",
        "Interface",
        "GeneSetScore",
    )
    if spatial_type not in valid_types:
        raise ValueError(
            f"spatial_type must be one of {valid_types}, got '{spatial_type}'."
        )

    # Get coordinates from spot IDs (row x col format)
    coords = _get_spot_coordinates(adata)

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

    # Single feature on provided axes
    if n_features == 1 and ax is not None:
        _plot_single(
            ax,
            coords,
            values_dict[features[0]],
            features[0],
            legend_name,
            is_discrete,
            colors,
            point_size,
            point_alpha,
            same_scale_fraction and spatial_type == "CellFraction",
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
            legend_name,
            is_discrete,
            colors,
            point_size,
            point_alpha,
            same_scale_fraction and spatial_type == "CellFraction",
        )

    # Hide unused axes
    for idx in range(n_features, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig


def _get_spot_coordinates(adata: ad.AnnData) -> np.ndarray:
    """Extract (x, y) coordinates from spot IDs or obsm['spatial']."""
    if "spatial" in adata.obsm:
        return adata.obsm["spatial"]

    # Parse from "row x col" spot IDs
    coords = []
    for sid in adata.obs_names:
        parts = sid.split("x")
        coords.append((float(parts[1]), float(parts[0])))
    return np.array(coords)


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
                row_sums = np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else X.sum(axis=1)
                vals = vals / row_sums * 1e5
                vals = np.log2(vals + 1)
            elif scale_type_gene == "LogTPM":
                row_sums = np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else X.sum(axis=1)
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

    elif spatial_type == "CellFraction":
        deconv = spacet.get("deconvolution", {})
        prop_mat = deconv.get("propMat")
        if prop_mat is None:
            raise ValueError("Run deconvolution() first.")

        if spatial_features is None or "All" in spatial_features:
            spatial_features = list(prop_mat.index)

        mat = {}
        for ct in spatial_features:
            if ct in prop_mat.index:
                mat[ct] = prop_mat.loc[ct].values.astype(float)
        return mat, "Fraction", False

    elif spatial_type == "LRNetworkScore":
        cci = spacet.get("CCI", {})
        lr_score = cci.get("LRNetworkScore")
        if lr_score is None:
            raise ValueError("Run cci_lr_network_score() first.")

        if spatial_features is None:
            spatial_features = ["Network_Score", "Network_Score_pv"]

        lr_index = cci.get("LRNetworkScore_index", [
            "Raw_expr", "Network_Score", "Network_Score_pv",
        ])
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

    elif spatial_type == "Interface":
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

    elif spatial_type == "GeneSetScore":
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

    return {}, "", False


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
) -> None:
    """Plot a single spatial feature on the given axes."""
    x = coords[:, 0]
    y = coords[:, 1]

    if is_discrete:
        categories = sorted(set(str(v) for v in values if pd.notna(v)))

        if colors is None:
            color_map = _DISCRETE_COLORS.get("Interface", {})
        elif isinstance(colors, list):
            color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
        else:
            color_map = {}

        for cat in categories:
            mask = np.array([str(v) == cat for v in values])
            c = color_map.get(cat, f"C{categories.index(cat)}")
            ax.scatter(x[mask], y[mask], c=c, s=point_size, alpha=point_alpha, label=cat)
        ax.legend(title=legend_name, fontsize=7, title_fontsize=8)
    else:
        vals = np.asarray(values, dtype=float)
        if colors:
            cmap = LinearSegmentedColormap.from_list("custom", colors)
        else:
            cmap = "viridis"

        vmin = 0.0 if fixed_scale else None
        vmax = 1.0 if fixed_scale else None

        sc = ax.scatter(
            x, y, c=vals, cmap=cmap, s=point_size, alpha=point_alpha,
            vmin=vmin, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label=legend_name, shrink=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
