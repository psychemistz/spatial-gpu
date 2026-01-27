"""
Spatial visualization functions.

Provides publication-quality plots for spatial omics data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def spatial_scatter(
    adata: ad.AnnData,
    color: Optional[str] = None,
    spatial_key: str = "spatial",
    spot_size: float = 1.0,
    alpha: float = 1.0,
    cmap: str = "viridis",
    palette: Optional[str | Sequence[str]] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 8),
    save: Optional[str] = None,
    **kwargs,
) -> Axes:
    """
    Create scatter plot of spatial coordinates.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates.
    color
        Key in `adata.obs` or `adata.var_names` to color by.
    spatial_key
        Key in `adata.obsm` containing spatial coordinates.
    spot_size
        Size of spots in plot.
    alpha
        Transparency of spots.
    cmap
        Colormap for continuous values.
    palette
        Color palette for categorical values.
    show_legend
        Whether to show legend.
    title
        Plot title.
    ax
        Matplotlib axes to plot on.
    figsize
        Figure size if creating new figure.
    save
        Path to save figure.
    **kwargs
        Additional matplotlib scatter kwargs.

    Returns
    -------
    Axes
        Matplotlib axes with the plot.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> sp.viz.spatial_scatter(adata, color="cell_type")
    >>> sp.viz.spatial_scatter(adata, color="GAPDH")
    """
    import matplotlib.pyplot as plt
    from spatialgpu.graph.utils import get_spatial_coords

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    coords = get_spatial_coords(adata, spatial_key=spatial_key)

    # Determine color values
    if color is None:
        c = None
    elif color in adata.obs.columns:
        c = adata.obs[color]
        if hasattr(c, "cat"):
            # Categorical
            categories = c.cat.categories
            c_numeric = c.cat.codes.values

            if palette is None:
                palette = plt.cm.tab20.colors[:len(categories)]

            colors = [palette[i % len(palette)] for i in c_numeric]

            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=colors,
                s=spot_size,
                alpha=alpha,
                **kwargs,
            )

            if show_legend:
                handles = [
                    plt.scatter([], [], c=[palette[i % len(palette)]], label=cat)
                    for i, cat in enumerate(categories)
                ]
                ax.legend(handles=handles, title=color, loc="best")

        else:
            # Continuous
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=c,
                s=spot_size,
                alpha=alpha,
                cmap=cmap,
                **kwargs,
            )
            if show_legend:
                plt.colorbar(scatter, ax=ax, label=color)

    elif color in adata.var_names:
        # Gene expression
        gene_idx = list(adata.var_names).index(color)
        if hasattr(adata.X, "toarray"):
            c = adata.X[:, gene_idx].toarray().flatten()
        else:
            c = adata.X[:, gene_idx].flatten()

        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=c,
            s=spot_size,
            alpha=alpha,
            cmap=cmap,
            **kwargs,
        )
        if show_legend:
            plt.colorbar(scatter, ax=ax, label=color)

    else:
        raise ValueError(f"'{color}' not found in adata.obs or adata.var_names")

    if color is None:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=spot_size,
            alpha=alpha,
            **kwargs,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if title:
        ax.set_title(title)

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def spatial_heatmap(
    adata: ad.AnnData,
    genes: Sequence[str],
    spatial_key: str = "spatial",
    n_cols: int = 4,
    spot_size: float = 1.0,
    cmap: str = "viridis",
    figsize_per_gene: tuple[float, float] = (4, 4),
    save: Optional[str] = None,
    **kwargs,
) -> Figure:
    """
    Create heatmap grid for multiple genes.

    Parameters
    ----------
    adata
        Annotated data object.
    genes
        List of genes to plot.
    spatial_key
        Key in `adata.obsm` for spatial coordinates.
    n_cols
        Number of columns in grid.
    spot_size
        Size of spots.
    cmap
        Colormap.
    figsize_per_gene
        Figure size per gene subplot.
    save
        Path to save figure.
    **kwargs
        Additional scatter kwargs.

    Returns
    -------
    Figure
        Matplotlib figure.

    Examples
    --------
    >>> sp.viz.spatial_heatmap(adata, genes=["GAPDH", "ACTB", "CD3D"])
    """
    import matplotlib.pyplot as plt

    n_genes = len(genes)
    n_rows = (n_genes + n_cols - 1) // n_cols

    figsize = (figsize_per_gene[0] * n_cols, figsize_per_gene[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_genes == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, gene in enumerate(genes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        spatial_scatter(
            adata, color=gene,
            spatial_key=spatial_key,
            spot_size=spot_size,
            cmap=cmap,
            ax=ax,
            title=gene,
            show_legend=True,
            **kwargs,
        )

    # Hide empty subplots
    for idx in range(n_genes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return fig


def nhood_enrichment_plot(
    adata: ad.AnnData,
    cluster_key: str,
    mode: Literal["zscore", "count"] = "zscore",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annot: bool = True,
    save: Optional[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot neighborhood enrichment heatmap.

    Parameters
    ----------
    adata
        Annotated data object with nhood enrichment results.
    cluster_key
        Key for cluster labels.
    mode
        Whether to plot z-scores or raw counts.
    ax
        Matplotlib axes.
    figsize
        Figure size.
    cmap
        Colormap.
    vmin, vmax
        Color scale limits.
    annot
        Show values in cells.
    save
        Path to save figure.
    **kwargs
        Additional seaborn heatmap kwargs.

    Returns
    -------
    Axes
        Matplotlib axes.

    Examples
    --------
    >>> sp.graph.nhood_enrichment(adata, cluster_key="cell_type")
    >>> sp.viz.nhood_enrichment_plot(adata, cluster_key="cell_type")
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    key = f"{cluster_key}_nhood_enrichment"
    if key not in adata.uns:
        raise ValueError(
            f"Neighborhood enrichment not found. Run "
            f"sp.graph.nhood_enrichment(adata, cluster_key='{cluster_key}') first."
        )

    data = adata.uns[key]
    matrix = data["zscore"] if mode == "zscore" else data["count"]
    categories = data["categories"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Set symmetric color scale for z-scores
    if mode == "zscore" and vmin is None and vmax is None:
        max_abs = np.abs(matrix).max()
        vmin, vmax = -max_abs, max_abs

    sns.heatmap(
        matrix,
        xticklabels=categories,
        yticklabels=categories,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f" if mode == "zscore" else ".0f",
        ax=ax,
        **kwargs,
    )

    ax.set_xlabel("Neighbor cluster")
    ax.set_ylabel("Source cluster")
    ax.set_title(f"Neighborhood enrichment ({mode})")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def co_occurrence_plot(
    adata: ad.AnnData,
    cluster_key: str,
    clusters: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10, 6),
    save: Optional[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot co-occurrence curves.

    Parameters
    ----------
    adata
        Annotated data object with co-occurrence results.
    cluster_key
        Key for cluster labels.
    clusters
        Specific cluster pairs to plot.
    ax
        Matplotlib axes.
    figsize
        Figure size.
    save
        Path to save figure.
    **kwargs
        Additional plot kwargs.

    Returns
    -------
    Axes
        Matplotlib axes.

    Examples
    --------
    >>> sp.graph.co_occurrence(adata, cluster_key="cell_type")
    >>> sp.viz.co_occurrence_plot(adata, cluster_key="cell_type")
    """
    import matplotlib.pyplot as plt

    key = f"{cluster_key}_co_occurrence"
    if key not in adata.uns:
        raise ValueError(
            f"Co-occurrence not found. Run "
            f"sp.graph.co_occurrence(adata, cluster_key='{cluster_key}') first."
        )

    data = adata.uns[key]
    occurrence = data["occurrence"]
    intervals = data["interval"]
    categories = data["categories"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(categories)

    for i in range(n_clusters):
        for j in range(n_clusters):
            if clusters is not None:
                if (categories[i], categories[j]) not in clusters:
                    continue

            ax.plot(
                intervals,
                occurrence[i, j, :],
                label=f"{categories[i]} - {categories[j]}",
                **kwargs,
            )

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Co-occurrence ratio")
    ax.set_title("Spatial co-occurrence")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def interaction_matrix_plot(
    adata: ad.AnnData,
    cluster_key: str,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    annot: bool = True,
    save: Optional[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot cell-cell interaction matrix.

    Parameters
    ----------
    adata
        Annotated data object with interaction matrix results.
    cluster_key
        Key for cluster labels.
    ax
        Matplotlib axes.
    figsize
        Figure size.
    cmap
        Colormap.
    annot
        Show values in cells.
    save
        Path to save figure.
    **kwargs
        Additional seaborn heatmap kwargs.

    Returns
    -------
    Axes
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    key = f"{cluster_key}_interaction_matrix"
    if key not in adata.uns:
        raise ValueError(
            f"Interaction matrix not found. Run "
            f"sp.graph.interaction_matrix(adata, cluster_key='{cluster_key}') first."
        )

    data = adata.uns[key]
    matrix = data["interaction"]
    categories = data["categories"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=categories,
        yticklabels=categories,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        ax=ax,
        **kwargs,
    )

    ax.set_xlabel("Target cluster")
    ax.set_ylabel("Source cluster")
    ax.set_title("Cell-cell interaction matrix")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def ripley_plot(
    adata: ad.AnnData,
    cluster: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8, 6),
    save: Optional[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot Ripley's statistics.

    Parameters
    ----------
    adata
        Annotated data object with Ripley results.
    cluster
        Specific cluster to plot (if computed per-cluster).
    ax
        Matplotlib axes.
    figsize
        Figure size.
    save
        Path to save figure.
    **kwargs
        Additional plot kwargs.

    Returns
    -------
    Axes
        Matplotlib axes.

    Examples
    --------
    >>> sp.graph.ripley(adata, mode="L")
    >>> sp.viz.ripley_plot(adata)
    """
    import matplotlib.pyplot as plt

    if "ripley" not in adata.uns:
        raise ValueError(
            "Ripley statistics not found. Run sp.graph.ripley(adata) first."
        )

    data = adata.uns["ripley"]
    radii = data["radii"]
    mode = data["mode"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    stats = data["stats"]
    if isinstance(stats, dict) and cluster is not None:
        stats = stats[cluster]
    elif isinstance(stats, dict):
        # Plot first cluster if not specified
        cluster = list(stats.keys())[0]
        stats = stats[cluster]

    # Plot observed
    ax.plot(radii, stats["observed"], "b-", label=f"Observed {mode}", **kwargs)

    # Plot simulation envelope
    ax.fill_between(
        radii,
        stats["simulated_lo"],
        stats["simulated_hi"],
        alpha=0.3,
        color="gray",
        label="95% CI (CSR)",
    )

    ax.plot(radii, stats["simulated_mean"], "k--", alpha=0.5, label="Expected (CSR)")

    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel("Distance (r)")
    ax.set_ylabel(f"{mode}(r) - r" if mode == "L" else f"{mode}(r)")
    ax.set_title(f"Ripley's {mode} function" + (f" ({cluster})" if cluster else ""))
    ax.legend()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax
