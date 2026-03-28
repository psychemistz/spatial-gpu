"""SecAct visualization functions.

Heatmaps, chord diagrams, Sankey plots, dot plots, bar/lollipop charts,
signaling velocity overlays, and Kaplan-Meier survival plots for SecAct
secreted-protein analysis results.

Reference: Ru et al., Nature Communications 14, 568 (2023)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from spatialgpu.deconvolution._keys import (
    COL_COUNT,
    COL_RECEIVER,
    COL_SECRETED_PROTEIN,
    COL_SENDER,
    KEY_SECACT,
    UNS_SPACET,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

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


def _get_secact_ccc(adata: ad.AnnData) -> pd.DataFrame:
    """Retrieve SecretedProteinCCC results from adata.uns, raising if absent."""
    spacet = adata.uns.get(UNS_SPACET, {})
    secact_out = spacet.get(KEY_SECACT, {})
    ccc = secact_out.get("SecretedProteinCCC")
    if ccc is None or len(ccc) == 0:
        raise ValueError("No CCC results. Run secact_spatial_ccc() first.")
    return ccc


def _build_ccc_count_matrix(ccc: pd.DataFrame) -> pd.DataFrame:
    """Build sender x receiver count matrix from CCC results."""
    all_types = sorted(set(ccc[COL_SENDER].tolist() + ccc[COL_RECEIVER].tolist()))
    mat = pd.DataFrame(0, index=all_types, columns=all_types, dtype=float)
    for _, row in ccc.iterrows():
        s, r = row[COL_SENDER], row[COL_RECEIVER]
        mat.loc[s, r] += 1
    return mat


def _default_cell_type_colors(labels: list[str]) -> dict[str, tuple]:
    """Generate default tab20 color palette for cell type labels."""
    tab20 = plt.cm.tab20(np.linspace(0, 1, max(20, len(labels))))
    return {lb: tab20[i] for i, lb in enumerate(labels)}


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

    ccc = _get_secact_ccc(adata)
    mat = _build_ccc_count_matrix(ccc)
    all_types = list(mat.index)

    # Set diagonal to NaN
    for ct in all_types:
        if ct in mat.index and ct in mat.columns:
            mat.loc[ct, ct] = np.nan

    if row_sorted:
        mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    if column_sorted:
        mat = mat[mat.sum(axis=0).sort_values(ascending=False).index]

    if colors_cell_type is None:
        colors_cell_type = _default_cell_type_colors(all_types)

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
    ccc = _get_secact_ccc(adata)
    mat = _build_ccc_count_matrix(ccc)
    all_types = list(mat.index)

    for ct in all_types:
        if ct in mat.index and ct in mat.columns:
            mat.loc[ct, ct] = 0

    if colors_cell_type is None:
        colors_cell_type = _default_cell_type_colors(all_types)

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

    sector_colors = {ct: colors_cell_type.get(ct, "gray") for ct in all_types}

    circos = Circos.initialize_from_matrix(
        mat,
        space=3,
        cmap=sector_colors,
        label_kws={"fontsize": 8},
        link_kws={"direction": 1, "ec": "white", "lw": 0.3},
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
    ccc = _get_secact_ccc(adata)

    # Filter
    mask = (
        ccc[COL_SENDER].isin(sender)
        & ccc[COL_SECRETED_PROTEIN].isin(secreted_protein)
        & ccc[COL_RECEIVER].isin(receiver)
    )
    ccc_sub = ccc[mask].copy()

    if len(ccc_sub) == 0:
        raise ValueError(
            "No CCC entries match the given sender/protein/receiver filters."
        )

    # All unique labels
    all_labels = sorted(
        set(
            ccc_sub[COL_SENDER].tolist()
            + ccc_sub[COL_SECRETED_PROTEIN].tolist()
            + ccc_sub[COL_RECEIVER].tolist()
        )
    )

    if colors_cell_type is None:
        colors_cell_type = _default_cell_type_colors(all_labels)

    fig, ax = plt.subplots(figsize=figsize)

    # Three columns: sender (x=0), protein (x=1), receiver (x=2)
    # Count flows
    s_to_p = (
        ccc_sub.groupby([COL_SENDER, COL_SECRETED_PROTEIN])
        .size()
        .reset_index(name=COL_COUNT)
    )
    p_to_r = (
        ccc_sub.groupby([COL_SECRETED_PROTEIN, COL_RECEIVER])
        .size()
        .reset_index(name=COL_COUNT)
    )

    # Compute node positions
    s_counts = ccc_sub[COL_SENDER].value_counts().sort_values(ascending=False)
    p_counts_l = (
        s_to_p.groupby(COL_SECRETED_PROTEIN)[COL_COUNT]
        .sum()
        .sort_values(ascending=False)
    )
    r_counts = ccc_sub[COL_RECEIVER].value_counts().sort_values(ascending=False)

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
        s, p, cnt = row[COL_SENDER], row[COL_SECRETED_PROTEIN], row[COL_COUNT]
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
        p, r, cnt = row[COL_SECRETED_PROTEIN], row[COL_RECEIVER], row[COL_COUNT]
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
    ccc = _get_secact_ccc(adata)

    mask = (
        ccc[COL_SENDER].isin(sender)
        & ccc[COL_SECRETED_PROTEIN].isin(secreted_protein)
        & ccc[COL_RECEIVER].isin(receiver)
    )
    ccc_sub = ccc[mask].copy()

    if len(ccc_sub) == 0:
        raise ValueError("No CCC entries match the given filters.")

    # Create s2r label
    ccc_sub["s2r"] = ccc_sub[COL_SENDER] + "->" + ccc_sub[COL_RECEIVER]

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
        sp = row[COL_SECRETED_PROTEIN]
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
    spacet = adata.uns.get(UNS_SPACET, {})
    secact_out = spacet.get(KEY_SECACT, {})
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


def _velocity_scst_filter_zoom(
    velocity_result: dict,
    customized_area: list[float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract cell/arrow DataFrames, optionally filtering to a zoom region."""
    arrows = velocity_result["arrows"]
    cell_df = velocity_result["cell_types"]

    if customized_area is None:
        return cell_df, arrows

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
    return cell_df, arrows


def _velocity_scst_draw_cells(
    ax: plt.Axes,
    cell_df: pd.DataFrame,
    colors: dict[str, str] | None,
    point_size: float,
    point_alpha: float,
) -> None:
    """Draw cells coloured by cell type onto *ax*."""
    for ct in cell_df["cell_type"].unique():
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


def _velocity_scst_draw_arrows(
    ax: plt.Axes,
    arrows: pd.DataFrame,
    arrow_color: str,
    arrow_size: float,
    arrow_width: float,
) -> None:
    """Overlay quiver arrows from sender to receiver cells onto *ax*."""
    if len(arrows) == 0:
        return
    x0 = arrows["x_start"].values
    y0 = arrows["y_start"].values
    x1 = arrows["x_end"].values
    y1 = arrows["y_end"].values
    dx = x1 - x0
    dy = y1 - y0

    ax.quiver(
        x0,
        y0,
        dx,
        dy,
        color=arrow_color,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=arrow_size * 0.01,
        headwidth=4,
        headlength=5,
        headaxislength=4.5,
        linewidth=arrow_width,
        alpha=0.7,
        zorder=2,
    )


def _velocity_scst_style_axes(
    ax: plt.Axes,
    *,
    is_zoomed: bool,
    show_coordinates: bool,
    legend_position: str,
    legend_size: float,
    point_size: float,
) -> None:
    """Apply axis styling and legend to the scST velocity plot."""
    # Don't force equal aspect for zoomed subregions (avoids stretched figures)
    if not is_zoomed:
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
    cell_df, arrows = _velocity_scst_filter_zoom(velocity_result, customized_area)

    if interactive:
        return _velocity_scst_plotly(
            cell_df,
            arrows,
            colors,
            point_size,
            point_alpha,
            arrow_color,
            arrow_size,
            arrow_width,
            figsize,
            save,
        )

    fig, ax = plt.subplots(figsize=figsize)

    _velocity_scst_draw_cells(ax, cell_df, colors, point_size, point_alpha)
    _velocity_scst_draw_arrows(ax, arrows, arrow_color, arrow_size, arrow_width)
    _velocity_scst_style_axes(
        ax,
        is_zoomed=customized_area is not None,
        show_coordinates=show_coordinates,
        legend_position=legend_position,
        legend_size=legend_size,
        point_size=point_size,
    )

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
                marker={
                    "size": max(2, point_size),
                    "color": c,
                    "opacity": point_alpha,
                },
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
                line={"color": arrow_color, "width": max(1, arrow_width)},
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
                line={"color": arrow_color, "width": 0.5},
                opacity=0.7,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        xaxis={"scaleanchor": "y", "scaleratio": 1, "showgrid": False},
        yaxis={"showgrid": False},
        plot_bgcolor="white",
        legend={"itemsizing": "constant"},
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
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
