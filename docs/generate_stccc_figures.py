"""Generate all figures for stCCC tutorial (Tutorial 7).

Run from project root: python docs/generate_stccc_figures.py

Generates:
  - cosmx_qc.png           (Section 1: QC metrics)
  - cosmx_celltypes.png     (Section 1: Cell type spatial distribution)
  - cosmx_heatmap.png       (Section 4: CCC heatmap)
  - cosmx_circle.png        (Section 4: CCC circle plot)
  - cosmx_dotplot.png       (Section 4: CCC dot plot)
  - cosmx_velocity.png      (Section 5: Full-tissue signaling velocity)
  - cosmx_velocity_cut.png  (Section 5: Zoomed signaling velocity)
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import KDTree

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CELL_COLORS = {
    "B": "#C88888",
    "Erythrocyte": "#fe666d",
    "T.alpha.beta": "#B95FBB",
    "T.gamma.delta": "#3288bd",
    "NK": "#bb8761",
    "Hepatocyte": "#63636d",
    "Cholangiocyte": "#de77ae",
    "Endothelial": "#D4D915",
    "Fibroblast": "#66c2a5",
    "Macrophage": "#ff9a36",
    "Tumor_core": "#A4DFF2",
    "Tumor_boundary": "blue",
    "Other": "#cccccc",
}


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_scst_velocity(
    adata,
    sender,
    secreted_protein,
    receiver,
    cell_type_col,
    colors,
    point_size=0.1,
    arrow_color="#ff0099",
    arrow_size=0.2,
    show_coordinates=True,
    customized_area=None,
    arrow_width=None,
    figsize=(12, 10),
):
    """Plot single-cell resolution signaling velocity.

    Mimics R SecAct.signaling.velocity.scST(): shows all cells colored by
    cell type, then overlays arrows from sender cells (high expression) to
    nearby receiver cells (high activity) for a given secreted protein.
    """
    secact_out = adata.uns.get("spacet", {}).get("SecAct_output", {})
    act_mat = secact_out.get("SecretedProteinActivity", {}).get("zscore")
    if act_mat is None:
        raise ValueError("No SecAct activity results. Run secact_inference() first.")

    # Coordinates
    coords_x = adata.obs["coordinate_x_um"].values.astype(float)
    coords_y = adata.obs["coordinate_y_um"].values.astype(float)
    cell_types = adata.obs[cell_type_col].values.astype(str)
    cell_names = adata.obs_names.tolist()

    # Get expression for the secreted protein
    from scipy import sparse
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    gene_names = adata.var_names.tolist()
    if secreted_protein in gene_names:
        gene_idx = gene_names.index(secreted_protein)
        expr_vals = X[:, gene_idx].astype(float)
        # TPM normalize
        total_counts = X.sum(axis=1).astype(float)
        total_counts[total_counts == 0] = 1
        expr_vals = expr_vals / total_counts * 1000
    else:
        expr_vals = np.zeros(len(cell_names))

    # Get activity for the secreted protein
    if secreted_protein in act_mat.index:
        act_series = act_mat.loc[secreted_protein]
        act_vals = np.array([
            max(0, act_series.get(name, 0)) for name in cell_names
        ])
    else:
        act_vals = np.zeros(len(cell_names))

    # Area subset
    if customized_area is not None:
        x_min, x_max, y_min, y_max = customized_area
        mask = (
            (coords_x >= x_min) & (coords_x <= x_max)
            & (coords_y >= y_min) & (coords_y <= y_max)
        )
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(cell_names))

    cx = coords_x[idx]
    cy = coords_y[idx]
    ct = cell_types[idx]
    ex = expr_vals[idx]
    ac = act_vals[idx]

    # Identify sender and receiver cells
    sender_mask = ct == sender
    receiver_mask = ct == receiver

    # Build KDTree for finding nearby sender-receiver pairs
    coords_sub = np.column_stack([cx, cy])
    sender_idx = np.where(sender_mask)[0]
    receiver_idx = np.where(receiver_mask)[0]

    # Filter sender cells: those with above-median expression
    sender_expr = ex[sender_idx]
    expr_thresh = np.percentile(sender_expr[sender_expr > 0], 50) if (sender_expr > 0).any() else 0

    # Filter receiver cells: those with above-median activity
    receiver_act = ac[receiver_idx]
    act_thresh = np.percentile(receiver_act[receiver_act > 0], 50) if (receiver_act > 0).any() else 0

    active_senders = sender_idx[sender_expr > expr_thresh]
    active_receivers = receiver_idx[receiver_act > act_thresh]

    # Find neighbor pairs within radius
    radius = 20.0 if customized_area is not None else 50.0
    if len(active_senders) > 0 and len(active_receivers) > 0:
        sender_coords = coords_sub[active_senders]
        receiver_coords = coords_sub[active_receivers]
        tree = KDTree(receiver_coords)

        arrows_x, arrows_y, arrows_dx, arrows_dy = [], [], [], []
        for i, si in enumerate(active_senders):
            nearby = tree.query_ball_point(coords_sub[si], r=radius)
            if not nearby:
                continue
            # Average direction to nearby receivers, weighted by activity
            for ni in nearby:
                ri = active_receivers[ni]
                dx = cx[ri] - cx[si]
                dy = cy[ri] - cy[si]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 1e-6:
                    continue
                weight = ex[si] * ac[ri]
                arrows_x.append(cx[si])
                arrows_y.append(cy[si])
                arrows_dx.append(dx * arrow_size)
                arrows_dy.append(dy * arrow_size)
    else:
        arrows_x, arrows_y, arrows_dx, arrows_dy = [], [], [], []

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Draw all cells colored by cell type
    unique_types = sorted(set(ct))
    # Draw "Other" types first, then important ones on top
    draw_order = [t for t in unique_types if t not in [sender, receiver]] + [receiver, sender]
    for cell_t in draw_order:
        if cell_t not in unique_types:
            continue
        tmask = ct == cell_t
        color = colors.get(cell_t, "#cccccc")
        ax.scatter(
            cx[tmask], cy[tmask],
            s=point_size, c=color, alpha=0.7 if cell_t in [sender, receiver] else 0.3,
            label=cell_t, rasterized=True,
        )

    # Draw arrows
    if arrows_x:
        # Subsample if too many arrows
        n_arrows = len(arrows_x)
        if n_arrows > 5000:
            rng = np.random.RandomState(42)
            sel = rng.choice(n_arrows, 5000, replace=False)
            arrows_x = [arrows_x[i] for i in sel]
            arrows_y = [arrows_y[i] for i in sel]
            arrows_dx = [arrows_dx[i] for i in sel]
            arrows_dy = [arrows_dy[i] for i in sel]

        aw = arrow_width if arrow_width is not None else 0.003
        ax.quiver(
            arrows_x, arrows_y, arrows_dx, arrows_dy,
            color=arrow_color, scale_units="xy", scale=1,
            width=aw, headwidth=4, headlength=5, alpha=0.6,
        )

    ax.set_aspect("equal")
    # Flip Y axis to match R convention (Y increases downward in image coords)
    ax.invert_yaxis()

    if show_coordinates:
        ax.set_xlabel("X (µm)", fontsize=10)
        ax.set_ylabel("Y (µm)", fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(
        f"{sender} → {secreted_protein} → {receiver}",
        fontsize=12, fontweight="bold",
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=7, markerscale=3, frameon=False,
        ncol=1,
    )

    fig.tight_layout()
    return fig


def main():
    import spatialgpu.deconvolution as spacet

    print("=== stCCC Figure Generation ===")

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print("\n1. Loading CosMx LIHC data...")
    adata = ad.read_h5ad("data/LIHC_CosMx/LIHC_CosMx.h5ad")
    print(f"   Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Rename 'NotDet' to 'Other' to match R tutorial
    if "cellType" in adata.obs.columns:
        adata.obs["cellType"] = adata.obs["cellType"].replace({"NotDet": "Other"})
        print(f"   Cell types: {sorted(adata.obs['cellType'].unique().tolist())}")

    # Quality control
    adata = spacet.quality_control(adata, min_genes=50)
    print(f"   After QC: {adata.shape[0]} cells")

    # ----------------------------------------------------------------
    # Section 1 figures: QC and cell types
    # ----------------------------------------------------------------
    print("\n2. Generating Section 1 figures...")

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"],
        point_size=0.1,
    )
    save(fig, "cosmx_qc.png")

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="metaData",
        spatial_features=["cellType"],
        colors=CELL_COLORS,
        point_size=0.1,
    )
    save(fig, "cosmx_celltypes.png")

    # ----------------------------------------------------------------
    # Section 2: SecAct inference
    # ----------------------------------------------------------------
    print("\n3. Running SecAct inference (this may take ~30 min)...")
    adata = spacet.secact_inference(
        adata,
        scale_factor=1000,
        is_spot_level=True,
    )
    secact_out = adata.uns["spacet"]["SecAct_output"]
    zscore = secact_out["SecretedProteinActivity"]["zscore"]
    print(f"   Inferred activity: {zscore.shape[0]} proteins x {zscore.shape[1]} cells")

    # ----------------------------------------------------------------
    # Section 3: Spatial CCC
    # ----------------------------------------------------------------
    print("\n4. Running spatial CCC (this may take ~20 min)...")
    adata = spacet.secact_spatial_ccc(
        adata,
        cell_type_col="cellType",
        scale_factor=1000,
        radius=20,
        ratio_cutoff=0.2,
        padj_cutoff=0.01,
        n_jobs=1,
    )
    ccc = adata.uns["spacet"]["SecAct_output"]["SecretedProteinCCC"]
    print(f"   Significant CCC interactions: {len(ccc)}")

    # ----------------------------------------------------------------
    # Section 4 figures: Heatmap, Circle, Dot plot
    # ----------------------------------------------------------------
    print("\n5. Generating Section 4 figures...")

    fig = spacet.visualize_secact_heatmap(
        adata,
        colors_cell_type=CELL_COLORS,
        row_sorted=True,
        column_sorted=True,
    )
    save(fig, "cosmx_heatmap.png")

    fig = spacet.visualize_secact_circle(
        adata,
        colors_cell_type=CELL_COLORS,
    )
    save(fig, "cosmx_circle.png")

    # Dot plot with selected cell types and proteins
    cell_types = ["Tumor_boundary", "Fibroblast", "Macrophage", "Endothelial"]
    proteins = [
        "BGN", "COL1A1", "COL1A2", "DCN", "IGFBP5",
        "LGALS1", "LGALS9", "LYZ", "LUM", "MGP",
        "SPP1", "THBS1", "THBS2",
    ]

    # Filter to proteins that exist in CCC results
    available_proteins = [p for p in proteins if p in ccc["secretedProtein"].values]
    available_senders = [ct for ct in cell_types if ct in ccc["sender"].values]
    available_receivers = [ct for ct in cell_types if ct in ccc["receiver"].values]

    if available_proteins and available_senders and available_receivers:
        fig = spacet.visualize_secact_dotplot(
            adata,
            sender=available_senders,
            secreted_protein=available_proteins,
            receiver=available_receivers,
        )
        save(fig, "cosmx_dotplot.png")
    else:
        print("  Warning: Not enough matching CCC entries for dot plot")
        print(f"    Available senders: {available_senders}")
        print(f"    Available receivers: {available_receivers}")
        print(f"    Available proteins: {available_proteins}")

    # ----------------------------------------------------------------
    # Section 5 figures: Signaling velocity
    # ----------------------------------------------------------------
    print("\n6. Generating Section 5 figures (signaling velocity)...")

    # Full tissue view
    fig = plot_scst_velocity(
        adata,
        sender="Fibroblast",
        secreted_protein="THBS2",
        receiver="Tumor_boundary",
        cell_type_col="cellType",
        colors=CELL_COLORS,
        point_size=0.1,
        arrow_color="#ff0099",
        arrow_size=0.2,
        show_coordinates=True,
        figsize=(12, 10),
    )
    save(fig, "cosmx_velocity.png")

    # Zoomed view
    fig = plot_scst_velocity(
        adata,
        sender="Fibroblast",
        secreted_protein="THBS2",
        receiver="Tumor_boundary",
        cell_type_col="cellType",
        colors=CELL_COLORS,
        point_size=5,
        arrow_color="#ff0099",
        arrow_size=0.7,
        arrow_width=0.008,
        show_coordinates=False,
        customized_area=[8290, 8366, 1100, 1400],
        figsize=(8, 8),
    )
    save(fig, "cosmx_velocity_cut.png")

    # ----------------------------------------------------------------
    # Save processed data for future use
    # ----------------------------------------------------------------
    print("\n7. Saving processed data...")
    adata.write_h5ad("data/LIHC_CosMx/LIHC_CosMx_processed.h5ad")
    print("   Saved data/LIHC_CosMx/LIHC_CosMx_processed.h5ad")

    print("\n=== All stCCC figures generated! ===")


if __name__ == "__main__":
    main()
