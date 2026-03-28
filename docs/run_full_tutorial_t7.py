"""Run full Tutorial 7 (stCCC / CosMx LIHC) — full dataset with SecAct + CCC + velocity.

Usage: python docs/run_full_tutorial_t7.py
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
import anndata as ad

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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


def write_output(name, text):
    path = os.path.join(OUTPUTS_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"  Output saved: {path}")


def main():
    import spatialgpu.deconvolution as spacet

    print("=== Tutorial 7: stCCC / CosMx LIHC (Full Dataset, GPU) ===\n")

    # 1. Load data
    print("1. Loading CosMx LIHC data...")
    adata = ad.read_h5ad("data/LIHC_CosMx/LIHC_CosMx.h5ad")
    print(f"   Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Rename 'NotDet' to 'Other'
    if "cellType" in adata.obs.columns:
        adata.obs["cellType"] = adata.obs["cellType"].replace({"NotDet": "Other"})
        ct_counts = adata.obs["cellType"].value_counts()
        write_output("t7_celltype_counts.txt", ct_counts.to_string())
        print(f"   Cell types: {sorted(adata.obs['cellType'].unique().tolist())}")

    # Quality control (full dataset)
    adata = spacet.quality_control(adata, min_genes=50)
    print(f"   After QC: {adata.shape[0]} cells x {adata.shape[1]} genes")
    write_output("t7_adata_print.txt", str(adata))

    # QC figures
    print("\n2. QC and cell type figures...")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"], point_size=0.1,
    )
    save(fig, "cosmx_qc.png")

    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="metaData",
        spatial_features=["cellType"], colors=CELL_COLORS, point_size=0.1,
    )
    save(fig, "cosmx_celltypes.png")

    # 2. SecAct inference (full dataset)
    print("\n3. SecAct inference (full dataset)...")
    adata = spacet.secact_inference(
        adata, scale_factor=1000, is_spot_level=True,
    )
    secact_out = adata.uns["spacet"]["SecAct_output"]
    zscore = secact_out["SecretedProteinActivity"]["zscore"]
    print(f"   Inferred activity: {zscore.shape[0]} proteins x {zscore.shape[1]} cells")
    write_output("t7_secact_shape.txt",
                 f"Inferred activity: {zscore.shape[0]} proteins x {zscore.shape[1]} cells")
    write_output("t7_secact_head.txt", zscore.iloc[:6, :3].to_string())

    # 3. Spatial CCC (full dataset)
    print("\n4. Spatial CCC (full dataset)...")
    adata = spacet.secact_spatial_ccc(
        adata, cell_type_col="cellType",
        scale_factor=1000, radius=20,
        ratio_cutoff=0.2, padj_cutoff=0.01,
        n_jobs=1,
    )
    ccc = adata.uns["spacet"]["SecAct_output"]["SecretedProteinCCC"]
    print(f"   Significant CCC interactions: {len(ccc)}")
    write_output("t7_ccc_count.txt", f"Significant CCC interactions: {len(ccc)}")
    write_output("t7_ccc_table.txt", ccc.head(20).to_string())

    # 4. CCC visualization figures
    print("\n5. CCC visualization figures...")

    fig = spacet.visualize_secact_heatmap(
        adata, colors_cell_type=CELL_COLORS,
        row_sorted=True, column_sorted=True,
    )
    save(fig, "cosmx_heatmap.png")

    try:
        fig = spacet.visualize_secact_circle(
            adata, colors_cell_type=CELL_COLORS,
        )
        save(fig, "cosmx_circle.png")
    except ImportError as e:
        print(f"  Skipping circle plot: {e}")

    # Dot plot
    cell_types = ["Tumor_boundary", "Fibroblast", "Macrophage", "Endothelial"]
    proteins = [
        "BGN", "COL1A1", "COL1A2", "DCN", "IGFBP5",
        "LGALS1", "LGALS9", "LYZ", "LUM", "MGP",
        "SPP1", "THBS1", "THBS2",
    ]
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
        print(f"  Warning: Not enough CCC entries for dot plot")
        print(f"    Senders: {available_senders}, Receivers: {available_receivers}, Proteins: {available_proteins}")

    # 5. Signaling velocity (full dataset)
    print("\n6. Signaling velocity (Fibroblast -> THBS2 -> Tumor_boundary)...")
    vel = spacet.secact_signaling_velocity_scst(
        adata, sender="Fibroblast", secreted_protein="THBS2",
        receiver="Tumor_boundary", cell_type_col="cellType",
        scale_factor=1e5, radius=20,
    )

    # Full view
    fig = spacet.visualize_secact_velocity_scst(
        vel, show_coordinates=True, colors=CELL_COLORS,
        point_size=0.1, legend_position="right", legend_size=2,
        arrow_color="#ff0099", arrow_size=0.2,
    )
    save(fig, "cosmx_velocity.png")

    # Zoomed view
    fig = spacet.visualize_secact_velocity_scst(
        vel, customized_area=[8290, 8366, 1100, 1400],
        show_coordinates=False, colors=CELL_COLORS,
        point_size=5, legend_position="right", legend_size=3,
        arrow_color="#ff0099", arrow_width=1, arrow_size=0.7,
    )
    save(fig, "cosmx_velocity_cut.png")

    # Interactive velocity (if available)
    try:
        print("   Generating interactive velocity plot...")
        fig_interactive = spacet.visualize_secact_velocity_scst(
            vel, interactive=True, colors=CELL_COLORS,
        )
        interactive_path = os.path.join(FIGURES_DIR, "cosmx_velocity_interactive.html")
        fig_interactive.write_html(interactive_path)
        print(f"  Saved {interactive_path}")
    except Exception as e:
        print(f"  Interactive plot skipped: {e}")

    # Session info
    print("\n7. Session info...")
    import spatialgpu
    import anndata, numpy, scipy, pandas
    session_lines = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        f"anndata:     {anndata.__version__}",
        f"numpy:       {numpy.__version__}",
        f"scipy:       {scipy.__version__}",
        f"pandas:      {pandas.__version__}",
        f"matplotlib:  {matplotlib.__version__}",
    ]
    try:
        import cupy
        session_lines.append(f"cupy:        {cupy.__version__} (GPU backend available)")
    except ImportError:
        session_lines.append("cupy:        not installed (CPU-only mode)")
    write_output("t7_session_info.txt", "\n".join(session_lines))

    print("\n=== Tutorial 7 complete! ===")


if __name__ == "__main__":
    main()
