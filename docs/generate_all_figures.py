"""Generate figures for ALL tutorials (2-7).

Tutorial 1 (visium_BC) figures already exist.
Run from project root: python docs/generate_all_figures.py

Generates figures for:
  - oldST_PDAC (tutorial 2)
  - hiresST_CRC (tutorial 3)
  - GeneSetScore (tutorial 4)
  - SpatialCorrelation (tutorial 5)
  - stPattern (tutorial 6) — requires SecAct/secactpy
  - stCCC (tutorial 7) — requires SecAct/secactpy, large dataset
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

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def generate_pdac():
    """Tutorial 2: oldST_PDAC figures."""
    print("\n=== Tutorial 2: oldST_PDAC ===")
    import anndata as ad
    import spatialgpu.deconvolution as spacet

    adata = ad.read_h5ad("data/oldST_PDAC/st_PDAC.h5ad")
    sc_adata = ad.read_h5ad("data/oldST_PDAC/sc_PDAC.h5ad")
    lineage_tree = sc_adata.uns["lineage_tree"]

    # Prepare scRNA-seq inputs
    from scipy import sparse as sp
    X = sc_adata.X
    if sp.issparse(X):
        X = X.toarray()
    sc_counts = pd.DataFrame(X.T, index=sc_adata.var_names, columns=sc_adata.obs_names)
    sc_annotation = pd.DataFrame({
        "cellID": sc_adata.obs_names,
        "cellType": sc_adata.obs["cell_type"].values,
    })

    print("  Running deconvolution_matched_scrnaseq...")
    adata = spacet.deconvolution_matched_scrnaseq(
        adata, sc_counts=sc_counts, sc_annotation=sc_annotation,
        sc_lineage_tree=lineage_tree, n_jobs=6,
    )

    # 4a. All cell types (oldST: use point_size=40 to match 4d style)
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction", spatial_features=["All"],
        same_scale_fraction=True, point_size=40, ncols=5,
    )
    save(fig, "pdac_fraction_all.png")

    # 4b. Composition pie
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellTypeComposition",
        spatial_features=["MajorLineage"], point_size=1.5,
    )
    save(fig, "pdac_composition.png")

    # 4c. Specific cell types — use types from propMat
    prop_mat = adata.uns["spacet"]["deconvolution"]["propMat"]
    available = [ct for ct in ["Malignant", "CAF", "Macrophage", "Endothelial"]
                 if ct in prop_mat.index]
    if not available:
        available = list(prop_mat.index[:4])
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=available, point_size=40, ncols=2,
    )
    save(fig, "pdac_specific_types.png")

    # 4d. Gene expression (larger spots for oldST ~100µm)
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneExpression",
        spatial_features=["TM4SF1", "S100A4", "PRSS1", "CRISP3"],
        point_size=40,
        ncols=2,
    )
    save(fig, "pdac_gene_expression.png")


def generate_crc():
    """Tutorial 3: hiresST_CRC figures."""
    print("\n=== Tutorial 3: hiresST_CRC ===")
    import anndata as ad
    import spatialgpu.deconvolution as spacet

    adata = ad.read_h5ad("data/hiresST_CRC/hiresST_CRC.h5ad")

    print("  Running deconvolution (CRC, ~30 min)...")
    adata = spacet.deconvolution(adata, cancer_type="CRC", n_jobs=6)

    # 3a. Cell fractions
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=["Malignant", "CAF", "Endothelial"],
        point_size=0.6, same_scale_fraction=True,
    )
    save(fig, "crc_fractions.png")

    # 3b. Most abundant cell type
    import json
    with open("data/hiresST_CRC/colors_vector.json") as f:
        colors_vector = json.load(f)

    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="MostAbundantCellType",
        spatial_features=["MajorLineage"],
        colors=colors_vector, point_size=0.6,
    )
    save(fig, "crc_most_abundant.png")


def generate_geneset():
    """Tutorial 4: GeneSetScore figures."""
    print("\n=== Tutorial 4: GeneSetScore ===")
    import spatialgpu.deconvolution as spacet

    adata = spacet.create_spacet_object_10x("data/Visium_BC")
    adata = spacet.quality_control(adata, min_genes=100)

    # Hallmark
    print("  Hallmark scores...")
    adata = spacet.gene_set_score(adata, gene_sets="Hallmark")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneSetScore",
        spatial_features=["HALLMARK_HYPOXIA", "HALLMARK_TGF_BETA_SIGNALING"],
    )
    save(fig, "gs_hallmark.png")

    # CancerCellState
    print("  CancerCellState scores...")
    adata = spacet.gene_set_score(adata, gene_sets="CancerCellState")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneSetScore",
        spatial_features=["CancerCellState_Cycle", "CancerCellState_cEMT"],
    )
    save(fig, "gs_cancer_state.png")

    # TLS
    print("  TLS score...")
    adata = spacet.gene_set_score(adata, gene_sets="TLS")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneSetScore",
        spatial_features=["TLS"],
    )
    save(fig, "gs_tls.png")


def generate_spatial_corr():
    """Tutorial 5: SpatialCorrelation figures."""
    print("\n=== Tutorial 5: SpatialCorrelation ===")
    import spatialgpu.deconvolution as spacet

    adata = spacet.create_spacet_object_10x("data/Visium_BC")
    adata = spacet.quality_control(adata, min_genes=100)

    # Co-expression visualization
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneExpression",
        spatial_features=["TGFB1", "TGFBR2"], ncols=2,
    )
    save(fig, "sc_coexpression.png")


def generate_stpattern():
    """Tutorial 6: stPattern figures."""
    print("\n=== Tutorial 6: stPattern ===")
    import spatialgpu.deconvolution as spacet

    adata = spacet.create_spacet_object_10x("data/Visium_HCC")
    adata = spacet.quality_control(adata, min_genes=1000)

    # QC
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"],
        point_size=15,
    )
    save(fig, "hcc_qc.png")

    # Deconvolution
    print("  Running deconvolution (LIHC)...")
    adata = spacet.deconvolution(adata, cancer_type="LIHC", n_jobs=8)

    # Cell fractions
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=["Malignant", "CAF", "Endothelial", "Macrophage",
                          "Hepatocyte", "B cell", "T CD4", "T CD8"],
        same_scale_fraction=True, point_size=15, ncols=4,
    )
    save(fig, "hcc_fractions.png")

    # Hallmark EMT
    print("  Hallmark EMT...")
    adata = spacet.gene_set_score(adata, gene_sets="Hallmark")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneSetScore",
        spatial_features=["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
        point_size=15,
    )
    save(fig, "hcc_hallmark_emt.png")

    # SecAct (if secactpy available)
    try:
        print("  SecAct inference...")
        adata = spacet.secact_inference(adata, scale_factor=1e5)

        fig = spacet.visualize_spatial_feature(
            adata, spatial_type="SecretedProteinActivity",
            spatial_features=["HDGF", "MYDGF"], point_size=15, ncols=2,
        )
        save(fig, "hcc_activity.png")

        print("  Signaling patterns...")
        adata = spacet.secact_signaling_patterns(adata, k=3)
        fig = spacet.visualize_spatial_feature(
            adata, spatial_type="SignalingPattern",
            spatial_features=["1", "2", "3"], point_size=15, ncols=3,
        )
        save(fig, "hcc_patterns.png")

        print("  Signaling velocity...")
        spacet.secact_signaling_velocity(adata, gene="SPARC")

        # Contour map
        fig = spacet.visualize_secact_velocity(
            adata, gene="SPARC", contour_map=True,
        )
        save(fig, "hcc_velocity_contour.png")

        # Spot-level
        fig = spacet.visualize_secact_velocity(
            adata, gene="SPARC", contour_map=False,
        )
        save(fig, "hcc_velocity.png")

        # Animated GIF
        anim = spacet.visualize_secact_velocity(
            adata, gene="SPARC", animated=True,
            save=os.path.join(FIGURES_DIR, "hcc_velocity_animated.gif"),
            dpi=150,
        )
        plt.close(anim._fig)
        print(f"  Saved {os.path.join(FIGURES_DIR, 'hcc_velocity_animated.gif')}")

    except ImportError:
        print("  Skipping SecAct figures (secactpy not installed)")


def generate_stccc():
    """Tutorial 7: stCCC figures — large dataset, SecAct required."""
    print("\n=== Tutorial 7: stCCC ===")
    import anndata as ad
    import spatialgpu.deconvolution as spacet

    adata = ad.read_h5ad("data/LIHC_CosMx/LIHC_CosMx.h5ad")
    adata = spacet.quality_control(adata, min_genes=50)

    # QC
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"], point_size=0.1,
    )
    save(fig, "cosmx_qc.png")

    # Cell types
    cell_colors = {
        'B': '#C88888', 'Erythrocyte': '#fe666d', 'T.alpha.beta': '#B95FBB',
        'T.gamma.delta': '#3288bd', 'NK': '#bb8761', 'Hepatocyte': '#63636d',
        'Cholangiocyte': '#de77ae', 'Endothelial': '#D4D915', 'Fibroblast': '#66c2a5',
        'Macrophage': '#ff9a36', 'Tumor_core': '#A4DFF2', 'Tumor_boundary': 'blue',
        'Other': '#cccccc',
    }
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="metaData",
        spatial_features=["cellType"], colors=cell_colors, point_size=0.1,
    )
    save(fig, "cosmx_celltypes.png")

    # SecAct + CCC + velocity
    try:
        print("  SecAct inference (scale_factor=1000)...")
        adata = spacet.secact_inference(adata, scale_factor=1000, is_filter_sig=True)

        print("  scST velocity (Fibroblast -> THBS2 -> Tumor_boundary)...")
        vel = spacet.secact_signaling_velocity_scst(
            adata, sender="Fibroblast", secreted_protein="THBS2",
            receiver="Tumor_boundary", cell_type_col="cellType",
            scale_factor=1e5, radius=20,
        )

        # Full view
        fig = spacet.visualize_secact_velocity_scst(
            vel, show_coordinates=True, colors=cell_colors,
            point_size=0.1, legend_position="right", legend_size=2,
            arrow_color="#ff0099", arrow_size=0.2,
        )
        save(fig, "cosmx_velocity.png")

        # Zoomed view
        fig = spacet.visualize_secact_velocity_scst(
            vel, customized_area=[8290, 8366, 1100, 1400],
            show_coordinates=False, colors=cell_colors,
            point_size=5, legend_position="right", legend_size=3,
            arrow_color="#ff0099", arrow_width=1, arrow_size=0.7,
        )
        save(fig, "cosmx_velocity_cut.png")

    except ImportError:
        print("  Skipping SecAct figures (secactpy not installed)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tutorials", nargs="*", default=["all"],
                        help="Which tutorials to generate: 2-7 or 'all'")
    args = parser.parse_args()

    targets = args.tutorials
    if "all" in targets:
        targets = ["2", "3", "4", "5", "6", "7"]

    if "2" in targets:
        generate_pdac()
    if "3" in targets:
        generate_crc()
    if "4" in targets:
        generate_geneset()
    if "5" in targets:
        generate_spatial_corr()
    if "6" in targets:
        generate_stpattern()
    if "7" in targets:
        generate_stccc()

    print("\nDone!")
