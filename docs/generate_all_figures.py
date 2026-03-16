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
        velocity = spacet.secact_signaling_velocity(adata, gene="SPARC")
        # Create a simple velocity plot
        fig, ax = plt.subplots(figsize=(8, 7))
        arrows = velocity["arrows"]
        coords_x = adata.obs["coordinate_x_um"].values
        coords_y = adata.obs["coordinate_y_um"].values
        ax.scatter(coords_x, coords_y, s=0.5, c="lightgrey", alpha=0.3)
        if len(arrows) > 0:
            ax.quiver(
                arrows["x_start"].values, arrows["y_start"].values,
                arrows["x_change"].values, arrows["y_change"].values,
                arrows["vec_len"].values, cmap="coolwarm", scale=None,
                width=0.003, alpha=0.7,
            )
        ax.set_title("SPARC Signaling Velocity")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_aspect("equal")
        save(fig, "hcc_velocity.png")

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

    # SecAct CCC figures require long computation — skip for now
    print("  SecAct CCC figures require ~50 min computation — skipping")
    print("  (heatmap, circle, dotplot, velocity will be placeholders)")


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
