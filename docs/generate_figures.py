"""Generate all figures for the visium_BC tutorial.

Run from the spatial-gpu project root:
    python docs/generate_figures.py
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import spatialgpu.deconvolution as spacet


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    # ---- 1. Create SpaCET Object ----
    print("1. Loading Visium BC data...")
    adata = spacet.create_spacet_object_10x("data/Visium_BC")

    # ---- 2. Quality Control ----
    print("2. Quality control...")
    adata = spacet.quality_control(adata, min_genes=100)

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"],
        point_size=15,
    )
    save(fig, "qc_umi_gene.png")

    # ---- 3. Deconvolution ----
    print("3. Deconvolution (this may take a few minutes)...")
    adata = spacet.deconvolution(adata, cancer_type="BRCA", n_jobs=5)

    # ---- 4. Visualize Cell Fractions ----
    print("4. Visualizing cell fractions...")

    # 4a. Selected cell types
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="CellFraction",
        spatial_features=["Malignant", "Macrophage"],
        point_size=15,
    )
    save(fig, "fraction_malignant_macrophage.png")

    # 4b. All cell types
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="CellFraction",
        spatial_features=["All"],
        same_scale_fraction=True,
        point_size=15,
        ncols=5,
        figsize=(20, 14),
    )
    save(fig, "fraction_all.png")

    # 4c. Cell type composition pie charts (smaller pies)
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="CellTypeComposition",
        spatial_features=["MajorLineage"],
        point_size=0.4,
    )
    save(fig, "composition_pie.png")

    # 4d. Most abundant cell type
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="MostAbundantCellType",
        point_size=15,
    )
    save(fig, "most_abundant.png")

    # ---- 5. Cell-Cell Interactions ----
    print("5. Cell-cell interactions...")

    # 5a. Colocalization
    adata = spacet.cci_colocalization(adata)
    fig = spacet.visualize_colocalization(adata)
    save(fig, "colocalization.png")

    # 5b. L-R Network Score
    adata = spacet.cci_lr_network_score(adata, n_jobs=6)
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="LRNetworkScore",
        spatial_features=["Network_Score", "Network_Score_pv"],
        point_size=15,
    )
    save(fig, "lr_network_score.png")

    # 5c. Cell-type pair
    adata = spacet.cci_cell_type_pair(
        adata, cell_type_pair=("CAF", "Macrophage M2")
    )
    fig = spacet.visualize_cell_type_pair(
        adata, cell_type_pair=("CAF", "Macrophage M2")
    )
    save(fig, "cell_type_pair_caf_m2.png")

    # ---- 6. Interface ----
    print("6. Tumor-immune interface...")
    adata = spacet.identify_interface(adata)
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="Interface",
        spatial_features=["Interface"],
        point_size=15,
    )
    save(fig, "interface.png")

    adata = spacet.combine_interface(
        adata, cell_type_pair=("CAF", "Macrophage M2")
    )
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="Interface",
        spatial_features=["Interface&CAF_Macrophage M2"],
        point_size=15,
    )
    save(fig, "interface_caf_m2.png")

    adata = spacet.distance_to_interface(
        adata, cell_type_pair=("CAF", "Macrophage M2")
    )
    fig = spacet.visualize_distance_to_interface(
        adata, cell_type_pair=("CAF", "Macrophage M2")
    )
    save(fig, "distance_to_interface.png")

    # ---- 7. Malignant cell states ----
    print("7. Malignant cell states...")
    adata = spacet.deconvolution_malignant(adata, n_jobs=6)
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="CellFraction",
        spatial_features=[
            "Malignant",
            "Malignant cell state A",
            "Malignant cell state B",
        ],
        point_size=15,
        ncols=3,
    )
    save(fig, "malignant_states.png")

    print("\nDone! All figures saved to docs/figures/")


if __name__ == "__main__":
    main()
