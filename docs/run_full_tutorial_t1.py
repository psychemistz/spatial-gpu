"""Run full Tutorial 1 (Visium BC) pipeline — figures + text outputs.

Usage: python docs/run_full_tutorial_t1.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


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

    print("=== Tutorial 1: Visium Breast Cancer (Full Pipeline) ===\n")

    # ---- 1. Create SpaCET Object ----
    print("1. Loading Visium BC data...")
    adata = spacet.create_spacet_object_10x("data/Visium_BC")
    print(f"   Loaded: {adata.shape[0]} spots x {adata.shape[1]} genes")

    # ---- 2. Quality Control ----
    print("2. Quality control...")
    adata = spacet.quality_control(adata, min_genes=100)
    print(f"   After QC: {adata.shape[0]} spots x {adata.shape[1]} genes")
    write_output("t1_adata_print.txt", str(adata))

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"],
        point_size=15,
    )
    save(fig, "qc_umi_gene.png")

    # ---- 3. Deconvolution ----
    print("3. Running deconvolution (full dataset)...")
    adata = spacet.deconvolution(adata, cancer_type="BRCA", n_jobs=8)

    # Capture deconvolution output table
    prop_mat = adata.uns["spacet"]["deconvolution"]["propMat"]
    write_output("t1_propmat_head.txt", prop_mat.iloc[:, :6].to_string())

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

    # 4c. Cell type composition pie charts
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

    # Capture CCI output
    cci_result = adata.uns["spacet"].get("CCI", {})
    ct_pair_key = "CAF_Macrophage M2"
    ct_pair_data = cci_result.get("cell_type_pair", {}).get(ct_pair_key, {})
    if "result" in ct_pair_data:
        write_output("t1_cci_result.txt", str(ct_pair_data["result"]))

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
    print("7. Malignant cell states (full dataset)...")
    adata = spacet.deconvolution_malignant(adata, n_jobs=6)

    # Capture malignant state fractions
    prop_mat2 = adata.uns["spacet"]["deconvolution"]["propMat"]
    mal_rows = [r for r in prop_mat2.index if "Malignant" in r]
    if mal_rows:
        write_output("t1_malignant_states.txt", prop_mat2.loc[mal_rows].iloc[:, :6].to_string())

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

    # ---- Session Info ----
    print("\n8. Session info...")
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
    session_text = "\n".join(session_lines)
    write_output("t1_session_info.txt", session_text)
    print(session_text)

    print("\n=== Tutorial 1 complete! ===")


if __name__ == "__main__":
    main()
