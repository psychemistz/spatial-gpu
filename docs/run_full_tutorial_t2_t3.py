"""Run full Tutorial 2 (oldST PDAC) + Tutorial 3 (hiresST CRC) — full datasets.

Requires extracted h5ad files from R data extraction step.

Usage: python docs/run_full_tutorial_t2_t3.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

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


def run_tutorial_2():
    """Tutorial 2: oldST_PDAC — full deconvolution with matched scRNA-seq."""
    import anndata as ad
    import spatialgpu.deconvolution as spacet
    from scipy import sparse as sp

    print("\n=== Tutorial 2: oldST_PDAC (Full Pipeline) ===\n")

    # Load ST data
    adata = ad.read_h5ad("data/oldST_PDAC/st_PDAC.h5ad")
    print(f"   ST data: {adata.shape[0]} spots x {adata.shape[1]} genes")
    write_output("t2_adata_print.txt", str(adata))

    # Load scRNA-seq reference
    sc_adata = ad.read_h5ad("data/oldST_PDAC/sc_PDAC.h5ad")
    print(f"   SC data: {sc_adata.shape[0]} cells x {sc_adata.shape[1]} genes")
    lineage_tree = sc_adata.uns["lineage_tree"]

    # Prepare scRNA-seq inputs
    X = sc_adata.X
    if sp.issparse(X):
        X = X.toarray()
    sc_counts = pd.DataFrame(X.T, index=sc_adata.var_names, columns=sc_adata.obs_names)
    sc_annotation = pd.DataFrame({
        "cellID": sc_adata.obs_names,
        "cellType": sc_adata.obs["cell_type"].values,
    })

    # Deconvolution with matched scRNA-seq (full dataset)
    print("   Running deconvolution_matched_scrnaseq (full dataset)...")
    adata = spacet.deconvolution_matched_scrnaseq(
        adata, sc_counts=sc_counts, sc_annotation=sc_annotation,
        sc_lineage_tree=lineage_tree, n_jobs=8,
    )

    # Capture deconvolution output
    prop_mat = adata.uns["spacet"]["deconvolution"]["propMat"]
    write_output("t2_propmat_head.txt", prop_mat.iloc[:, :6].to_string())
    print(f"   Deconvolution complete: {prop_mat.shape[0]} cell types x {prop_mat.shape[1]} spots")

    # Lineage tree output
    write_output("t2_lineage_tree.txt", json.dumps(lineage_tree, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))

    # Figures
    print("   Generating figures...")

    # All cell types
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction", spatial_features=["All"],
        same_scale_fraction=True, point_size=40, ncols=5,
    )
    save(fig, "pdac_fraction_all.png")

    # Composition pie
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellTypeComposition",
        spatial_features=["MajorLineage"], point_size=1.5,
    )
    save(fig, "pdac_composition.png")

    # Specific cell types
    available = [ct for ct in ["Malignant", "CAF", "Macrophage", "Endothelial"]
                 if ct in prop_mat.index]
    if not available:
        available = list(prop_mat.index[:4])
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=available, point_size=40, ncols=2,
    )
    save(fig, "pdac_specific_types.png")

    # Gene expression
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneExpression",
        spatial_features=["TM4SF1", "S100A4", "PRSS1", "CRISP3"],
        point_size=40, ncols=2,
    )
    save(fig, "pdac_gene_expression.png")

    print("   Tutorial 2 complete!")


def run_tutorial_3():
    """Tutorial 3: hiresST_CRC — full deconvolution."""
    import anndata as ad
    import spatialgpu.deconvolution as spacet

    print("\n=== Tutorial 3: hiresST_CRC (Full Pipeline) ===\n")

    adata = ad.read_h5ad("data/hiresST_CRC/hiresST_CRC.h5ad")
    print(f"   Data: {adata.shape[0]} spots x {adata.shape[1]} genes")
    write_output("t3_adata_print.txt", str(adata))

    # Full deconvolution
    print("   Running deconvolution (CRC, full dataset)...")
    adata = spacet.deconvolution(adata, cancer_type="CRC", n_jobs=8)

    prop_mat = adata.uns["spacet"]["deconvolution"]["propMat"]
    write_output("t3_propmat_head.txt", prop_mat.iloc[:, :6].to_string())
    print(f"   Deconvolution complete: {prop_mat.shape[0]} cell types x {prop_mat.shape[1]} spots")

    # Figures
    print("   Generating figures...")

    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=["Malignant", "CAF", "Endothelial"],
        point_size=0.6, same_scale_fraction=True,
    )
    save(fig, "crc_fractions.png")

    # Most abundant cell type
    colors_path = "data/hiresST_CRC/colors_vector.json"
    if os.path.exists(colors_path):
        with open(colors_path) as f:
            colors_vector = json.load(f)
    else:
        colors_vector = None

    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="MostAbundantCellType",
        spatial_features=["MajorLineage"],
        colors=colors_vector, point_size=0.6,
    )
    save(fig, "crc_most_abundant.png")

    # CCI on full CRC dataset
    print("   Running CCI colocalization (full dataset)...")
    adata = spacet.cci_colocalization(adata)
    coloc = adata.uns["spacet"].get("CCI", {}).get("colocalization", None)
    if coloc is not None:
        write_output("t3_colocalization.txt", str(coloc))

    print("   Tutorial 3 complete!")


def main():
    run_tutorial_2()
    run_tutorial_3()
    print("\n=== All T2/T3 pipelines complete! ===")


if __name__ == "__main__":
    main()
