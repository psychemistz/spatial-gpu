"""Convert extracted SpaCET R data (CSV/MTX) to h5ad files.

Usage: python scripts/convert_to_h5ad.py

Reads from data/{oldST_PDAC,hiresST_CRC}/ and writes .h5ad files.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix

DATA_DIR = Path("data")


def convert_oldst_pdac():
    """Convert oldST_PDAC ST + scRNA-seq to h5ad."""
    d = DATA_DIR / "oldST_PDAC"
    print("=== Converting oldST_PDAC ===")

    # ---- ST data ----
    counts = mmread(str(d / "st_counts.mtx"))
    counts = csc_matrix(counts)  # genes x spots
    genes = pd.read_csv(d / "st_genes.csv")["gene"].values
    spots = pd.read_csv(d / "st_spots.csv")["spot"].values
    coords = pd.read_csv(d / "st_coordinates.csv", index_col=0)

    # Build AnnData (spots x genes)
    adata = ad.AnnData(
        X=counts.T.tocsr(),
        obs=pd.DataFrame(
            {
                "coordinate_x_um": coords["coordinate_x_um"].values,
                "coordinate_y_um": coords["coordinate_y_um"].values,
            },
            index=pd.Index(spots),
        ),
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    adata.obsm["spatial"] = np.column_stack(
        [coords["coordinate_x_um"].values, coords["coordinate_y_um"].values]
    )
    adata.uns["spacet_platform"] = "oldST"
    adata.uns["spacet"] = {}

    out = d / "st_PDAC.h5ad"
    adata.write_h5ad(out)
    print(f"  ST: {adata.n_obs} spots x {adata.n_vars} genes -> {out}")

    # ---- scRNA-seq data ----
    sc_counts = mmread(str(d / "sc_counts.mtx"))
    sc_counts = csc_matrix(sc_counts)  # genes x cells
    sc_genes = pd.read_csv(d / "sc_genes.csv")["gene"].values
    sc_cells = pd.read_csv(d / "sc_cells.csv")["cell"].values
    sc_annot = pd.read_csv(d / "sc_annotation.csv", index_col=0)

    sc_adata = ad.AnnData(
        X=sc_counts.T.tocsr(),
        obs=pd.DataFrame(
            {"cell_type": sc_annot["bio_celltype"].values},
            index=pd.Index(sc_cells),
        ),
        var=pd.DataFrame(index=pd.Index(sc_genes)),
    )

    # Store lineage tree
    with open(d / "sc_lineage_tree.json") as f:
        sc_adata.uns["lineage_tree"] = json.load(f)

    out = d / "sc_PDAC.h5ad"
    sc_adata.write_h5ad(out)
    print(f"  SC: {sc_adata.n_obs} cells x {sc_adata.n_vars} genes -> {out}")

    # ---- Colors ----
    with open(d / "colors_vector.json") as f:
        colors = json.load(f)
    print(f"  Colors: {len(colors)} entries")


def convert_hiresst_crc():
    """Convert hiresST_CRC to h5ad."""
    d = DATA_DIR / "hiresST_CRC"
    print("=== Converting hiresST_CRC ===")

    counts = mmread(str(d / "counts.mtx"))
    counts = csc_matrix(counts)  # genes x spots
    genes = pd.read_csv(d / "genes.csv")["gene"].values
    spots = pd.read_csv(d / "spots.csv")["spot"].values
    coords = pd.read_csv(d / "coordinates.csv", index_col=0)

    adata = ad.AnnData(
        X=counts.T.tocsr(),
        obs=pd.DataFrame(
            {
                "coordinate_x_um": coords["coordinate_x_um"].values,
                "coordinate_y_um": coords["coordinate_y_um"].values,
            },
            index=pd.Index(spots),
        ),
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    adata.obsm["spatial"] = np.column_stack(
        [coords["coordinate_x_um"].values, coords["coordinate_y_um"].values]
    )
    adata.uns["spacet_platform"] = "SlideSeq"
    adata.uns["spacet"] = {}

    out = d / "hiresST_CRC.h5ad"
    adata.write_h5ad(out)
    print(f"  {adata.n_obs} spots x {adata.n_vars} genes -> {out}")

    # Colors
    with open(d / "colors_vector.json") as f:
        colors = json.load(f)
    print(f"  Colors: {len(colors)} entries")


if __name__ == "__main__":
    convert_oldst_pdac()
    convert_hiresst_crc()
    print("\n=== All conversions complete ===")
