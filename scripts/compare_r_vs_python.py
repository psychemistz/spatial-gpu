#!/usr/bin/env python
"""Compare Python SpaCET deconvolution against R SpaCET output.

Runs Python deconvolution on all 3 VST datasets and reports absolute
error vs R for both major and minor lineage cell types.
"""
from __future__ import annotations

import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from spatialgpu.deconvolution.core import deconvolution

VST_DIR = "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"
VAL_DIR = "/Users/seongyongpark/project/psychemist/spatial-gpu/validation"


def load_vst_dataset(i: int) -> ad.AnnData:
    """Load VST dataset i as AnnData."""
    counts_file = f"{VST_DIR}/{i}_counts.tsv"
    counts = pd.read_csv(counts_file, sep="\t", index_col=0)

    gene_names = np.array(counts.index)
    spot_names = np.array(counts.columns)

    # Parse coordinates from spot IDs
    parts = [s.split("x") for s in spot_names]
    array_row = np.array([float(p[0]) for p in parts])
    array_col = np.array([float(p[1]) for p in parts])

    coord_x_um = array_col * 0.5 * 100.0
    coord_y_um = array_row * 0.5 * np.sqrt(3) * 100.0
    coord_y_um = coord_y_um.max() - coord_y_um

    counts_sparse = sparse.csc_matrix(counts.values.astype(np.float64))

    adata = ad.AnnData(
        X=counts_sparse.T.tocsr(),
        obs=pd.DataFrame(
            {
                "coordinate_x_um": coord_x_um,
                "coordinate_y_um": coord_y_um,
            },
            index=pd.Index(spot_names),
        ),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.uns["spacet"] = {}
    adata.uns["spacet_platform"] = "Visium"

    return adata


def compute_errors(
    py_prop: pd.DataFrame,
    r_prop: pd.DataFrame,
    label: str,
) -> dict:
    """Compute absolute error statistics between Python and R propMat."""
    common_types = py_prop.index.intersection(r_prop.index)
    common_spots = py_prop.columns.intersection(r_prop.columns)

    if len(common_types) == 0 or len(common_spots) == 0:
        return {"label": label, "n_types": 0, "n_spots": 0, "error": "No overlap"}

    py_vals = py_prop.loc[common_types, common_spots].values.astype(np.float64)
    r_vals = r_prop.loc[common_types, common_spots].values.astype(np.float64)

    abs_err = np.abs(py_vals - r_vals)

    # Per cell-type max absolute error
    per_type_max = pd.Series(
        abs_err.max(axis=1), index=common_types, name="max_abs_err"
    )
    per_type_mean = pd.Series(
        abs_err.mean(axis=1), index=common_types, name="mean_abs_err"
    )

    return {
        "label": label,
        "n_types": len(common_types),
        "n_spots": len(common_spots),
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "median_abs_err": float(np.median(abs_err)),
        "per_type_max": per_type_max,
        "per_type_mean": per_type_mean,
    }


def main():
    # Major lineage types (from R output)
    major_types = [
        "Malignant", "CAF", "Endothelial", "Plasma", "B cell", "T CD4",
        "T CD8", "NK", "cDC", "pDC", "Macrophage", "Mast", "Neutrophil",
    ]

    # Minor lineage types
    minor_types = [
        "CAF", "Endothelial", "Plasma",
        "B cell naive", "B cell non-switched memory",
        "B cell switched memory", "B cell exhausted",
        "T CD4 naive", "Th1", "Th2", "Th17", "Tfh", "Treg",
        "T CD8 naive", "T CD8 central memory", "T CD8 effector memory",
        "T CD8 effector", "T CD8 exhausted",
        "NK",
        "cDC1 CLEC9A", "cDC2 CD1C", "cDC3 LAMP3",
        "pDC",
        "Macrophage M1", "Macrophage M2",
        "Mast", "Neutrophil",
    ]

    print("=" * 80)
    print("SpaCET R vs Python Numerical Equivalence Report")
    print("=" * 80)

    for i in range(1, 4):
        print(f"\n{'='*80}")
        print(f"DATASET {i}")
        print(f"{'='*80}")

        # Load R results
        r_major = pd.read_csv(f"{VAL_DIR}/vst{i}_propMat_major.csv", index_col=0)
        r_minor = pd.read_csv(f"{VAL_DIR}/vst{i}_propMat_minor.csv", index_col=0)
        r_full = pd.read_csv(f"{VAL_DIR}/vst{i}_propMat.csv", index_col=0)

        print(f"  R propMat: {r_full.shape[0]} cell types x {r_full.shape[1]} spots")

        # Load and run Python
        print(f"  Loading dataset {i}...")
        adata = load_vst_dataset(i)
        print(f"  AnnData: {adata.n_obs} spots x {adata.n_vars} genes")

        print(f"  Running Python deconvolution...")
        t0 = time.time()
        adata = deconvolution(adata, cancer_type="BRCA")
        elapsed = time.time() - t0
        print(f"  Python deconvolution completed in {elapsed:.1f}s")

        py_prop = adata.uns["spacet"]["deconvolution"]["propMat"]
        print(f"  Python propMat: {py_prop.shape[0]} cell types x {py_prop.shape[1]} spots")

        # Major lineage comparison
        py_major = py_prop.loc[py_prop.index.isin(major_types)]
        res_major = compute_errors(py_major, r_major, f"VST{i} Major Lineage")

        print(f"\n  --- MAJOR LINEAGE ({res_major['n_types']} types, {res_major['n_spots']} spots) ---")
        print(f"  Max absolute error:    {res_major['max_abs_err']:.6e}")
        print(f"  Mean absolute error:   {res_major['mean_abs_err']:.6e}")
        print(f"  Median absolute error: {res_major['median_abs_err']:.6e}")
        print(f"\n  Per cell-type max absolute error:")
        for ct, err in res_major["per_type_max"].items():
            flag = " <<<" if err > 1e-10 else ""
            print(f"    {ct:30s}  {err:.6e}{flag}")

        # Minor lineage comparison
        py_minor = py_prop.loc[py_prop.index.isin(minor_types)]
        res_minor = compute_errors(py_minor, r_minor, f"VST{i} Minor Lineage")

        print(f"\n  --- MINOR LINEAGE ({res_minor['n_types']} types, {res_minor['n_spots']} spots) ---")
        print(f"  Max absolute error:    {res_minor['max_abs_err']:.6e}")
        print(f"  Mean absolute error:   {res_minor['mean_abs_err']:.6e}")
        print(f"  Median absolute error: {res_minor['median_abs_err']:.6e}")
        print(f"\n  Per cell-type max absolute error:")
        for ct, err in res_minor["per_type_max"].items():
            flag = " <<<" if err > 1e-10 else ""
            print(f"    {ct:30s}  {err:.6e}{flag}")

    print(f"\n{'='*80}")
    print("NUMERICAL EQUIVALENCE CRITERIA: max |Python - R| < 1e-10 (float64 machine eps)")
    print("=" * 80)


if __name__ == "__main__":
    main()
