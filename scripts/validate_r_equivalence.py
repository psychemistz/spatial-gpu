"""Validate Python deconvolution against R outputs for vst1, vst2, vst3.

Run from project root: python scripts/validate_r_equivalence.py [dataset]
  dataset: vst1, vst2, vst3, or all (default: vst1)

Compares propMat and malProp at machine precision (rtol=1e-10, atol=1e-12).
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

RTOL = 1e-10
ATOL = 1e-12

DATASETS = {
    # min_genes=0 for vst1 matches R validation (no QC filtering)
    # vst2/vst3 use _matched.csv files regenerated from same input data
    "vst1": {
        "path": "data/Visium_BC",
        "cancer": "BRCA",
        "min_genes": 0,
        "r_propmat": "validation/vst1_propMat.csv",
        "r_malprop": "validation/vst1_malProp.csv",
    },
    "vst2": {
        "path": "data/Visium_HCC",
        "cancer": "LIHC",
        "min_genes": 1000,
        "r_propmat": "validation/vst2_propMat_matched.csv",
        "r_malprop": "validation/vst2_malProp_matched.csv",
    },
    "vst3": {
        "path": "data/hiresST_CRC/hiresST_CRC.h5ad",
        "cancer": "CRC",
        "min_genes": 1000,
        "r_propmat": "validation/vst3_propMat_matched.csv",
        "r_malprop": "validation/vst3_malProp_matched.csv",
    },
}


def validate_dataset(name, cfg):
    import spatialgpu.deconvolution as spacet

    print(f"\n{'='*60}")
    print(f"  Validating: {name} ({cfg['cancer']})")
    print(f"{'='*60}")

    # Load data
    t0 = time.time()
    if cfg["path"].endswith(".h5ad"):
        import anndata as ad

        adata = ad.read_h5ad(cfg["path"])
    else:
        adata = spacet.create_spacet_object_10x(cfg["path"])
    # min_genes=0 adds QC columns without filtering (matches R validation)
    adata = spacet.quality_control(adata, min_genes=max(cfg["min_genes"], 0))
    print(
        f"  Loaded: {adata.shape[0]} spots x {adata.shape[1]} genes ({time.time()-t0:.1f}s)"
    )

    # Run Python deconvolution
    t1 = time.time()
    adata = spacet.deconvolution(adata, cancer_type=cfg["cancer"], n_jobs=8)
    print(f"  Deconvolution: {time.time()-t1:.1f}s")

    # Get Python results
    py_propmat = adata.uns["spacet"]["deconvolution"]["propMat"]
    py_malprop = adata.uns["spacet"]["deconvolution"]["malRes"]["malProp"]

    # Load R reference
    r_propmat = pd.read_csv(cfg["r_propmat"], index_col=0)
    r_malprop = pd.read_csv(cfg["r_malprop"], index_col=0)

    # Align
    common_types = sorted(set(r_propmat.index) & set(py_propmat.index))
    common_spots = sorted(set(r_propmat.columns) & set(py_propmat.columns))

    print(
        f"\n  Cell types: R={r_propmat.shape[0]}, Py={py_propmat.shape[0]}, common={len(common_types)}"
    )
    print(
        f"  Spots: R={r_propmat.shape[1]}, Py={py_propmat.shape[1]}, common={len(common_spots)}"
    )

    if not common_types or not common_spots:
        print("  ERROR: No common types/spots!")
        return False

    r_sub = r_propmat.loc[common_types, common_spots].values.astype(np.float64)
    py_sub = py_propmat.loc[common_types, common_spots].values.astype(np.float64)

    # Per-cell-type analysis
    print(f"\n  {'Cell Type':<30} {'MaxDiff':>12} {'MeanDiff':>12} {'Corr':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    for i, ct in enumerate(common_types):
        r_row = r_sub[i]
        py_row = py_sub[i]
        max_diff = np.max(np.abs(r_row - py_row))
        mean_diff = np.mean(np.abs(r_row - py_row))
        if np.std(r_row) > 1e-15 and np.std(py_row) > 1e-15:
            corr = np.corrcoef(r_row, py_row)[0, 1]
        else:
            corr = 1.0 if np.allclose(r_row, py_row) else 0.0
        flag = " *" if max_diff > ATOL else ""
        print(f"  {ct:<30} {max_diff:>12.2e} {mean_diff:>12.2e} {corr:>10.6f}{flag}")

    # Overall metrics
    max_diff = np.max(np.abs(r_sub - py_sub))
    mean_diff = np.mean(np.abs(r_sub - py_sub))
    flat_r, flat_py = r_sub.ravel(), py_sub.ravel()
    mask = (np.abs(flat_r) > 1e-15) | (np.abs(flat_py) > 1e-15)
    if mask.any():
        corr = np.corrcoef(flat_r[mask], flat_py[mask])[0, 1]
    else:
        corr = 1.0

    allclose = np.allclose(r_sub, py_sub, rtol=RTOL, atol=ATOL)

    print(f"\n  Overall max diff: {max_diff:.2e}")
    print(f"  Overall mean diff: {mean_diff:.2e}")
    print(f"  Overall correlation: {corr:.10f}")
    print(f"  allclose(rtol={RTOL}, atol={ATOL}): {allclose}")

    # malProp comparison
    if hasattr(py_malprop, "values"):
        py_mal = py_malprop.values
    else:
        py_mal = np.array(py_malprop)

    r_mal_col = (
        r_malprop.iloc[:, 0].values if r_malprop.shape[1] > 0 else r_malprop.values
    )
    if len(py_mal) == len(r_mal_col):
        mal_diff = np.max(np.abs(py_mal - r_mal_col))
        mal_corr = np.corrcoef(py_mal, r_mal_col)[0, 1]
        mal_close = np.allclose(py_mal, r_mal_col, rtol=RTOL, atol=ATOL)
        print(f"\n  malProp max diff: {mal_diff:.2e}")
        print(f"  malProp correlation: {mal_corr:.10f}")
        print(f"  malProp allclose: {mal_close}")

    result = "PASS" if allclose else "FAIL"
    print(f"\n  Result: {result}")
    return allclose


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "vst1"

    if target == "all":
        datasets = list(DATASETS.items())
    elif target in DATASETS:
        datasets = [(target, DATASETS[target])]
    else:
        print(f"Unknown dataset: {target}. Use: vst1, vst2, vst3, or all")
        sys.exit(1)

    results = {}
    for name, cfg in datasets:
        try:
            results[name] = validate_dataset(name, cfg)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n  ALL DATASETS PASS!")
    else:
        print(f"\n  {sum(not v for v in results.values())} dataset(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
