"""Run SpaCET (NNLS / IRWLS) on a DREAM validation bulk matrix.

Loads the reference bundle from `build_dream_reference.py` and an expression
matrix (genes x samples). Writes predicted cell-type fractions per method.

The two methods we benchmark map to `bench_spacet_trial.py`'s `_METHOD_MAP`:
  - v0_none   : NNLS, no cross-subject weighting        (cross_subject_weighting=False)
  - v3_irwls  : IRWLS, with cross-subject weighting     (cross_subject_weighting=True)

Usage:
    python scripts/bench_real/run_spacet_dream.py \
        --bulk /vf/.../dream/in_vitro/DS1_hugo_tpm.tsv \
        --reference /vf/.../dream/reference/reference_coarse.pkl \
        --method v3_irwls \
        --out /vf/.../bench_outputs/DS1_v3_irwls.csv \
        --no-malignant  # DREAM ground truth has no malignant compartment
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402

_METHOD_MAP = {
    "v0_none":  {"cross_subject_weighting": False},
    "v3_irwls": {"cross_subject_weighting": True},
}


def _load_bulk(path: str) -> ad.AnnData:
    sep = "," if path.endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep, index_col=0)
    # df is genes x samples; AnnData wants samples x genes
    return ad.AnnData(
        X=df.T.values,
        obs=pd.DataFrame(index=df.columns.astype(str)),
        var=pd.DataFrame(index=df.index.astype(str)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", required=True, help="expression matrix, genes x samples (tsv/csv)")
    ap.add_argument("--reference", required=True, help="pickled reference bundle from build_dream_reference.py")
    ap.add_argument("--method", required=True, choices=list(_METHOD_MAP))
    ap.add_argument("--out", required=True, help="output CSV (samples x cell types)")
    ap.add_argument("--no-malignant", action="store_true",
                    help="Set sc_include_malignant=False (DREAM has no malignant target).")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    print(f"=== SpaCET {args.method} on {os.path.basename(args.bulk)} ===")
    t0 = time.time()

    bulk = _load_bulk(args.bulk)
    print(f"  bulk: {bulk.n_obs} samples x {bulk.n_vars} genes")

    with open(args.reference, "rb") as f:
        ref = pickle.load(f)
    print(f"  reference: {ref['sc_counts'].shape[1]} cells x "
          f"{ref['sc_counts'].shape[0]} genes; "
          f"{ref['sc_annotation']['cellType'].nunique()} cell types")

    import spatialgpu.deconvolution as spacet

    method_kwargs = _METHOD_MAP[args.method]
    result = spacet.deconvolution_bulk(
        bulk,
        sc_counts=ref["sc_counts"],
        sc_annotation=ref["sc_annotation"],
        sc_lineage_tree=ref["sc_lineage_tree"],
        sc_include_malignant=not args.no_malignant,
        cancer_type=None,
        sc_downsampling=False,  # pure profiles, no need to downsample
        subject_col="subject_id" if method_kwargs["cross_subject_weighting"] else None,
        n_jobs=args.n_jobs,
        **method_kwargs,
    )

    prop = pd.DataFrame(
        result.obsm["spacet_propMat"],
        index=result.obs_names,
        columns=result.uns["spacet"]["cell_types"],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prop.to_csv(args.out)
    elapsed = time.time() - t0
    print(f"  wrote {prop.shape[0]}x{prop.shape[1]} predictions to {args.out}")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
