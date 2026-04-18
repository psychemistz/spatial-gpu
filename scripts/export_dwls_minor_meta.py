"""Annotate the T8 train-set reference cells with celltype_minor for fair DWLS.

Uses the IDENTICAL cells already exported to t8_real_sc_counts.csv (the same
4500-cell train reference that MuSiC and DWLS-broad saw), and adds the
celltype_minor column from the raw AnnData. Guarantees equal-data fairness —
only the label resolution changes.

Writes:
  - docs/outputs/t8_real_sc_meta_minor.csv   (cell_id, celltype_minor, celltype_major, subject_id)
  - docs/outputs/t8_minor_to_major.json      (mapping for eval-time collapse)

Prints per-minor cell counts so we can set the drop-threshold before MAST.
"""

import json
import os

import anndata as ad
import pandas as pd

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUTS_DIR = os.path.join(DOCS_DIR, "outputs")

EXISTING_META = os.path.join(OUTPUTS_DIR, "t8_real_sc_meta.csv")
H5AD_PATH = os.path.join(DATA_DIR, "BRCA_scRNA", "BRCA_scRNA_full.h5ad")
OUT_META = os.path.join(OUTPUTS_DIR, "t8_real_sc_meta_minor.csv")
OUT_MAP = os.path.join(OUTPUTS_DIR, "t8_minor_to_major.json")

MIN_CELLS_PER_MINOR = 30


def main():
    existing = pd.read_csv(EXISTING_META, index_col=0)
    print(f"Existing train meta: {len(existing)} cells, cols={list(existing.columns)}")

    raw = ad.read_h5ad(H5AD_PATH, backed="r")
    raw_obs = raw.obs[["celltype_major", "celltype_minor", "orig.ident"]].copy()
    raw_obs.columns = ["celltype_major", "celltype_minor", "subject_id"]

    missing = set(existing.index) - set(raw_obs.index)
    if missing:
        raise RuntimeError(f"{len(missing)} train cells not found in raw AnnData")

    meta = raw_obs.loc[existing.index].copy()
    meta["celltype_minor"] = meta["celltype_minor"].astype(str)
    meta["celltype_major"] = meta["celltype_major"].astype(str)

    sanity = (meta["celltype_major"] == existing["cell_type"]).all()
    if not sanity:
        raise RuntimeError("celltype_major mismatch between raw AnnData and existing meta")

    counts = meta["celltype_minor"].value_counts()
    print(f"\nPer-minor cell counts ({len(counts)} minors):")
    print(counts.to_string())

    to_drop = counts[counts < MIN_CELLS_PER_MINOR].index.tolist()
    to_keep = counts[counts >= MIN_CELLS_PER_MINOR].index.tolist()
    print(f"\nKeeping {len(to_keep)} minors (>= {MIN_CELLS_PER_MINOR} cells):")
    print(sorted(to_keep))
    print(f"\nDropping {len(to_drop)} minors (< {MIN_CELLS_PER_MINOR} cells):")
    print(sorted(to_drop))
    print(f"Cells kept: {meta['celltype_minor'].isin(to_keep).sum()} / {len(meta)}")

    meta = meta[meta["celltype_minor"].isin(to_keep)].copy()
    meta.index.name = ""
    meta.to_csv(OUT_META)
    print(f"\nWrote: {OUT_META}")

    minor_to_major = (
        meta.groupby("celltype_minor")["celltype_major"].agg(lambda s: s.mode().iat[0]).to_dict()
    )
    with open(OUT_MAP, "w") as f:
        json.dump(minor_to_major, f, indent=2, sort_keys=True)
    print(f"Wrote: {OUT_MAP}")
    print("\nMinor -> major map:")
    for k, v in sorted(minor_to_major.items()):
        print(f"  {k:40s} -> {v}")


if __name__ == "__main__":
    main()
