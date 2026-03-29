#!/usr/bin/env python3
"""Side-by-side validation: R SpaCET vs native Python spatial-gpu.

Runs both R and Python deconvolution on the same Visium sample(s) and
compares cell type fractions. Python runs in NATIVE mode only (no R subprocess).

Usage:
    python validate_r_vs_python_native.py [--sample BRCA_10x_Datasets/Version1.0.0_Breast.Cancer_rep1]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# Paths (configurable via environment variables)
VISIUM_DIR = Path(os.environ.get(
    "SPACET_VISIUM_DIR",
    "/data/parks34/projects/0sigdiscov/pkg_dev/sigdiscov-main/visium_pipeline",
))
VST_DIR = VISIUM_DIR / "results" / "1vst"
SPACET_DECONV_DIR = VISIUM_DIR / "results" / "0spacet"
OUTPUT_DIR = Path(os.environ.get(
    "SPACET_OUTPUT_DIR",
    "/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results",
))

# Default test samples (BRCA, well-characterized)
DEFAULT_SAMPLES = [
    "BRCA_10x_Datasets/Version1.0.0_Breast.Cancer_rep1",
    "BRCA_10x_Datasets/Version1.0.0_Breast.Cancer_rep2",
]


def find_raw_counts(dataset, sample):
    """Find raw count matrix for a Visium sample."""
    # Check common locations
    candidates = [
        VISIUM_DIR / "data" / "raw" / dataset / f"{sample}.h5ad",
        VISIUM_DIR / "data" / "raw" / dataset / sample / "filtered_feature_bc_matrix.h5",
        Path(os.environ.get(
            "SPACET_CYTOATLAS_DIR",
            "/data/parks34/projects/2cytoatlas/data/0_Collection/Visium",
        )) / dataset / sample,
    ]
    # The VST file exists — we can reconstruct counts from it
    vst_path = VST_DIR / dataset / f"{sample}_vst.tsv"
    if vst_path.exists():
        return vst_path
    for c in candidates:
        if c.exists():
            return c
    return None


def run_r_spacet(counts_path, cancer_type, output_path):
    """Run R SpaCET deconvolution and save results."""
    r_script = f"""
library(SpaCET)

# Load counts
counts <- as.matrix(read.csv("{counts_path}", sep="\\t", row.names=1, check.names=FALSE))
cat(sprintf("Loaded: %d genes x %d spots\\n", nrow(counts), ncol(counts)))

# Parse coordinates from spot names
spots <- colnames(counts)
parts <- strsplit(spots, "x")
rows <- sapply(parts, function(x) as.numeric(x[1]))
cols <- sapply(parts, function(x) as.numeric(x[2]))

# Create SpaCET object
spacet <- SpaCET.obj.new(
    counts = counts,
    spotCoordinates = data.frame(row = rows, col = cols, row.names = spots),
    platform = "Visium"
)

# Run deconvolution
cat("Running SpaCET deconvolution...\\n")
spacet <- SpaCET.deconvolution(spacet, cancerType = "{cancer_type}")

# Save results
propMat <- spacet@results$deconvolution$propMat
write.csv(propMat, "{output_path}/r_propMat.csv")

# Save major lineage
major_types <- c("Malignant", "CAF", "Endothelial", "Plasma", "B cell",
                 "T CD4", "T CD8", "NK", "cDC", "pDC", "Macrophage", "Mast", "Neutrophil")
major <- propMat[rownames(propMat) %in% major_types, , drop=FALSE]
write.csv(major, "{output_path}/r_propMat_major.csv")

# Save malignant proportion
malProp <- spacet@results$deconvolution$malProp
write.csv(data.frame(spot=names(malProp), malProp=malProp), "{output_path}/r_malProp.csv", row.names=FALSE)

cat("R SpaCET done.\\n")
"""
    r_script_path = output_path / "run_spacet.R"
    r_script_path.write_text(r_script)

    log.info("Running R SpaCET...")
    t0 = time.time()
    result = subprocess.run(
        ["Rscript", str(r_script_path)],
        capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error("R SpaCET failed:\n%s", result.stderr[-500:])
        return None

    log.info("R SpaCET completed in %.1fs", elapsed)
    return pd.read_csv(output_path / "r_propMat.csv", index_col=0)


def run_python_native(counts_path, cancer_type, output_path):
    """Run Python spatial-gpu deconvolution in native mode (no R subprocess)."""
    # Force native Python mode
    os.environ["SPATIALGPU_FORCE_PYTHON"] = "1"

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)
    from spatialgpu.deconvolution.core import deconvolution

    # Load counts
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    log.info("Loaded: %d genes × %d spots", counts.shape[0], counts.shape[1])

    # Parse coordinates
    spots = counts.columns.tolist()
    parts = [s.split("x") for s in spots]
    array_row = np.array([float(p[0]) for p in parts])
    array_col = np.array([float(p[1]) for p in parts])

    coord_x = array_col * 0.5 * 100.0
    coord_y = array_row * 0.5 * np.sqrt(3) * 100.0
    coord_y = coord_y.max() - coord_y

    # Create AnnData
    adata = ad.AnnData(
        X=sparse.csc_matrix(counts.values.astype(np.float64)).T.tocsr(),
        obs=pd.DataFrame(
            {"coordinate_x_um": coord_x, "coordinate_y_um": coord_y},
            index=pd.Index(spots),
        ),
        var=pd.DataFrame(index=pd.Index(counts.index)),
    )
    adata.uns["spacet"] = {}
    adata.uns["spacet_platform"] = "Visium"

    log.info("Running Python deconvolution (native mode)...")
    t0 = time.time()
    adata = deconvolution(adata, cancer_type=cancer_type)
    elapsed = time.time() - t0
    log.info("Python deconvolution completed in %.1fs", elapsed)

    py_prop = adata.uns["spacet"]["deconvolution"]["propMat"]
    py_prop.to_csv(output_path / "py_propMat.csv")

    return py_prop


def compare_results(r_prop, py_prop, label):
    """Compare R vs Python deconvolution results."""
    common_types = sorted(set(r_prop.index) & set(py_prop.index))
    common_spots = sorted(set(r_prop.columns) & set(py_prop.columns))

    if not common_types or not common_spots:
        log.error("No overlap: R types=%d, Python types=%d", len(r_prop.index), len(py_prop.index))
        return

    r_vals = r_prop.loc[common_types, common_spots].values.astype(np.float64)
    py_vals = py_prop.loc[common_types, common_spots].values.astype(np.float64)
    diff = np.abs(r_vals - py_vals)

    print(f"\n{'='*70}")
    print(f"  {label}: {len(common_types)} cell types × {len(common_spots)} spots")
    print(f"{'='*70}")
    print(f"  Max |diff|:    {diff.max():.6e}")
    print(f"  Mean |diff|:   {diff.mean():.6e}")
    print(f"  Median |diff|: {np.median(diff):.6e}")
    print(f"  Correlation:   {np.corrcoef(r_vals.ravel(), py_vals.ravel())[0,1]:.10f}")

    print(f"\n  Per cell-type max |diff|:")
    for i, ct in enumerate(common_types):
        ct_diff = diff[i].max()
        flag = " <<<" if ct_diff > 0.01 else ""
        print(f"    {ct:30s}  {ct_diff:.6e}{flag}")

    # R median vs Python median per cell type
    print(f"\n  Per cell-type median fraction:")
    print(f"    {'Cell type':30s} {'R median':>10} {'Py median':>10} {'|diff|':>10}")
    for i, ct in enumerate(common_types):
        r_med = np.median(r_vals[i])
        py_med = np.median(py_vals[i])
        print(f"    {ct:30s} {r_med:>10.4f} {py_med:>10.4f} {abs(r_med-py_med):>10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES,
                        help="dataset/sample paths relative to VST_DIR")
    parser.add_argument("--cancer-type", default="BRCA")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_path in args.samples:
        dataset = sample_path.split("/")[0]
        sample = sample_path.split("/")[1]

        log.info("Processing %s/%s", dataset, sample)
        out_dir = OUTPUT_DIR / f"{dataset}__{sample}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find counts
        vst_path = VST_DIR / dataset / f"{sample}_vst.tsv"
        if not vst_path.exists():
            log.warning("VST not found: %s — skipping", vst_path)
            continue

        # Run R SpaCET
        r_prop = run_r_spacet(vst_path, args.cancer_type, out_dir)

        # Run Python (native)
        py_prop = run_python_native(vst_path, args.cancer_type, out_dir)

        if r_prop is not None and py_prop is not None:
            compare_results(r_prop, py_prop, f"{dataset}/{sample}")

    print(f"\n{'='*70}")
    print("DONE — Results saved to:", OUTPUT_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
