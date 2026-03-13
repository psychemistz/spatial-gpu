#!/usr/bin/env python
"""Visual equivalence comparison between R SpaCET and Python visualization.

Generates reference plots from R (via subprocess) and Python, then creates
side-by-side comparison panels for manual inspection.

Usage:
    python scripts/compare_visualizations.py [--output-dir OUTPUT_DIR]
    python scripts/compare_visualizations.py --skip-r  # reuse existing R data

Requires: R with SpaCET, ggplot2, patchwork installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _run_r(r_code: str, label: str, timeout: int = 1800) -> None:
    """Run R code via Rscript with error handling."""
    print(f"  [{label}] Running R code (timeout={timeout}s)...")
    result = subprocess.run(
        ["Rscript", "-e", r_code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"  [{label}] R stderr: {result.stderr[:1000]}")
        raise RuntimeError(f"R step '{label}' failed")
    print(f"  [{label}] Done.")


def generate_r_plots(output_dir: str) -> dict[str, str]:
    """Generate reference plots from R SpaCET in multiple phases."""
    r_plots = {}

    # Phase 1: Deconvolution + data export (slowest part)
    rds_path = os.path.join(output_dir, "spacet_obj.rds")

    if not os.path.exists(rds_path):
        r_phase1 = f"""
        suppressPackageStartupMessages({{
            library(SpaCET)
        }})
        visiumPath <- system.file(
            "extdata", "Visium_BC", package = "SpaCET"
        )
        SpaCET_obj <- create.SpaCET.object.10X(visiumPath)
        SpaCET_obj <- SpaCET.quality.control(
            SpaCET_obj, min.genes=100
        )
        cat("Running deconvolution...\\n")
        SpaCET_obj <- SpaCET.deconvolution(
            SpaCET_obj, cancerType="BRCA", coreNo=1
        )
        cat("Running CCI...\\n")
        SpaCET_obj <- SpaCET.CCI.colocalization(SpaCET_obj)
        SpaCET_obj <- SpaCET.CCI.LRNetworkScore(
            SpaCET_obj, coreNo=1
        )
        SpaCET_obj <- SpaCET.identify.interface(SpaCET_obj)

        # Save full object
        saveRDS(SpaCET_obj, "{rds_path}")

        # Export data for Python
        write.csv(
            as.matrix(SpaCET_obj@input$counts),
            "{output_dir}/counts.csv"
        )
        write.csv(
            SpaCET_obj@input$spotCoordinates,
            "{output_dir}/coords.csv"
        )
        write.csv(
            SpaCET_obj@results$deconvolution$propMat,
            "{output_dir}/propMat.csv"
        )
        write.csv(
            SpaCET_obj@results$metrics,
            "{output_dir}/metrics.csv"
        )
        write.csv(
            SpaCET_obj@results$CCI$LRNetworkScore,
            "{output_dir}/lr_score.csv"
        )
        write.csv(
            as.data.frame(SpaCET_obj@results$CCI$interface),
            "{output_dir}/interface.csv"
        )
        write.csv(
            SpaCET_obj@results$CCI$colocalization,
            "{output_dir}/colocalization.csv"
        )
        lt <- SpaCET_obj@results$deconvolution$Ref$lineageTree
        writeLines(
            jsonlite::toJSON(lt, auto_unbox=FALSE),
            "{output_dir}/lineageTree.json"
        )
        cat("Phase 1 complete.\\n")
        """
        _run_r(r_phase1, "Phase 1: Deconvolution+CCI", timeout=1800)
    else:
        print("  [Phase 1] Reusing cached RDS file.")

    # Phase 2: Generate all plots from saved object
    r_phase2 = f"""
    suppressPackageStartupMessages({{
        library(SpaCET)
        library(ggplot2)
        library(patchwork)
    }})
    SpaCET_obj <- readRDS("{rds_path}")

    # QualityControl
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="QualityControl",
        spatialFeatures=c("UMI","Gene"), imageBg=FALSE
    )
    ggsave(
        "{output_dir}/r_qc.png", p,
        width=10, height=5, dpi=150
    )

    # CellFraction
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="CellFraction",
        spatialFeatures=c("Malignant","CAF"), imageBg=FALSE
    )
    ggsave(
        "{output_dir}/r_cellfraction.png", p,
        width=10, height=5, dpi=150
    )

    # MostAbundantCellType (needs explicit colors for discrete scale)
    ct_names <- unlist(
        SpaCET_obj@results$deconvolution$Ref$lineageTree
    )
    if("Malignant" %in% rownames(
        SpaCET_obj@results$deconvolution$propMat
    ) & !"Malignant" %in% ct_names) {{
        ct_names <- c("Malignant", ct_names)
    }}
    tab20 <- c(
        "#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c",
        "#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
        "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f",
        "#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5"
    )
    ct_colors <- setNames(
        tab20[seq_along(ct_names)], ct_names
    )
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="MostAbundantCellType",
        spatialFeatures="SubLineage", imageBg=FALSE,
        colors=ct_colors
    )
    ggsave(
        "{output_dir}/r_mostabundant.png", p,
        width=8, height=6, dpi=150
    )

    # GeneExpression
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="GeneExpression",
        spatialFeatures=c("EPCAM","MS4A1"), imageBg=FALSE
    )
    ggsave(
        "{output_dir}/r_geneexpr.png", p,
        width=10, height=5, dpi=150
    )

    # LRNetworkScore
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="LRNetworkScore",
        spatialFeatures=c("Network_Score","Network_Score_pv"),
        imageBg=FALSE
    )
    ggsave(
        "{output_dir}/r_lrnetwork.png", p,
        width=10, height=5, dpi=150
    )

    # Colocalization (2-panel)
    p <- SpaCET.visualize.colocalization(SpaCET_obj)
    ggsave(
        "{output_dir}/r_colocalization.png", p,
        width=14, height=6, dpi=150
    )

    # Interface
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="Interface",
        spatialFeatures="Interface", imageBg=FALSE
    )
    ggsave(
        "{output_dir}/r_interface.png", p,
        width=8, height=6, dpi=150
    )

    # CellTypeComposition (pie charts, needs explicit colors)
    p <- SpaCET.visualize.spatialFeature(
        SpaCET_obj, spatialType="CellTypeComposition",
        spatialFeatures="SubLineage", imageBg=FALSE,
        colors=ct_colors
    )
    ggsave(
        "{output_dir}/r_composition.png", p,
        width=8, height=6, dpi=150
    )

    cat("Phase 2 complete: all plots generated.\\n")
    """
    _run_r(r_phase2, "Phase 2: Plot generation", timeout=300)

    # Collect generated R plots
    for name in [
        "qc", "cellfraction", "mostabundant", "geneexpr",
        "lrnetwork", "colocalization", "interface", "composition",
    ]:
        path = os.path.join(output_dir, f"r_{name}.png")
        if os.path.exists(path):
            r_plots[name] = path

    return r_plots


def generate_python_plots(
    output_dir: str, data_dir: str,
) -> dict[str, str]:
    """Generate equivalent plots from Python using R's data."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    from spatialgpu.deconvolution.visualization import (
        visualize_colocalization,
        visualize_spatial_feature,
    )

    py_plots = {}

    # Load R-exported data
    counts = pd.read_csv(
        os.path.join(data_dir, "counts.csv"), index_col=0,
    )
    pd.read_csv(os.path.join(data_dir, "coords.csv"), index_col=0)
    prop_mat = pd.read_csv(
        os.path.join(data_dir, "propMat.csv"), index_col=0,
    )
    metrics = pd.read_csv(
        os.path.join(data_dir, "metrics.csv"), index_col=0,
    )

    # Create AnnData (spots x genes)
    X = csr_matrix(counts.values.T.astype(np.float64))
    spot_ids = counts.columns.tolist()
    gene_names = counts.index.tolist()

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=spot_ids),
        var=pd.DataFrame(index=gene_names),
    )

    # Add QC metrics
    adata.obs["UMI"] = metrics.loc[
        "UMI", spot_ids
    ].values.astype(float)
    adata.obs["Gene"] = metrics.loc[
        "Gene", spot_ids
    ].values.astype(float)

    # Load lineage tree
    with open(os.path.join(data_dir, "lineageTree.json")) as f:
        lineage_tree = json.load(f)

    # Add deconvolution results
    adata.uns["spacet"] = {
        "platform": "Visium",
        "deconvolution": {
            "propMat": prop_mat,
            "Ref": {"lineageTree": lineage_tree},
        },
    }

    print("Generating Python plots...")

    def _save(fig, name):
        path = os.path.join(output_dir, f"py_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        py_plots[name] = path
        print(f"  Saved py_{name}.png")

    # 1. Quality Control
    fig = visualize_spatial_feature(
        adata, "QualityControl", figsize=(10, 5),
    )
    _save(fig, "qc")

    # 2. Cell Fraction
    fig = visualize_spatial_feature(
        adata, "CellFraction",
        spatial_features=["Malignant", "CAF"],
        figsize=(10, 5),
    )
    _save(fig, "cellfraction")

    # 3. Most Abundant Cell Type
    fig = visualize_spatial_feature(
        adata, "MostAbundantCellType",
        spatial_features=["SubLineage"],
        figsize=(8, 6),
    )
    _save(fig, "mostabundant")

    # 4. Gene Expression
    genes_to_plot = [
        g for g in ["EPCAM", "MS4A1"] if g in adata.var_names
    ]
    if genes_to_plot:
        fig = visualize_spatial_feature(
            adata, "GeneExpression",
            spatial_features=genes_to_plot,
            figsize=(10, 5),
        )
        _save(fig, "geneexpr")

    # 5. LR Network Score
    lr_path = os.path.join(data_dir, "lr_score.csv")
    if os.path.exists(lr_path):
        lr_df = pd.read_csv(lr_path, index_col=0)
        adata.uns["spacet"]["CCI"] = {
            "LRNetworkScore": lr_df.values.astype(float),
            "LRNetworkScore_index": lr_df.index.tolist(),
        }
        fig = visualize_spatial_feature(
            adata, "LRNetworkScore", figsize=(10, 5),
        )
        _save(fig, "lrnetwork")

    # 6. Interface
    iface_path = os.path.join(data_dir, "interface.csv")
    if os.path.exists(iface_path):
        iface_df = pd.read_csv(iface_path, index_col=0)
        if "CCI" not in adata.uns["spacet"]:
            adata.uns["spacet"]["CCI"] = {}
        adata.uns["spacet"]["CCI"]["interface"] = iface_df
        fig = visualize_spatial_feature(
            adata, "Interface",
            spatial_features=["Interface"],
            figsize=(8, 6),
        )
        _save(fig, "interface")

    # 7. Colocalization
    coloc_path = os.path.join(data_dir, "colocalization.csv")
    if os.path.exists(coloc_path):
        coloc_df = pd.read_csv(coloc_path, index_col=0)
        adata.uns["spacet"]["CCI"]["colocalization"] = coloc_df
        fig = visualize_colocalization(adata)
        _save(fig, "colocalization")

    # 8. Cell Type Composition
    fig = visualize_spatial_feature(
        adata, "CellTypeComposition",
        spatial_features=["SubLineage"],
        figsize=(8, 6),
    )
    _save(fig, "composition")

    return py_plots


def create_comparison(
    r_plots: dict[str, str],
    py_plots: dict[str, str],
    output_path: str,
) -> None:
    """Create side-by-side comparison panels."""
    common = sorted(set(r_plots.keys()) & set(py_plots.keys()))
    if not common:
        print("No common plots to compare.")
        return

    n = len(common)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, name in enumerate(common):
        r_img = mpimg.imread(r_plots[name])
        axes[i, 0].imshow(r_img)
        axes[i, 0].set_title(
            f"R: {name}", fontsize=12, fontweight="bold",
        )
        axes[i, 0].axis("off")

        py_img = mpimg.imread(py_plots[name])
        axes[i, 1].imshow(py_img)
        axes[i, 1].set_title(
            f"Python: {name}", fontsize=12, fontweight="bold",
        )
        axes[i, 1].axis("off")

    fig.suptitle(
        "Visual Equivalence: R SpaCET vs Python",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare R vs Python SpaCET visualizations",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/viz_comparison",
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R generation, reuse existing data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        if not args.skip_r:
            # Step 1: Generate R reference plots
            r_plots = generate_r_plots(data_dir)
            print(f"Generated {len(r_plots)} R plots")
        else:
            print("Skipping R generation, using existing data.")
            r_plots = {}
            for name in [
                "qc", "cellfraction", "mostabundant",
                "geneexpr", "lrnetwork", "colocalization",
                "interface", "composition",
            ]:
                path = os.path.join(data_dir, f"r_{name}.png")
                if os.path.exists(path):
                    r_plots[name] = path

        # Step 2: Generate Python plots using same data
        py_plots = generate_python_plots(args.output_dir, data_dir)
        print(f"Generated {len(py_plots)} Python plots")

        # Step 3: Side-by-side comparison
        comparison_path = os.path.join(
            args.output_dir, "comparison.png",
        )
        create_comparison(r_plots, py_plots, comparison_path)

        # Copy R vignette reference images
        ref_dir = (
            "/Users/seongyongpark/project/psychemist/"
            "SpaCET/vignettes/img"
        )
        if os.path.isdir(ref_dir):
            ref_dest = os.path.join(
                args.output_dir, "r_vignette_ref",
            )
            os.makedirs(ref_dest, exist_ok=True)
            for f in os.listdir(ref_dir):
                if f.endswith(".png"):
                    shutil.copy2(
                        os.path.join(ref_dir, f), ref_dest,
                    )
            print(
                f"Copied R vignette reference images to {ref_dest}"
            )

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        print(
            "Ensure R with SpaCET, ggplot2, patchwork installed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
