"""Run full Tutorial 4 (GeneSetScore) + Tutorial 5 (SpatialCorrelation) — full datasets.

Uses Visium_BC data. Runs genome-wide Moran's I, full LR database, and pairwise analysis.

Usage: python docs/run_full_tutorial_t4_t5.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run_tutorial_4():
    """Tutorial 4: GeneSetScore — full dataset."""
    import spatialgpu.deconvolution as spacet

    print("\n=== Tutorial 4: GeneSetScore (Full Pipeline) ===\n")

    adata = spacet.create_spacet_object_10x("data/Visium_BC")
    adata = spacet.quality_control(adata, min_genes=100)
    print(f"   Data: {adata.shape[0]} spots x {adata.shape[1]} genes")

    # Hallmark
    print("   Computing Hallmark scores (full gene set collection)...")
    adata = spacet.gene_set_score(adata, gene_sets="Hallmark")

    # Capture Hallmark output
    gs_data = adata.uns["spacet"].get("GeneSetScore", {})
    if "Hallmark" in gs_data:
        hallmark_df = gs_data["Hallmark"]
        write_output("t4_hallmark_head.txt", hallmark_df.iloc[:, :6].head(10).to_string())

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="GeneSetScore",
        spatial_features=["HALLMARK_HYPOXIA", "HALLMARK_TGF_BETA_SIGNALING"],
    )
    save(fig, "gs_hallmark.png")

    # CancerCellState
    print("   Computing CancerCellState scores...")
    adata = spacet.gene_set_score(adata, gene_sets="CancerCellState")

    gs_data = adata.uns["spacet"].get("GeneSetScore", {})
    if "CancerCellState" in gs_data:
        css_df = gs_data["CancerCellState"]
        write_output("t4_cancercellstate_head.txt", css_df.iloc[:, :6].head(10).to_string())

    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="GeneSetScore",
        spatial_features=["CancerCellState_Cycle", "CancerCellState_cEMT"],
    )
    save(fig, "gs_cancer_state.png")

    # TLS
    print("   Computing TLS score...")
    adata = spacet.gene_set_score(adata, gene_sets="TLS")
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="GeneSetScore",
        spatial_features=["TLS"],
    )
    save(fig, "gs_tls.png")

    print("   Tutorial 4 complete!")
    return adata


def run_tutorial_5(adata=None):
    """Tutorial 5: SpatialCorrelation — full genome-wide analysis."""
    import spatialgpu.deconvolution as spacet

    print("\n=== Tutorial 5: SpatialCorrelation (Full Pipeline) ===\n")

    if adata is None:
        adata = spacet.create_spacet_object_10x("data/Visium_BC")
        adata = spacet.quality_control(adata, min_genes=100)

    print(f"   Data: {adata.shape[0]} spots x {adata.shape[1]} genes")
    write_output("t5_adata_print.txt", str(adata))

    # 2. Compute Weight Matrix
    print("   Computing spatial weight matrix...")
    W = spacet.cal_weights(adata, radius=200, sigma=100, diag_as_zero=True)
    w_text = f"Weight matrix shape: {W.shape}\nNon-zero entries: {(W > 0).sum()}"
    write_output("t5_weight_matrix.txt", w_text)
    print(f"   {w_text}")

    # 3a. Targeted univariate Moran's I
    print("   Univariate Moran's I — targeted TGF-beta genes...")
    genes = ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TGFBR3"]
    adata = spacet.spatial_correlation(
        adata, mode="univariate", item=genes, W=W, n_permutation=1000,
    )
    uni_results = adata.uns["spacet"]["SpatialCorrelation"]["univariate"]
    write_output("t5_univariate_targeted.txt", uni_results.to_string())
    print(f"   Results:\n{uni_results}")

    # 3b. Genome-wide univariate Moran's I (ALL genes)
    print("   Univariate Moran's I — GENOME-WIDE (all genes, full dataset)...")
    adata = spacet.spatial_correlation(
        adata, mode="univariate", item=None, W=W, n_permutation=1000,
    )
    uni_all = adata.uns["spacet"]["SpatialCorrelation"]["univariate"]
    write_output("t5_univariate_genomewide.txt", uni_all.to_string())
    print(f"   Genome-wide results: {uni_all.shape[0]} genes tested")
    # Top 20 spatially variable genes
    top_svgs = uni_all.sort_values("p.Moran_Padj").head(20)
    write_output("t5_top20_svgs.txt", top_svgs.to_string())
    print(f"   Top 20 SVGs:\n{top_svgs}")

    # 4a. Custom bivariate pairs
    print("   Bivariate Moran's I — custom TGFB1 pairs...")
    gene_pairs = pd.DataFrame({
        "gene1": ["TGFB1", "TGFB1"],
        "gene2": ["TGFBR1", "TGFBR2"],
    })
    adata = spacet.spatial_correlation(
        adata, mode="bivariate", item=gene_pairs, W=W, n_permutation=1000,
    )
    biv_results = adata.uns["spacet"]["SpatialCorrelation"]["bivariate"]
    write_output("t5_bivariate_custom.txt", biv_results.to_string())
    print(f"   Custom bivariate results:\n{biv_results}")

    # 4b. Full L-R database (ALL pairs)
    print("   Bivariate Moran's I — FULL L-R database (all pairs, full dataset)...")
    adata = spacet.spatial_correlation(
        adata, mode="bivariate", item=None, W=W, n_permutation=1000,
    )
    biv_all = adata.uns["spacet"]["SpatialCorrelation"]["bivariate"]
    write_output("t5_bivariate_full_lr.txt", biv_all.to_string())
    print(f"   Full L-R results: {biv_all.shape[0]} pairs tested")
    top_lr = biv_all.sort_values("p.Moran_Padj").head(20)
    write_output("t5_top20_lr_pairs.txt", top_lr.to_string())
    print(f"   Top 20 L-R pairs:\n{top_lr}")

    # 4c. Visualize co-expression
    fig = spacet.visualize_spatial_feature(
        adata,
        spatial_type="GeneExpression",
        spatial_features=["TGFB1", "TGFBR2"],
        ncols=2,
    )
    save(fig, "sc_coexpression.png")

    # 5. Pairwise co-expression (full gene × gene matrix)
    print("   Pairwise Moran's I — ALL gene pairs (full dataset)...")
    adata = spacet.spatial_correlation(
        adata, mode="pairwise", W=W,
    )
    pairwise = adata.uns["spacet"]["SpatialCorrelation"]["pairwise"]
    pw_text = f"Pairwise matrix shape: {pairwise.shape}"
    # Extract TGF-beta submatrix
    tgf_genes = ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TGFBR3"]
    available_tgf = [g for g in tgf_genes if g in pairwise.index]
    if available_tgf:
        submat = pairwise.loc[available_tgf, available_tgf]
        pw_text += f"\n\n{submat.to_string()}"
    write_output("t5_pairwise.txt", pw_text)
    print(f"   {pw_text}")

    # Session info
    import spatialgpu
    import anndata, numpy, scipy
    session_lines = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        f"anndata:     {anndata.__version__}",
        f"numpy:       {numpy.__version__}",
        f"scipy:       {scipy.__version__}",
        f"pandas:      {pd.__version__}",
        f"matplotlib:  {matplotlib.__version__}",
    ]
    try:
        import cupy
        session_lines.append(f"cupy:        {cupy.__version__} (GPU backend available)")
    except ImportError:
        session_lines.append("cupy:        not installed (CPU-only mode)")
    session_text = "\n".join(session_lines)
    write_output("t5_session_info.txt", session_text)

    print("\n   Tutorial 5 complete!")


def main():
    adata = run_tutorial_4()
    run_tutorial_5(adata)
    print("\n=== All T4/T5 pipelines complete! ===")


if __name__ == "__main__":
    main()
