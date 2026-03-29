"""Run full Tutorial 6 (stPattern/HCC) — full dataset with SecAct + velocity.

Usage: python docs/run_full_tutorial_t6.py
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

    print("=== Tutorial 6: stPattern/HCC (Full Pipeline) ===\n")

    # 1. Load data
    print("1. Loading Visium HCC data...")
    adata = spacet.create_spacet_object_10x("data/Visium_HCC")
    adata = spacet.quality_control(adata, min_genes=1000)
    print(f"   Data: {adata.shape[0]} spots x {adata.shape[1]} genes")
    write_output("t6_adata_print.txt", str(adata))

    # QC figure
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="QualityControl",
        spatial_features=["UMI", "Gene"], point_size=15,
    )
    save(fig, "hcc_qc.png")

    # 2. Deconvolution (full dataset)
    print("2. Running deconvolution (LIHC, full dataset)...")
    adata = spacet.deconvolution(adata, cancer_type="LIHC", n_jobs=8)

    prop_mat = adata.uns["spacet"]["deconvolution"]["propMat"]
    write_output("t6_propmat_head.txt", prop_mat.iloc[:, :6].to_string())

    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="CellFraction",
        spatial_features=["Malignant", "CAF", "Endothelial", "Macrophage",
                          "Hepatocyte", "B cell", "T CD4", "T CD8"],
        same_scale_fraction=True, point_size=15, ncols=4,
    )
    save(fig, "hcc_fractions.png")

    # 3. Hallmark EMT
    print("3. Computing Hallmark gene set scores...")
    adata = spacet.gene_set_score(adata, gene_sets="Hallmark")
    fig = spacet.visualize_spatial_feature(
        adata, spatial_type="GeneSetScore",
        spatial_features=["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
        point_size=15,
    )
    save(fig, "hcc_hallmark_emt.png")

    # 4. SecAct inference (full dataset)
    try:
        print("4. SecAct inference (full dataset)...")
        adata = spacet.secact_inference(adata, scale_factor=1e5)

        secact_out = adata.uns["spacet"]["SecAct_output"]
        zscore = secact_out["SecretedProteinActivity"]["zscore"]
        write_output("t6_secact_zscore_head.txt", zscore.iloc[:6, :3].to_string())
        print(f"   Inferred activity: {zscore.shape[0]} proteins x {zscore.shape[1]} spots")

        fig = spacet.visualize_spatial_feature(
            adata, spatial_type="SecretedProteinActivity",
            spatial_features=["HDGF", "MYDGF"], point_size=15, ncols=2,
        )
        save(fig, "hcc_activity.png")

        # 5. Signaling patterns (full dataset)
        print("5. Signaling patterns (k=3, full dataset)...")
        adata = spacet.secact_signaling_patterns(adata, k=3)

        patterns = adata.uns["spacet"]["SecAct_output"].get("SignalingPattern", {})
        if "gene_weights" in patterns:
            gw = patterns["gene_weights"]
            write_output("t6_pattern_weights.txt", gw.head(10).to_string())

        fig = spacet.visualize_spatial_feature(
            adata, spatial_type="SignalingPattern",
            spatial_features=["1", "2", "3"], point_size=15, ncols=3,
        )
        save(fig, "hcc_patterns.png")

        # 6. Signaling velocity (full dataset)
        print("6. Computing signaling velocity for SPARC...")
        spacet.secact_signaling_velocity(adata, gene="SPARC")

        # Contour map
        fig = spacet.visualize_secact_velocity(
            adata, gene="SPARC", contour_map=True,
        )
        save(fig, "hcc_velocity_contour.png")

        # Spot-level
        fig = spacet.visualize_secact_velocity(
            adata, gene="SPARC", contour_map=False,
        )
        save(fig, "hcc_velocity.png")

        # Animated GIF
        print("   Generating animated velocity GIF...")
        anim = spacet.visualize_secact_velocity(
            adata, gene="SPARC", animated=True,
            save=os.path.join(FIGURES_DIR, "hcc_velocity_animated.gif"),
            dpi=150,
        )
        plt.close(anim._fig)
        print(f"  Saved {os.path.join(FIGURES_DIR, 'hcc_velocity_animated.gif')}")

    except ImportError:
        print("  Skipping SecAct figures (secactpy not installed)")
    except Exception as e:
        print(f"  SecAct error: {e}")
        import traceback
        traceback.print_exc()

    # Session info
    print("\n7. Session info...")
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
    write_output("t6_session_info.txt", "\n".join(session_lines))

    print("\n=== Tutorial 6 complete! ===")


if __name__ == "__main__":
    main()
