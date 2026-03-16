"""Tests for visualization module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt

from spatialgpu.deconvolution.visualization import (
    _flatten_lineage_tree,
    _get_spot_coordinates,
    visualize_cell_type_pair,
    visualize_colocalization,
    visualize_spatial_feature,
)


@pytest.fixture
def mock_adata():
    """Create a minimal AnnData with SpaCET results for testing."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    n_spots = 50
    n_genes = 100
    rng = np.random.default_rng(42)

    X = csr_matrix(rng.poisson(5, (n_spots, n_genes)).astype(np.float64))

    # Create spot IDs in "rowxcol" format
    spot_ids = [f"{i // 10}x{i % 10 * 2}" for i in range(n_spots)]
    gene_names = [f"Gene{i}" for i in range(n_genes)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=spot_ids),
        var=pd.DataFrame(index=gene_names),
    )

    # Add QC metrics
    adata.obs["UMI"] = np.asarray(X.sum(axis=1)).ravel()
    adata.obs["Gene"] = np.asarray((X > 0).sum(axis=1)).ravel()

    # Add deconvolution results
    cell_types = ["Malignant", "CAF", "Macrophage M1", "Macrophage M2", "T cell CD4"]
    prop_mat = pd.DataFrame(
        rng.dirichlet(np.ones(len(cell_types)), n_spots).T,
        index=cell_types,
        columns=spot_ids,
    )

    lineage_tree = {
        "Fibroblast": ["CAF"],
        "Macrophage": ["Macrophage M1", "Macrophage M2"],
        "T cell": ["T cell CD4"],
    }

    adata.uns["spacet"] = {
        "platform": "Visium",
        "deconvolution": {
            "propMat": prop_mat,
            "Ref": {
                "lineageTree": lineage_tree,
                "sigGenes": {ct: [f"Gene{i}" for i in range(5)] for ct in cell_types},
                "refProfiles": pd.DataFrame(
                    rng.random((20, 3)),
                    columns=["CAF", "Macrophage M1", "Macrophage M2"],
                ),
            },
        },
    }

    return adata


@pytest.fixture
def mock_adata_with_cci(mock_adata):
    """Add CCI results to mock adata."""
    n_spots = mock_adata.n_obs
    rng = np.random.default_rng(42)

    # LR Network Score: 3 rows (Raw_expr, Network_Score, Network_Score_pv)
    lr_score = np.vstack(
        [
            rng.random(n_spots) * 10,
            rng.random(n_spots) * 1.5 + 0.5,
            rng.random(n_spots),
        ]
    )
    mock_adata.uns["spacet"]["CCI"] = {
        "LRNetworkScore": lr_score,
        "LRNetworkScore_index": ["Raw_expr", "Network_Score", "Network_Score_pv"],
    }

    # Interface
    labels = np.array(["Tumor"] * 20 + ["Stroma"] * 20 + ["Interface"] * 10)
    interface_df = pd.DataFrame(
        [labels],
        index=["Interface"],
        columns=mock_adata.obs_names,
    )
    mock_adata.uns["spacet"]["CCI"]["interface"] = interface_df

    # Colocalization
    cell_types = list(mock_adata.uns["spacet"]["deconvolution"]["propMat"].index)
    pairs = []
    for ct1 in cell_types:
        for ct2 in cell_types:
            if ct1 != ct2:
                pairs.append(
                    {
                        "cell_type_1": ct1,
                        "cell_type_2": ct2,
                        "fraction_product": rng.random() * 0.02,
                        "fraction_rho": rng.uniform(-0.5, 0.5),
                        "fraction_pv": rng.random(),
                        "reference_rho": rng.uniform(-0.5, 0.5),
                        "reference_pv": rng.random(),
                    }
                )
    coloc_df = pd.DataFrame(pairs)
    coloc_df.index = coloc_df["cell_type_1"] + "_" + coloc_df["cell_type_2"]
    mock_adata.uns["spacet"]["CCI"]["colocalization"] = coloc_df

    # Interaction test results and group mat
    group_vals = rng.choice(
        ["Both", "CAF", "Macrophage M2", "Other"],
        size=n_spots,
        p=[0.2, 0.2, 0.2, 0.4],
    )
    group_mat = pd.DataFrame(
        [group_vals],
        index=["CAF_Macrophage M2"],
        columns=mock_adata.obs_names,
    )
    test_res = pd.DataFrame(
        {
            "colocalization_rho": [0.45],
            "colocalization_pv": [0.001],
            "groupCompare_cohen.d": [-0.3],
            "groupCompare_pv": [0.01],
            "Interaction": [True],
        },
        index=["CAF_Macrophage M2"],
    )
    mock_adata.uns["spacet"]["CCI"]["interaction"] = {
        "testRes": test_res,
        "groupMat": group_mat,
    }

    return mock_adata


class TestVisualizeSpatialFeature:
    def test_quality_control(self, mock_adata):
        fig = visualize_spatial_feature(mock_adata, "QualityControl")
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_quality_control_single(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata, "QualityControl", spatial_features=["UMI"]
        )
        assert fig is not None
        plt.close(fig)

    def test_gene_expression(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "GeneExpression",
            spatial_features=["Gene0", "Gene1"],
        )
        assert fig is not None
        plt.close(fig)

    def test_gene_expression_scales(self, mock_adata):
        for scale in ["RawCounts", "LogRawCounts", "LogTPM/10", "LogTPM"]:
            fig = visualize_spatial_feature(
                mock_adata,
                "GeneExpression",
                spatial_features=["Gene0"],
                scale_type_gene=scale,
            )
            assert fig is not None
            plt.close(fig)

    def test_cell_fraction(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "CellFraction",
            spatial_features=["Malignant", "CAF"],
        )
        assert fig is not None
        plt.close(fig)

    def test_cell_fraction_all(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "CellFraction",
            spatial_features=["All"],
        )
        assert fig is not None
        plt.close(fig)

    def test_cell_fraction_same_scale(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "CellFraction",
            spatial_features=["Malignant", "CAF"],
            same_scale_fraction=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_most_abundant_cell_type(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "MostAbundantCellType",
            spatial_features=["MajorLineage"],
        )
        assert fig is not None
        plt.close(fig)

    def test_most_abundant_sublineage(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "MostAbundantCellType",
            spatial_features=["SubLineage"],
        )
        assert fig is not None
        plt.close(fig)

    def test_lr_network_score(self, mock_adata_with_cci):
        fig = visualize_spatial_feature(
            mock_adata_with_cci,
            "LRNetworkScore",
        )
        assert fig is not None
        plt.close(fig)

    def test_interface(self, mock_adata_with_cci):
        fig = visualize_spatial_feature(
            mock_adata_with_cci,
            "Interface",
            spatial_features=["Interface"],
        )
        assert fig is not None
        plt.close(fig)

    def test_cell_type_composition(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "CellTypeComposition",
            spatial_features=["MajorLineage"],
        )
        assert fig is not None
        plt.close(fig)

    def test_invalid_type(self, mock_adata):
        with pytest.raises(ValueError, match="spatial_type must be one of"):
            visualize_spatial_feature(mock_adata, "InvalidType")

    def test_single_feature_on_axes(self, mock_adata):
        fig, ax = plt.subplots()
        result = visualize_spatial_feature(
            mock_adata,
            "QualityControl",
            spatial_features=["UMI"],
            ax=ax,
        )
        assert result is None  # returns None when ax provided
        plt.close(fig)

    def test_custom_colors(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "QualityControl",
            colors=["white", "red"],
        )
        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "QualityControl",
            figsize=(10, 5),
        )
        assert fig is not None
        assert fig.get_figwidth() == 10
        plt.close(fig)

    def test_gene_not_found(self, mock_adata):
        fig = visualize_spatial_feature(
            mock_adata,
            "GeneExpression",
            spatial_features=["Gene0", "NONEXISTENT"],
        )
        assert fig is not None  # Should still plot Gene0
        plt.close(fig)

    def test_gene_all_not_found(self, mock_adata):
        with pytest.raises(ValueError, match="No valid features"):
            visualize_spatial_feature(
                mock_adata,
                "GeneExpression",
                spatial_features=["NONEXISTENT"],
            )


class TestVisualizeColocalization:
    def test_basic(self, mock_adata_with_cci):
        fig = visualize_colocalization(mock_adata_with_cci)
        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_missing_data(self, mock_adata):
        with pytest.raises(ValueError, match="Run cci_colocalization"):
            visualize_colocalization(mock_adata)


class TestVisualizeCellTypePair:
    def test_basic(self, mock_adata_with_cci):
        fig = visualize_cell_type_pair(
            mock_adata_with_cci,
            cell_type_pair=("CAF", "Macrophage M2"),
        )
        assert fig is not None
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_missing_data(self, mock_adata):
        with pytest.raises(ValueError, match="Run cci_cell_type_pair"):
            visualize_cell_type_pair(mock_adata, ("CAF", "Macrophage M2"))


class TestColorMaps:
    def test_lr_network_score_7_stops(self):
        from spatialgpu.deconvolution.visualization import _COLORMAPS

        assert len(_COLORMAPS["LRNetworkScore"]) == 7

    def test_secreted_protein_activity_7_stops(self):
        from spatialgpu.deconvolution.visualization import _COLORMAPS

        assert len(_COLORMAPS["SecretedProteinActivity"]) == 7

    def test_interface_discrete_colors(self):
        from spatialgpu.deconvolution.visualization import _DISCRETE_COLORS

        colors = _DISCRETE_COLORS["Interface"]
        assert colors["Tumor"] == "black"
        assert colors["Stroma"] == "darkgrey"
        assert colors["Interface"] == "#f3c300"


class TestCoordinates:
    def test_spot_id_parsing(self):
        import anndata as ad

        adata = ad.AnnData(
            X=np.zeros((3, 2)),
            obs=pd.DataFrame(index=["0x0", "1x2", "2x4"]),
        )
        coords = _get_spot_coordinates(adata)
        assert coords.shape == (3, 2)
        # col is x, row is y
        assert coords[0, 0] == 0  # col=0
        assert coords[0, 1] == 0  # row=0
        assert coords[1, 0] == 2  # col=2
        assert coords[1, 1] == 1  # row=1

    def test_obsm_spatial(self):
        import anndata as ad

        coords_in = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = ad.AnnData(
            X=np.zeros((2, 2)),
            obs=pd.DataFrame(index=["s1", "s2"]),
            obsm={"spatial": coords_in},
        )
        coords = _get_spot_coordinates(adata)
        np.testing.assert_array_equal(coords, coords_in)


class TestFlattenLineageTree:
    def test_basic(self):
        tree = {
            "Fibroblast": ["CAF", "NF"],
            "Macrophage": ["M1", "M2"],
        }
        result = _flatten_lineage_tree(tree)
        assert result == ["CAF", "NF", "M1", "M2"]

    def test_single_values(self):
        tree = {"T cell": "CD4"}
        result = _flatten_lineage_tree(tree)
        assert result == ["CD4"]

    def test_empty(self):
        assert _flatten_lineage_tree({}) == []
