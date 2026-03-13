"""Tests for SecAct analysis and visualization functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spatialgpu.deconvolution.secact import (
    _ensure_secact,
    _get_expression_matrix,
    _normalize_tpm,
    _rm_duplicates,
    _scalar1,
    secact_coxph_regression,
    secact_pattern_genes,
    secact_signaling_patterns,
    secact_signaling_velocity,
    secact_spatial_ccc,
    secact_survival_data,
)
from spatialgpu.deconvolution.visualization import (
    visualize_secact_bar,
    visualize_secact_dotplot,
    visualize_secact_heatmap,
    visualize_secact_heatmap_activity,
    visualize_secact_lollipop,
    visualize_secact_velocity,
)


@pytest.fixture
def mock_adata_secact():
    """Create a minimal AnnData with SecAct results for testing."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    n_spots = 100
    n_genes = 200
    rng = np.random.default_rng(42)

    X = csr_matrix(rng.poisson(5, (n_spots, n_genes)).astype(np.float64))

    spot_ids = [f"spot_{i}" for i in range(n_spots)]
    gene_names = [f"Gene{i}" for i in range(n_genes)]

    # Add some "secreted protein" names
    sp_names = ["TGFB1", "IL6", "VEGFA", "CCL2", "CXCL12"]
    for i, sp in enumerate(sp_names):
        gene_names[i] = sp

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "coordinate_x_um": rng.uniform(0, 1000, n_spots),
                "coordinate_y_um": rng.uniform(0, 1000, n_spots),
                "cell_type": rng.choice(["TypeA", "TypeB", "TypeC"], n_spots),
            },
            index=spot_ids,
        ),
        var=pd.DataFrame(index=gene_names),
    )

    # Add SecAct activity results
    zscore = pd.DataFrame(
        rng.standard_normal((len(sp_names), n_spots)),
        index=sp_names,
        columns=spot_ids,
    )
    pvalue = pd.DataFrame(
        rng.uniform(0, 1, (len(sp_names), n_spots)),
        index=sp_names,
        columns=spot_ids,
    )

    adata.uns["spacet"] = {
        "SecAct_output": {
            "SecretedProteinActivity": {
                "zscore": zscore,
                "pvalue": pvalue,
            }
        }
    }

    return adata


@pytest.fixture
def mock_ccc_adata(mock_adata_secact):
    """AnnData with CCC results added."""
    adata = mock_adata_secact
    ccc_data = pd.DataFrame(
        {
            "sender": ["TypeA", "TypeB", "TypeA", "TypeC", "TypeB"],
            "secretedProtein": ["TGFB1", "IL6", "VEGFA", "TGFB1", "CCL2"],
            "receiver": ["TypeB", "TypeC", "TypeC", "TypeA", "TypeA"],
            "sender_count": [33, 34, 33, 33, 34],
            "receiver_count": [34, 33, 33, 33, 33],
            "neighboringCellPairs": [50, 40, 45, 35, 42],
            "communicatingCellPairs": [20, 15, 18, 12, 16],
            "ratio": [0.4, 0.375, 0.4, 0.343, 0.381],
            "pv": [0.001, 0.005, 0.002, 0.01, 0.003],
            "pv_adj": [0.005, 0.008, 0.005, 0.01, 0.005],
        }
    )
    adata.uns["spacet"]["SecAct_output"]["SecretedProteinCCC"] = ccc_data
    return adata


class TestHelpers:
    def test_ensure_secact_creates_namespace(self):
        import anndata as ad

        adata = ad.AnnData()
        result = _ensure_secact(adata)
        assert isinstance(result, dict)
        assert "SecAct_output" in adata.uns["spacet"]

    def test_ensure_secact_existing(self):
        import anndata as ad

        adata = ad.AnnData()
        adata.uns["spacet"] = {"SecAct_output": {"key": "val"}}
        result = _ensure_secact(adata)
        assert result["key"] == "val"

    def test_get_expression_matrix(self, mock_adata_secact):
        expr = _get_expression_matrix(mock_adata_secact)
        assert isinstance(expr, pd.DataFrame)
        assert expr.shape == (200, 100)  # genes × spots
        assert expr.index[0] == "TGFB1"

    def test_normalize_tpm(self):
        df = pd.DataFrame(
            {"s1": [10, 20, 30], "s2": [5, 10, 15]},
            index=["g1", "g2", "g3"],
        )
        result = _normalize_tpm(df, scale_factor=1e5)
        assert result.shape == df.shape
        # After TPM: each column sums to scale_factor (before log)
        # log2(val + 1) should be > 0 for non-zero values
        assert (result.values >= 0).all()

    def test_rm_duplicates(self):
        df = pd.DataFrame(
            {"s1": [10, 20, 5], "s2": [1, 2, 3]},
            index=["A", "B", "A"],
        )
        result = _rm_duplicates(df)
        assert len(result) == 2
        assert "A" in result.index
        assert "B" in result.index

    def test_scalar1(self):
        v = np.array([3.0, 4.0])
        result = _scalar1(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)

    def test_scalar1_zero(self):
        v = np.array([0.0, 0.0])
        result = _scalar1(v)
        np.testing.assert_array_equal(result, v)


class TestSignalingPatterns:
    def test_basic(self, mock_adata_secact):
        result = secact_signaling_patterns(mock_adata_secact, k=2, seed=42)
        # May or may not have patterns depending on filter
        # The function should not error
        assert result is mock_adata_secact
        assert "SecAct_output" in result.uns["spacet"]

    def test_pattern_genes(self, mock_adata_secact):
        # Add pattern results manually
        W = pd.DataFrame(
            [[0.5, 0.1], [0.2, 0.8], [0.7, 0.3]],
            index=["TGFB1", "IL6", "VEGFA"],
            columns=["1", "2"],
        )
        H = pd.DataFrame(
            np.random.rand(2, 100),
            index=["1", "2"],
            columns=mock_adata_secact.obs_names,
        )
        mock_adata_secact.uns["spacet"]["SecAct_output"]["pattern"] = {
            "weight_W": W,
            "signal_H": H,
            "ccc_SP": pd.DataFrame(),
        }

        result = secact_pattern_genes(mock_adata_secact, n=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestSignalingVelocity:
    def test_basic(self, mock_adata_secact):
        result = secact_signaling_velocity(
            mock_adata_secact, gene="TGFB1", signal_mode="receiving"
        )
        assert "arrows" in result
        assert "points" in result
        assert isinstance(result["arrows"], pd.DataFrame)
        assert isinstance(result["points"], pd.DataFrame)
        assert result["gene"] == "TGFB1"
        assert result["signal_mode"] == "receiving"

    def test_sending_mode(self, mock_adata_secact):
        result = secact_signaling_velocity(
            mock_adata_secact, gene="TGFB1", signal_mode="sending"
        )
        assert result["signal_mode"] == "sending"

    def test_missing_gene(self, mock_adata_secact):
        result = secact_signaling_velocity(
            mock_adata_secact, gene="NONEXISTENT", signal_mode="receiving"
        )
        # Should not error, just produce empty arrows
        assert len(result["arrows"]) == 0


class TestSpatialCCC:
    def test_basic(self, mock_adata_secact):
        result = secact_spatial_ccc(
            mock_adata_secact,
            cell_type_col="cell_type",
            radius=500.0,
            n_background=100,
        )
        secact_out = result.uns["spacet"]["SecAct_output"]
        assert "SecretedProteinCCC" in secact_out
        assert "ccc_SP" in secact_out

    def test_no_activity(self):
        import anndata as ad

        adata = ad.AnnData()
        adata.uns["spacet"] = {"SecAct_output": {}}
        with pytest.raises(ValueError, match="Run secact_inference"):
            secact_spatial_ccc(adata, cell_type_col="ct")


class TestCoxPH:
    def test_basic(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(42)
        activity = pd.DataFrame(
            rng.standard_normal((5, 50)),
            index=[f"P{i}" for i in range(5)],
            columns=[f"S{i}" for i in range(50)],
        )
        survival = pd.DataFrame(
            {
                "Time": rng.uniform(1, 100, 50),
                "Event": rng.choice([0, 1], 50),
            },
            index=[f"S{i}" for i in range(50)],
        )
        result = secact_coxph_regression(activity, survival)
        assert isinstance(result, pd.DataFrame)
        assert "risk_score_z" in result.columns
        assert "p_value" in result.columns
        assert len(result) == 5

    def test_no_overlap(self):
        pytest.importorskip("lifelines")
        activity = pd.DataFrame(
            np.zeros((2, 3)),
            index=["P1", "P2"],
            columns=["A", "B", "C"],
        )
        survival = pd.DataFrame(
            {"Time": [1, 2], "Event": [1, 0]},
            index=["X", "Y"],
        )
        with pytest.raises(ValueError, match="No overlapping"):
            secact_coxph_regression(activity, survival)


class TestSurvivalData:
    def test_basic(self):
        pytest.importorskip("lifelines")
        rng = np.random.default_rng(42)
        activity = pd.DataFrame(
            rng.standard_normal((3, 50)),
            index=["P1", "P2", "P3"],
            columns=[f"S{i}" for i in range(50)],
        )
        survival = pd.DataFrame(
            {
                "Time": rng.uniform(1, 100, 50),
                "Event": rng.choice([0, 1], 50),
            },
            index=[f"S{i}" for i in range(50)],
        )
        result = secact_survival_data(activity, survival, protein="P1")
        assert "high" in result
        assert "low" in result
        assert "logrank_p" in result
        assert result["protein"] == "P1"
        assert len(result["high"]) + len(result["low"]) == 50


class TestSecActVisualization:
    def test_heatmap(self, mock_ccc_adata):
        fig = visualize_secact_heatmap(mock_ccc_adata)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dotplot(self, mock_ccc_adata):
        fig = visualize_secact_dotplot(
            mock_ccc_adata,
            sender=["TypeA", "TypeB"],
            secreted_protein=["TGFB1", "IL6"],
            receiver=["TypeB", "TypeC"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_activity(self):
        data = pd.DataFrame(
            np.random.rand(5, 4),
            index=[f"P{i}" for i in range(5)],
            columns=[f"CT{j}" for j in range(4)],
        )
        fig = visualize_secact_heatmap_activity(data, title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar(self):
        data = pd.Series(
            {"TGFB1": 2.5, "IL6": -1.2, "VEGFA": 0.8, "CCL2": -0.5},
        )
        fig = visualize_secact_bar(data, title="Risk Scores")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_lollipop(self):
        data = pd.Series(
            {"TGFB1": 2.5, "IL6": -1.2, "VEGFA": 0.8},
        )
        fig = visualize_secact_lollipop(data, title="Lollipop")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_velocity(self, mock_adata_secact):
        # First compute velocity data
        secact_signaling_velocity(
            mock_adata_secact, gene="TGFB1", signal_mode="receiving"
        )
        fig = visualize_secact_velocity(
            mock_adata_secact, gene="TGFB1", signal_mode="receiving"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_velocity_no_data(self, mock_adata_secact):
        with pytest.raises(ValueError, match="No velocity data"):
            visualize_secact_velocity(mock_adata_secact, gene="NONEXISTENT")
