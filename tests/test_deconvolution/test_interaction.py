"""Tests for cell-cell interaction analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialgpu.deconvolution.interaction import (
    _bipartite_edge_swap,
    _cohens_d,
    identify_interface,
)


class TestBipartiteEdgeSwap:
    def test_preserves_degree(self):
        """Edge swap should preserve row and column degrees."""
        rng = np.random.RandomState(42)
        mat = np.zeros((10, 15), dtype=np.int32)
        # Create ~30 random edges
        for _ in range(30):
            i, j = rng.randint(0, 10), rng.randint(0, 15)
            mat[i, j] = 1

        row_deg_before = mat.sum(axis=1)
        col_deg_before = mat.sum(axis=0)

        result = _bipartite_edge_swap(mat.copy(), rng)

        row_deg_after = result.sum(axis=1)
        col_deg_after = result.sum(axis=0)

        np.testing.assert_array_equal(row_deg_before, row_deg_after)
        np.testing.assert_array_equal(col_deg_before, col_deg_after)

    def test_preserves_edge_count(self):
        """Total number of edges should be preserved."""
        rng = np.random.RandomState(42)
        mat = np.zeros((8, 12), dtype=np.int32)
        edges = [(0, 1), (0, 3), (2, 5), (3, 7), (5, 2), (7, 9)]
        for i, j in edges:
            mat[i, j] = 1

        result = _bipartite_edge_swap(mat.copy(), rng)
        assert result.sum() == mat.sum()

    def test_single_edge_unchanged(self):
        """With only one edge, no swap is possible."""
        rng = np.random.RandomState(42)
        mat = np.zeros((5, 5), dtype=np.int32)
        mat[2, 3] = 1

        result = _bipartite_edge_swap(mat.copy(), rng)
        np.testing.assert_array_equal(result, mat)


class TestCohensD:
    def test_identical_groups(self):
        x = np.ones(10)
        d = _cohens_d(x, x)
        assert d == 0.0

    def test_known_effect(self):
        g1 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = _cohens_d(g1, g2)
        # Large positive d (g1 > g2)
        assert d > 1.0

    def test_negative_effect(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([10.0, 11.0, 12.0])
        d = _cohens_d(g1, g2)
        assert d < 0


class TestIdentifyInterface:
    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData with deconvolution results for interface testing."""
        import anndata as ad

        spots = [
            "10x10", "10x12", "10x14",
            "11x11", "11x13",
            "12x10", "12x12", "12x14",
        ]
        n_spots = len(spots)

        # Malignant fractions: some spots are tumor, some are stroma
        mal_fracs = [0.8, 0.1, 0.7, 0.9, 0.05, 0.6, 0.02, 0.85]

        prop_mat = pd.DataFrame(
            {spot: [mf, 1 - mf] for spot, mf in zip(spots, mal_fracs)},
            index=["Malignant", "NonMalignant"],
        )

        adata = ad.AnnData(
            X=np.zeros((n_spots, 5)),
            obs=pd.DataFrame(index=pd.Index(spots)),
        )
        adata.uns["spacet"] = {
            "deconvolution": {"propMat": prop_mat},
        }
        return adata

    def test_classifies_tumor_stroma(self, mock_adata):
        result = identify_interface(mock_adata, malignant_cutoff=0.5)
        cci = result.uns["spacet"]["CCI"]
        interface = cci["interface"]
        assert "Interface" in interface.index

        # Spots with mal >= 0.5 should be Tumor
        assert interface.loc["Interface", "10x10"] == "Tumor"
        assert interface.loc["Interface", "11x13"] != "Tumor"

    def test_interface_spots_detected(self, mock_adata):
        result = identify_interface(mock_adata, malignant_cutoff=0.5)
        interface = result.uns["spacet"]["CCI"]["interface"]
        values = interface.loc["Interface"].values
        # Should have mix of Tumor, Stroma, and possibly Interface
        unique_vals = set(values)
        assert "Tumor" in unique_vals

    def test_invalid_cutoff_raises(self, mock_adata):
        with pytest.raises(ValueError, match="between 0 and 1"):
            identify_interface(mock_adata, malignant_cutoff=1.5)

    def test_missing_malignant_raises(self, mock_adata):
        with pytest.raises(ValueError, match="not found"):
            identify_interface(mock_adata, malignant="NonExistent")
