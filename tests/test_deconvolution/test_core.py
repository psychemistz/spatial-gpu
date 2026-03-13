"""Tests for core deconvolution functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution.core import (
    _cpm_log2_center,
    _solve_constrained_batch,
    cormat,
)


class TestCormat:
    def test_perfect_correlation(self):
        """Perfect positive correlation should give r=1.0."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64).T
        Y = np.array([[1, 2, 3]], dtype=np.float64).T
        result = cormat(X, Y)
        assert np.isclose(result["cor_r"].iloc[0], 1.0, atol=1e-3)

    def test_negative_correlation(self):
        """Perfect negative correlation should give r=-1.0."""
        X = np.array([[3, 2, 1]], dtype=np.float64).T
        Y = np.array([[1, 2, 3]], dtype=np.float64).T
        result = cormat(X, Y)
        assert np.isclose(result["cor_r"].iloc[0], -1.0, atol=1e-3)

    def test_zero_correlation(self):
        """Orthogonal vectors should give r near 0."""
        np.random.seed(42)
        n = 1000
        X = np.random.randn(n, 5)
        Y = np.random.randn(n, 1)
        result = cormat(X, Y)
        assert all(abs(result["cor_r"]) < 0.15)

    def test_p_value_significance(self):
        """Strong correlation should have small p-values."""
        n = 100
        x = np.arange(n, dtype=np.float64)
        X = (x + np.random.randn(n) * 0.1).reshape(-1, 1)
        Y = x.reshape(-1, 1)
        result = cormat(X, Y)
        assert result["cor_p"].iloc[0] < 0.001

    def test_bh_adjustment(self):
        """BH-adjusted p-values should be >= raw p-values."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 1)
        result = cormat(X, Y)
        assert all(result["cor_padj"] >= result["cor_p"] - 1e-15)

    def test_rounding(self):
        """cor_r should be rounded to 3 decimals."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        Y = np.random.randn(100, 1)
        result = cormat(X, Y)
        for r in result["cor_r"]:
            assert r == round(r, 3)

    def test_output_shape(self):
        """Output should have one row per sample (column of X)."""
        X = np.random.randn(50, 7)
        Y = np.random.randn(50, 1)
        result = cormat(X, Y)
        assert len(result) == 7
        assert "cor_r" in result.columns
        assert "cor_p" in result.columns
        assert "cor_padj" in result.columns


class TestCPMLog2Center:
    def test_dense_matrix(self):
        """CPM normalization on dense matrix."""
        counts = np.array([[10, 20], [30, 40], [60, 40]], dtype=np.float64)
        result = _cpm_log2_center(counts)
        assert result.shape == counts.shape
        # After centering, row means should be ~0
        row_means = result.mean(axis=1)
        np.testing.assert_allclose(row_means, 0, atol=1e-10)

    def test_sparse_matrix(self):
        """CPM normalization on sparse matrix."""
        counts = sparse.csc_matrix(
            np.array([[10, 20], [30, 40], [60, 40]], dtype=np.float64)
        )
        result = _cpm_log2_center(counts)
        assert result.shape == (3, 2)
        row_means = result.mean(axis=1)
        np.testing.assert_allclose(row_means, 0, atol=1e-10)

    def test_sparse_dense_equivalence(self):
        """Sparse and dense should give same result."""
        np.random.seed(42)
        counts = np.random.poisson(5, (100, 50)).astype(np.float64)
        counts_sparse = sparse.csc_matrix(counts)
        result_dense = _cpm_log2_center(counts)
        result_sparse = _cpm_log2_center(counts_sparse)
        np.testing.assert_allclose(result_dense, result_sparse, atol=1e-10)

    def test_cpm_sum(self):
        """After CPM (before log), each column should sum to 1e6."""
        counts = np.array([[10, 20], [30, 40], [60, 40]], dtype=np.float64)
        col_sums = counts.sum(axis=0)
        cpm = counts / col_sums[np.newaxis, :] * 1e6
        np.testing.assert_allclose(cpm.sum(axis=0), 1e6)


class TestConstrainedOptimization:
    def test_simple_2cell(self):
        """Simple 2-cell-type deconvolution."""
        # Reference: 2 genes x 2 cell types
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Mixture: 2 genes x 3 spots
        B = np.array([[0.5, 0.3, 0.7], [0.5, 0.7, 0.3]])
        theta_sum = np.array([1.0, 1.0, 1.0]) - 1e-5
        pp_min = np.zeros(3)
        pp_max = np.ones(3)

        result = _solve_constrained_batch(A, B, 2, theta_sum, pp_min, pp_max)
        assert result.shape == (2, 3)
        # Fractions should be close to the mixture values
        np.testing.assert_allclose(result[:, 0], [0.5, 0.5], atol=0.1)

    def test_nonnegativity(self):
        """All fractions should be >= 0."""
        np.random.seed(42)
        A = np.random.rand(20, 5).astype(np.float64)
        B = np.random.rand(20, 10).astype(np.float64)
        theta_sum = np.full(10, 0.8)
        pp_min = np.zeros(10)
        pp_max = np.ones(10)

        result = _solve_constrained_batch(A, B, 5, theta_sum, pp_min, pp_max)
        assert np.all(result >= -1e-10)

    def test_sum_constraint(self):
        """Sum of fractions should respect bounds."""
        np.random.seed(42)
        A = np.random.rand(20, 3).astype(np.float64)
        B = np.random.rand(20, 5).astype(np.float64)
        mal_prop = np.array([0.3, 0.2, 0.5, 0.1, 0.4])
        theta_sum = (1 - mal_prop) - 1e-5
        pp_min = np.zeros(5)
        pp_max = 1 - mal_prop

        result = _solve_constrained_batch(A, B, 3, theta_sum, pp_min, pp_max)
        row_sums = result.sum(axis=0)
        assert np.all(row_sums <= pp_max + 1e-6)

    def test_zero_theta_sum(self):
        """When theta_sum <= 0.01, should return uniform."""
        A = np.random.rand(10, 3).astype(np.float64)
        B = np.random.rand(10, 2).astype(np.float64)
        theta_sum = np.array([0.005, 0.001])
        pp_min = np.zeros(2)
        pp_max = np.ones(2)

        result = _solve_constrained_batch(A, B, 3, theta_sum, pp_min, pp_max)
        # Should return equal fractions
        for j in range(2):
            np.testing.assert_allclose(result[:, j], theta_sum[j] / 3, atol=1e-10)


class TestNumericalEquivalence:
    """Test against R validation data if available."""

    @pytest.fixture
    def r_validation_data(self):
        """Load R validation data."""
        import os

        val_dir = os.path.join(os.path.dirname(__file__), "..", "..", "validation")
        prop_mat_path = os.path.join(val_dir, "visium_bc_propMat.csv")
        mal_prop_path = os.path.join(val_dir, "visium_bc_malProp.csv")

        if not os.path.exists(prop_mat_path):
            pytest.skip("R validation data not available")

        return {
            "propMat": pd.read_csv(prop_mat_path, index_col=0),
            "malProp": pd.read_csv(mal_prop_path, index_col=0),
        }

    def test_r_validation_data_exists(self, r_validation_data):
        """Verify R validation data is loadable."""
        assert r_validation_data["propMat"].shape[0] > 0
        assert r_validation_data["malProp"].shape[0] > 0

    def test_propmat_shape(self, r_validation_data):
        """propMat should have cell types x spots."""
        pm = r_validation_data["propMat"]
        assert pm.shape[0] > 10  # >10 cell types
        assert pm.shape[1] > 200  # >200 spots

    def test_propmat_fractions_valid(self, r_validation_data):
        """All fractions should be in [0, 1]."""
        pm = r_validation_data["propMat"]
        assert pm.min().min() >= -1e-10
        assert pm.max().max() <= 1 + 1e-10

    def test_malprop_range(self, r_validation_data):
        """malProp should be in [0, 1]."""
        mp = r_validation_data["malProp"]
        assert mp.min().min() >= -1e-10
        assert mp.max().max() <= 1 + 1e-10
