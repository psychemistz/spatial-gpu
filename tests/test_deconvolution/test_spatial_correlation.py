"""Tests for spatial correlation (Moran's I)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution.spatial_correlation import cal_weights, spatial_correlation


class TestCalWeights:
    @pytest.fixture
    def adata_grid(self):
        """Create AnnData with grid spatial coordinates."""
        import anndata as ad

        n = 25  # 5x5 grid
        rows, cols = np.meshgrid(range(5), range(5))
        x_um = cols.ravel().astype(np.float64) * 100.0
        y_um = rows.ravel().astype(np.float64) * 100.0

        adata = ad.AnnData(
            X=np.random.poisson(5, (n, 50)).astype(np.float64),
            obs=pd.DataFrame(
                {"coordinate_x_um": x_um, "coordinate_y_um": y_um},
                index=[f"spot_{i}" for i in range(n)],
            ),
        )
        adata.var_names = pd.Index([f"gene_{i}" for i in range(50)])
        return adata

    def test_symmetric(self, adata_grid):
        W = cal_weights(adata_grid, radius=150, sigma=100)
        diff = W - W.T
        assert abs(diff).max() < 1e-10

    def test_zero_diagonal(self, adata_grid):
        W = cal_weights(adata_grid, radius=150, sigma=100)
        diag = W.diagonal()
        np.testing.assert_array_equal(diag, 0)

    def test_rbf_decay(self, adata_grid):
        """Closer spots should have higher weights."""
        W = cal_weights(adata_grid, radius=500, sigma=100)
        W_dense = W.toarray()
        # Spot 0 and spot 1 are distance 100 apart
        # Spot 0 and spot 5 are distance 100 apart
        # Spot 0 and spot 6 are distance ~141 apart
        w_close = W_dense[0, 1]  # distance 100
        w_far = W_dense[0, 6]  # distance ~141
        assert w_close > w_far

    def test_radius_cutoff(self, adata_grid):
        """No edges beyond radius."""
        W = cal_weights(adata_grid, radius=110, sigma=100)
        W_dense = W.toarray()
        # Corner spots (0,0) and (4,4) are far apart
        assert W_dense[0, 24] == 0

    def test_sparse_output(self, adata_grid):
        W = cal_weights(adata_grid, radius=150, sigma=100)
        assert sparse.issparse(W)

    def test_shape(self, adata_grid):
        W = cal_weights(adata_grid, radius=150, sigma=100)
        assert W.shape == (25, 25)


class TestSpatialCorrelation:
    @pytest.fixture
    def adata_spatial(self):
        """AnnData with spatially structured expression."""
        import anndata as ad

        np.random.seed(42)
        n = 100
        x_um = np.random.rand(n) * 1000.0
        y_um = np.random.rand(n) * 1000.0

        # Create spatially autocorrelated gene
        X = np.random.poisson(5, (n, 20)).astype(np.float64)
        # Gene 0: spatial gradient (correlated with x position)
        X[:, 0] = (x_um / 1000 * 20).astype(int)

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"coordinate_x_um": x_um, "coordinate_y_um": y_um},
                index=[f"s{i}" for i in range(n)],
            ),
        )
        adata.var_names = pd.Index([f"g{i}" for i in range(20)])
        return adata

    def test_univariate_output_format(self, adata_spatial):
        W = cal_weights(adata_spatial, radius=300, sigma=100)
        result = spatial_correlation(
            adata_spatial, mode="univariate",
            item=["g0", "g1"], W=W, n_permutation=100,
        )
        sc_res = result.uns["spacet"]["SpatialCorrelation"]["univariate"]
        assert isinstance(sc_res, pd.DataFrame)
        assert "p.Moran_I" in sc_res.columns
        assert "p.Moran_Z" in sc_res.columns
        assert "p.Moran_P" in sc_res.columns
        assert "p.Moran_Padj" in sc_res.columns

    def test_spatially_correlated_gene_detected(self, adata_spatial):
        """Spatially structured gene should have high Moran's I."""
        W = cal_weights(adata_spatial, radius=300, sigma=100)
        result = spatial_correlation(
            adata_spatial, mode="univariate",
            item=["g0"], W=W, n_permutation=100,
        )
        sc_res = result.uns["spacet"]["SpatialCorrelation"]["univariate"]
        # Gene with spatial gradient should have positive Moran's I
        assert sc_res.loc["g0", "p.Moran_I"] > 0

    def test_pairwise_output_format(self, adata_spatial):
        W = cal_weights(adata_spatial, radius=300, sigma=100)
        result = spatial_correlation(
            adata_spatial, mode="pairwise", W=W,
        )
        sc_res = result.uns["spacet"]["SpatialCorrelation"]["pairwise"]
        assert isinstance(sc_res, pd.DataFrame)
        # Should be square
        assert sc_res.shape[0] == sc_res.shape[1]

    def test_pairwise_diagonal(self, adata_spatial):
        """Diagonal of pairwise Moran's I should be positive (self-correlation)."""
        W = cal_weights(adata_spatial, radius=300, sigma=100)
        result = spatial_correlation(
            adata_spatial, mode="pairwise", W=W,
        )
        sc_res = result.uns["spacet"]["SpatialCorrelation"]["pairwise"]
        diag = np.diag(sc_res.values)
        # Most diagonal values should be positive (univariate Moran's I)
        assert np.mean(diag > 0) > 0.3

    def test_invalid_mode_raises(self, adata_spatial):
        with pytest.raises(ValueError, match="Invalid mode"):
            spatial_correlation(adata_spatial, mode="invalid")
