"""Tests for I/O utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution.io import (
    create_spacet_object,
    quality_control,
)


class TestCreateSpacetObject:
    @pytest.fixture
    def simple_data(self):
        """Simple count matrix and coordinates."""
        np.random.seed(42)
        n_genes, n_spots = 100, 50
        counts = np.random.poisson(5, (n_genes, n_spots)).astype(np.float64)
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
        spot_names = [
            f"{r}x{c}" for r, c in zip(range(n_spots), range(0, n_spots * 2, 2))
        ]
        counts_df = pd.DataFrame(counts, index=gene_names, columns=spot_names)
        coords = pd.DataFrame(
            {"X": np.random.rand(n_spots) * 1000, "Y": np.random.rand(n_spots) * 1000},
            index=spot_names,
        )
        return counts_df, coords

    def test_basic_creation(self, simple_data):
        counts_df, coords = simple_data
        adata = create_spacet_object(counts_df, coords, "Visium")
        assert adata.n_obs == 50
        assert adata.n_vars == 100
        assert "spatial" in adata.obsm
        assert adata.uns["spacet_platform"] == "Visium"

    def test_sparse_input(self, simple_data):
        counts_df, coords = simple_data
        sparse.csc_matrix(counts_df.values)
        # Need to pass gene names via DataFrame
        adata = create_spacet_object(counts_df, coords, "Visium")
        assert sparse.issparse(adata.X)

    def test_spot_order_preserved(self, simple_data):
        counts_df, coords = simple_data
        adata = create_spacet_object(counts_df, coords, "Visium")
        np.testing.assert_array_equal(adata.obs_names, coords.index)

    def test_spatial_coordinates(self, simple_data):
        counts_df, coords = simple_data
        adata = create_spacet_object(counts_df, coords, "Visium")
        np.testing.assert_allclose(adata.obsm["spatial"][:, 0], coords["X"].values)
        np.testing.assert_allclose(adata.obsm["spatial"][:, 1], coords["Y"].values)

    def test_mismatched_spots_raises(self, simple_data):
        counts_df, coords = simple_data
        coords_bad = coords.copy()
        coords_bad.index = [f"bad_{i}" for i in range(len(coords))]
        with pytest.raises(ValueError, match="not identical"):
            create_spacet_object(counts_df, coords_bad, "Visium")


class TestQualityControl:
    @pytest.fixture
    def adata_with_zeros(self):
        import anndata as ad

        np.random.seed(42)
        n_spots, n_genes = 20, 50
        X = np.random.poisson(3, (n_spots, n_genes)).astype(np.float64)
        # Make some spots have zero genes
        X[0, :] = 0
        X[1, :] = 0
        adata = ad.AnnData(X=sparse.csr_matrix(X))
        adata.obs_names = pd.Index([f"spot_{i}" for i in range(n_spots)])
        adata.var_names = pd.Index([f"gene_{i}" for i in range(n_genes)])
        return adata

    def test_removes_zero_spots(self, adata_with_zeros):
        result = quality_control(adata_with_zeros, min_genes=1)
        assert result.n_obs == 18  # 2 spots removed

    def test_umi_and_gene_counts(self, adata_with_zeros):
        result = quality_control(adata_with_zeros, min_genes=1)
        assert "UMI" in result.obs.columns
        assert "Gene" in result.obs.columns
        assert all(result.obs["Gene"] >= 1)

    def test_min_genes_filter(self, adata_with_zeros):
        result = quality_control(adata_with_zeros, min_genes=30)
        # More spots should be filtered with higher threshold
        assert result.n_obs <= 18
