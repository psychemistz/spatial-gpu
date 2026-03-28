"""Tests for pseudobulk benchmark utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialgpu.benchmarks.pseudobulk import (
    generate_pseudobulk_dirichlet,
    generate_semi_synthetic_scrna,
)


class TestGenerateSemiSyntheticScrna:
    """Test semi-synthetic scRNA-seq generation."""

    @pytest.fixture(scope="class")
    def scrna_with_mal(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=True, cancer_type="BRCA", seed=42
        )

    @pytest.fixture(scope="class")
    def scrna_no_mal(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=False, seed=42
        )

    def test_returns_anndata(self, scrna_with_mal):
        import anndata as ad

        assert isinstance(scrna_with_mal, ad.AnnData)

    def test_cell_type_column_exists(self, scrna_with_mal):
        assert "cell_type" in scrna_with_mal.obs.columns

    def test_correct_n_cells_per_type(self, scrna_with_mal):
        counts = scrna_with_mal.obs["cell_type"].value_counts()
        assert all(counts == 50)

    def test_includes_malignant_type(self, scrna_with_mal):
        types = set(scrna_with_mal.obs["cell_type"])
        assert "Malignant_BRCA" in types

    def test_excludes_malignant_when_off(self, scrna_no_mal):
        types = set(scrna_no_mal.obs["cell_type"])
        assert not any("Malignant" in t for t in types)

    def test_has_level1_cell_types(self, scrna_with_mal):
        types = set(scrna_with_mal.obs["cell_type"])
        expected = {"CAF", "Endothelial", "B cell", "T CD4", "T CD8", "Macrophage"}
        assert expected.issubset(types)

    def test_counts_are_nonneg_integers(self, scrna_with_mal):
        X = scrna_with_mal.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert np.all(X >= 0)
        assert np.allclose(X, X.astype(int))

    def test_has_dropout(self, scrna_with_mal):
        """Substantial fraction of entries should be zero (dropout)."""
        X = scrna_with_mal.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        zero_frac = (X == 0).sum() / X.size
        assert zero_frac > 0.3

    def test_deterministic_with_seed(self):
        a = generate_semi_synthetic_scrna(n_cells_per_type=20, seed=99)
        b = generate_semi_synthetic_scrna(n_cells_per_type=20, seed=99)
        Xa = a.X.toarray() if hasattr(a.X, "toarray") else a.X
        Xb = b.X.toarray() if hasattr(b.X, "toarray") else b.X
        np.testing.assert_array_equal(Xa, Xb)

    def test_gene_names_from_reference(self, scrna_with_mal):
        from spatialgpu.deconvolution.reference import load_comb_ref

        ref = load_comb_ref()
        ref_genes = set(ref["refProfiles"].index)
        scrna_genes = set(scrna_with_mal.var_names)
        assert scrna_genes.issubset(ref_genes)


class TestGeneratePseudobulkDirichlet:
    @pytest.fixture(scope="class")
    def scrna(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=True, seed=42
        )

    @pytest.fixture(scope="class")
    def bulk_and_truth(self, scrna):
        return generate_pseudobulk_dirichlet(
            scrna, n_samples=20, n_cells_per_sample=200, alpha=1.0, seed=42
        )

    def test_returns_tuple(self, bulk_and_truth):
        adata_bulk, ground_truth = bulk_and_truth
        import anndata as ad

        assert isinstance(adata_bulk, ad.AnnData)
        assert isinstance(ground_truth, pd.DataFrame)

    def test_bulk_shape(self, bulk_and_truth, scrna):
        adata_bulk, _ = bulk_and_truth
        assert adata_bulk.n_obs == 20
        assert adata_bulk.n_vars == scrna.n_vars

    def test_ground_truth_shape(self, bulk_and_truth):
        _, gt = bulk_and_truth
        assert gt.shape[0] == 20
        assert gt.shape[1] > 5

    def test_ground_truth_sums_to_one(self, bulk_and_truth):
        _, gt = bulk_and_truth
        np.testing.assert_allclose(gt.sum(axis=1).values, 1.0, atol=1e-10)

    def test_ground_truth_nonnegative(self, bulk_and_truth):
        _, gt = bulk_and_truth
        assert gt.min().min() >= 0.0

    def test_bulk_counts_nonneg_integers(self, bulk_and_truth):
        adata_bulk, _ = bulk_and_truth
        X = adata_bulk.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert np.all(X >= 0)
        assert np.allclose(X, X.astype(int))

    def test_deterministic_with_seed(self, scrna):
        _, gt1 = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=77)
        _, gt2 = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=77)
        np.testing.assert_array_equal(gt1.values, gt2.values)

    def test_cell_type_columns_match_scrna(self, bulk_and_truth, scrna):
        _, gt = bulk_and_truth
        scrna_types = set(scrna.obs["cell_type"].unique())
        gt_types = set(gt.columns)
        assert gt_types == scrna_types
