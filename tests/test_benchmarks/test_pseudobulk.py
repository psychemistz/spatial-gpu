"""Tests for pseudobulk benchmark utilities."""

from __future__ import annotations

import numpy as np
import pytest

from spatialgpu.benchmarks.pseudobulk import generate_semi_synthetic_scrna


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
