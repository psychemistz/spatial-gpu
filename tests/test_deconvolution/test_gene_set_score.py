"""Tests for gene set scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution.gene_set_score import gene_set_score


class TestGeneSetScore:
    @pytest.fixture
    def adata_simple(self):
        import anndata as ad

        np.random.seed(42)
        n_spots, n_genes = 30, 200
        X = np.random.poisson(5, (n_spots, n_genes)).astype(np.float64)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        adata = ad.AnnData(X=sparse.csr_matrix(X))
        adata.var_names = pd.Index(gene_names)
        adata.obs_names = pd.Index([f"spot_{i}" for i in range(n_spots)])
        return adata

    def test_custom_gene_sets(self, adata_simple):
        gene_sets = {
            "SetA": ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"],
            "SetB": ["GENE10", "GENE11", "GENE12"],
        }
        result = gene_set_score(adata_simple, gene_sets)
        assert "spacet" in result.uns
        assert "GeneSetScore" in result.uns["spacet"]
        scores = result.uns["spacet"]["GeneSetScore"]
        assert scores.shape == (2, 30)
        assert "SetA" in scores.index
        assert "SetB" in scores.index

    def test_scores_range(self, adata_simple):
        gene_sets = {
            "SetA": ["GENE0", "GENE1", "GENE2"],
        }
        result = gene_set_score(adata_simple, gene_sets)
        scores = result.uns["spacet"]["GeneSetScore"]
        # UCell scores should be in [0, 1]
        assert scores.values.min() >= 0
        assert scores.values.max() <= 1

    def test_empty_gene_set(self, adata_simple):
        gene_sets = {
            "Empty": ["NONEXISTENT_GENE1", "NONEXISTENT_GENE2"],
        }
        result = gene_set_score(adata_simple, gene_sets)
        scores = result.uns["spacet"]["GeneSetScore"]
        # All zeros for non-existent genes
        np.testing.assert_array_equal(scores.loc["Empty"].values, 0)

    def test_invalid_builtin_raises(self, adata_simple):
        with pytest.raises(ValueError, match="must be a dict"):
            gene_set_score(adata_simple, "InvalidName")

    def test_accumulation(self, adata_simple):
        gs1 = {"SetA": ["GENE0", "GENE1"]}
        gs2 = {"SetB": ["GENE10", "GENE11"]}
        result = gene_set_score(adata_simple, gs1)
        result = gene_set_score(result, gs2)
        scores = result.uns["spacet"]["GeneSetScore"]
        assert "SetA" in scores.index
        assert "SetB" in scores.index
