"""Tests for bulk RNA-seq deconvolution wrapper.

`deconvolution_bulk` is a thin wrapper around `deconvolution_matched_scrnaseq`:
each bulk sample is a mixture deconvolved jointly against a matched scRNA-seq
reference (no malignant-first cascade, no within-cohort purity rescaling).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution import (
    deconvolution_bulk,
    deconvolution_matched_scrnaseq,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_matched_data(
    n_samples=15, n_cells=300, n_genes=800, include_malignant=True, seed=42
):
    """Build a synthetic (bulk, scRNA-seq) pair sharing the same gene space."""
    import anndata as ad

    rng = np.random.RandomState(seed)

    gene_names = np.array([f"Gene_{i:04d}" for i in range(n_genes)])

    cell_types = ["T CD8", "B cell", "Macrophage", "CAF", "Endothelial"]
    if include_malignant:
        cell_types = cell_types + ["Malignant"]

    cells_per_type = n_cells // len(cell_types)
    type_rates = {ct: np.exp(rng.randn(n_genes) * 1.2 + 1.5) for ct in cell_types}

    sc_counts_list = []
    sc_labels = []
    sc_ids = []
    for ct in cell_types:
        base = type_rates[ct]
        cells = rng.poisson(base[np.newaxis, :], size=(cells_per_type, n_genes))
        sc_counts_list.append(cells)
        sc_labels.extend([ct] * cells_per_type)
        sc_ids.extend([f"{ct}_{i:04d}" for i in range(cells_per_type)])
    sc_counts = np.vstack(sc_counts_list).astype(np.float64).T  # genes x cells
    sc_counts_df = pd.DataFrame(sc_counts, index=gene_names, columns=sc_ids)
    sc_annotation = pd.DataFrame(
        {"cellID": sc_ids, "cellType": sc_labels}, index=sc_ids
    )

    bulk_counts = np.zeros((n_samples, n_genes))
    for s in range(n_samples):
        props = rng.dirichlet(np.ones(len(cell_types)))
        for j, ct in enumerate(cell_types):
            bulk_counts[s] += props[j] * type_rates[ct] * 1000
    bulk_counts = rng.poisson(bulk_counts).astype(np.float64)

    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    bulk_adata = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=pd.Index(sample_names)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )

    lineage_tree = {ct: [ct] for ct in cell_types}

    return bulk_adata, sc_counts_df, sc_annotation, lineage_tree


@pytest.fixture(scope="module")
def matched_data():
    return _make_matched_data()


@pytest.fixture(scope="module")
def matched_data_no_mal():
    return _make_matched_data(include_malignant=False, seed=43)


# ---------------------------------------------------------------------------
# Required-args contract
# ---------------------------------------------------------------------------


class TestBulkWrapperContract:
    def test_missing_sc_counts_raises(self, matched_data):
        adata, _, sc_ann, tree = matched_data
        with pytest.raises(TypeError):
            deconvolution_bulk(adata.copy(), sc_annotation=sc_ann, sc_lineage_tree=tree)

    def test_missing_sc_annotation_raises(self, matched_data):
        adata, sc_counts, _, tree = matched_data
        with pytest.raises(TypeError):
            deconvolution_bulk(adata.copy(), sc_counts=sc_counts, sc_lineage_tree=tree)

    def test_missing_lineage_tree_raises(self, matched_data):
        adata, sc_counts, sc_ann, _ = matched_data
        with pytest.raises(TypeError):
            deconvolution_bulk(
                adata.copy(), sc_counts=sc_counts, sc_annotation=sc_ann
            )


# ---------------------------------------------------------------------------
# Delegation parity with deconvolution_matched_scrnaseq
# ---------------------------------------------------------------------------


class TestBulkWrapperDelegation:
    """The bulk wrapper must produce results identical to a direct
    matched_scrnaseq call on the same inputs."""

    def test_output_matches_matched_scrnaseq(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data

        via_wrapper = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
            sc_include_malignant=True,
        )
        direct = deconvolution_matched_scrnaseq(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
            sc_include_malignant=True,
        )

        pm_w = via_wrapper.uns["spacet"]["deconvolution"]["propMat"]
        pm_d = direct.uns["spacet"]["deconvolution"]["propMat"]
        np.testing.assert_allclose(pm_w.values, pm_d.values, atol=1e-10)
        assert list(pm_w.index) == list(pm_d.index)
        assert list(pm_w.columns) == list(pm_d.columns)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestBulkWrapperOutput:
    def test_uns_spacet_populated(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data
        result = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )
        assert "spacet" in result.uns
        d = result.uns["spacet"]["deconvolution"]
        assert "propMat" in d

    def test_propmat_shape_and_nonneg(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data
        result = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )
        pm = result.uns["spacet"]["deconvolution"]["propMat"]
        assert pm.shape[1] == result.n_obs
        assert pm.shape[0] >= 5
        assert pm.min().min() >= -1e-8

    def test_obsm_present(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data
        result = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )
        assert "spacet_propMat" in result.obsm
        assert result.obsm["spacet_propMat"].shape[0] == result.n_obs

    def test_malignant_included_when_requested(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data
        result = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
            sc_include_malignant=True,
        )
        pm = result.uns["spacet"]["deconvolution"]["propMat"]
        assert "Malignant" in pm.index

    def test_no_malignant_when_not_in_reference(self, matched_data_no_mal):
        adata, sc_counts, sc_ann, tree = matched_data_no_mal
        result = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
            sc_include_malignant=True,
        )
        pm = result.uns["spacet"]["deconvolution"]["propMat"]
        assert "Malignant" not in pm.index


# ---------------------------------------------------------------------------
# Dense/sparse input equivalence
# ---------------------------------------------------------------------------


class TestBulkWrapperSparse:
    def test_dense_sparse_equivalence(self, matched_data):
        adata, sc_counts, sc_ann, tree = matched_data

        adata_sparse = adata.copy()
        adata_sparse.X = sparse.csr_matrix(adata_sparse.X)

        r_dense = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )
        r_sparse = deconvolution_bulk(
            adata_sparse,
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )
        pm_d = r_dense.uns["spacet"]["deconvolution"]["propMat"]
        pm_s = r_sparse.uns["spacet"]["deconvolution"]["propMat"]
        np.testing.assert_allclose(pm_d.values, pm_s.values, atol=1e-8)


# ---------------------------------------------------------------------------
# GPU vs CPU equivalence
# ---------------------------------------------------------------------------


def _gpu_available():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not _gpu_available(), reason="No GPU available")


@skipno_gpu
class TestBulkWrapperGPU:
    def test_gpu_cpu_match(self, matched_data):
        import spatialgpu as sp

        adata, sc_counts, sc_ann, tree = matched_data

        sp.set_backend("cpu")
        r_cpu = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )

        sp.set_backend("auto")
        if not sp.get_backend().is_gpu_active:
            pytest.skip("GPU not active after set_backend('auto')")
        r_gpu = deconvolution_bulk(
            adata.copy(),
            sc_counts=sc_counts,
            sc_annotation=sc_ann,
            sc_lineage_tree=tree,
        )

        pm_cpu = r_cpu.uns["spacet"]["deconvolution"]["propMat"]
        pm_gpu = r_gpu.uns["spacet"]["deconvolution"]["propMat"]
        np.testing.assert_allclose(pm_gpu.values, pm_cpu.values, atol=1e-4)
        sp.set_backend("auto")
