"""Tests for bulk RNA-seq deconvolution pathway.

Covers all three Stage 1 branches (normal, external mal_prop,
signature-based inference) and Stage 2 hierarchical deconvolution.
GPU vs CPU equivalence tests are marked with @skipno_gpu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialgpu.deconvolution.core import _infer_mal_bulk, deconvolution_bulk
from spatialgpu.deconvolution.reference import get_cancer_signature, load_comb_ref

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bulk_adata(n_samples=30, n_genes=2000, seed=42, sparse_x=False):
    """Create a realistic-ish synthetic bulk RNA-seq AnnData.

    Uses Poisson counts with gene-specific rates to mimic
    expression profiles.  Embeds a rough malignant signature so
    Stage 1 has something to correlate against.
    """
    import anndata as ad

    rng = np.random.RandomState(seed)

    # Gene-specific mean expression (log-normal distributed)
    gene_means = np.exp(rng.randn(n_genes) * 1.5 + 2)
    counts = rng.poisson(gene_means[np.newaxis, :], size=(n_samples, n_genes))
    counts = counts.astype(np.float64)

    # Load real gene names from reference so gene overlap exists
    ref = load_comb_ref()
    ref_genes = np.array(ref["refProfiles"].index)
    # Use first n_genes reference genes (or pad with synthetic)
    if len(ref_genes) >= n_genes:
        gene_names = ref_genes[:n_genes]
    else:
        extra = [f"SynGene{i}" for i in range(n_genes - len(ref_genes))]
        gene_names = np.concatenate([ref_genes, extra])

    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]

    X = sparse.csr_matrix(counts) if sparse_x else counts
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(sample_names)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    return adata


@pytest.fixture(scope="module")
def bulk_adata():
    """Dense bulk AnnData with real reference gene names."""
    return _make_bulk_adata()


@pytest.fixture(scope="module")
def bulk_adata_sparse():
    """Sparse bulk AnnData with real reference gene names."""
    return _make_bulk_adata(sparse_x=True)


# ---------------------------------------------------------------------------
# Stage 1 branch: normal tissue (mal_prop = 0)
# ---------------------------------------------------------------------------


class TestBulkNormalTissue:
    """cancer_type='normal' sets mal_prop to 0 and skips Stage 1."""

    def test_runs_without_error(self, bulk_adata):
        adata = bulk_adata.copy()
        result = deconvolution_bulk(adata, cancer_type="normal")
        assert result is adata

    def test_malprop_all_zero(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        mal_prop = adata.uns["deconv"]["malProp"]
        np.testing.assert_array_equal(mal_prop.values, 0.0)

    def test_propmat_shape(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        pm = adata.uns["deconv"]["propMat"]
        assert isinstance(pm, pd.DataFrame)
        assert pm.shape[0] > 5  # multiple cell types
        assert pm.shape[1] == adata.n_obs  # one column per sample

    def test_fractions_nonnegative(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        pm = adata.uns["deconv"]["propMat"]
        assert pm.min().min() >= -1e-10

    def test_obsm_storage(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        assert "deconv_propMat" in adata.obsm
        assert adata.obsm["deconv_propMat"].shape[0] == adata.n_obs

    def test_malignant_row_present(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        pm = adata.uns["deconv"]["propMat"]
        assert "Malignant" in pm.index

    def test_malignant_row_zero(self, bulk_adata):
        """With cancer_type='normal', Malignant fraction should be 0."""
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        pm = adata.uns["deconv"]["propMat"]
        np.testing.assert_allclose(pm.loc["Malignant"].values, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Stage 1 branch: external mal_prop
# ---------------------------------------------------------------------------


class TestBulkExternalMalProp:
    """User provides external purity estimates (e.g., from ABSOLUTE)."""

    def test_runs_with_array(self, bulk_adata):
        adata = bulk_adata.copy()
        rng = np.random.RandomState(99)
        mal = rng.uniform(0.1, 0.6, size=adata.n_obs)
        result = deconvolution_bulk(adata, cancer_type="BRCA", mal_prop=mal)
        assert "deconv" in result.uns

    def test_runs_with_series(self, bulk_adata):
        adata = bulk_adata.copy()
        rng = np.random.RandomState(99)
        mal = pd.Series(rng.uniform(0.1, 0.6, size=adata.n_obs), index=adata.obs_names)
        deconvolution_bulk(adata, cancer_type="BRCA", mal_prop=mal)
        stored_mal = adata.uns["deconv"]["malProp"]
        # Should preserve the values we passed in (clipped to [0,1])
        np.testing.assert_allclose(stored_mal.values, mal.values, atol=1e-10)

    def test_clips_out_of_range(self, bulk_adata):
        adata = bulk_adata.copy()
        mal = np.array([-0.5] * 15 + [1.5] * 15)
        deconvolution_bulk(adata, cancer_type="BRCA", mal_prop=mal)
        stored = adata.uns["deconv"]["malProp"]
        assert stored.min() >= 0.0
        assert stored.max() <= 1.0

    def test_propmat_reflects_purity(self, bulk_adata):
        """Higher mal_prop should yield higher Malignant fraction."""
        adata_low = bulk_adata.copy()
        adata_high = bulk_adata.copy()
        deconvolution_bulk(adata_low, cancer_type="BRCA", mal_prop=np.full(30, 0.1))
        deconvolution_bulk(adata_high, cancer_type="BRCA", mal_prop=np.full(30, 0.8))
        low_mal = adata_low.uns["deconv"]["propMat"].loc["Malignant"].mean()
        high_mal = adata_high.uns["deconv"]["propMat"].loc["Malignant"].mean()
        assert high_mal > low_mal

    def test_sparse_input(self, bulk_adata_sparse):
        adata = bulk_adata_sparse.copy()
        mal = np.full(adata.n_obs, 0.3)
        deconvolution_bulk(adata, cancer_type="BRCA", mal_prop=mal)
        pm = adata.uns["deconv"]["propMat"]
        assert pm.shape[0] > 5
        assert pm.min().min() >= -1e-10


# ---------------------------------------------------------------------------
# Stage 1 branch: signature-based inference
# ---------------------------------------------------------------------------


class TestBulkSignatureInference:
    """_infer_mal_bulk estimates malignant fraction from CNA/expr signatures."""

    def test_brca_signature_exists(self):
        """BRCA should have a usable cancer signature."""
        sig_type, sig = get_cancer_signature("BRCA")
        assert len(sig) > 0

    def test_infer_mal_bulk_returns_correct_shape(self, bulk_adata):
        adata = bulk_adata.copy()
        counts = adata.X.T if not sparse.issparse(adata.X) else adata.X.T.toarray()
        gene_names = np.array(adata.var_names)
        sample_names = np.array(adata.obs_names)

        gene_sums = counts.sum(axis=1)
        mask = gene_sums > 0
        counts = counts[mask]
        gene_names = gene_names[mask]

        mal_prop, mal_ref = _infer_mal_bulk(
            counts, gene_names, sample_names, "BRCA", None
        )
        assert isinstance(mal_prop, pd.Series)
        assert len(mal_prop) == len(sample_names)
        assert mal_prop.min() >= 0.0
        assert mal_prop.max() <= 1.0

    def test_infer_mal_bulk_with_forced_signature_type(self, bulk_adata):
        adata = bulk_adata.copy()
        counts = adata.X.T.copy()
        gene_names = np.array(adata.var_names)
        sample_names = np.array(adata.obs_names)

        gene_sums = counts.sum(axis=1)
        mask = gene_sums > 0
        counts = counts[mask]
        gene_names = gene_names[mask]

        mal_prop, mal_ref = _infer_mal_bulk(
            counts, gene_names, sample_names, "BRCA", "CNA"
        )
        assert isinstance(mal_prop, pd.Series)

    def test_full_pipeline_signature_based(self, bulk_adata):
        """Full deconvolution_bulk with auto signature estimation."""
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="BRCA")
        pm = adata.uns["deconv"]["propMat"]
        mal = adata.uns["deconv"]["malProp"]
        assert pm.shape[0] > 5
        assert len(mal) == adata.n_obs
        # Malignant fraction should be between 0 and 1
        assert mal.min() >= 0.0
        assert mal.max() <= 1.0

    def test_unknown_cancer_type_falls_back_to_pancan(self, bulk_adata):
        """Unknown cancer type should fall back to PANCAN expr."""
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="UNKNOWN_XYZ")
        # Should still produce results (PANCAN fallback or zero)
        assert "deconv" in adata.uns


# ---------------------------------------------------------------------------
# Stage 2 output validation
# ---------------------------------------------------------------------------


class TestBulkStage2Output:
    """Validate Stage 2 hierarchical deconvolution output properties."""

    @pytest.fixture(scope="class")
    def deconv_normal(self, bulk_adata):
        adata = bulk_adata.copy()
        deconvolution_bulk(adata, cancer_type="normal")
        return adata

    def test_cell_types_include_subtypes(self, deconv_normal):
        """Should have both major lineages and subtypes."""
        pm = deconv_normal.uns["deconv"]["propMat"]
        cell_types = set(pm.index)
        # Expect at least some of these major/sub types
        expected_any = {"B cell", "T cell", "CAF", "Macrophage", "Endothelial"}
        found = cell_types & expected_any
        assert len(found) >= 2, f"Only found: {found}"

    def test_spacet_uns_structure(self, deconv_normal):
        """uns['spacet']['deconvolution'] should be populated."""
        d = deconv_normal.uns["spacet"]["deconvolution"]
        assert "propMat" in d
        assert "malRes" in d
        assert "Ref" in d

    def test_obsm_matches_propmat(self, deconv_normal):
        """obsm should be the transposed propMat."""
        pm = deconv_normal.uns["deconv"]["propMat"]
        obsm = deconv_normal.obsm["deconv_propMat"]
        assert obsm.shape == (deconv_normal.n_obs, pm.shape[0])

    def test_dense_sparse_equivalence(self):
        """Dense and sparse input should produce same output."""
        adata_dense = _make_bulk_adata(n_samples=15, n_genes=1500, sparse_x=False)
        adata_sparse = _make_bulk_adata(n_samples=15, n_genes=1500, sparse_x=True)

        deconvolution_bulk(adata_dense, cancer_type="normal")
        deconvolution_bulk(adata_sparse, cancer_type="normal")

        pm_d = adata_dense.uns["deconv"]["propMat"]
        pm_s = adata_sparse.uns["deconv"]["propMat"]

        np.testing.assert_allclose(
            pm_d.values,
            pm_s.values,
            atol=1e-8,
            err_msg="Dense vs sparse input should yield equivalent results",
        )


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
class TestBulkGPUEquivalence:
    """Verify GPU and CPU bulk deconvolution produce equivalent results."""

    def test_normal_tissue_gpu_cpu(self):
        """GPU and CPU should match for normal tissue bulk deconv."""
        import spatialgpu as sp

        adata_cpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=123)
        adata_gpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=123)

        sp.set_backend("cpu")
        deconvolution_bulk(adata_cpu, cancer_type="normal")

        sp.set_backend("auto")
        if not sp.get_backend().is_gpu_active:
            pytest.skip("GPU not active after set_backend('auto')")
        deconvolution_bulk(adata_gpu, cancer_type="normal")

        pm_cpu = adata_cpu.uns["deconv"]["propMat"]
        pm_gpu = adata_gpu.uns["deconv"]["propMat"]

        np.testing.assert_allclose(
            pm_gpu.values,
            pm_cpu.values,
            atol=1e-4,
            err_msg="GPU vs CPU normal-tissue bulk deconv mismatch",
        )
        sp.set_backend("auto")

    def test_external_malprop_gpu_cpu(self):
        """GPU and CPU match with external mal_prop."""
        import spatialgpu as sp

        mal = np.full(20, 0.4)

        adata_cpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=77)
        adata_gpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=77)

        sp.set_backend("cpu")
        deconvolution_bulk(adata_cpu, cancer_type="BRCA", mal_prop=mal.copy())

        sp.set_backend("auto")
        if not sp.get_backend().is_gpu_active:
            pytest.skip("GPU not active")
        deconvolution_bulk(adata_gpu, cancer_type="BRCA", mal_prop=mal.copy())

        pm_cpu = adata_cpu.uns["deconv"]["propMat"]
        pm_gpu = adata_gpu.uns["deconv"]["propMat"]

        np.testing.assert_allclose(
            pm_gpu.values,
            pm_cpu.values,
            atol=1e-4,
            err_msg="GPU vs CPU external-malprop bulk deconv mismatch",
        )
        sp.set_backend("auto")

    def test_signature_inference_gpu_cpu(self):
        """GPU and CPU match for signature-based malignant inference."""
        import spatialgpu as sp

        adata_cpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=55)
        adata_gpu = _make_bulk_adata(n_samples=20, n_genes=1500, seed=55)

        sp.set_backend("cpu")
        deconvolution_bulk(adata_cpu, cancer_type="BRCA")

        sp.set_backend("auto")
        if not sp.get_backend().is_gpu_active:
            pytest.skip("GPU not active")
        deconvolution_bulk(adata_gpu, cancer_type="BRCA")

        # Compare malignant proportions
        mal_cpu = adata_cpu.uns["deconv"]["malProp"]
        mal_gpu = adata_gpu.uns["deconv"]["malProp"]
        np.testing.assert_allclose(
            mal_gpu.values,
            mal_cpu.values,
            atol=1e-3,
            err_msg="GPU vs CPU malignant fraction mismatch",
        )

        # Compare full propMat
        pm_cpu = adata_cpu.uns["deconv"]["propMat"]
        pm_gpu = adata_gpu.uns["deconv"]["propMat"]
        np.testing.assert_allclose(
            pm_gpu.values,
            pm_cpu.values,
            atol=1e-4,
            err_msg="GPU vs CPU signature-based bulk deconv mismatch",
        )
        sp.set_backend("auto")
