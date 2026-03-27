"""End-to-end GPU pipeline tests.

Runs core pipeline functions with GPU backend and compares
output against CPU backend to verify numerical equivalence.
"""

import numpy as np
import pytest


def gpu_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not gpu_available(), reason="No GPU available")


@skipno_gpu
class TestGPUDeconvolutionEquivalence:

    def test_nnls_solver_equivalence(self):
        """GPU NNLS solver matches CPU for synthetic deconvolution."""
        import spatialgpu as sp
        from spatialgpu.deconvolution.core import _solve_nnls

        rng = np.random.RandomState(42)
        n_genes, n_cell, n_spots = 100, 8, 50
        A = np.abs(rng.randn(n_genes, n_cell))
        B = np.abs(rng.randn(n_genes, n_spots))
        theta_sum = np.full(n_spots, 0.8)
        pp_max_arr = np.full(n_spots, 0.9)

        sp.set_backend("cpu")
        cpu_result = _solve_nnls(A, B, n_cell, theta_sum, pp_max_arr)

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_result = _solve_nnls(A, B, n_cell, theta_sum, pp_max_arr)
            np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-4)
        sp.set_backend("auto")

    def test_cormat_equivalence(self):
        """GPU cormat matches CPU."""
        import spatialgpu as sp
        from spatialgpu.deconvolution.core import cormat

        rng = np.random.RandomState(42)
        X = rng.randn(200, 100)
        Y = rng.randn(200, 1)

        sp.set_backend("cpu")
        cpu_result = cormat(X, Y, method="spearman")

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_result = cormat(X, Y, method="spearman")
            np.testing.assert_allclose(
                gpu_result["cor_r"].values, cpu_result["cor_r"].values, atol=1e-3
            )
        sp.set_backend("auto")

    def test_pairwise_spearmanr_equivalence(self):
        """GPU pairwise Spearman matches CPU."""
        import spatialgpu as sp
        from spatialgpu.deconvolution.interaction import _pairwise_spearmanr

        rng = np.random.RandomState(42)
        mat = rng.randn(10, 200)

        sp.set_backend("cpu")
        cpu_rho, cpu_pval = _pairwise_spearmanr(mat)

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_rho, gpu_pval = _pairwise_spearmanr(mat)
            np.testing.assert_allclose(gpu_rho, cpu_rho, atol=1e-5)
            np.testing.assert_allclose(gpu_pval, cpu_pval, atol=1e-4)
        sp.set_backend("auto")


@skipno_gpu
class TestGPUGeneSetScoreEquivalence:

    def test_ucell_equivalence(self):
        """GPU UCell scoring matches CPU."""
        import anndata as ad
        import spatialgpu as sp
        from spatialgpu.deconvolution.gene_set_score import _ucell_score

        rng = np.random.RandomState(42)
        X = rng.randint(0, 100, (200, 500)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = [f"Gene{i}" for i in range(500)]
        adata.obs_names = [f"Spot{i}" for i in range(200)]

        gene_sets = {"TestSet": [f"Gene{i}" for i in range(10)]}

        sp.set_backend("cpu")
        cpu_scores = _ucell_score(adata, gene_sets)

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_scores = _ucell_score(adata, gene_sets)
            np.testing.assert_allclose(gpu_scores.values, cpu_scores.values, atol=1e-5)
        sp.set_backend("auto")
