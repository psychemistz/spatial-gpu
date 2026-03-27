"""CPU-vs-GPU equivalence tests for spatialgpu.core.gpu_ops primitives."""

import numpy as np
import pytest
from scipy.stats import rankdata
from scipy.stats import spearmanr as scipy_spearmanr


def gpu_available():
    """Return True if a CUDA-capable GPU is accessible via CuPy."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not gpu_available(), reason="No GPU available")


class TestGPURankdata:
    """CPU-vs-GPU equivalence tests for gpu_rankdata."""

    @skipno_gpu
    def test_1d_average(self):
        """1-D array with ties: compare to scipy.stats.rankdata."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        data = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5], dtype=np.float64)
        expected = rankdata(data, method="average")

        result_gpu = gpu_rankdata(cp.asarray(data))
        result_cpu = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)

    @skipno_gpu
    def test_2d_columnwise(self):
        """2-D float32 array, axis=0: compare to np.apply_along_axis."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        rng = np.random.default_rng(42)
        data = rng.random((100, 20)).astype(np.float32)
        expected = np.apply_along_axis(
            lambda col: rankdata(col, method="average"), axis=0, arr=data
        )

        result_gpu = gpu_rankdata(cp.asarray(data), axis=0)
        result_cpu = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_2d_rowwise(self):
        """2-D float32 array, axis=1: compare to np.apply_along_axis."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        rng = np.random.default_rng(42)
        data = rng.random((50, 200)).astype(np.float32)
        expected = np.apply_along_axis(
            lambda row: rankdata(row, method="average"), axis=1, arr=data
        )

        result_gpu = gpu_rankdata(cp.asarray(data), axis=1)
        result_cpu = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_all_ties(self):
        """All identical values: average rank should be 2.5 for all."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        data = np.array([5, 5, 5, 5], dtype=np.float64)
        expected = np.full(4, 2.5)

        result_gpu = gpu_rankdata(cp.asarray(data))
        result_cpu = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)

    @skipno_gpu
    def test_no_ties(self):
        """No ties: average ranks equal ordinal ranks."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        data = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        expected = rankdata(data, method="average")

        result_gpu = gpu_rankdata(cp.asarray(data))
        result_cpu = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)


class TestGPUSpearmanr:
    """CPU-vs-GPU equivalence tests for gpu_pairwise_spearmanr and gpu_cormat."""

    @skipno_gpu
    def test_pairwise_small(self):
        """Small (5, 30) matrix: compare rho and pval to scipy_spearmanr."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr

        rng = np.random.default_rng(42)
        mat_np = rng.random((5, 30)).astype(np.float64)

        rho_gpu, pval_gpu = gpu_pairwise_spearmanr(cp.asarray(mat_np))
        rho_cpu = cp.asnumpy(rho_gpu)
        pval_cpu = cp.asnumpy(pval_gpu)

        # Build reference from scipy: correlate each pair of rows
        n_vars = mat_np.shape[0]
        rho_ref = np.zeros((n_vars, n_vars))
        pval_ref = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                r, p = scipy_spearmanr(mat_np[i], mat_np[j])
                rho_ref[i, j] = r
                pval_ref[i, j] = p

        np.testing.assert_allclose(rho_cpu, rho_ref, atol=1e-6)
        np.testing.assert_allclose(pval_cpu, pval_ref, atol=1e-6)

    @skipno_gpu
    def test_pairwise_larger(self):
        """Larger (20, 500) matrix: compare rho atol=1e-5, pval atol=1e-4."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr

        rng = np.random.default_rng(123)
        mat_np = rng.random((20, 500)).astype(np.float64)

        rho_gpu, pval_gpu = gpu_pairwise_spearmanr(cp.asarray(mat_np))
        rho_cpu = cp.asnumpy(rho_gpu)
        pval_cpu = cp.asnumpy(pval_gpu)

        n_vars = mat_np.shape[0]
        rho_ref = np.zeros((n_vars, n_vars))
        pval_ref = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                r, p = scipy_spearmanr(mat_np[i], mat_np[j])
                rho_ref[i, j] = r
                pval_ref[i, j] = p

        np.testing.assert_allclose(rho_cpu, rho_ref, atol=1e-5)
        np.testing.assert_allclose(pval_cpu, pval_ref, atol=1e-4)

    @skipno_gpu
    def test_cormat_spearman(self):
        """gpu_cormat matches CPU cormat for (100, 50) X and (100, 1) Y."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_cormat
        from spatialgpu.deconvolution.core import cormat

        rng = np.random.default_rng(42)
        X_np = rng.random((100, 50)).astype(np.float64)
        Y_np = rng.random((100, 1)).astype(np.float64)

        # CPU reference
        ref_df = cormat(X_np, Y_np, method="spearman")
        ref_r = ref_df["cor_r"].to_numpy()
        ref_p = ref_df["cor_p"].to_numpy()

        # GPU
        rs_gpu, ps_gpu = gpu_cormat(cp.asarray(X_np), cp.asarray(Y_np), method="spearman")
        rs_cpu = cp.asnumpy(rs_gpu)
        ps_cpu = cp.asnumpy(ps_gpu)

        np.testing.assert_allclose(rs_cpu, ref_r, atol=1e-3)
        np.testing.assert_allclose(ps_cpu, ref_p, atol=1e-3)

    @skipno_gpu
    def test_diagonal_is_one(self):
        """Diagonal of pairwise rho matrix must be exactly 1.0."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr

        rng = np.random.default_rng(42)
        mat_np = rng.random((10, 100)).astype(np.float64)

        rho_gpu, _ = gpu_pairwise_spearmanr(cp.asarray(mat_np))
        diag = cp.asnumpy(cp.diag(rho_gpu))

        np.testing.assert_array_equal(diag, np.ones(10))
