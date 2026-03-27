"""CPU-vs-GPU equivalence tests for spatialgpu.core.gpu_ops primitives."""

import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls
from scipy.spatial.distance import cdist as scipy_cdist
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


class TestGPUNNLS:
    """CPU-vs-GPU equivalence tests for gpu_nnls and gpu_nnls_batch."""

    @skipno_gpu
    def test_basic_nnls(self):
        """Basic NNLS: non-negative true solution, compare to scipy nnls."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nnls

        rng = np.random.default_rng(42)
        A_np = rng.random((50, 5)).astype(np.float64)
        x_true = np.abs(rng.standard_normal(5))
        b_np = A_np @ x_true + 0.1 * rng.standard_normal(50)

        x_ref, _ = scipy_nnls(A_np, b_np)
        x_gpu = gpu_nnls(cp.asarray(A_np), cp.asarray(b_np))
        x_cpu = cp.asnumpy(x_gpu)

        np.testing.assert_allclose(x_cpu, x_ref, atol=1e-5)

    @skipno_gpu
    def test_batch_nnls(self):
        """Batch NNLS: gpu_nnls_batch matches 20 individual scipy nnls calls."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nnls_batch

        rng = np.random.default_rng(42)
        A_np = rng.random((50, 5)).astype(np.float64)
        B_np = np.abs(rng.standard_normal((50, 20)))

        X_ref = np.column_stack([scipy_nnls(A_np, B_np[:, j])[0] for j in range(20)])
        X_gpu = gpu_nnls_batch(cp.asarray(A_np), cp.asarray(B_np))
        X_cpu = cp.asnumpy(X_gpu)

        np.testing.assert_allclose(X_cpu, X_ref, atol=1e-5)

    @skipno_gpu
    def test_nnls_all_zero_rhs(self):
        """Zero right-hand side: solution should be all zeros."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nnls

        A_np = np.eye(5, dtype=np.float64)
        b_np = np.zeros(5, dtype=np.float64)

        x_gpu = gpu_nnls(cp.asarray(A_np), cp.asarray(b_np))
        x_cpu = cp.asnumpy(x_gpu)

        np.testing.assert_array_equal(x_cpu, np.zeros(5))


class TestGPUNMF:
    """Tests for gpu_nmf: Non-negative Matrix Factorization on GPU."""

    @skipno_gpu
    def test_reconstruction(self):
        """Reconstruction error should be less than 0.5 relative to ||V||."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nmf

        rng = np.random.default_rng(42)
        V_np = np.abs(rng.standard_normal((50, 100)))
        V = cp.asarray(V_np)

        W, H = gpu_nmf(V, n_components=5, seed=42, max_iter=500)

        reconstruction = cp.asnumpy(W @ H)
        err = np.linalg.norm(V_np - reconstruction)
        rel_err = err / np.linalg.norm(V_np)

        assert rel_err < 0.5, f"Relative reconstruction error {rel_err:.4f} >= 0.5"

    @skipno_gpu
    def test_matches_sklearn(self):
        """GPU NMF reconstruction error should be within 1.5x of sklearn NMF."""
        import cupy as cp
        from sklearn.decomposition import NMF
        from spatialgpu.core.gpu_ops import gpu_nmf

        rng = np.random.default_rng(42)
        V_np = np.abs(rng.standard_normal((30, 80)))
        V = cp.asarray(V_np)

        # GPU NMF
        W_gpu, H_gpu = gpu_nmf(V, n_components=3, seed=42, max_iter=500)
        gpu_err = float(cp.linalg.norm(V - W_gpu @ H_gpu))

        # sklearn NMF reference
        model = NMF(n_components=3, random_state=42, max_iter=500)
        W_sk = model.fit_transform(V_np)
        H_sk = model.components_
        sk_err = np.linalg.norm(V_np - W_sk @ H_sk)

        assert gpu_err <= 1.5 * sk_err, (
            f"GPU NMF error {gpu_err:.4f} is more than 1.5x sklearn error {sk_err:.4f}"
        )

    @skipno_gpu
    def test_non_negative(self):
        """W and H must be non-negative after factorization."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nmf

        rng = np.random.default_rng(7)
        V_np = np.abs(rng.standard_normal((40, 60)))
        V = cp.asarray(V_np)

        W, H = gpu_nmf(V, n_components=3, seed=7, max_iter=200)

        assert float(cp.min(W)) >= 0.0, "W contains negative values"
        assert float(cp.min(H)) >= 0.0, "H contains negative values"


class TestGPUCdist:
    """CPU-vs-GPU equivalence tests for gpu_cdist."""

    @skipno_gpu
    def test_euclidean(self):
        """Compare gpu_cdist to scipy.spatial.distance.cdist on float64 arrays."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_cdist

        rng = np.random.default_rng(42)
        A_np = rng.random((50, 2)).astype(np.float64)
        B_np = rng.random((30, 2)).astype(np.float64)

        ref = scipy_cdist(A_np, B_np, metric="euclidean")
        result = cp.asnumpy(gpu_cdist(cp.asarray(A_np), cp.asarray(B_np)))

        np.testing.assert_allclose(result, ref, atol=1e-6)

    @skipno_gpu
    def test_self_distance(self):
        """Self-distance matrix diagonal must be zero."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_cdist

        rng = np.random.default_rng(42)
        A_np = rng.random((20, 2)).astype(np.float64)

        result = cp.asnumpy(gpu_cdist(cp.asarray(A_np), cp.asarray(A_np)))
        diag = np.diag(result)

        np.testing.assert_allclose(diag, np.zeros(20), atol=1e-10)


class TestGPUBipartiteEdgeSwap:
    """Degree-preservation tests for gpu_bipartite_edge_swap."""

    @skipno_gpu
    def test_degree_preservation(self):
        """Row and column sums must be preserved after rewiring."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_bipartite_edge_swap

        rng = np.random.default_rng(42)
        mat_np = (rng.random((10, 8)) > 0.7).astype(np.int32)
        mat_gpu = cp.asarray(mat_np)

        row_sums_before = mat_np.sum(axis=1)
        col_sums_before = mat_np.sum(axis=0)

        rewired = gpu_bipartite_edge_swap(mat_gpu, seed=42)
        rewired_np = cp.asnumpy(rewired)

        np.testing.assert_array_equal(rewired_np.sum(axis=1), row_sums_before)
        np.testing.assert_array_equal(rewired_np.sum(axis=0), col_sums_before)

    @skipno_gpu
    def test_edge_count_preserved(self):
        """Total number of edges must be unchanged after rewiring."""
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_bipartite_edge_swap

        rng = np.random.default_rng(42)
        mat_np = (rng.random((15, 12)) > 0.6).astype(np.int32)
        mat_gpu = cp.asarray(mat_np)

        n_edges_before = int(mat_np.sum())

        rewired = gpu_bipartite_edge_swap(mat_gpu, seed=99)
        n_edges_after = int(cp.asnumpy(rewired).sum())

        assert n_edges_after == n_edges_before, (
            f"Edge count changed: {n_edges_before} -> {n_edges_after}"
        )
