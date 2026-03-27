"""CPU-vs-GPU equivalence tests for spatialgpu.core.gpu_ops primitives."""

import numpy as np
import pytest
from scipy.stats import rankdata


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
