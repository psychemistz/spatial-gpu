"""Tests for core functionality."""

import numpy as np
import pytest


class TestBackend:
    """Tests for Backend class."""

    def test_singleton(self):
        """Test backend is singleton."""
        from spatialgpu.core.backend import Backend

        b1 = Backend()
        b2 = Backend()

        assert b1 is b2

    def test_get_backend(self):
        """Test get_backend function."""
        import spatialgpu as sp

        backend = sp.get_backend()
        assert backend is not None
        assert hasattr(backend, "is_gpu_available")
        assert hasattr(backend, "is_gpu_active")

    def test_set_backend_cpu(self):
        """Test setting CPU backend."""
        import spatialgpu as sp

        sp.set_backend("cpu")
        backend = sp.get_backend()

        assert not backend.is_gpu_active
        assert backend.xp is np

    def test_set_backend_auto(self):
        """Test auto backend selection."""
        import spatialgpu as sp

        sp.set_backend("auto")
        backend = sp.get_backend()

        # Should be GPU if available, else CPU
        if backend.is_gpu_available:
            assert backend.is_gpu_active
        else:
            assert not backend.is_gpu_active

    def test_invalid_backend(self):
        """Test error on invalid backend."""
        import spatialgpu as sp

        with pytest.raises(ValueError, match="Unknown backend"):
            sp.set_backend("invalid")


class TestArrayUtils:
    """Tests for array utilities."""

    def test_get_array_module_numpy(self):
        """Test get_array_module with numpy array."""
        from spatialgpu.core.array_utils import get_array_module

        x = np.array([1, 2, 3])
        xp = get_array_module(x)

        assert xp is np

    def test_is_gpu_array_numpy(self):
        """Test is_gpu_array with numpy array."""
        from spatialgpu.core.array_utils import is_gpu_array

        x = np.array([1, 2, 3])
        assert not is_gpu_array(x)

    def test_to_cpu_numpy(self):
        """Test to_cpu with numpy array."""
        from spatialgpu.core.array_utils import to_cpu

        x = np.array([1, 2, 3])
        result = to_cpu(x)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, x)

    def test_to_cpu_copy(self):
        """Test to_cpu with copy."""
        from spatialgpu.core.array_utils import to_cpu

        x = np.array([1, 2, 3])
        result = to_cpu(x, copy=True)

        assert result is not x
        np.testing.assert_array_equal(result, x)

    def test_to_cpu_dtype(self):
        """Test to_cpu with dtype conversion."""
        from spatialgpu.core.array_utils import to_cpu

        x = np.array([1, 2, 3], dtype=np.int32)
        result = to_cpu(x, dtype=np.float32)

        assert result.dtype == np.float32

    def test_ensure_contiguous(self):
        """Test ensure_contiguous."""
        from spatialgpu.core.array_utils import ensure_contiguous

        x = np.array([[1, 2], [3, 4]]).T  # Fortran-contiguous
        result = ensure_contiguous(x, order="C")

        assert result.flags.c_contiguous

    def test_as_float32(self):
        """Test as_float32."""
        from spatialgpu.core.array_utils import as_float32

        x = np.array([1, 2, 3], dtype=np.int32)
        result = as_float32(x)

        assert result.dtype == np.float32

    def test_chunked_operation(self):
        """Test chunked_operation."""
        from spatialgpu.core.array_utils import chunked_operation

        x = np.random.randn(1000, 10)
        result = chunked_operation(
            lambda c: c**2,
            x,
            chunk_size=100,
        )

        np.testing.assert_array_almost_equal(result, x**2)


class TestConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration."""
        import spatialgpu as sp

        assert sp.config.gpu.device_id == 0
        assert sp.config.compute.backend == "auto"
        assert sp.config.compute.precision == "float32"

    def test_config_modification(self):
        """Test configuration modification."""
        import spatialgpu as sp

        sp.config.compute.precision = "float64"
        assert sp.config.compute.precision == "float64"

        # Reset
        sp.config.compute.precision = "float32"

    def test_config_reset(self):
        """Test configuration reset."""
        import spatialgpu as sp

        sp.config.compute.n_jobs = 4
        sp.config.reset()

        assert sp.config.compute.n_jobs == -1

    def test_config_to_dict(self):
        """Test configuration serialization."""
        import spatialgpu as sp

        d = sp.config.to_dict()

        assert "gpu" in d
        assert "compute" in d
        assert "graph" in d
        assert "segmentation" in d

    def test_config_from_dict(self):
        """Test configuration deserialization."""
        from spatialgpu.core.config import Config

        d = {
            "gpu": {"device_id": 1},
            "compute": {"n_jobs": 4},
        }

        config = Config.from_dict(d)

        assert config.gpu.device_id == 1
        assert config.compute.n_jobs == 4
