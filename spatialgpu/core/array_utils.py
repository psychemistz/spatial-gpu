"""
Array utilities for CPU/GPU interoperability.

Provides seamless conversion between NumPy and CuPy arrays,
with automatic backend detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    import cupy as cp
    from numpy.typing import ArrayLike

# Type alias for arrays that can be on CPU or GPU
Array = Union[np.ndarray, "cp.ndarray"]


def get_array_module(x: ArrayLike):
    """
    Get the array module (numpy or cupy) for an array.

    Parameters
    ----------
    x : array-like
        Input array

    Returns
    -------
    module
        numpy or cupy module

    Examples
    --------
    >>> import numpy as np
    >>> xp = get_array_module(np.array([1, 2, 3]))
    >>> xp.__name__
    'numpy'
    """
    try:
        import cupy as cp

        return cp.get_array_module(x)
    except ImportError:
        return np


def is_gpu_array(x: ArrayLike) -> bool:
    """
    Check if array is on GPU.

    Parameters
    ----------
    x : array-like
        Input array

    Returns
    -------
    bool
        True if array is a CuPy array
    """
    try:
        import cupy as cp

        return isinstance(x, cp.ndarray)
    except ImportError:
        return False


def to_gpu(
    x: ArrayLike,
    dtype: str | np.dtype | None = None,
    copy: bool = False,
) -> Array:
    """
    Transfer array to GPU.

    If GPU is not available, returns the input unchanged (or a copy).

    Parameters
    ----------
    x : array-like
        Input array (numpy array, scipy sparse, or list)
    dtype : str or dtype, optional
        Data type for the output array
    copy : bool
        Force copy even if already on GPU

    Returns
    -------
    array
        CuPy array if GPU available, otherwise NumPy array

    Examples
    --------
    >>> import numpy as np
    >>> import spatialgpu as sp
    >>> x = np.random.randn(1000, 100)
    >>> x_gpu = sp.core.to_gpu(x)
    >>> type(x_gpu)
    <class 'cupy.ndarray'>
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if not backend.is_gpu_active:
        # CPU mode: just ensure numpy array
        if sparse.issparse(x):
            x = x.toarray()
        arr = np.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype, copy=copy)
        elif copy:
            arr = arr.copy()
        return arr

    # GPU mode
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    if is_gpu_array(x):
        # Already on GPU
        if dtype is not None:
            return x.astype(dtype, copy=copy)
        return x.copy() if copy else x

    if sparse.issparse(x):
        # Handle scipy sparse matrices
        if sparse.isspmatrix_csr(x):
            gpu_arr = cp_sparse.csr_matrix(x)
        elif sparse.isspmatrix_csc(x):
            gpu_arr = cp_sparse.csc_matrix(x)
        elif sparse.isspmatrix_coo(x):
            gpu_arr = cp_sparse.coo_matrix(x)
        else:
            # Convert to CSR first
            gpu_arr = cp_sparse.csr_matrix(x.tocsr())

        if dtype is not None:
            gpu_arr = gpu_arr.astype(dtype)
        return gpu_arr

    # Regular array
    gpu_arr = cp.asarray(x)
    if dtype is not None:
        gpu_arr = gpu_arr.astype(dtype, copy=False)

    return gpu_arr


def to_cpu(
    x: ArrayLike,
    dtype: str | np.dtype | None = None,
    copy: bool = False,
) -> np.ndarray:
    """
    Transfer array to CPU.

    Parameters
    ----------
    x : array-like
        Input array (numpy or cupy array)
    dtype : str or dtype, optional
        Data type for the output array
    copy : bool
        Force copy even if already on CPU

    Returns
    -------
    numpy.ndarray
        NumPy array on CPU

    Examples
    --------
    >>> import spatialgpu as sp
    >>> x_gpu = sp.core.to_gpu(np.random.randn(1000, 100))
    >>> x_cpu = sp.core.to_cpu(x_gpu)
    >>> type(x_cpu)
    <class 'numpy.ndarray'>
    """
    if is_gpu_array(x):
        import cupy as cp

        arr = cp.asnumpy(x)
    elif sparse.issparse(x):
        arr = x.toarray()
    else:
        arr = np.asarray(x)

    if dtype is not None:
        arr = arr.astype(dtype, copy=copy)
    elif copy and not is_gpu_array(x):
        arr = arr.copy()

    return arr


def ensure_contiguous(x: Array, order: str = "C") -> Array:
    """
    Ensure array is contiguous in memory.

    Parameters
    ----------
    x : array
        Input array
    order : {"C", "F"}
        Memory layout order

    Returns
    -------
    array
        Contiguous array (may be a copy)
    """
    xp = get_array_module(x)

    if order == "C":
        if x.flags.c_contiguous:
            return x
        return xp.ascontiguousarray(x)
    else:
        if x.flags.f_contiguous:
            return x
        return xp.asfortranarray(x)


def as_float32(x: Array) -> Array:
    """Convert array to float32."""
    return x.astype(np.float32, copy=False)


def as_float64(x: Array) -> Array:
    """Convert array to float64."""
    return x.astype(np.float64, copy=False)


def chunked_operation(
    func,
    x: Array,
    chunk_size: int | None = None,
    axis: int = 0,
    **kwargs,
) -> Array:
    """
    Apply function to array in chunks to manage memory.

    Parameters
    ----------
    func : callable
        Function to apply to each chunk
    x : array
        Input array
    chunk_size : int, optional
        Size of each chunk. If None, uses config default.
    axis : int
        Axis along which to chunk
    **kwargs
        Additional arguments passed to func

    Returns
    -------
    array
        Concatenated results

    Examples
    --------
    >>> import numpy as np
    >>> from spatialgpu.core.array_utils import chunked_operation
    >>> x = np.random.randn(10000, 100)
    >>> result = chunked_operation(lambda c: c ** 2, x, chunk_size=1000)
    """
    from spatialgpu.core.config import config

    if chunk_size is None:
        chunk_size = config.gpu.chunk_size

    xp = get_array_module(x)
    n = x.shape[axis]

    if n <= chunk_size:
        return func(x, **kwargs)

    results = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = xp.take(x, xp.arange(start, end), axis=axis)
        result = func(chunk, **kwargs)
        results.append(result)

    return xp.concatenate(results, axis=axis)


def sparse_to_dense_chunked(
    sparse_matrix,
    chunk_size: int = 10000,
    dtype=np.float32,
) -> Array:
    """
    Convert sparse matrix to dense in chunks to manage memory.

    Parameters
    ----------
    sparse_matrix : sparse matrix
        Input sparse matrix (scipy or cupyx)
    chunk_size : int
        Number of rows per chunk
    dtype : dtype
        Output data type

    Returns
    -------
    array
        Dense array
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()
    n_rows = sparse_matrix.shape[0]

    if backend.is_gpu_active:
        import cupy as cp

        xp = cp
    else:
        xp = np

    # Pre-allocate output
    out = xp.zeros(sparse_matrix.shape, dtype=dtype)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = sparse_matrix[start:end].toarray()
        out[start:end] = xp.asarray(chunk)

    return out
