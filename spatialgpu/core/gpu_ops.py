"""
Reusable GPU primitives for spatial transcriptomics analysis.

This module provides low-level GPU-accelerated operations that serve as
building blocks for higher-level analyses. All functions accept and return
CuPy arrays and are intended to be called from code paths that have already
confirmed GPU availability via the Backend singleton.
"""

from __future__ import annotations

import numpy as np


def _rankdata_1d(x):
    """
    Compute average-method ranks for a 1-D CuPy array.

    Parameters
    ----------
    x : cupy.ndarray
        1-D input array. Must be a CuPy array.

    Returns
    -------
    cupy.ndarray
        Float64 array of average ranks (1-based), same length as x.
    """
    import cupy as cp

    n = x.size
    # Step 1: stable argsort to get ordering permutation
    order = cp.argsort(x, kind="stable")
    # Step 2: sorted values
    sorted_x = x[order]
    # Step 3: ordinal ranks 1..n
    ordinal = cp.arange(1, n + 1, dtype=np.float64)
    # Step 4: boolean mask where value changes (first element always True)
    diff = cp.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = sorted_x[1:] != sorted_x[:-1]
    # Step 5: group labels (0-based) via cumsum
    group_labels = cp.cumsum(diff) - 1
    n_groups = int(group_labels[-1]) + 1 if n > 0 else 0
    # Step 6: accumulate sum and count of ordinal ranks per group
    group_sum = cp.zeros(n_groups, dtype=np.float64)
    group_count = cp.zeros(n_groups, dtype=np.float64)
    cp.add.at(group_sum, group_labels, ordinal)
    cp.add.at(group_count, group_labels, cp.ones(n, dtype=np.float64))
    # Step 7: average rank per group
    avg_ranks = group_sum / group_count
    # Step 8: map average ranks back to sorted positions
    result_sorted = avg_ranks[group_labels]
    # Step 9: invert the permutation so ranks align with original positions
    result = cp.empty(n, dtype=np.float64)
    result[order] = result_sorted
    return result


def gpu_rankdata(x, method="average", axis=None):
    """
    Compute ranks of elements in a CuPy array, with tie-breaking.

    Parameters
    ----------
    x : cupy.ndarray
        Input array. Must be a CuPy array.
    method : str, optional
        Tie-breaking method. Only ``"average"`` is currently supported.
        Default is ``"average"``.
    axis : {None, 0, 1}, optional
        Axis along which to rank.

        - ``None`` (default): rank all elements in the flattened array.
        - ``0``: rank each column independently (operate along rows).
        - ``1``: rank each row independently (operate along columns).

    Returns
    -------
    cupy.ndarray
        Float64 array of ranks with the same shape as *x* (when *axis* is
        not ``None``) or a 1-D array of length ``x.size`` (when
        ``axis=None``).

    Raises
    ------
    NotImplementedError
        If *method* is not ``"average"``.
    ValueError
        If *axis* is not ``None``, ``0``, or ``1``.
    """
    import cupy as cp

    if method != "average":
        raise NotImplementedError(
            f"method={method!r} is not supported. Only 'average' is implemented."
        )

    if axis is None:
        return _rankdata_1d(x.ravel())

    if axis == 0:
        # Rank each column: iterate over columns
        if x.ndim != 2:
            raise ValueError("axis=0 requires a 2-D input array.")
        n_rows, n_cols = x.shape
        result = cp.empty((n_rows, n_cols), dtype=np.float64)
        for j in range(n_cols):
            result[:, j] = _rankdata_1d(x[:, j])
        return result

    if axis == 1:
        # Rank each row: iterate over rows
        if x.ndim != 2:
            raise ValueError("axis=1 requires a 2-D input array.")
        n_rows, n_cols = x.shape
        result = cp.empty((n_rows, n_cols), dtype=np.float64)
        for i in range(n_rows):
            result[i, :] = _rankdata_1d(x[i, :])
        return result

    raise ValueError(f"axis={axis!r} is not supported. Use None, 0, or 1.")
