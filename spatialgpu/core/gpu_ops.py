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


def gpu_pairwise_spearmanr(mat):
    """
    Compute pairwise Spearman correlations for all row pairs in a matrix.

    Parameters
    ----------
    mat : cupy.ndarray
        2-D input array of shape (n_vars, n_obs). Each row is a variable
        whose pairwise Spearman correlation with every other row is computed.

    Returns
    -------
    rho : cupy.ndarray
        (n_vars, n_vars) float64 array of Spearman correlation coefficients.
    pval : cupy.ndarray
        (n_vars, n_vars) float64 array of two-sided p-values.
    """
    import cupy as cp
    from scipy import stats as scipy_stats

    n_vars, n_obs = mat.shape

    # Rank each row using average method
    ranked = gpu_rankdata(mat, method="average", axis=1)

    # Center and normalize
    ranked_mean = ranked.mean(axis=1, keepdims=True)
    ranked_centered = ranked - ranked_mean
    ranked_std = ranked_centered.std(axis=1, keepdims=True)
    # Avoid division by zero for constant rows
    ranked_std[ranked_std == 0] = 1.0
    ranked_norm = ranked_centered / ranked_std

    # Correlation via matmul
    rho = ranked_norm @ ranked_norm.T / n_obs

    # Clip to valid correlation range
    rho = cp.clip(rho, -1.0, 1.0)

    # Compute t-statistics for p-values
    t_stat = rho * cp.sqrt((n_obs - 2) / (1.0 - rho**2 + 1e-300))

    # Transfer to CPU for scipy p-value calculation
    t_stat_cpu = cp.asnumpy(t_stat)
    pval_cpu = 2.0 * scipy_stats.t.sf(np.abs(t_stat_cpu), df=n_obs - 2)

    pval = cp.asarray(pval_cpu)

    # Fix diagonal: rho=1, pval=0
    diag_idx = cp.arange(n_vars)
    rho[diag_idx, diag_idx] = 1.0
    pval[diag_idx, diag_idx] = 0.0

    # Replace NaN: rho->0, pval->1
    rho = cp.where(cp.isnan(rho), cp.float64(0.0), rho)
    pval = cp.where(cp.isnan(pval), cp.float64(1.0), pval)

    return rho, pval


def gpu_cormat(X, Y, method="spearman"):
    """
    Compute correlation between columns of X and each column of Y on GPU.

    Equivalent to the CPU ``cormat()`` in ``spatialgpu.deconvolution.core``,
    but operates on CuPy arrays and returns CuPy arrays.

    Parameters
    ----------
    X : cupy.ndarray
        (n_genes, n_samples) matrix. Correlations are computed across genes
        (rows) for every sample (column).
    Y : cupy.ndarray
        (n_genes, n_features) matrix. Usually a single-column signature.
    method : str, optional
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    rs : cupy.ndarray
        1-D array of length n_samples with rounded (3 decimals) correlation
        coefficients against the first column of Y.
    ps : cupy.ndarray
        1-D array of length n_samples with two-sided p-values against the
        first column of Y.
    """
    import cupy as cp
    from scipy import stats as scipy_stats

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_obs = X.shape[0]

    # Rank column-wise for Spearman
    if method == "spearman":
        X_prep = gpu_rankdata(X, method="average", axis=0)
    elif method == "pearson":
        X_prep = X.astype(np.float64)
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method!r}'")

    X_centered = X_prep - X_prep.mean(axis=0, keepdims=True)
    std_x = cp.sqrt((X_centered**2).sum(axis=0))

    # Use first column of Y only (matches CPU cormat behaviour)
    y_col = Y[:, 0]
    if method == "spearman":
        y_prep = gpu_rankdata(y_col.reshape(1, -1), method="average", axis=1).ravel()
    else:
        y_prep = y_col.astype(np.float64)

    y_centered = y_prep - y_prep.mean()
    cov_xy = (X_centered * y_centered[:, None]).sum(axis=0)
    std_y = cp.sqrt((y_centered**2).sum())

    denom = std_x * std_y
    denom = cp.where(denom == 0, cp.float64(1.0), denom)
    rs_raw = cov_xy / denom

    t_stat = rs_raw * cp.sqrt((n_obs - 2) / (1.0 - rs_raw**2 + 1e-300))

    # P-values via scipy on CPU
    t_stat_cpu = cp.asnumpy(t_stat)
    ps_cpu = 2.0 * scipy_stats.t.sf(np.abs(t_stat_cpu), df=n_obs - 2)

    # Round rs to 3 decimals (matches CPU cormat)
    rs = cp.round(rs_raw, 3)
    ps = cp.asarray(ps_cpu)

    return rs, ps
