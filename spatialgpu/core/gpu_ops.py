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


def gpu_nnls(A, b, max_iter=None):
    """
    Solve non-negative least squares on GPU using the Lawson-Hanson active-set algorithm.

    Solves: min ||A @ x - b||^2  subject to  x >= 0

    Parameters
    ----------
    A : cupy.ndarray
        (m, n) matrix of shape (n_observations, n_variables). Must be a CuPy array.
    b : cupy.ndarray
        (m,) right-hand-side vector. Must be a CuPy array.
    max_iter : int, optional
        Maximum number of outer iterations. Default is ``3 * n``.

    Returns
    -------
    x : cupy.ndarray
        (n,) non-negative solution vector.
    """
    import cupy as cp

    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n

    # Precompute Gram matrix and projected gradient
    AtA = A.T @ A
    Atb = A.T @ b

    x = cp.zeros(n, dtype=np.float64)
    passive = cp.zeros(n, dtype=bool)  # True = variable is in passive (free) set
    w = Atb - AtA @ x  # gradient: Atb - AtA @ x

    for _ in range(max_iter):
        # Find active variables with positive gradient component
        active_mask = ~passive
        active_w = cp.where(active_mask, w, cp.float64(-cp.inf))
        t = int(cp.argmax(active_w))

        if float(active_w[t]) <= 0.0:
            # No active variable has a positive gradient — converged
            break

        # Move t into the passive (free) set
        passive[t] = True

        # Inner loop: ensure non-negativity on passive set
        while True:
            # Solve unconstrained subproblem restricted to passive set
            passive_idx = cp.where(passive)[0]
            AtA_sub = AtA[cp.ix_(passive_idx, passive_idx)]
            Atb_sub = Atb[passive_idx]

            # Use CPU linalg.solve for small subproblem stability
            AtA_sub_cpu = cp.asnumpy(AtA_sub)
            Atb_sub_cpu = cp.asnumpy(Atb_sub)
            z_sub_cpu = np.linalg.solve(AtA_sub_cpu, Atb_sub_cpu)
            z_sub = cp.asarray(z_sub_cpu)

            # Build full z vector (zero outside passive set)
            z = cp.zeros(n, dtype=np.float64)
            z[passive_idx] = z_sub

            # Check if all passive variables are positive
            if cp.all(z[passive] > 0.0):
                break

            # Find the boundary-limiting alpha among infeasible passive variables
            infeasible = passive & (z <= 0.0)
            alpha = cp.min(x[infeasible] / (x[infeasible] - z[infeasible]))
            alpha = float(alpha)

            # Move x toward z by alpha
            x = x + alpha * (z - x)

            # Remove passive variables that have become zero (or negative)
            newly_active = passive & (cp.abs(x) < 1e-12)
            passive[newly_active] = False

        x = z

        # Update gradient
        w = Atb - AtA @ x

    return cp.maximum(x, 0.0)


def gpu_nnls_batch(A, B):
    """
    Solve NNLS for each column of B against A on GPU.

    Solves: min ||A @ x_k - b_k||^2  s.t.  x_k >= 0  for each column b_k of B.

    Parameters
    ----------
    A : cupy.ndarray
        (m, n) CuPy array.
    B : cupy.ndarray
        (m, k) CuPy array. Each column is an independent right-hand-side vector.

    Returns
    -------
    X : cupy.ndarray
        (n, k) CuPy array. Column j contains the NNLS solution for B[:, j].
    """
    import cupy as cp

    m, n = A.shape
    k = B.shape[1]
    X = cp.empty((n, k), dtype=np.float64)

    for j in range(k):
        X[:, j] = gpu_nnls(A, B[:, j])

    return X


def gpu_cdist(A, B):
    """
    Compute pairwise Euclidean distances between rows of A and rows of B on GPU.

    Parameters
    ----------
    A : cupy.ndarray
        (n, d) CuPy array of n points in d-dimensional space.
    B : cupy.ndarray
        (m, d) CuPy array of m points in d-dimensional space.

    Returns
    -------
    cupy.ndarray
        (n, m) CuPy array where entry [i, j] is the Euclidean distance
        between A[i] and B[j].
    """
    import cupy as cp

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b.T
    A_sq = (A * A).sum(axis=1, keepdims=True)  # (n, 1)
    B_sq = (B * B).sum(axis=1, keepdims=True)  # (m, 1)
    cross = A @ B.T  # (n, m)
    dist_sq = A_sq + B_sq.T - 2.0 * cross
    return cp.sqrt(cp.maximum(dist_sq, 0.0))


def gpu_bipartite_edge_swap(mat, n_swaps=None, seed=None):
    """
    Degree-preserving bipartite edge swap on GPU.

    Randomly rewires a binary bipartite adjacency matrix while preserving
    the degree (row and column sums) of every node. Mirrors the CPU
    ``_bipartite_edge_swap`` in ``spatialgpu.deconvolution.interaction``.

    Parameters
    ----------
    mat : cupy.ndarray
        Binary int32 adjacency matrix of shape (n_ligands, n_receptors).
        Entry [i, j] == 1 indicates an edge between ligand i and receptor j.
    n_swaps : int, optional
        Number of swap attempts. Default is ``5 * n_edges``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    cupy.ndarray
        Rewired binary adjacency matrix with the same shape and preserved
        row/column degree sums.

    Notes
    -----
    The swap loop is inherently sequential because each attempt depends on
    the current state of the matrix. Random pairs are pre-generated in bulk
    for efficiency; scalar indices are extracted with ``int()`` to avoid
    per-iteration CuPy overhead.
    """
    import cupy as cp

    if seed is not None:
        cp.random.seed(seed)

    mat = mat.copy()
    edges_l, edges_r = cp.where(mat == 1)
    n_edges = int(edges_l.size)

    if n_edges < 2:
        return mat

    if n_swaps is None:
        n_swaps = 5 * n_edges

    # Pre-generate all random pairs at once
    rand_pairs = cp.random.randint(0, n_edges, size=(n_swaps, 2))

    for i in range(n_swaps):
        idx1 = int(rand_pairs[i, 0])
        idx2 = int(rand_pairs[i, 1])

        if idx1 == idx2:
            continue

        l1 = int(edges_l[idx1])
        r1 = int(edges_r[idx1])
        l2 = int(edges_l[idx2])
        r2 = int(edges_r[idx2])

        # Skip if same ligand or same receptor
        if l1 == l2 or r1 == r2:
            continue

        # Check that swapped edges don't already exist
        if int(mat[l1, r2]) == 1 or int(mat[l2, r1]) == 1:
            continue

        # Perform swap
        mat[l1, r1] = 0
        mat[l2, r2] = 0
        mat[l1, r2] = 1
        mat[l2, r1] = 1

        # Update edge lists
        edges_r[idx1] = r2
        edges_r[idx2] = r1

    return mat


def gpu_nmf(V, n_components, seed=None, max_iter=500, tol=1e-4):
    """
    Non-negative Matrix Factorization using multiplicative update rules on GPU.

    Factorizes V ≈ W @ H where W and H are non-negative, using the
    Lee & Seung multiplicative update algorithm.

    Parameters
    ----------
    V : cupy.ndarray
        (n, m) non-negative CuPy array to factorize.
    n_components : int
        Number of components (k). Determines the inner dimension of W and H.
    seed : int or None, optional
        Random seed for reproducible initialization. Default is None.
    max_iter : int, optional
        Maximum number of multiplicative update iterations. Default is 500.
    tol : float, optional
        Convergence tolerance on relative change in reconstruction error.
        Default is 1e-4.

    Returns
    -------
    W : cupy.ndarray
        (n, k) non-negative factor matrix.
    H : cupy.ndarray
        (k, m) non-negative factor matrix.
    """
    import cupy as cp

    rng = cp.random.RandomState(seed)

    n, m = V.shape
    k = n_components
    eps = 1e-16

    # Initialize W and H from scaled random normal (absolute value)
    avg = cp.sqrt(V.mean() / k)
    W = cp.abs(avg * rng.randn(n, k))
    H = cp.abs(avg * rng.randn(k, m))

    prev_err = None

    for i in range(max_iter):
        # Update H: H *= (W.T @ V) / (W.T @ W @ H + eps)
        WtV = W.T @ V
        WtW = W.T @ W
        H *= WtV / (WtW @ H + eps)

        # Update W: W *= (V @ H.T) / (W @ (H @ H.T) + eps)
        VHt = V @ H.T
        HHt = H @ H.T
        W *= VHt / (W @ HHt + eps)

        # Check convergence every 10 iterations
        if i % 10 == 0:
            err = float(cp.linalg.norm(V - W @ H))
            if prev_err is not None and prev_err > 0:
                if abs(prev_err - err) / prev_err < tol:
                    break
            prev_err = err

    return W, H


def gpu_solve_qp(AtA, Atb, n_cell, ppmin, ppmax, max_iter=500, gtol=1e-15):
    """
    Solve a constrained quadratic program via projected gradient descent.

    Minimizes:  x^T AtA x - 2 Atb^T x
    Subject to: x >= 0,  ppmin <= sum(x) <= ppmax

    Parameters
    ----------
    AtA : cupy.ndarray
        (n_cell, n_cell) precomputed Gram matrix A^T A.
    Atb : cupy.ndarray
        (n_cell,) precomputed A^T b vector.
    n_cell : int
        Number of cell types (dimension of x).
    ppmin : float
        Minimum allowed sum of x.
    ppmax : float
        Maximum allowed sum of x.
    max_iter : int, optional
        Maximum number of projected gradient iterations. Default is 500.
    gtol : float, optional
        Gradient norm convergence threshold. Default is 1e-15.

    Returns
    -------
    x : cupy.ndarray
        (n_cell,) non-negative solution vector satisfying the sum constraint.
    """
    import cupy as cp

    # Step size from Lipschitz constant: L = max eigenvalue of 2*AtA
    eigvals = cp.linalg.eigvalsh(2.0 * AtA)
    L = float(cp.max(eigvals))
    if L <= 0.0:
        L = 1.0
    step = 1.0 / L

    # Initialise x uniformly within the feasible sum range
    x = cp.full(n_cell, 0.5 * ppmax / n_cell, dtype=np.float64)

    for _ in range(max_iter):
        grad = 2.0 * AtA @ x - 2.0 * Atb

        if float(cp.linalg.norm(grad)) < gtol:
            break

        x_new = x - step * grad

        # Project onto non-negative orthant
        x_new = cp.maximum(x_new, 0.0)

        # Project onto sum constraint [ppmin, ppmax]
        s = float(cp.sum(x_new))
        if s > ppmax:
            # Scale uniformly so sum == ppmax
            x_new = x_new * (ppmax / s)
        elif s < ppmin:
            # Add uniform deficit to reach ppmin
            deficit = ppmin - s
            x_new = x_new + deficit / n_cell

        x = x_new

    return cp.maximum(x, 0.0)


def gpu_solve_qp_batch(A, AtA, B, n_cell, ppmin_arr, ppmax_arr):
    """
    Solve a constrained QP for each spot (column) of B.

    Solves min x^T AtA x - 2 (A^T b)^T x  s.t. x >= 0, ppmin <= sum(x) <= ppmax
    independently for every column b of B.

    Parameters
    ----------
    A : cupy.ndarray
        (n_genes, n_cell) signature matrix.
    AtA : cupy.ndarray
        (n_cell, n_cell) precomputed A^T A.
    B : cupy.ndarray
        (n_genes, n_spots) observed expression matrix.
    n_cell : int
        Number of cell types.
    ppmin_arr : cupy.ndarray
        (n_spots,) minimum sum constraint per spot.
    ppmax_arr : cupy.ndarray
        (n_spots,) maximum sum constraint per spot.

    Returns
    -------
    X : cupy.ndarray
        (n_cell, n_spots) solution matrix. Column i is the QP solution for spot i.
        Spots with ppmax <= 0.01 receive a uniform allocation (1/n_cell each).
    """
    import cupy as cp

    n_spots = B.shape[1]
    X = cp.empty((n_cell, n_spots), dtype=np.float64)

    for i in range(n_spots):
        ppmax_i = float(ppmax_arr[i])
        ppmin_i = float(ppmin_arr[i])

        if ppmax_i <= 0.01:
            # Degenerate spot: return uniform allocation
            X[:, i] = cp.full(n_cell, 1.0 / n_cell, dtype=np.float64)
            continue

        Atb_i = A.T @ B[:, i]
        X[:, i] = gpu_solve_qp(AtA, Atb_i, n_cell, ppmin_i, ppmax_i)

    return X
