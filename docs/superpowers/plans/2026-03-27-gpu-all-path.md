# GPU All-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU code paths to all CPU-only functions in the deconvolution pipeline (tutorials t1–t3, t6) so the entire SpaCET workflow runs on GPU when available.

**Architecture:** Build reusable GPU primitives in `spatialgpu/core/gpu_ops.py`, then integrate them into each pipeline function via the existing `backend.is_gpu_active` dispatch pattern. Each primitive has a CPU fallback and must match CPU output within float32 tolerance (~1e-6). The primitives are: rankdata, Spearman correlation, NNLS, pairwise distance, NMF, and constrained QP solver.

**Tech Stack:** CuPy (array ops, custom CUDA kernels), cupyx.scipy (sparse, linalg), existing Backend singleton for dispatch.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `spatialgpu/core/gpu_ops.py` | **Create** | All reusable GPU primitives |
| `tests/test_gpu_ops.py` | **Create** | CPU-vs-GPU equivalence tests for each primitive |
| `spatialgpu/deconvolution/core.py` | Modify | GPU dispatch in `cormat`, `_solve_nnls`, `_solve_trust_constr` |
| `spatialgpu/deconvolution/interaction.py` | Modify | GPU dispatch in `_pairwise_spearmanr`, `cci_lr_network_score`, `distance_to_interface` |
| `spatialgpu/deconvolution/gene_set_score.py` | Modify | GPU dispatch in `_ucell_score` |
| `spatialgpu/deconvolution/secact.py` | Modify | GPU dispatch in `secact_signaling_patterns`, `secact_signaling_velocity` |
| `spatialgpu/deconvolution/spatial_correlation.py` | Modify | GPU dispatch in `cal_weights` |
| `tests/test_deconvolution/test_gpu_integration.py` | **Create** | End-to-end GPU pipeline tests |

---

### Task 1: GPU rankdata primitive

**Files:**
- Create: `spatialgpu/core/gpu_ops.py`
- Create: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_ops.py
"""CPU-vs-GPU equivalence tests for GPU primitives."""

import numpy as np
import pytest
from scipy.stats import rankdata as scipy_rankdata


def gpu_available():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not gpu_available(), reason="No GPU available")


class TestGPURankdata:
    """Tests for GPU rankdata primitive."""

    @skipno_gpu
    def test_1d_average(self):
        """GPU rankdata matches scipy for 1D with ties."""
        from spatialgpu.core.gpu_ops import gpu_rankdata
        import cupy as cp

        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0])
        expected = scipy_rankdata(x, method="average")

        x_gpu = cp.asarray(x)
        result = gpu_rankdata(x_gpu, method="average")
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)

    @skipno_gpu
    def test_2d_columnwise(self):
        """GPU rankdata matches scipy column-wise on a matrix."""
        from spatialgpu.core.gpu_ops import gpu_rankdata
        import cupy as cp

        rng = np.random.RandomState(42)
        X = rng.randn(100, 20).astype(np.float32)

        expected = np.apply_along_axis(
            lambda col: scipy_rankdata(col, method="average"), 0, X
        )

        X_gpu = cp.asarray(X)
        result = gpu_rankdata(X_gpu, method="average", axis=0)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_2d_rowwise(self):
        """GPU rankdata matches scipy row-wise on a matrix."""
        from spatialgpu.core.gpu_ops import gpu_rankdata
        import cupy as cp

        rng = np.random.RandomState(42)
        X = rng.randn(50, 200).astype(np.float32)

        expected = np.apply_along_axis(
            lambda row: scipy_rankdata(row, method="average"), 1, X
        )

        X_gpu = cp.asarray(X)
        result = gpu_rankdata(X_gpu, method="average", axis=1)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_all_ties(self):
        """GPU rankdata handles all-tied values."""
        from spatialgpu.core.gpu_ops import gpu_rankdata
        import cupy as cp

        x = np.array([5.0, 5.0, 5.0, 5.0])
        expected = scipy_rankdata(x, method="average")

        x_gpu = cp.asarray(x)
        result = gpu_rankdata(x_gpu, method="average")
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)

    @skipno_gpu
    def test_no_ties(self):
        """GPU rankdata handles no ties (ordinal = average)."""
        from spatialgpu.core.gpu_ops import gpu_rankdata
        import cupy as cp

        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        expected = scipy_rankdata(x, method="average")

        x_gpu = cp.asarray(x)
        result = gpu_rankdata(x_gpu, method="average")
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPURankdata -v 2>&1 | head -30`
Expected: FAIL with "ImportError" or "cannot import name 'gpu_rankdata'"

- [ ] **Step 3: Write minimal implementation**

```python
# spatialgpu/core/gpu_ops.py
"""
Reusable GPU primitives for the deconvolution pipeline.

Each function takes CuPy arrays and returns CuPy arrays.
These are low-level building blocks — pipeline functions should
call them only when ``backend.is_gpu_active`` is True.
"""

from __future__ import annotations

from typing import Literal


def gpu_rankdata(
    x,
    method: Literal["average", "ordinal"] = "average",
    axis: int | None = None,
):
    """Rank data on GPU, matching scipy.stats.rankdata semantics.

    Parameters
    ----------
    x : cupy.ndarray
        Input array (1-D or 2-D).
    method : str
        Tie-breaking method. Only "average" is supported.
    axis : int or None
        Axis along which to rank. None = flatten.

    Returns
    -------
    cupy.ndarray
        Ranks (1-based, float64 for average ties).
    """
    import cupy as cp

    if method != "average":
        raise NotImplementedError(f"method={method!r} not supported, use 'average'")

    if axis is None:
        return _rankdata_1d(x.ravel())

    if x.ndim != 2:
        raise ValueError("axis parameter requires 2-D input")

    if axis == 0:
        # Rank each column
        out = cp.empty_like(x, dtype=cp.float64)
        for j in range(x.shape[1]):
            out[:, j] = _rankdata_1d(x[:, j])
        return out
    elif axis == 1:
        # Rank each row
        out = cp.empty_like(x, dtype=cp.float64)
        for i in range(x.shape[0]):
            out[i, :] = _rankdata_1d(x[i, :])
        return out
    else:
        raise ValueError(f"axis must be 0, 1, or None, got {axis}")


def _rankdata_1d(x):
    """Rank a 1-D CuPy array with average tie-breaking.

    Algorithm: argsort to get ordinal ranks, then average tied groups.
    """
    import cupy as cp

    n = x.shape[0]
    if n == 0:
        return cp.array([], dtype=cp.float64)

    # Sort and get inverse permutation
    order = cp.argsort(x, kind="stable")
    sorted_x = x[order]

    # Ordinal ranks (1-based)
    ordinal = cp.arange(1, n + 1, dtype=cp.float64)

    # Find tie groups: where sorted values differ
    # For ties, replace ordinal ranks with their average
    # diff[i] = True means sorted_x[i] != sorted_x[i-1]
    diff = cp.ones(n, dtype=cp.bool_)
    diff[1:] = sorted_x[1:] != sorted_x[:-1]

    # Label each tie group
    group_labels = cp.cumsum(diff) - 1  # 0-based group ID

    # For each group, compute average rank
    n_groups = int(group_labels[-1]) + 1
    group_sum = cp.zeros(n_groups, dtype=cp.float64)
    group_count = cp.zeros(n_groups, dtype=cp.float64)

    # Use scatter_add for group aggregation
    cp.add.at(group_sum, group_labels, ordinal)
    cp.add.at(group_count, group_labels, 1.0)

    avg_ranks = group_sum / group_count

    # Map averaged ranks back to original positions
    result_sorted = avg_ranks[group_labels]

    # Invert the sort permutation
    result = cp.empty(n, dtype=cp.float64)
    result[order] = result_sorted

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPURankdata -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU rankdata primitive with CPU equivalence tests"
```

---

### Task 2: GPU Spearman correlation primitive

**Files:**
- Modify: `spatialgpu/core/gpu_ops.py`
- Modify: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_ops.py`:

```python
from scipy.stats import spearmanr as scipy_spearmanr


class TestGPUSpearmanr:
    """Tests for GPU pairwise Spearman correlation."""

    @skipno_gpu
    def test_pairwise_small(self):
        """GPU pairwise Spearman matches scipy on small matrix."""
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr
        import cupy as cp

        rng = np.random.RandomState(42)
        mat = rng.randn(5, 30).astype(np.float64)  # 5 variables, 30 observations

        expected_rho, expected_pval = scipy_spearmanr(mat, axis=1)

        mat_gpu = cp.asarray(mat)
        rho_gpu, pval_gpu = gpu_pairwise_spearmanr(mat_gpu)
        rho = cp.asnumpy(rho_gpu)
        pval = cp.asnumpy(pval_gpu)

        np.testing.assert_allclose(rho, expected_rho, atol=1e-6)
        np.testing.assert_allclose(pval, expected_pval, atol=1e-6)

    @skipno_gpu
    def test_pairwise_larger(self):
        """GPU Spearman matches scipy on 20x500 matrix."""
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr
        import cupy as cp

        rng = np.random.RandomState(123)
        mat = rng.randn(20, 500).astype(np.float64)

        expected_rho, expected_pval = scipy_spearmanr(mat, axis=1)

        mat_gpu = cp.asarray(mat)
        rho_gpu, pval_gpu = gpu_pairwise_spearmanr(mat_gpu)
        rho = cp.asnumpy(rho_gpu)
        pval = cp.asnumpy(pval_gpu)

        np.testing.assert_allclose(rho, expected_rho, atol=1e-5)
        np.testing.assert_allclose(pval, expected_pval, atol=1e-4)

    @skipno_gpu
    def test_cormat_spearman(self):
        """GPU cormat_spearman matches CPU cormat for X-vs-Y correlation."""
        from spatialgpu.core.gpu_ops import gpu_cormat
        import cupy as cp

        rng = np.random.RandomState(42)
        X = rng.randn(100, 50).astype(np.float64)  # 100 genes x 50 samples
        Y = rng.randn(100, 1).astype(np.float64)

        # CPU reference
        from spatialgpu.deconvolution.core import cormat
        expected = cormat(X, Y, method="spearman")

        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)
        rs, ps = gpu_cormat(X_gpu, Y_gpu, method="spearman")
        rs_cpu = cp.asnumpy(rs)
        ps_cpu = cp.asnumpy(ps)

        np.testing.assert_allclose(rs_cpu, expected["cor_r"].values, atol=1e-3)
        np.testing.assert_allclose(ps_cpu, expected["cor_p"].values, atol=1e-3)

    @skipno_gpu
    def test_diagonal_is_one(self):
        """Pairwise Spearman has 1.0 on diagonal."""
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr
        import cupy as cp

        rng = np.random.RandomState(42)
        mat = rng.randn(10, 100).astype(np.float64)

        mat_gpu = cp.asarray(mat)
        rho_gpu, _ = gpu_pairwise_spearmanr(mat_gpu)
        rho = cp.asnumpy(rho_gpu)

        np.testing.assert_allclose(np.diag(rho), 1.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUSpearmanr -v 2>&1 | head -20`
Expected: FAIL with "cannot import name 'gpu_pairwise_spearmanr'"

- [ ] **Step 3: Write implementation**

Append to `spatialgpu/core/gpu_ops.py`:

```python
def gpu_pairwise_spearmanr(mat):
    """Pairwise Spearman correlation for rows of mat (n_vars x n_obs).

    Parameters
    ----------
    mat : cupy.ndarray
        Shape (n_vars, n_obs). Each row is a variable.

    Returns
    -------
    rho : cupy.ndarray
        (n_vars, n_vars) Spearman correlation matrix.
    pval : cupy.ndarray
        (n_vars, n_vars) two-sided p-value matrix.
    """
    import cupy as cp
    from cupyx.scipy.special import betainc  # noqa: F401 — check availability

    n_vars, n_obs = mat.shape

    # Rank each row
    ranked = gpu_rankdata(mat, method="average", axis=1)

    # Center and standardize
    ranked_centered = ranked - ranked.mean(axis=1, keepdims=True)
    std = cp.sqrt((ranked_centered ** 2).sum(axis=1, keepdims=True))
    std = cp.where(std == 0, 1.0, std)
    ranked_norm = ranked_centered / std

    # Correlation via matmul
    rho = ranked_norm @ ranked_norm.T / n_obs

    # Clip to [-1, 1] to avoid numerical issues
    rho = cp.clip(rho, -1.0, 1.0)

    # P-values from t-distribution: t = r * sqrt((n-2)/(1-r^2))
    t_stat = rho * cp.sqrt((n_obs - 2) / (1 - rho ** 2 + 1e-300))
    # Two-sided p-value via regularized incomplete beta function
    # p = 2 * betainc(df/2, 0.5, df/(df + t^2))  where df = n_obs - 2
    df = float(n_obs - 2)
    x = df / (df + t_stat ** 2)

    try:
        from cupyx.scipy import special as cp_special
        pval = 2.0 * cp_special.betainc(
            cp.float64(df / 2.0), cp.float64(0.5), x
        )
    except (ImportError, AttributeError):
        # Fallback: compute p-values on CPU
        from scipy import stats as sp_stats
        t_cpu = cp.asnumpy(t_stat)
        pval_cpu = 2.0 * sp_stats.t.sf(np.abs(t_cpu), df=n_obs - 2)
        pval = cp.asarray(pval_cpu)

    # Fix diagonal
    cp.fill_diagonal(rho, 1.0)
    cp.fill_diagonal(pval, 0.0)

    # Replace NaN
    rho = cp.where(cp.isnan(rho), 0.0, rho)
    pval = cp.where(cp.isnan(pval), 1.0, pval)

    return rho, pval


def gpu_cormat(X, Y, method="spearman"):
    """GPU version of cormat: correlate columns of X with columns of Y.

    Parameters
    ----------
    X : cupy.ndarray
        (n_genes, n_samples)
    Y : cupy.ndarray
        (n_genes, n_features)
    method : str
        'pearson' or 'spearman'

    Returns
    -------
    rs : cupy.ndarray
        Correlation coefficients, shape (n_samples,) per feature.
    ps : cupy.ndarray
        P-values, shape (n_samples,) per feature.
    """
    import cupy as cp
    from scipy import stats as sp_stats

    n_obs = X.shape[0]

    if method == "spearman":
        X_prep = gpu_rankdata(X, method="average", axis=0)
    elif method == "pearson":
        X_prep = X.astype(cp.float64)
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman', got '{method}'")

    X_centered = X_prep - X_prep.mean(axis=0, keepdims=True)
    std_x = cp.sqrt((X_centered ** 2).sum(axis=0))

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    all_rs = []
    all_ps = []

    for j in range(Y.shape[1]):
        y_col = Y[:, j]
        if method == "spearman":
            y_prep = gpu_rankdata(y_col, method="average")
        else:
            y_prep = y_col.astype(cp.float64)

        y_centered = y_prep - y_prep.mean()
        cov_xy = (X_centered * y_centered[:, None]).sum(axis=0)
        std_y = cp.sqrt((y_centered ** 2).sum())
        denom = std_x * std_y
        denom = cp.where(denom == 0, 1.0, denom)
        rs = cov_xy / denom

        t_stat = rs * cp.sqrt((n_obs - 2) / (1 - rs ** 2 + 1e-300))
        # P-values on CPU (scipy t-distribution)
        t_cpu = cp.asnumpy(t_stat)
        ps_cpu = 2 * sp_stats.t.sf(np.abs(t_cpu), df=n_obs - 2)
        ps = cp.asarray(ps_cpu)

        all_rs.append(rs)
        all_ps.append(ps)

    return cp.round(all_rs[0], 3), all_ps[0]
```

Also add at the top of `gpu_ops.py`:

```python
import numpy as np
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUSpearmanr -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU Spearman correlation primitives with equivalence tests"
```

---

### Task 3: GPU NNLS primitive

**Files:**
- Modify: `spatialgpu/core/gpu_ops.py`
- Modify: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_ops.py`:

```python
from scipy.optimize import nnls as scipy_nnls


class TestGPUNNLS:
    """Tests for GPU NNLS solver."""

    @skipno_gpu
    def test_basic_nnls(self):
        """GPU NNLS matches scipy for a single system."""
        from spatialgpu.core.gpu_ops import gpu_nnls
        import cupy as cp

        rng = np.random.RandomState(42)
        A = rng.randn(50, 5).astype(np.float64)
        x_true = np.abs(rng.randn(5))
        b = A @ x_true + 0.1 * rng.randn(50)

        expected, _ = scipy_nnls(A, b)

        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        result = gpu_nnls(A_gpu, b_gpu)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_batch_nnls(self):
        """GPU batch NNLS matches scipy for multiple RHS columns."""
        from spatialgpu.core.gpu_ops import gpu_nnls_batch
        import cupy as cp

        rng = np.random.RandomState(42)
        A = rng.randn(50, 5).astype(np.float64)
        B = np.abs(rng.randn(50, 20))

        expected = np.column_stack([scipy_nnls(A, B[:, i])[0] for i in range(20)])

        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        result = gpu_nnls_batch(A_gpu, B_gpu)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-5)

    @skipno_gpu
    def test_nnls_all_zero_rhs(self):
        """GPU NNLS returns zeros for zero RHS."""
        from spatialgpu.core.gpu_ops import gpu_nnls
        import cupy as cp

        A = cp.eye(5, dtype=cp.float64)
        b = cp.zeros(5, dtype=cp.float64)
        result = gpu_nnls(A, b)

        np.testing.assert_allclose(cp.asnumpy(result), 0.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUNNLS -v 2>&1 | head -20`
Expected: FAIL with "cannot import name 'gpu_nnls'"

- [ ] **Step 3: Write implementation**

Append to `spatialgpu/core/gpu_ops.py`:

```python
def gpu_nnls(A, b, max_iter=None):
    """Non-negative least squares on GPU (active-set, matching scipy.optimize.nnls).

    Solves: min ||A @ x - b||^2  s.t.  x >= 0

    Uses the Lawson-Hanson active-set algorithm on GPU.

    Parameters
    ----------
    A : cupy.ndarray
        Coefficient matrix (m, n).
    b : cupy.ndarray
        Right-hand side (m,).
    max_iter : int or None
        Maximum iterations. Default: 3 * n.

    Returns
    -------
    x : cupy.ndarray
        Solution (n,).
    """
    import cupy as cp

    m, n = A.shape
    if max_iter is None:
        max_iter = 3 * n

    AtA = A.T @ A
    Atb = A.T @ b

    x = cp.zeros(n, dtype=cp.float64)
    passive = cp.zeros(n, dtype=cp.bool_)  # passive set (unconstrained)

    w = Atb - AtA @ x  # gradient

    for _ in range(max_iter):
        # Check if any active variable wants to enter passive set
        active_mask = ~passive
        if not cp.any(active_mask):
            break

        w_active = w.copy()
        w_active[passive] = -cp.inf

        if cp.max(w_active) <= 0:
            break

        # Move variable with largest gradient to passive set
        t = int(cp.argmax(w_active))
        passive[t] = True

        # Solve unconstrained subproblem on passive set
        while True:
            passive_idx = cp.where(passive)[0]
            A_sub = AtA[cp.ix_(passive_idx, passive_idx)]
            b_sub = Atb[passive_idx]

            z = cp.zeros(n, dtype=cp.float64)
            try:
                z[passive_idx] = cp.linalg.solve(A_sub, b_sub)
            except cp.linalg.LinAlgError:
                # Singular: use pseudoinverse
                z[passive_idx] = cp.linalg.lstsq(
                    A[..., cp.asnumpy(passive_idx).tolist()],
                    b, rcond=None
                )[0]

            # Check for negative entries in passive set
            neg_mask = passive & (z <= 0)
            if not cp.any(neg_mask):
                x = z.copy()
                break

            # Find alpha: step to boundary
            neg_idx = cp.where(neg_mask)[0]
            alpha_vals = x[neg_idx] / (x[neg_idx] - z[neg_idx])
            alpha = float(cp.min(alpha_vals))

            x = x + alpha * (z - x)

            # Remove variables that hit zero from passive set
            passive = passive & ~((cp.abs(x) < 1e-15) & passive)

        w = Atb - AtA @ x

    return cp.maximum(x, 0.0)


def gpu_nnls_batch(A, B):
    """Batch NNLS: solve NNLS for each column of B.

    Parameters
    ----------
    A : cupy.ndarray
        Coefficient matrix (m, n).
    B : cupy.ndarray
        RHS matrix (m, k).

    Returns
    -------
    X : cupy.ndarray
        Solutions (n, k).
    """
    import cupy as cp

    n = A.shape[1]
    k = B.shape[1]
    X = cp.empty((n, k), dtype=cp.float64)

    for i in range(k):
        X[:, i] = gpu_nnls(A, B[:, i])

    return X
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUNNLS -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU NNLS solver primitive with equivalence tests"
```

---

### Task 4: GPU pairwise distance and permutation primitives

**Files:**
- Modify: `spatialgpu/core/gpu_ops.py`
- Modify: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_ops.py`:

```python
from scipy.spatial.distance import cdist as scipy_cdist


class TestGPUCdist:
    """Tests for GPU pairwise distance."""

    @skipno_gpu
    def test_euclidean(self):
        """GPU cdist matches scipy for Euclidean distance."""
        from spatialgpu.core.gpu_ops import gpu_cdist
        import cupy as cp

        rng = np.random.RandomState(42)
        A = rng.randn(50, 2).astype(np.float64)
        B = rng.randn(30, 2).astype(np.float64)

        expected = scipy_cdist(A, B, metric="euclidean")

        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        result = gpu_cdist(A_gpu, B_gpu)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-6)

    @skipno_gpu
    def test_self_distance(self):
        """GPU cdist self-distance has zero diagonal."""
        from spatialgpu.core.gpu_ops import gpu_cdist
        import cupy as cp

        rng = np.random.RandomState(42)
        A = rng.randn(20, 2).astype(np.float64)

        A_gpu = cp.asarray(A)
        result = gpu_cdist(A_gpu, A_gpu)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(np.diag(result_cpu), 0.0, atol=1e-10)


class TestGPUBipartiteEdgeSwap:
    """Tests for GPU bipartite edge swap."""

    @skipno_gpu
    def test_degree_preservation(self):
        """GPU edge swap preserves row and column degrees."""
        from spatialgpu.core.gpu_ops import gpu_bipartite_edge_swap
        import cupy as cp

        rng = np.random.RandomState(42)
        mat = (rng.rand(10, 8) > 0.7).astype(np.int32)

        row_deg_before = mat.sum(axis=1)
        col_deg_before = mat.sum(axis=0)

        mat_gpu = cp.asarray(mat)
        result = gpu_bipartite_edge_swap(mat_gpu, seed=42)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_array_equal(result_cpu.sum(axis=1), row_deg_before)
        np.testing.assert_array_equal(result_cpu.sum(axis=0), col_deg_before)

    @skipno_gpu
    def test_edge_count_preserved(self):
        """GPU edge swap preserves total edge count."""
        from spatialgpu.core.gpu_ops import gpu_bipartite_edge_swap
        import cupy as cp

        rng = np.random.RandomState(42)
        mat = (rng.rand(15, 12) > 0.6).astype(np.int32)
        total_edges = mat.sum()

        mat_gpu = cp.asarray(mat)
        result = gpu_bipartite_edge_swap(mat_gpu, seed=42)

        assert int(cp.sum(result)) == total_edges
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUCdist tests/test_gpu_ops.py::TestGPUBipartiteEdgeSwap -v 2>&1 | head -20`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Append to `spatialgpu/core/gpu_ops.py`:

```python
def gpu_cdist(A, B):
    """Pairwise Euclidean distance on GPU.

    Parameters
    ----------
    A : cupy.ndarray
        (n, d) array.
    B : cupy.ndarray
        (m, d) array.

    Returns
    -------
    cupy.ndarray
        (n, m) distance matrix.
    """
    import cupy as cp

    A_sq = cp.sum(A ** 2, axis=1, keepdims=True)
    B_sq = cp.sum(B ** 2, axis=1, keepdims=True).T
    dist_sq = A_sq + B_sq - 2.0 * A @ B.T
    return cp.sqrt(cp.maximum(dist_sq, 0.0))


def gpu_bipartite_edge_swap(mat, n_swaps=None, seed=None):
    """Degree-preserving bipartite edge swap on GPU.

    Matches _bipartite_edge_swap from interaction.py.

    Parameters
    ----------
    mat : cupy.ndarray
        Binary adjacency matrix (ligands x receptors), int32.
    n_swaps : int or None
        Number of swap attempts. Default: 5 * n_edges.
    seed : int or None
        Random seed.

    Returns
    -------
    cupy.ndarray
        Rewired adjacency matrix with preserved degrees.
    """
    import cupy as cp

    if seed is not None:
        cp.random.seed(seed)

    mat = mat.copy()
    edges_l, edges_r = cp.where(mat == 1)
    n_edges = len(edges_l)

    if n_edges < 2:
        return mat

    if n_swaps is None:
        n_swaps = 5 * int(n_edges)

    # Generate all random indices at once
    rand_pairs = cp.random.randint(0, int(n_edges), size=(n_swaps, 2))

    for s in range(n_swaps):
        idx1, idx2 = int(rand_pairs[s, 0]), int(rand_pairs[s, 1])
        if idx1 == idx2:
            continue

        l1, r1 = int(edges_l[idx1]), int(edges_r[idx1])
        l2, r2 = int(edges_l[idx2]), int(edges_r[idx2])

        if l1 == l2 or r1 == r2:
            continue

        if mat[l1, r2] == 1 or mat[l2, r1] == 1:
            continue

        mat[l1, r1] = 0
        mat[l2, r2] = 0
        mat[l1, r2] = 1
        mat[l2, r1] = 1

        edges_r[idx1] = r2
        edges_r[idx2] = r1

    return mat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUCdist tests/test_gpu_ops.py::TestGPUBipartiteEdgeSwap -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU cdist and bipartite edge swap primitives"
```

---

### Task 5: GPU NMF primitive

**Files:**
- Modify: `spatialgpu/core/gpu_ops.py`
- Modify: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_ops.py`:

```python
class TestGPUNMF:
    """Tests for GPU NMF."""

    @skipno_gpu
    def test_reconstruction(self):
        """GPU NMF reconstructs input within tolerance."""
        from spatialgpu.core.gpu_ops import gpu_nmf
        import cupy as cp

        rng = np.random.RandomState(42)
        V = np.abs(rng.randn(50, 100)).astype(np.float64)

        V_gpu = cp.asarray(V)
        W, H = gpu_nmf(V_gpu, n_components=5, seed=42, max_iter=500)
        W_cpu = cp.asnumpy(W)
        H_cpu = cp.asnumpy(H)

        reconstruction = W_cpu @ H_cpu
        error = np.linalg.norm(V - reconstruction) / np.linalg.norm(V)
        assert error < 0.5  # NMF is approximate

    @skipno_gpu
    def test_matches_sklearn(self):
        """GPU NMF matches sklearn NMF output closely."""
        from spatialgpu.core.gpu_ops import gpu_nmf
        from sklearn.decomposition import NMF
        import cupy as cp

        rng = np.random.RandomState(42)
        V = np.abs(rng.randn(30, 80)).astype(np.float64)

        # sklearn reference
        model = NMF(n_components=3, random_state=42, max_iter=500)
        W_sk = model.fit_transform(V)
        H_sk = model.components_
        err_sk = np.linalg.norm(V - W_sk @ H_sk)

        # GPU
        V_gpu = cp.asarray(V)
        W_gpu, H_gpu = gpu_nmf(V_gpu, n_components=3, seed=42, max_iter=500)
        W_cu = cp.asnumpy(W_gpu)
        H_cu = cp.asnumpy(H_gpu)
        err_cu = np.linalg.norm(V - W_cu @ H_cu)

        # GPU error should be comparable to sklearn
        assert err_cu < err_sk * 1.5

    @skipno_gpu
    def test_non_negative(self):
        """GPU NMF produces non-negative W and H."""
        from spatialgpu.core.gpu_ops import gpu_nmf
        import cupy as cp

        V = cp.abs(cp.random.randn(20, 50))
        W, H = gpu_nmf(V, n_components=3, seed=42)

        assert float(cp.min(W)) >= 0.0
        assert float(cp.min(H)) >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUNMF -v 2>&1 | head -20`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Append to `spatialgpu/core/gpu_ops.py`:

```python
def gpu_nmf(V, n_components, seed=None, max_iter=500, tol=1e-4):
    """Non-negative Matrix Factorization on GPU (multiplicative update).

    Matches sklearn's NMF with solver='mu' (multiplicative update rules).

    Parameters
    ----------
    V : cupy.ndarray
        Non-negative input matrix (n, m).
    n_components : int
        Number of components (k).
    seed : int or None
        Random seed for initialization.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    W : cupy.ndarray
        Basis matrix (n, k).
    H : cupy.ndarray
        Coefficient matrix (k, m).
    """
    import cupy as cp

    if seed is not None:
        cp.random.seed(seed)

    n, m = V.shape
    eps = 1e-16

    # NNDSVD-like initialization (simplified: random + scale)
    avg = cp.sqrt(cp.mean(V) / n_components)
    W = cp.abs(avg * cp.random.randn(n, n_components).astype(cp.float64))
    H = cp.abs(avg * cp.random.randn(n_components, m).astype(cp.float64))

    prev_err = float("inf")

    for iteration in range(max_iter):
        # Update H
        WtV = W.T @ V
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH

        # Update W
        VHt = V @ H.T
        WHHt = W @ (H @ H.T) + eps
        W *= VHt / WHHt

        # Check convergence every 10 iterations
        if (iteration + 1) % 10 == 0:
            err = float(cp.linalg.norm(V - W @ H))
            if abs(prev_err - err) / (prev_err + eps) < tol:
                break
            prev_err = err

    return W, H
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUNMF -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU NMF primitive with multiplicative update rules"
```

---

### Task 6: GPU trust-constr QP solver primitive

**Files:**
- Modify: `spatialgpu/core/gpu_ops.py`
- Modify: `tests/test_gpu_ops.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_ops.py`:

```python
class TestGPUTrustConstr:
    """Tests for GPU constrained QP solver."""

    @skipno_gpu
    def test_matches_scipy_trust_constr(self):
        """GPU QP solver matches scipy trust-constr for deconvolution-like problem."""
        from spatialgpu.core.gpu_ops import gpu_solve_qp
        import cupy as cp
        from scipy.optimize import Bounds, LinearConstraint, minimize

        rng = np.random.RandomState(42)
        n_genes, n_cell = 50, 5
        A = rng.randn(n_genes, n_cell).astype(np.float64)
        x_true = np.abs(rng.randn(n_cell)) * 0.2
        b = A @ x_true + 0.05 * rng.randn(n_genes)

        AtA = A.T @ A
        Atb = A.T @ b
        ppmin, ppmax = 0.1, 0.8

        # scipy reference
        def f(th):
            return float(th @ AtA @ th - 2 * Atb @ th)
        def g(th):
            return 2 * (AtA @ th - Atb)
        def h(th):
            return 2 * AtA

        theta0 = np.full(n_cell, 0.5 * ppmax / n_cell)
        bnds = Bounds(lb=np.zeros(n_cell), ub=np.full(n_cell, np.inf))
        lc = LinearConstraint(np.ones((1, n_cell)), lb=ppmin, ub=ppmax)

        res = minimize(f, theta0, jac=g, hess=h, method="trust-constr",
                       bounds=bnds, constraints=lc,
                       options={"maxiter": 500, "gtol": 1e-15})
        expected = np.clip(res.x, 0, None)

        # GPU
        AtA_gpu = cp.asarray(AtA)
        Atb_gpu = cp.asarray(Atb)
        result = gpu_solve_qp(AtA_gpu, Atb_gpu, n_cell, ppmin, ppmax)
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-4)

    @skipno_gpu
    def test_batch_qp(self):
        """GPU batch QP solver matches per-spot scipy solutions."""
        from spatialgpu.core.gpu_ops import gpu_solve_qp_batch
        import cupy as cp
        from scipy.optimize import Bounds, LinearConstraint, minimize

        rng = np.random.RandomState(42)
        n_genes, n_cell, n_spots = 30, 4, 10
        A = rng.randn(n_genes, n_cell).astype(np.float64)
        B = rng.randn(n_genes, n_spots).astype(np.float64)

        AtA = A.T @ A
        ppmin_arr = np.full(n_spots, 0.1)
        ppmax_arr = np.full(n_spots, 0.8)

        # scipy per-spot reference
        expected = np.zeros((n_cell, n_spots))
        for i in range(n_spots):
            Atb = A.T @ B[:, i]
            def f(th, _Atb=Atb):
                return float(th @ AtA @ th - 2 * _Atb @ th)
            def g(th, _Atb=Atb):
                return 2 * (AtA @ th - _Atb)
            def h(th):
                return 2 * AtA
            theta0 = np.full(n_cell, 0.05)
            bnds = Bounds(lb=np.zeros(n_cell), ub=np.full(n_cell, np.inf))
            lc = LinearConstraint(np.ones((1, n_cell)), lb=ppmin_arr[i], ub=ppmax_arr[i])
            res = minimize(f, theta0, jac=g, hess=h, method="trust-constr",
                           bounds=bnds, constraints=lc,
                           options={"maxiter": 500, "gtol": 1e-15})
            expected[:, i] = np.clip(res.x, 0, None)

        # GPU batch
        AtA_gpu = cp.asarray(AtA)
        B_gpu = cp.asarray(B)
        A_gpu = cp.asarray(A)
        result = gpu_solve_qp_batch(
            A_gpu, AtA_gpu, B_gpu, n_cell,
            cp.asarray(ppmin_arr), cp.asarray(ppmax_arr)
        )
        result_cpu = cp.asnumpy(result)

        np.testing.assert_allclose(result_cpu, expected, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUTrustConstr -v 2>&1 | head -20`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Append to `spatialgpu/core/gpu_ops.py`:

```python
def gpu_solve_qp(AtA, Atb, n_cell, ppmin, ppmax, max_iter=500, gtol=1e-15):
    """Solve a constrained QP on GPU matching scipy trust-constr behavior.

    min  x^T AtA x - 2 Atb^T x
    s.t. x >= 0,  ppmin <= sum(x) <= ppmax

    Uses projected gradient descent with sum constraint.

    Parameters
    ----------
    AtA : cupy.ndarray
        (n_cell, n_cell) Gram matrix.
    Atb : cupy.ndarray
        (n_cell,) target vector.
    n_cell : int
        Number of variables.
    ppmin : float
        Lower bound on sum(x).
    ppmax : float
        Upper bound on sum(x).
    max_iter : int
        Maximum iterations.
    gtol : float
        Gradient tolerance.

    Returns
    -------
    cupy.ndarray
        Solution (n_cell,).
    """
    import cupy as cp

    two_AtA = 2.0 * AtA

    # Initialize
    x = cp.full(n_cell, 0.5 * ppmax / n_cell, dtype=cp.float64)

    # Step size from Lipschitz constant of gradient
    L = float(cp.max(cp.abs(cp.linalg.eigvalsh(two_AtA))))
    step = 1.0 / max(L, 1e-10)

    for _ in range(max_iter):
        grad = two_AtA @ x - 2.0 * Atb

        if float(cp.linalg.norm(grad)) < gtol:
            break

        # Gradient step
        x_new = x - step * grad

        # Project onto non-negative orthant
        x_new = cp.maximum(x_new, 0.0)

        # Project onto sum constraint [ppmin, ppmax]
        s = float(cp.sum(x_new))
        if s > ppmax and s > 0:
            x_new = x_new * (ppmax / s)
        elif s < ppmin and s >= 0:
            # Add uniform to meet minimum
            deficit = ppmin - s
            x_new = x_new + deficit / n_cell

        x = x_new

    return cp.maximum(x, 0.0)


def gpu_solve_qp_batch(A, AtA, B, n_cell, ppmin_arr, ppmax_arr):
    """Batch QP solver: solve for each column of B.

    Parameters
    ----------
    A : cupy.ndarray
        (n_genes, n_cell)
    AtA : cupy.ndarray
        (n_cell, n_cell)
    B : cupy.ndarray
        (n_genes, n_spots)
    n_cell : int
        Number of cell types.
    ppmin_arr : cupy.ndarray
        (n_spots,) per-spot lower bounds.
    ppmax_arr : cupy.ndarray
        (n_spots,) per-spot upper bounds.

    Returns
    -------
    cupy.ndarray
        (n_cell, n_spots) solutions.
    """
    import cupy as cp

    At = A.T
    n_spots = B.shape[1]
    result = cp.empty((n_cell, n_spots), dtype=cp.float64)

    for i in range(n_spots):
        Atb = At @ B[:, i]
        ppmin = float(ppmin_arr[i])
        ppmax = float(ppmax_arr[i])

        if ppmax <= 0.01:
            result[:, i] = max(ppmax, 0) / n_cell
            continue

        result[:, i] = gpu_solve_qp(AtA, Atb, n_cell, ppmin, ppmax)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_ops.py::TestGPUTrustConstr -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/core/gpu_ops.py tests/test_gpu_ops.py
git commit -m "Add GPU constrained QP solver primitive for deconvolution Level 2"
```

---

### Task 7: Integrate GPU into cormat (core.py)

**Files:**
- Modify: `spatialgpu/deconvolution/core.py`

- [ ] **Step 1: Add GPU dispatch to cormat**

In `spatialgpu/deconvolution/core.py`, modify `cormat()` (line 379) to check backend and dispatch:

```python
def cormat(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation between columns of X and columns of Y."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        return _cormat_gpu(X, Y, method)

    # --- existing CPU code below (unchanged) ---
    from statsmodels.stats.multitest import multipletests
    # ... rest of existing implementation ...
```

Then add the GPU function:

```python
def _cormat_gpu(X, Y, method):
    """GPU implementation of cormat."""
    import cupy as cp
    from statsmodels.stats.multitest import multipletests

    from spatialgpu.core.gpu_ops import gpu_cormat

    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

    cor_r, cor_p = gpu_cormat(X_gpu, Y_gpu, method=method)
    cor_r = cp.asnumpy(cor_r)
    cor_p = cp.asnumpy(cor_p)

    cor_p = np.array([float(f"{p:.3g}") for p in cor_p])
    _, cor_padj, _, _ = multipletests(cor_p, method="fdr_bh")

    return pd.DataFrame({"cor_r": cor_r, "cor_p": cor_p, "cor_padj": cor_padj})
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `python -m pytest tests/test_deconvolution/test_core.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add spatialgpu/deconvolution/core.py
git commit -m "Add GPU dispatch to cormat in deconvolution core"
```

---

### Task 8: Integrate GPU into NNLS and trust-constr solvers (core.py)

**Files:**
- Modify: `spatialgpu/deconvolution/core.py`

- [ ] **Step 1: Add GPU dispatch to _solve_nnls**

Modify `_solve_nnls()` (line 1141):

```python
def _solve_nnls(
    A: np.ndarray,
    B: np.ndarray,
    n_cell: int,
    theta_sum: np.ndarray,
    pp_max_arr: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Fast NNLS solver for Level 1 (ppmin=0)."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        return _solve_nnls_gpu(A, B, n_cell, theta_sum, pp_max_arr)

    # --- existing CPU code (unchanged) ---
    from joblib import Parallel, delayed
    from scipy.optimize import nnls
    # ... rest of existing implementation ...
```

Add the GPU function:

```python
def _solve_nnls_gpu(A, B, n_cell, theta_sum, pp_max_arr):
    """GPU implementation of two-pass NNLS solver."""
    import cupy as cp

    from spatialgpu.core.gpu_ops import gpu_nnls

    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)

    n_spots = B.shape[1]
    result = cp.empty((n_cell, n_spots), dtype=cp.float64)

    for i in range(n_spots):
        ts = float(theta_sum[i])
        if ts <= 0.01:
            result[:, i] = max(ts, 0) / n_cell
            continue

        b = B_gpu[:, i]
        ppmax = float(pp_max_arr[i])

        # Pass 1: unweighted NNLS
        prop = gpu_nnls(A_gpu, b)
        s = float(cp.sum(prop))
        if s > ppmax and s > 0:
            prop = prop * (ppmax / s)

        # Pass 2: weighted NNLS
        bhat = A_gpu @ prop
        w = 1.0 / cp.sqrt(bhat + 1.0)
        Aw = A_gpu * w[:, None]
        bw = b * w
        prop2 = gpu_nnls(Aw, bw)
        s2 = float(cp.sum(prop2))
        if s2 > ppmax and s2 > 0:
            prop2 = prop2 * (ppmax / s2)

        result[:, i] = prop2

    return cp.asnumpy(result)
```

- [ ] **Step 2: Add GPU dispatch to _solve_trust_constr**

Modify `_solve_trust_constr()` (line 1189):

```python
def _solve_trust_constr(
    A: np.ndarray,
    B: np.ndarray,
    n_cell: int,
    theta_sum: np.ndarray,
    pp_min_arr: np.ndarray,
    pp_max_arr: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Trust-constr solver for Level 2 (ppmin>0, needs both bounds)."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        return _solve_trust_constr_gpu(A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr)

    # --- existing CPU code (unchanged) ---
    import warnings
    from joblib import Parallel, delayed
    from scipy.optimize import Bounds, LinearConstraint, minimize
    # ... rest of existing implementation ...
```

Add:

```python
def _solve_trust_constr_gpu(A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr):
    """GPU implementation of two-pass trust-constr solver."""
    import cupy as cp

    from spatialgpu.core.gpu_ops import gpu_nnls, gpu_solve_qp

    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    AtA = A_gpu.T @ A_gpu

    n_spots = B.shape[1]
    result = cp.empty((n_cell, n_spots), dtype=cp.float64)

    for i in range(n_spots):
        ts = float(theta_sum[i])
        if ts <= 0.01:
            result[:, i] = max(ts, 0) / n_cell
            continue

        b = B_gpu[:, i]
        ppmin = float(pp_min_arr[i])
        ppmax = float(pp_max_arr[i])

        Atb = A_gpu.T @ b

        # Pass 1: constrained QP
        prop = gpu_solve_qp(AtA, Atb, n_cell, ppmin, ppmax)

        # Pass 2: weighted
        bhat = A_gpu @ prop
        w = 1.0 / cp.sqrt(bhat + 1.0)
        Aw = A_gpu * w[:, None]
        bw = b * w
        AtA_w = Aw.T @ Aw
        Atb_w = Aw.T @ bw

        prop2 = gpu_solve_qp(AtA_w, Atb_w, n_cell, ppmin, ppmax)
        result[:, i] = prop2

    return cp.asnumpy(result)
```

- [ ] **Step 3: Run existing tests**

Run: `python -m pytest tests/test_deconvolution/test_core.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add spatialgpu/deconvolution/core.py
git commit -m "Add GPU dispatch to NNLS and trust-constr solvers"
```

---

### Task 9: Integrate GPU into interaction.py

**Files:**
- Modify: `spatialgpu/deconvolution/interaction.py`

- [ ] **Step 1: Add GPU dispatch to _pairwise_spearmanr**

Modify `_pairwise_spearmanr()` (line 540):

```python
def _pairwise_spearmanr(
    mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Spearman correlation for rows of mat."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_pairwise_spearmanr

        mat_gpu = cp.asarray(mat)
        rho_gpu, pval_gpu = gpu_pairwise_spearmanr(mat_gpu)
        rho = cp.asnumpy(rho_gpu)
        pval = cp.asnumpy(pval_gpu)

        np.fill_diagonal(rho, 1.0)
        np.fill_diagonal(pval, 0.0)
        return rho, pval

    # --- existing CPU code (unchanged) ---
    n = mat.shape[0]
    result = stats.spearmanr(mat, axis=1)
    # ...
```

- [ ] **Step 2: Add GPU dispatch to cci_lr_network_score**

In `cci_lr_network_score()` (line 172), add GPU paths for the two expensive loops. After the line that generates `lr_mat` and before the permutation loop (line 265):

```python
    from spatialgpu.core.backend import get_backend
    backend = get_backend()

    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_bipartite_edge_swap

        # GPU permuted networks
        logger.info("  Generating 1000 permuted networks (GPU)...")
        for _perm_i in range(1000):
            lr_mat_gpu = cp.asarray(lr_mat.copy().astype(np.int32))
            perm_mat = gpu_bipartite_edge_swap(lr_mat_gpu, seed=123456 + _perm_i)
            perm_mat_cpu = cp.asnumpy(perm_mat)
            li_arr, ri_arr = np.where(perm_mat_cpu == 1)
            all_perm_l_indices.append(lig_gene_indices[li_arr])
            all_perm_r_indices.append(rec_gene_indices[ri_arr])
    else:
        # existing CPU loop
        for _perm_i in range(1000):
            perm_mat = _bipartite_edge_swap(lr_mat.copy(), rng)
            # ...
```

And for the score computation loop (line 292), GPU-vectorize:

```python
    if backend.is_gpu_active:
        import cupy as cp
        st_sub_gpu = cp.asarray(st_sub)
        idx_map_gpu = cp.asarray(idx_map)

        perm_scores_all = cp.empty((1000, n_spots), dtype=cp.float64)
        for p in range(1000):
            perm_l = cp.asarray(all_perm_l_indices[p])
            perm_r = cp.asarray(all_perm_r_indices[p])
            perm_scores_all[p] = cp.mean(
                st_sub_gpu[idx_map_gpu[perm_l], :] *
                st_sub_gpu[idx_map_gpu[perm_r], :],
                axis=0,
            )
        perm_scores_all = cp.asnumpy(perm_scores_all)
    else:
        # existing CPU loop
        perm_scores_all = np.empty((1000, n_spots), dtype=np.float64)
        for p in range(1000):
            # ...
```

- [ ] **Step 3: Add GPU dispatch to distance_to_interface**

In `distance_to_interface()`, replace the permutation loop with GPU cdist. After computing `border_coords` and `both_spots`:

```python
    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_cdist

        border_gpu = cp.asarray(border_coords)

        def _mean_min_distance_gpu(spots):
            if len(spots) == 0:
                return np.inf
            coords = np.array([_spot_to_coords(s) for s in spots])
            coords_gpu = cp.asarray(coords)
            dists = gpu_cdist(coords_gpu, border_gpu)
            return float(cp.mean(cp.min(dists, axis=1)))

        d_observed = _mean_min_distance_gpu(both_spots)
        # permutation loop uses GPU cdist too
    else:
        d_observed = _mean_min_distance(both_spots)
```

- [ ] **Step 4: Run existing tests**

Run: `python -m pytest tests/test_deconvolution/test_interaction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/deconvolution/interaction.py
git commit -m "Add GPU dispatch to Spearman, LR network score, and distance_to_interface"
```

---

### Task 10: Integrate GPU into gene_set_score.py

**Files:**
- Modify: `spatialgpu/deconvolution/gene_set_score.py`

- [ ] **Step 1: Add GPU dispatch to _ucell_score**

Modify `_ucell_score()` (line 68):

```python
def _ucell_score(
    adata: ad.AnnData,
    gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute UCell-like gene set scores."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    X = adata.X
    gene_names = np.array(adata.var_names)
    spot_names = np.array(adata.obs_names)
    n_spots = X.shape[0]
    n_genes = X.shape[1]

    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        X_gpu = cp.asarray(X_dense.astype(np.float32))
        ranks_gpu = n_genes + 1 - gpu_rankdata(X_gpu, method="average", axis=1)
        ranks = cp.asnumpy(ranks_gpu)
    else:
        from scipy.stats import rankdata

        ranks = np.apply_along_axis(
            lambda row: n_genes + 1 - rankdata(row, method="average"),
            axis=1,
            arr=X_dense,
        )

    # Rest of the function is the same (set scoring from ranks)
    results = {}
    for set_name, genes in gene_sets.items():
        gene_mask = np.isin(gene_names, genes)
        n_set = gene_mask.sum()
        if n_set == 0:
            results[set_name] = np.zeros(n_spots)
            continue
        set_ranks = ranks[:, gene_mask]
        mean_rank = set_ranks.mean(axis=1)
        scores = 1 - mean_rank / n_genes
        results[set_name] = scores

    score_df = pd.DataFrame(results, index=spot_names).T
    return score_df
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_deconvolution/test_gene_set_score.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add spatialgpu/deconvolution/gene_set_score.py
git commit -m "Add GPU dispatch to UCell gene set scoring"
```

---

### Task 11: Integrate GPU into secact.py

**Files:**
- Modify: `spatialgpu/deconvolution/secact.py`

- [ ] **Step 1: Add GPU dispatch to secact_signaling_patterns**

In `secact_signaling_patterns()` (line 199), add GPU path for the Spearman correlation loop (line 278) and NMF (line 349):

```python
    # Spearman correlation filtering
    from spatialgpu.core.backend import get_backend
    backend = get_backend()

    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_rankdata

        # Vectorized GPU Spearman for all genes at once
        act_vals = cp.asarray(act_new.values)  # (n_proteins, n_spots)
        exp_vals = cp.asarray(expr_new_aggr.reindex(act_new.index).values)

        # Rank both matrices row-wise
        act_ranked = gpu_rankdata(act_vals, method="average", axis=1)
        exp_ranked = gpu_rankdata(exp_vals, method="average", axis=1)

        # Centered
        act_c = act_ranked - act_ranked.mean(axis=1, keepdims=True)
        exp_c = exp_ranked - exp_ranked.mean(axis=1, keepdims=True)

        # Per-row correlation
        num = (act_c * exp_c).sum(axis=1)
        den = cp.sqrt((act_c**2).sum(axis=1) * (exp_c**2).sum(axis=1))
        den = cp.where(den == 0, 1.0, den)
        r_vals = cp.asnumpy(num / den)

        n_obs = act_vals.shape[1]
        t_vals = r_vals * np.sqrt((n_obs - 2) / (1 - r_vals**2 + 1e-300))
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n_obs - 2)

        corr_df = pd.DataFrame({
            "r": r_vals,
            "p": p_vals,
        }, index=act_new.index)
        corr_df.index.name = "gene"
    else:
        # existing CPU loop
        corr_data = []
        for gene in act_new.index:
            # ...
```

For NMF (line 349), add GPU path:

```python
    if backend.is_gpu_active:
        import cupy as cp
        from spatialgpu.core.gpu_ops import gpu_nmf

        act_nneg_gpu = cp.asarray(act_nneg)

        if isinstance(k, list):
            from sklearn.metrics import silhouette_score
            best_k = k[0]
            best_sil = -1.0
            for ki in k:
                W_gpu, H_gpu = gpu_nmf(act_nneg_gpu, n_components=ki, seed=seed, max_iter=500)
                labels = cp.asnumpy(W_gpu.argmax(axis=1))
                if len(set(labels)) > 1:
                    sil = silhouette_score(act_nneg, labels)
                else:
                    sil = 0.0
                if sil > best_sil:
                    best_sil = sil
                    best_k = ki
            k_final = best_k
        else:
            k_final = k

        W_gpu, H_gpu = gpu_nmf(act_nneg_gpu, n_components=k_final, seed=seed, max_iter=500)
        W = cp.asnumpy(W_gpu)
        H = cp.asnumpy(H_gpu)
    else:
        # existing sklearn NMF code
        # ...
```

- [ ] **Step 2: Add GPU dispatch to secact_signaling_velocity**

In `secact_signaling_velocity()`, replace the `cal_weights` + matrix multiply with GPU versions. The spatial weight computation and the weighted matrix `weights_new` benefit from GPU cdist:

```python
    if backend.is_gpu_active:
        import cupy as cp

        # GPU-accelerated weight matrix multiply
        if sparse.issparse(weights):
            weights_dense = cp.asarray(weights.toarray())
        else:
            weights_dense = cp.asarray(weights)
        # ... continue with GPU matmul for weights_new
    else:
        # existing CPU code
```

- [ ] **Step 3: Run existing tests**

Run: `python -m pytest tests/test_deconvolution/test_secact.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add spatialgpu/deconvolution/secact.py
git commit -m "Add GPU dispatch to SecAct signaling patterns and velocity"
```

---

### Task 12: Integrate GPU into spatial_correlation.py

**Files:**
- Modify: `spatialgpu/deconvolution/spatial_correlation.py`

- [ ] **Step 1: Add GPU dispatch to cal_weights**

In `cal_weights()` (line 33), add GPU path for KDTree + RBF computation:

```python
def cal_weights(adata, radius=200.0, k=None, sigma=100.0, diag_as_zero=True):
    """Compute spatial weight matrix using RBF kernel."""
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    coords = np.column_stack([
        adata.obs["coordinate_x_um"].values.astype(np.float64),
        adata.obs["coordinate_y_um"].values.astype(np.float64),
    ])
    n_spots = coords.shape[0]

    if backend.is_gpu_active and n_spots > 1000:
        return _cal_weights_gpu(coords, n_spots, radius, sigma, diag_as_zero)

    # --- existing CPU code (unchanged) ---
    tree = KDTree(coords)
    # ...
```

Add:

```python
def _cal_weights_gpu(coords, n_spots, radius, sigma, diag_as_zero):
    """GPU implementation of cal_weights using chunked cdist."""
    import cupy as cp
    from spatialgpu.core.gpu_ops import gpu_cdist

    coords_gpu = cp.asarray(coords)

    # Process in chunks to manage memory
    chunk_size = min(5000, n_spots)
    rows_list, cols_list, vals_list = [], [], []

    for i in range(0, n_spots, chunk_size):
        end_i = min(i + chunk_size, n_spots)
        dists = gpu_cdist(coords_gpu[i:end_i], coords_gpu)

        # Mask: within radius and not self
        mask = (dists <= radius) & (dists > 0)
        local_rows, local_cols = cp.where(mask)

        d_vals = dists[mask]
        w_vals = cp.exp(-d_vals ** 2 / (2 * sigma ** 2))

        rows_list.append(cp.asnumpy(local_rows) + i)
        cols_list.append(cp.asnumpy(local_cols))
        vals_list.append(cp.asnumpy(w_vals))

    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_vals = np.concatenate(vals_list)
    else:
        all_rows = np.array([], dtype=np.int64)
        all_cols = np.array([], dtype=np.int64)
        all_vals = np.array([], dtype=np.float64)

    W = sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_spots, n_spots))

    # Row-normalize
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W = sparse.diags(1.0 / row_sums) @ W

    if diag_as_zero:
        W.setdiag(0)
        W.eliminate_zeros()

    return W
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_deconvolution/test_spatial_correlation.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add spatialgpu/deconvolution/spatial_correlation.py
git commit -m "Add GPU dispatch to spatial weight matrix computation"
```

---

### Task 13: End-to-end GPU integration test

**Files:**
- Create: `tests/test_deconvolution/test_gpu_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_deconvolution/test_gpu_integration.py
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
    """Test GPU vs CPU deconvolution equivalence."""

    def test_nnls_solver_equivalence(self):
        """GPU NNLS solver matches CPU for synthetic deconvolution problem."""
        import spatialgpu as sp
        from spatialgpu.deconvolution.core import _solve_nnls

        rng = np.random.RandomState(42)
        n_genes, n_cell, n_spots = 100, 8, 50
        A = np.abs(rng.randn(n_genes, n_cell)).astype(np.float64)
        B = np.abs(rng.randn(n_genes, n_spots)).astype(np.float64)
        theta_sum = np.full(n_spots, 0.8)
        pp_max_arr = np.full(n_spots, 0.9)

        # CPU
        sp.set_backend("cpu")
        cpu_result = _solve_nnls(A, B, n_cell, theta_sum, pp_max_arr)

        # GPU
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
        X = rng.randn(200, 100).astype(np.float64)
        Y = rng.randn(200, 1).astype(np.float64)

        sp.set_backend("cpu")
        cpu_result = cormat(X, Y, method="spearman")

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_result = cormat(X, Y, method="spearman")
            np.testing.assert_allclose(
                gpu_result["cor_r"].values,
                cpu_result["cor_r"].values,
                atol=1e-3,
            )

        sp.set_backend("auto")

    def test_pairwise_spearmanr_equivalence(self):
        """GPU pairwise Spearman matches CPU."""
        import spatialgpu as sp
        from spatialgpu.deconvolution.interaction import _pairwise_spearmanr

        rng = np.random.RandomState(42)
        mat = rng.randn(10, 200).astype(np.float64)

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
    """Test GPU vs CPU gene set scoring."""

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

        gene_sets = {
            "TestSet": [f"Gene{i}" for i in range(10)],
        }

        sp.set_backend("cpu")
        cpu_scores = _ucell_score(adata, gene_sets)

        sp.set_backend("auto")
        if sp.get_backend().is_gpu_active:
            gpu_scores = _ucell_score(adata, gene_sets)
            np.testing.assert_allclose(
                gpu_scores.values, cpu_scores.values, atol=1e-5
            )

        sp.set_backend("auto")
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_deconvolution/test_gpu_integration.py -v`
Expected: All tests PASS (or skip on CPU-only nodes)

- [ ] **Step 3: Commit**

```bash
git add tests/test_deconvolution/test_gpu_integration.py
git commit -m "Add end-to-end GPU vs CPU equivalence integration tests"
```

---

### Task 14: Update tutorial SLURM scripts for correct GPU/CPU partitions

**Files:**
- Modify: `scripts/slurm_tutorial_t1_gpu.sh`
- Modify: `scripts/slurm_tutorial_t2t3_gpu.sh`
- Modify: `scripts/slurm_tutorial_t6_gpu.sh`

- [ ] **Step 1: Verify tutorials actually invoke GPU backend**

Add `SPATIALGPU_BACKEND=auto` and a GPU status print to each tutorial script. Before the `python docs/run_full_tutorial_*.py` line, add:

```bash
export SPATIALGPU_BACKEND=auto
echo "Backend: GPU (A100)"
python -c "import spatialgpu; print(f'GPU active: {spatialgpu.get_backend().is_gpu_active}')"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/slurm_tutorial_t1_gpu.sh scripts/slurm_tutorial_t2t3_gpu.sh scripts/slurm_tutorial_t6_gpu.sh
git commit -m "Add GPU backend verification to tutorial SLURM scripts"
```

---

### Task 15: Run full GPU tests on SLURM

**Files:**
- None (execution only)

- [ ] **Step 1: Run GPU primitive tests on GPU node**

```bash
sbatch --job-name=gpu_test --partition=gpu --gres=gpu:a100:1 --mem=32g \
  --cpus-per-task=4 --time=1:00:00 \
  --output=validation_results/gpu_tests_%j.out \
  --error=validation_results/gpu_tests_%j.err \
  --wrap="source ~/bin/myconda && conda activate secactpy && module load CUDA/12 cuDNN && \
  cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu && \
  pip install -e . --quiet && \
  python -m pytest tests/test_gpu_ops.py tests/test_deconvolution/test_gpu_integration.py -v 2>&1"
```

- [ ] **Step 2: Review output and fix any failures**

Check: `cat validation_results/gpu_tests_*.out`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "GPU all-path: complete GPU acceleration for deconvolution pipeline"
```
