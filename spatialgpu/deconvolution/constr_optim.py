"""R-compatible constrOptim: log-barrier + Nelder-Mead.

Faithful Python port of R's stats::constrOptim() with grad=NULL and
R's C-level nmmin() from src/appl/optim.c (exact same algorithm).
"""

from __future__ import annotations

import numpy as np

_R_EPS = np.finfo(np.float64).eps
_R_RELTOL = np.sqrt(_R_EPS)
_BIG = 1.0e35


def _nmmin(
    f: callable,
    x0: np.ndarray,
    *,
    reltol: float = _R_RELTOL,
    abstol: float = -np.inf,
    maxiter: int = 500,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
) -> tuple[np.ndarray, float, int, bool]:
    """Nelder-Mead matching R's nmmin() exactly.

    Ported from R source: src/appl/optim.c
    Uses 1-based indexing for L, H (matching R's C code).
    """
    n = len(x0)
    n1 = n + 1
    C_col = n1  # 0-based index of extra column (centroid/temp)

    # P[row][col]: rows 0..n-1 = parameters, row n = function value
    # Columns 0..n = simplex vertices, column n+1 = centroid/temp
    P = np.zeros((n1, n1 + 1), dtype=np.float64)

    Bvec = x0.copy().astype(np.float64)
    fval = f(Bvec)
    if not np.isfinite(fval):
        return x0.copy(), float(fval), 1, False

    funcount = 1
    convtol = reltol * (abs(fval) + reltol)

    P[n, 0] = fval
    P[:n, 0] = Bvec

    L = 1  # 1-indexed

    step = 0.0
    for i in range(n):
        if 0.1 * abs(Bvec[i]) > step:
            step = 0.1 * abs(Bvec[i])
    if step == 0.0:
        step = 0.1

    # Build initial simplex
    size = 0.0
    for j in range(2, n1 + 1):
        P[:n, j - 1] = Bvec
        trystep = step
        while P[j - 2, j - 1] == Bvec[j - 2]:
            P[j - 2, j - 1] = Bvec[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size

    calcvert = True
    fail = False

    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    Bvec[:] = P[:n, j]
                    fv = f(Bvec)
                    if not np.isfinite(fv):
                        fv = _BIG
                    funcount += 1
                    P[n, j] = fv
            calcvert = False

        # Find best (L) and worst (H)
        VL = P[n, L - 1]
        VH = VL
        H = L

        for j in range(1, n1 + 1):
            if j != L:
                fv = P[n, j - 1]
                if fv < VL:
                    L = j
                    VL = fv
                if fv > VH:
                    H = j
                    VH = fv

        if VH <= VL + convtol or VL <= abstol:
            break

        # Centroid (excluding H)
        # R: temp = -P[i][H-1]; for j in 0..n: temp += P[i][j]; /= n
        P[:n, C_col] = (P[:n, :n1].sum(axis=1) - P[:n, H - 1]) / n

        # Reflection
        Bvec[:] = (1.0 + alpha) * P[:n, C_col] - alpha * P[:n, H - 1]
        fr = f(Bvec)
        if not np.isfinite(fr):
            fr = _BIG
        funcount += 1
        VR = fr

        if VR < VL:
            # Try expansion
            P[n, C_col] = fr
            # R computes expansion element-by-element, storing reflected in C_col
            xr_copy = Bvec.copy()
            Bvec[:] = gamma * xr_copy + (1 - gamma) * P[:n, C_col]
            P[:n, C_col] = xr_copy  # store reflected point

            fe = f(Bvec)
            if not np.isfinite(fe):
                fe = _BIG
            funcount += 1

            if fe < VR:
                P[:n, H - 1] = Bvec
                P[n, H - 1] = fe
            else:
                P[:n, H - 1] = P[:n, C_col]
                P[n, H - 1] = VR
        else:
            if VR < VH:
                P[:n, H - 1] = Bvec.copy()
                P[n, H - 1] = VR

            # Contraction
            Bvec[:] = (1 - beta) * P[:n, H - 1] + beta * P[:n, C_col]
            fc = f(Bvec)
            if not np.isfinite(fc):
                fc = _BIG
            funcount += 1

            if fc < P[n, H - 1]:
                P[:n, H - 1] = Bvec
                P[n, H - 1] = fc
            else:
                if VR >= VH:
                    # Shrink
                    calcvert = True
                    size = 0.0
                    for j in range(n1):
                        if j + 1 != L:
                            P[:n, j] = beta * (P[:n, j] - P[:n, L - 1]) + P[:n, L - 1]
                            size += np.sum(np.abs(P[:n, j] - P[:n, L - 1]))
                    if size < oldsize:
                        oldsize = size
                    else:
                        fail = True
                        break

        # R's do-while condition
        if funcount > maxiter:
            break

    x_opt = P[:n, L - 1].copy()
    f_opt = float(P[n, L - 1])
    return x_opt, f_opt, funcount, funcount <= maxiter and not fail


def constr_optim(
    theta0: np.ndarray,
    f: callable,
    ui: np.ndarray,
    ci: np.ndarray,
    args: tuple = (),
    mu: float = 1e-4,
    outer_iterations: int = 100,
    outer_eps: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Constrained optimization matching R's stats::constrOptim(grad=NULL)."""
    theta = theta0.copy().astype(np.float64)

    if np.any(ui @ theta - ci <= 0):
        raise ValueError(
            "Initial value is not in the interior of the feasible region"
        )

    theta_old = theta.copy()
    obj = f(theta, *args)

    def make_barrier(t_old):
        gi_old = (ui @ t_old - ci).copy()

        def bfn(x):
            ui_x = ui @ x
            gi = ui_x - ci
            if np.any(gi < 0):
                return np.nan
            log_gi = np.log(gi)
            bar = np.sum(gi_old * log_gi - ui_x)
            if not np.isfinite(bar):
                bar = -np.inf
            return f(x, *args) - mu * bar

        return bfn

    r = make_barrier(theta_old)(theta)
    if not np.isfinite(r):
        r = _BIG
    s_mu = 1.0 if mu > 0 else -1.0

    for i in range(outer_iterations):
        r_old = r
        theta_old = theta.copy()

        current_barrier = make_barrier(theta_old)
        theta_new, r_new, _, _ = _nmmin(current_barrier, theta_old)
        r = r_new

        if (
            np.isfinite(r)
            and np.isfinite(r_old)
            and abs(r - r_old) < (0.001 + abs(r)) * outer_eps
        ):
            break

        theta = theta_new
        obj_new = f(theta, *args)

        if s_mu * obj_new > s_mu * obj:
            break

        obj = obj_new

    return theta, f(theta, *args)
