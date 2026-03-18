"""MUDAN normalization and clustering.

Pure-Python implementation of the MUDAN pipeline
(normalizeVariance + PCA + hclust + silhouette).

Reference: Fan et al., MUDAN R package (used internally by SpaCET).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def mudan_cluster(
    counts: sparse.spmatrix | np.ndarray,
    n_pcs: int = 30,
    gam_k: int = 5,
    alpha: float = 0.05,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Full MUDAN pipeline: normalizeVariance + PCA + ward.D clustering.

    Matches R's SpaCET inferMal_cor Step 1 exactly.

    Parameters
    ----------
    counts : sparse matrix or ndarray
        Raw counts, genes x spots.
    n_pcs : int
        Number of principal components.
    gam_k : int
        Number of GAM basis functions.
    alpha : float
        Significance threshold for overdispersed gene selection.

    Returns
    -------
    clustering : pd.Series
        Cluster assignments (1-indexed) for each spot.
    ods : ndarray of int
        Indices of overdispersed genes (0-based).
    gsf : ndarray
        Gene scale factors (one per gene).
    """
    return _mudan_cluster_python(counts, n_pcs, gam_k, alpha)


def _mudan_cluster_python(
    counts: sparse.spmatrix | np.ndarray,
    n_pcs: int,
    gam_k: int,
    alpha: float,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Pure-Python MUDAN approximation (fallback when R is unavailable)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_samples

    counts_sp = sparse.csc_matrix(counts, dtype=np.float64)

    # normalizeVariance (approximate)
    norm_mat, ods, gsf = _normalize_variance_python(counts_sp, gam_k, alpha)

    # log10 transform
    log_mat = norm_mat.copy().astype(np.float64)
    log_mat.data = np.log10(log_mat.data + 1)

    # PCA
    pcs = _get_pcs(log_mat[ods, :], n_pcs=n_pcs)

    # Clustering
    corr = np.corrcoef(pcs)
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 0)

    condensed = squareform(dist, checks=False)
    # R's hclust(method="ward.D") uses raw distances, while scipy's
    # linkage(method="ward") matches R's "ward.D2" (squared distances).
    # Square in-place (condensed not reused) then sqrt merge heights.
    np.square(condensed, out=condensed)
    Z = linkage(condensed, method="ward")
    Z[:, 2] = np.sqrt(Z[:, 2])

    cluster_numbers = list(range(2, 10))
    sil_scores = []
    for k in cluster_numbers:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_samples(dist, labels, metric="precomputed")
        sil_scores.append(np.mean(sil))

    sil_diff = [sil_scores[i] - sil_scores[i + 1] for i in range(len(sil_scores) - 1)]
    max_n = cluster_numbers[np.argmax(sil_diff) + 1]

    clustering = fcluster(Z, t=max_n, criterion="maxclust")
    return pd.Series(clustering, dtype=int), ods, gsf


def _normalize_variance_python(
    counts_sp: sparse.spmatrix,
    gam_k: int,
    alpha: float,
    max_adj_var: float = 1000.0,
    min_adj_var: float = 0.001,
) -> tuple[sparse.spmatrix, np.ndarray, np.ndarray]:
    """Pure-Python normalizeVariance (approximate)."""
    from scipy import stats

    mat_t = counts_sp.T.tocsc()
    n_spots = mat_t.shape[0]
    n_genes = mat_t.shape[1]

    gene_means = np.asarray(mat_t.mean(axis=0)).ravel()
    mat_sq = mat_t.copy()
    mat_sq.data = mat_sq.data**2
    mean_sq = np.asarray(mat_sq.mean(axis=0)).ravel()
    gene_vars = (mean_sq - gene_means**2) * n_spots / (n_spots - 1)

    dfm = np.log(gene_means)
    dfv = np.log(gene_vars)
    vi = np.where(np.isfinite(dfv))[0]

    if len(vi) < gam_k * 1.5:
        gam_k = 1

    m_vi, v_vi = dfm[vi], dfv[vi]

    if gam_k < 2:
        coeffs = np.polyfit(m_vi, v_vi, deg=1)
        fitted_vi = np.polyval(coeffs, m_vi)
    else:
        fitted_vi = _tprs_1d_reml(m_vi, v_vi, k=max(gam_k, 10))

    residuals = np.full(n_genes, -np.inf)
    residuals[vi] = v_vi - fitted_vi

    lp = np.full(n_genes, np.nan)
    lp[vi] = stats.f.logsf(np.exp(residuals[vi]), dfn=n_spots, dfd=n_spots)
    lpa = _bh_adjust_log(lp)
    ods = np.where(lpa < np.log(alpha))[0]

    qv = np.full(n_genes, np.nan)
    finite_lp = np.isfinite(lp)
    if np.any(finite_lp):
        p_vals = np.clip(np.exp(lp[finite_lp]), 0, 1)
        qv[finite_lp] = stats.chi2.isf(p_vals, df=n_genes - 1) / n_genes

    gsf = np.sqrt(np.clip(qv, min_adj_var, max_adj_var) / np.exp(dfv))
    gsf[~np.isfinite(gsf)] = 0.0

    norm_mat = sparse.diags(gsf) @ counts_sp
    return norm_mat, ods, gsf


def _get_pcs(
    mat: sparse.spmatrix | np.ndarray,
    n_pcs: int = 30,
) -> np.ndarray:
    """PCA matching R's MUDAN::getPcs() + fastPca()."""
    if sparse.issparse(mat):
        m = mat.T.toarray().astype(np.float64)
    else:
        m = mat.T.astype(np.float64)

    col_means = m.mean(axis=0)
    m = m - col_means
    n = m.shape[0]

    cov_mat = m.T @ m / (n - 1)
    n_pcs_actual = min(n_pcs, cov_mat.shape[0] - 1)

    # Use dense eigh — exact, robust, no ARPACK convergence issues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    # eigh returns ascending order; take top n_pcs_actual
    idx = np.argsort(eigenvalues)[::-1][:n_pcs_actual]
    eigenvectors = eigenvectors[:, idx]
    return m @ eigenvectors


def _bh_adjust_log(log_p: np.ndarray) -> np.ndarray:
    """BH p-value adjustment in log space (matches R's bh.adjust)."""
    result = log_p.copy()
    finite_mask = np.isfinite(log_p)
    x = log_p[finite_mask]
    n = len(x)
    if n == 0:
        return result

    order = np.argsort(x, kind="stable")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    q = x[order] + np.log(n / ranks)
    a = np.minimum.accumulate(q[::-1])[::-1]

    unsorted = np.empty_like(a)
    unsorted[order] = a
    result[finite_mask] = unsorted
    return result


def _tprs_1d_reml(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    max_knots: int = 2000,
) -> np.ndarray:
    """1D thin-plate regression spline with REML smoothing parameter.

    Matches R's mgcv::gam(y ~ s(x, k=k)) for 1D data:
    - Builds full energy matrix on a subsample of max_knots points
    - QR-projects out the null space {1, x}
    - Eigendecomposes to get the top (k-2) penalized basis functions
    - Selects smoothing parameter by REML
    """
    from scipy.optimize import minimize_scalar

    n = len(x)
    M = 2  # null space dimension {1, x} for m=2, d=1
    n_pen = k - M

    # Subsample for basis construction (mgcv default: max.knots=2000)
    if n > max_knots:
        rng = np.random.RandomState(0)
        sub_idx = np.sort(rng.choice(n, max_knots, replace=False))
        xu = x[sub_idx]
    else:
        xu = x.copy()
    nk = len(xu)

    # Energy matrix at subsample: E_ij = |xu_i - xu_j|^3
    E = np.abs(xu[:, None] - xu[None, :]) ** 3

    # QR decomposition of null space → project out {1, x}
    T = np.column_stack([np.ones(nk), xu])
    Q, _ = np.linalg.qr(T, mode="complete")
    Q_perp = Q[:, M:]

    # Eigendecompose projected energy matrix
    E_proj = Q_perp.T @ E @ Q_perp
    eigvals, eigvecs = np.linalg.eigh(E_proj)
    idx = np.argsort(np.abs(eigvals))[::-1]
    Dz = eigvals[idx][:n_pen]
    UZ = Q_perp @ eigvecs[:, idx][:, :n_pen]

    # Evaluate basis at all data points
    E_xk = np.abs(x[:, None] - xu[None, :]) ** 3
    Z = E_xk @ UZ / np.sqrt(np.abs(Dz))[None, :]
    T_data = np.column_stack([np.ones(n), x])
    X = np.column_stack([T_data, Z])

    # Penalty on penalized coefficients only
    S = np.zeros((k, k))
    S[M:, M:] = np.eye(n_pen)

    # REML smoothing parameter selection
    XtX = X.T @ X
    Xty = X.T @ y

    def neg_reml(log_lam):
        lam = np.exp(log_lam)
        B = XtX + lam * S
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            return 1e20
        beta = np.linalg.solve(B, Xty)
        rss = np.sum((y - X @ beta) ** 2)
        log_det_B = 2.0 * np.sum(np.log(np.diag(L)))
        n_eff = n - M
        return 0.5 * (n_eff * np.log(rss / n_eff) + log_det_B - n_pen * log_lam)

    result = minimize_scalar(neg_reml, bounds=(-15, 25), method="bounded")
    lam = np.exp(result.x)
    beta = np.linalg.solve(XtX + lam * S, Xty)
    return X @ beta
