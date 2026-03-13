"""MUDAN normalization and clustering — exact replication via R subprocess.

For exact numerical equivalence with SpaCET, the entire MUDAN pipeline
(normalizeVariance + PCA + hclust + silhouette) is executed in R via
subprocess. This is necessary because:
  1. R's mgcv::gam uses thin plate regression splines (no Python equivalent)
  2. R's pf/qchisq with log.p=TRUE has different precision than scipy
  3. R's hclust(ward.D) merges differ from scipy's linkage at machine precision

R, MUDAN, and cluster packages must be installed.

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
    try:
        return _mudan_cluster_via_r(counts, n_pcs, gam_k, alpha)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"R/MUDAN not available ({e}), using Python approximation. "
            "Results may differ from SpaCET.",
            stacklevel=2,
        )
        return _mudan_cluster_python(counts, n_pcs, gam_k, alpha)


def _mudan_cluster_via_r(
    counts: sparse.spmatrix | np.ndarray,
    n_pcs: int,
    gam_k: int,
    alpha: float,
) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Run entire MUDAN pipeline in R via subprocess."""
    import os
    import shutil
    import subprocess
    import tempfile

    from scipy.io import mmwrite

    counts_sp = sparse.csc_matrix(counts, dtype=np.float64)
    n_spots = counts_sp.shape[1]

    tmpdir = tempfile.mkdtemp()
    input_mtx = os.path.join(tmpdir, "counts.mtx")
    output_gsf = os.path.join(tmpdir, "gsf.csv")
    output_ods = os.path.join(tmpdir, "ods.csv")
    output_clust = os.path.join(tmpdir, "clustering.csv")

    mmwrite(input_mtx, counts_sp)

    r_code = f"""
    suppressPackageStartupMessages({{
        library(Matrix)
        library(MUDAN)
        library(cluster)
    }})

    set.seed(123)

    # Read counts
    counts <- readMM("{input_mtx}")
    counts <- as(counts, "dgCMatrix")

    # 1. normalizeVariance
    info <- normalizeVariance(counts, gam.k = {gam_k}, alpha = {alpha},
                              details = TRUE, verbose = FALSE)
    gsf <- info$df$gsf
    ods <- info$ods

    write.csv(data.frame(gsf = gsf), "{output_gsf}", row.names = FALSE)
    write.csv(data.frame(idx = as.integer(ods)), "{output_ods}", row.names = FALSE)

    # 2. log10 transform + PCA
    matnorm <- log10(info$mat + 1)
    nPcs <- min({n_pcs}, length(ods) - 1, ncol(counts) - 1)
    pcs <- getPcs(matnorm[ods, ], nGenes = length(ods), nPcs = nPcs, verbose = FALSE)

    # 3. Hierarchical clustering (ward.D on correlation distance)
    d <- as.dist(1 - cor(t(pcs)))
    hc <- hclust(d, method = "ward.D")

    # 4. Silhouette analysis for k=2:9
    cluster_numbers <- 2:9
    sil_values <- c()
    for (k in cluster_numbers) {{
        cl <- cutree(hc, k = k)
        sil <- silhouette(cl, d, Fun = mean)
        sil_values <- c(sil_values, mean(sil[, 3]))
    }}

    # 5. Optimal k (max silhouette decrease)
    sil_diff <- sil_values[1:(length(sil_values) - 1)] - sil_values[2:length(sil_values)]
    maxN <- which(sil_diff == max(sil_diff)) + 1
    clustering <- cutree(hc, k = cluster_numbers[maxN])

    write.csv(data.frame(cluster = as.integer(clustering)),
              "{output_clust}", row.names = FALSE)
    """

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R MUDAN pipeline failed: {result.stderr}")

        gsf = pd.read_csv(output_gsf)["gsf"].values
        ods = pd.read_csv(output_ods)["idx"].values - 1  # R 1-indexed → 0-indexed
        clustering = pd.read_csv(output_clust)["cluster"].values
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    clustering_series = pd.Series(clustering, dtype=int)
    return clustering_series, ods, gsf


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
    Z = linkage(condensed, method="ward")

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
    mat_sq.data = mat_sq.data ** 2
    mean_sq = np.asarray(mat_sq.mean(axis=0)).ravel()
    gene_vars = (mean_sq - gene_means ** 2) * n_spots / (n_spots - 1)

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
        try:
            from pygam import LinearGAM, s

            gam = LinearGAM(s(0, n_splines=gam_k, spline_order=3)).fit(
                m_vi.reshape(-1, 1), v_vi
            )
            fitted_vi = gam.predict(m_vi.reshape(-1, 1))
        except ImportError:
            from scipy.interpolate import UnivariateSpline

            order = np.argsort(m_vi)
            spl = UnivariateSpline(m_vi[order], v_vi[order], k=3)
            fitted_vi = spl(m_vi)

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
    from scipy.sparse.linalg import eigsh

    if sparse.issparse(mat):
        m = mat.T.toarray().astype(np.float64)
    else:
        m = mat.T.astype(np.float64)

    col_means = m.mean(axis=0)
    m = m - col_means
    n = m.shape[0]

    cov_mat = m.T @ m / (n - 1)
    n_pcs_actual = min(n_pcs, cov_mat.shape[0] - 1)
    eigenvalues, eigenvectors = eigsh(cov_mat, k=n_pcs_actual)

    idx = np.argsort(eigenvalues)[::-1]
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
