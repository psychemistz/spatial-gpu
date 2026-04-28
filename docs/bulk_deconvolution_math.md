# Bulk Deconvolution: Formal Derivations

This tutorial states the bulk RNA-seq deconvolution problem, then derives the
estimator used by each of the four methods compared in the T8 benchmark:
**SpaCET** (original), **MuSiC**, **DWLS**, and **SpaCET-IRWLS** (the
residual-reweighting variant in this repo). The goal is to make the
mathematical assumptions of each method explicit so that benchmark results
can be interpreted in terms of *what each estimator is actually optimizing*
rather than as a black box.

## 1. Problem Statement

Let:

- $G$ = number of genes, $K$ = number of cell types, $N$ = number of bulk samples.
- $Y \in \mathbb{R}^{G \times N}_{\geq 0}$ — bulk expression matrix (one column per sample).
- $X \in \mathbb{R}^{G \times K}_{\geq 0}$ — cell-type **signature matrix**, where
  $X_{gk}$ is the mean expression of gene $g$ in cell type $k$. Built from a
  single-cell RNA-seq reference.
- $p \in \Delta^{K-1}$ — unknown cell-type proportions for one bulk sample,
  on the simplex $\{p : p_k \ge 0, \sum_k p_k = 1\}$.

The **linear mixture model** assumed by all four methods:

$$
y \;=\; X\,p \;+\; \varepsilon, \qquad \varepsilon \in \mathbb{R}^G,
$$

where $y$ is one column of $Y$ and $\varepsilon$ is a residual capturing
biological + technical noise. Methods differ in (a) how $X$ is constructed,
(b) the noise model on $\varepsilon$ (equivalently: the gene-weight matrix
$W$), and (c) how the simplex constraint is imposed.

The generic weighted estimator is:

$$
\hat p \;=\; \arg\min_{p \in \Delta^{K-1}} \;\; (y - Xp)^\top W (y - Xp),
$$

with $W \in \mathbb{R}^{G \times G}$ a positive-semidefinite gene-weight
matrix (almost always diagonal). The four methods correspond to four
choices of $W$ and four ways to handle the constraint.

---

## 2. SpaCET (Original) — Hierarchical Constrained NNLS

Source: Ru et al., *Nat. Commun.* **14**, 568 (2023). Implemented in this
repo at `spatialgpu/deconvolution/extensions.py:deconvolution_matched_scrnaseq`
with `cross_subject_weighting=False`.

### 2.1 Signature

For each (sub-)lineage $k$, $X_{gk}$ is the **mean count-per-million** of
gene $g$ across all single cells annotated as type $k$ in the reference,
optionally after per-lineage downsampling to balance class sizes. Genes
with zero row-sum are filtered.

### 2.2 Loss

OLS in CPM-normalized space, with non-negativity *and* simplex constraint
imposed jointly:

$$
\hat p \;=\; \arg\min_{p \geq 0,\; \mathbf 1^\top p = 1} \; \| y - Xp \|_2^2.
$$

This is $W = I$ in the generic form. Solved by an active-set NNLS
(`scipy.optimize.nnls`-style) followed by simplex projection / re-solve;
in this codebase the joint constraint is enforced by the QP path in
`spatialgpu/deconvolution/constr_optim.py`.

### 2.3 Hierarchical Cascade

The distinctive piece. Cell types are organized into a **lineage tree**
(e.g. `T_cells -> {CD4, CD8, NK}`). The cascade:

1. Solve at the **major** lineage level (lump CD4+CD8+NK into one column
   $X_{:,T}$ that averages their fine-type signatures). Obtain
   $\hat p^{\text{major}}$.
2. For each major lineage with sub-types, solve the same NNLS over only its
   sub-columns of $X$, then **rescale** sub-fractions so they sum to
   $\hat p^{\text{major}}_k$.

Equivalently, the final fine-type proportions satisfy

$$
\hat p^{\text{fine}}_{k_{\text{sub}}}
\;=\;
\hat p^{\text{major}}_{k_{\text{major}}} \cdot
\frac{\hat q_{k_{\text{sub}}}}{\sum_{k' \in k_{\text{major}}} \hat q_{k'}},
$$

where $\hat q$ is the unconstrained NNLS solution restricted to that
lineage's columns. This is a **constraint-by-construction** scheme: the
hierarchy guarantees the simplex is respected without a global QP at the
fine-type level. It is also the source of one known failure mode — when a
major lineage is *under-predicted* at step 1 (e.g. malignant in a
tumor-purity = 90% sample), no amount of step-2 rescaling can recover it.
See § 5 for why IRWLS does not fix this.

---

## 3. MuSiC — Cross-Subject Inverse-Variance Weighted NNLS

Source: Wang, Hao et al., *Nat. Commun.* **10**, 380 (2019). The MuSiC
weights are reproduced in this repo at
`spatialgpu/deconvolution/extensions.py:_compute_cross_subject_weights_absolute`
and used when `weighting_method="irwls"` (initialization step).

### 3.1 Insight

A gene with a stable mean expression across donors carries reliable
information about cell-type identity; a gene whose mean *jumps between
donors* is an unreliable mixture indicator (its variation is donor effect,
not cell-type effect). MuSiC exploits this by reweighting the loss with an
inverse-variance term computed **per-(gene, cell-type) across subjects**.

### 3.2 Variance Components

Let $D$ index donors/subjects. For each cell type $k$, let
$\bar x^{(d)}_{gk}$ be the donor-$d$ mean of gene $g$ across cells of
type $k$. Define:

$$
\sigma^2_{\text{between}}(g,k) \;=\; \mathrm{Var}_{d}\!\left[\bar x^{(d)}_{gk}\right]
\quad\text{(donor-level variance of within-type means)},
$$

$$
\sigma^2_{\text{within}}(g,k) \;=\; \mathbb{E}_d\!\left[\mathrm{Var}_{c \in d,k}\!\left[x_{gc}\right]\right]
\quad\text{(within-donor cell-level variance, averaged over donors)}.
$$

Both are computed in CPM space; this codebase implements them in
`_subject_variance_components` (extensions.py:958–1010).

### 3.3 Weight

MuSiC's published per-gene weight is

$$
w_g \;\propto\; \frac{1}{\sigma^2_{\text{between}}(g,k) + \sigma^2_{\text{within}}(g,k)},
$$

averaged across cell types $k$ that have $\geq 2$ donors of data, then
normalized so $\max_g w_g = 1$. This is the form returned by
`_compute_cross_subject_weights_absolute`. The diagonal weight matrix is
$W = \mathrm{diag}(w_1, \dots, w_G)$.

### 3.4 Estimator

$$
\hat p^{\text{MuSiC}} \;=\; \arg\min_{p \geq 0,\; \mathbf 1^\top p = 1}
\;\sum_{g=1}^G w_g \,(y_g - (Xp)_g)^2.
$$

The constraint is handled by a hierarchical recursive selection step in the
original MuSiC R package (drop-out of low-effect cell types and re-fit), but
the **core estimator** is the WNLS above. Whether one uses recursive cell-type
selection is orthogonal to the weighting scheme.

### 3.5 Variant: SpaCET "ratio" Form

The SpaCET R package re-derives a related but **scale-free** weight,
implemented here in `_compute_cross_subject_weights`:

$$
w^{\text{ratio}}_g \;\propto\; \frac{1}{1 + \sigma^2_{\text{between}}(g,k)\,/\,\sigma^2_{\text{within}}(g,k)}.
$$

Two genes with the same between/within *ratio* receive the same weight,
regardless of absolute variance. This is empirically more robust on
tumor-dominated samples than the published MuSiC absolute-inverse form
(see § 6 of the bench results), at the cost of giving up the
inverse-variance interpretation.

---

## 4. DWLS — Dampened Weighted Least Squares

Source: Tsoucas et al., *Nat. Commun.* **10**, 2975 (2019). Used in the
benchmark via the CRAN/GitHub `DWLS` package
(`scripts/_dwls_common.R:deconvolve_scenarios`).

### 4.1 Insight

If we adopt a Poisson noise model $\mathrm{Var}(y_g) \propto y_g$ (or more
generally if the variance scales with the predicted bulk expression), the
optimal weight is **inverse to the predicted value** — high-expression
genes get *lower* weight because their absolute residuals are large by
construction. Plain WLS with $w_g = 1 / \hat y_g^2$ is unstable when $\hat
y_g$ is small (a single low-expression gene can dominate the loss). DWLS
fixes this with a **damping** scheme.

### 4.2 Iteration

Initialize $\hat p^{(0)}$ via OLS NNLS. At iteration $t$:

1. Predict: $\hat y^{(t)} = X \hat p^{(t)}$.
2. Compute raw weights: $w_g^{(t)} = 1 / (\hat y_g^{(t)})^2$.
3. **Damp**: rescale so the largest weight is at most $j$ times the median
   weight, where $j \in \{2^0, 2^1, \dots, 2^J\}$ is selected by an internal
   search that minimizes the residual sum of squares of $p^{(t+1)}$. The
   damping cap is

   $$
   \tilde w_g^{(t)} \;=\; \min\!\left(w_g^{(t)},\; j \cdot \mathrm{median}(w^{(t)})\right).
   $$

4. Solve

   $$
   \hat p^{(t+1)} \;=\; \arg\min_{p \geq 0,\; \mathbf 1^\top p = 1}
   \sum_g \tilde w_g^{(t)} \,(y_g - (Xp)_g)^2.
   $$

5. Stop when $\|\hat p^{(t+1)} - \hat p^{(t)}\|_\infty < \tau$ (default
   $\tau = 0.01$) or after a max iter count.

### 4.3 Signature

DWLS builds its signature differently from SpaCET/MuSiC: per-cell-type
**MAST** differential expression (zero-inflated GLM) selects discriminative
genes ($\log\mathrm{FC} > 0.5$, $p < 0.01$), and the signature column for
type $k$ is the mean of those genes' expression in cells of type $k$. See
`DWLS::buildSignatureMatrixMAST` and the wrapper
`build_or_load_signature` in `scripts/_dwls_common.R:45–63`.

### 4.4 Estimator (Compact Form)

In closed form, DWLS converges to the fixed point of:

$$
\hat p^\star \;=\; \arg\min_{p \in \Delta} \;
(y - Xp)^\top \,W^\star(p)\, (y - Xp), \qquad
W^\star(p) = \mathrm{diag}\!\left(\min\!\left(\frac{1}{(Xp)_g^2},\; j^\star \cdot M\right)\right),
$$

where $M = \mathrm{median}_g\!\left[1/(Xp)_g^2\right]$ and $j^\star$ is the
damping factor selected per iteration. This is a **self-weighted** WLS:
the weights depend on $p$, the unknown.

---

## 5. SpaCET-IRWLS (this repo)

Implemented in `extensions.py:_irwls_lite_updated_weights` and triggered by
`weighting_method="irwls"`. This is a compact, single-iteration variant of
**iteratively reweighted least squares** that combines the MuSiC
between/within variance structure with a DWLS-style residual update —
matched to the SpaCET hierarchical-cascade infrastructure.

### 5.1 Motivation

MuSiC's weights are **prior-only**: they depend on the scRNA-seq reference
alone, not on how well a given gene fits the *bulk* sample. DWLS's weights
are **posterior-only**: they depend on the predicted bulk, not on
cross-subject reliability. The IRWLS variant combines both information
sources in a single residual-informed update.

### 5.2 Initial Weights

$$
w_g^{(0)} \;\propto\; \frac{1}{\sigma^2_{\text{between}}(g) + \sigma^2_{\text{within}}(g) + \epsilon},
$$

i.e. the MuSiC absolute-inverse form (§ 3.3), averaged across cell types.

### 5.3 First Solve

Run the SpaCET hierarchical cascade with $W^{(0)} =
\mathrm{diag}(w^{(0)})$ to obtain $\hat p^{(1)}$.

### 5.4 Residual Variance

For each gene $g$, compute the **across-sample residual variance** in CPM
space:

$$
r_{g,n} \;=\; y_{g,n} - (X \hat p^{(1)})_{g,n},
\qquad
\sigma^2_{\text{resid}}(g) \;=\; \mathrm{Var}_n[r_{g,n}].
$$

Genes that the first solve fits poorly across samples accumulate large
$\sigma^2_{\text{resid}}$ and will be downweighted on the next pass.

### 5.5 Updated Weights

$$
w_g^{(1)} \;\propto\; \frac{1}{\sigma^2_{\text{between}}(g) + \sigma^2_{\text{within}}(g) + \sigma^2_{\text{resid}}(g) + \epsilon},
$$

normalized so $\max_g w_g^{(1)} = 1$.

### 5.6 Second Solve

Re-run the hierarchical cascade with $W^{(1)} = \mathrm{diag}(w^{(1)})$ to
obtain the final $\hat p^{(2)}$.

### 5.7 Why Stop at One Update

Empirically (T8 BRCA benchmark, 4 scenarios × N=100 paired trials), one
residual-reweighting pass strictly Pareto-improves over both `none` and
`ratio` MuSiC-style weighting on overall Pearson $r$ and per-cell-type
metrics. Additional iterations *hurt* tumor-dominated scenarios because
the malignant under-prediction in those cases is a **hierarchical-cascade
artifact** (§ 2.3) — once `Cancer Epithelial` is under-predicted at the
major-lineage step, residuals at malignant-marker genes blow up,
downweighting them, which makes the *next* malignant prediction even
worse. Residual reweighting cannot fix a constraint-induced bias; it can
only refine fits that the constraint already permits. Stopping at one
update preserves the gain from refining well-conditioned cases without
amplifying the cascade artifact.

### 5.8 Compact Form

$$
\hat p^{\text{IRWLS-lite}} \;=\; \mathcal{H}\!\left(y, X, W^{(1)}\right),
\qquad
W^{(1)} \;=\; W^{(1)}\!\left(W^{(0)},\, \mathcal{H}(y, X, W^{(0)})\right),
$$

where $\mathcal{H}(\cdot)$ denotes the SpaCET hierarchical constrained
NNLS operator and $W^{(1)}$ is computed from the first-pass residuals as
in § 5.5. One outer iteration; the constraint is handled inside
$\mathcal{H}$ at each level of the lineage tree.

---

## 6. Side-by-Side Summary

| Method            | Weights $W$                                                                     | Constraint handling                  | Reference signature                | Iteration |
|-------------------|----------------------------------------------------------------------------------|--------------------------------------|------------------------------------|-----------|
| SpaCET            | $I$                                                                              | Hierarchical NNLS by lineage tree    | Per-type CPM mean, downsampled     | 1 pass    |
| MuSiC             | $\mathrm{diag}\!\left(1/(\sigma^2_b + \sigma^2_w)\right)$                        | NNLS + recursive type selection      | Per-(type, donor) mean, hierarchical| 1 pass    |
| DWLS              | $\mathrm{diag}\!\left(\min\!\left(1/\hat y_g^2,\; jM\right)\right)$               | NNLS + simplex                       | MAST DE genes, per-type mean       | Iterative until $\Delta p < \tau$ |
| SpaCET-IRWLS      | $\mathrm{diag}\!\left(1/(\sigma^2_b + \sigma^2_w + \sigma^2_{\text{resid}})\right)$ | Hierarchical NNLS by lineage tree    | Per-type CPM mean, downsampled     | 2 passes (one residual update) |

Key implementation pointers in this repo:

- SpaCET hierarchical cascade: `spatialgpu/deconvolution/core.py:_spatial_deconv`
- MuSiC weights (ratio form): `extensions.py:_compute_cross_subject_weights`
- MuSiC weights (absolute form): `extensions.py:_compute_cross_subject_weights_absolute`
- IRWLS-lite update: `extensions.py:_irwls_lite_updated_weights`
- DWLS wrapper: `scripts/_dwls_common.R:deconvolve_scenarios`
- Benchmark driver: `scripts/bench_spacet_weighting.py`

---

## 7. What Each Method Cannot Do

A short list of **failure modes that follow from the math**, useful for
interpreting benchmark plots:

- **SpaCET:** if the lineage tree mis-groups types whose bulk-level
  signatures are not separable, the major-step solve aliases mass between
  them; no fine-step rescaling can correct this.
- **MuSiC:** when $\sigma^2_w$ is small (rare type with few cells per
  donor), $w_g$ explodes for that type's markers, dominating the loss and
  biasing the estimate toward fitting those genes at the cost of every
  other type.
- **DWLS:** dependence of $W^\star$ on $\hat y = X\hat p$ creates local
  minima for samples where one cell type's signature genes overlap
  another's; the damping constant $j^\star$ trades robustness against
  responsiveness, not both.
- **SpaCET-IRWLS:** inherits the SpaCET cascade artifact at the major
  level; the residual update *cannot* recover from a wrongly-assigned
  major-lineage proportion (see § 5.7).

These are not bugs to patch — they are properties of the loss + constraint.
A correctness argument for any of these methods must therefore include the
regime in which it holds, not just a parity claim against an R reference.
