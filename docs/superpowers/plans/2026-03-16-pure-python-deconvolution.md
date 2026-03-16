# Pure-Python Deconvolution Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all R subprocess dependencies from deconvolution, making Python the sole execution path, and validate numerical equivalence against R outputs for 3 Visium datasets at machine precision (rtol=1e-10, atol=1e-12).

**Architecture:** Replace the current R-first/Python-fallback pattern with Python-only execution. The existing Python implementations (`_deconvolution_python`, `_solve_constrained_batch_python`, `_mudan_cluster_python`, `constr_optim.py`) become the sole code paths. All `_via_r()` functions are deleted. Validation tests compare per-spot outputs against R reference CSVs in `validation/`.

**Tech Stack:** Python, NumPy, SciPy, pandas, pygam (for MUDAN GAM), joblib (parallel), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `spatialgpu/deconvolution/core.py` | Modify | Remove 4 `_via_r()` functions, rewire entry points to Python-only |
| `spatialgpu/deconvolution/mudan.py` | Modify | Remove `_mudan_cluster_via_r()`, make Python the default |
| `spatialgpu/deconvolution/spatial_correlation.py` | Modify | Remove `_vst_normalize_via_r()` |
| `spatialgpu/deconvolution/extensions.py` | Modify | Remove `_de_limma_via_r()` |
| `spatialgpu/deconvolution/gene_set_score.py` | Modify | Remove `_ucell_score_via_r()` |
| `spatialgpu/deconvolution/constr_optim.py` | Keep | Already a faithful R port, no changes needed |
| `tests/test_deconvolution/test_r_equivalence.py` | Create | Strict numerical equivalence tests against R validation data |

---

### Task 1: Write Strict Numerical Equivalence Tests

**Files:**
- Create: `tests/test_deconvolution/test_r_equivalence.py`
- Read: `validation/intermediates/*.csv`, `validation/vst{1,2,3}_*.csv`

- [ ] **Step 1: Write test for constrOptim per-spot equivalence**

```python
"""Tests for numerical equivalence with R at machine precision."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

VALIDATION = Path("validation")
INTERMEDIATES = VALIDATION / "intermediates"
RTOL = 1e-10
ATOL = 1e-12


def _skip_if_no_validation():
    if not VALIDATION.exists():
        pytest.skip("No validation/ directory")


class TestConstrOptimEquivalence:
    """Verify Python constrOptim matches R spot-by-spot."""

    def setup_method(self):
        _skip_if_no_validation()
        self.A = pd.read_csv(INTERMEDIATES / "A_L1.csv", index_col=0).values
        self.spots = [1, 4, 7, 9, 10]

    @pytest.mark.parametrize("spot_id", [1, 4, 7, 9, 10])
    def test_spot_pass2_matches_r(self, spot_id):
        from spatialgpu.deconvolution.constr_optim import constr_optim

        r_result = pd.read_csv(
            INTERMEDIATES / f"spot{spot_id}_L1_result.csv", index_col=0
        )
        b_df = pd.read_csv(
            INTERMEDIATES / f"spot{spot_id}_b_L1.csv", index_col=0
        )

        A = self.A
        b = b_df.iloc[:, 0].values
        r_pass2 = r_result["pass2"].values
        n_cell = A.shape[1]

        mal_prop = pd.read_csv(INTERMEDIATES / "malProp.csv", index_col=0)
        # Get the spot name from r_result columns or use index
        theta_sum = 1.0  # approximate; exact value from malProp

        theta0 = np.full(n_cell, 0.05)
        ui = np.vstack([np.eye(n_cell), np.ones((1, n_cell)), -np.ones((1, n_cell))])
        ci = np.concatenate([np.zeros(n_cell), [0.0], [-1.0]])

        def f0(theta, A, b):
            return np.sum((A @ theta - b) ** 2)

        prop, _ = constr_optim(theta0, f0, ui, ci, args=(A, b))
        bhat = A @ prop

        def f_w(theta, A, b):
            return np.sum((A @ theta - b) ** 2 / (bhat + 1))

        prop2, _ = constr_optim(theta0, f_w, ui, ci, args=(A, b))

        np.testing.assert_allclose(prop2, r_pass2, rtol=RTOL, atol=ATOL)
```

- [ ] **Step 2: Write test for L1 propMat equivalence**

```python
class TestSpatialDeconvL1Equivalence:
    """Verify Level-1 propMat matches R at machine precision."""

    def setup_method(self):
        _skip_if_no_validation()

    def test_propmat_l1_matches_r(self):
        r_propmat = pd.read_csv(INTERMEDIATES / "propMat_L1_R.csv", index_col=0)
        # This will be filled in once we can run the pipeline
        assert r_propmat.shape[0] > 0, "R reference data loaded"
```

- [ ] **Step 3: Write test for full pipeline equivalence (vst1, vst2, vst3)**

```python
class TestFullPipelineEquivalence:
    """Verify full deconvolution matches R for 3 Visium datasets."""

    @pytest.mark.parametrize("dataset", ["vst1", "vst2", "vst3"])
    def test_propmat_matches_r(self, dataset):
        r_propmat = pd.read_csv(VALIDATION / f"{dataset}_propMat.csv", index_col=0)
        r_malprop = pd.read_csv(VALIDATION / f"{dataset}_malProp.csv", index_col=0)
        assert r_propmat.shape[0] > 0
        assert r_malprop.shape[0] > 0
        # Pipeline execution will be added after R removal
```

- [ ] **Step 4: Run tests to confirm they load data correctly**

Run: `pytest tests/test_deconvolution/test_r_equivalence.py -v`
Expected: Tests pass (data loading) or skip (no validation dir)

- [ ] **Step 5: Commit**

```bash
git add tests/test_deconvolution/test_r_equivalence.py
git commit -m "Add strict R equivalence tests for deconvolution"
```

---

### Task 2: Remove R Subprocess from core.py

**Files:**
- Modify: `spatialgpu/deconvolution/core.py`

- [ ] **Step 1: Replace deconvolution() entry point**

Change lines 68-74 from R-first/Python-fallback to Python-only:

```python
def deconvolution(adata, cancer_type, signature_type=None, adjacent_normal=False, n_jobs=1):
    # Remove try/except R block. Call Python directly:
    prop_mat, mal_res = _deconvolution_python(
        adata, cancer_type, signature_type, adjacent_normal, n_jobs
    )
    # ... rest stays the same
```

- [ ] **Step 2: Replace _spatial_deconv() to call Python directly**

Change lines 720-766: remove R try/except, call `_spatial_deconv_python()` directly.

```python
def _spatial_deconv(ST, gene_names, spot_names, ref, mal_prop, mal_ref,
                    mode="standard", unidentifiable=True, macrophage_other=True, n_jobs=1):
    return _spatial_deconv_python(
        ST, gene_names, spot_names, ref, mal_prop, mal_ref,
        mode, unidentifiable, macrophage_other, n_jobs,
    )
```

- [ ] **Step 3: Replace _solve_constrained_batch() to call Python directly**

Change lines 1153-1179: remove R try/except, call `_solve_constrained_batch_python()` directly.

```python
def _solve_constrained_batch(A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr, n_jobs=1):
    return _solve_constrained_batch_python(
        A, B, n_cell, theta_sum, pp_min_arr, pp_max_arr, n_jobs
    )
```

- [ ] **Step 4: Replace _compute_mal_ref() to use Python directly**

Change lines 1337-1359: remove R try/except, inline the Python logic.

```python
def _compute_mal_ref(counts, gene_names, spot_mal_idx):
    mal_counts = counts[:, spot_mal_idx]
    if sparse.issparse(mal_counts):
        mal_col_sums = np.asarray(mal_counts.sum(axis=0)).ravel()
        mal_cpm = mal_counts.toarray().astype(np.float64)
        mal_cpm = mal_cpm / mal_col_sums[np.newaxis, :] * 1e6
    else:
        mal_col_sums = mal_counts.sum(axis=0)
        mal_cpm = mal_counts.astype(np.float64) / mal_col_sums[np.newaxis, :] * 1e6
    mal_ref = np.nanmean(mal_cpm, axis=1)
    return pd.Series(mal_ref, index=gene_names)
```

- [ ] **Step 5: Delete all `_via_r()` functions**

Delete these functions entirely:
- `_deconvolution_via_r()` (lines 96-201)
- `_spatial_deconv_via_r()` (lines 769-895)
- `_solve_constrained_batch_via_r()` (lines 1182-1268)
- `_compute_mal_ref_via_r()` (lines 1362-1413)

- [ ] **Step 6: Run existing tests**

Run: `pytest tests/test_deconvolution/test_core.py -v`
Expected: All pass (unit tests don't depend on R)

- [ ] **Step 7: Commit**

```bash
git add spatialgpu/deconvolution/core.py
git commit -m "Remove R subprocess from core deconvolution"
```

---

### Task 3: Remove R Subprocess from mudan.py

**Files:**
- Modify: `spatialgpu/deconvolution/mudan.py`

- [ ] **Step 1: Replace mudan_cluster() entry point**

Make Python the sole path (remove R try/except):

```python
def mudan_cluster(counts, n_pcs=30, gam_k=5, alpha=0.05):
    return _mudan_cluster_python(counts, n_pcs, gam_k, alpha)
```

- [ ] **Step 2: Delete `_mudan_cluster_via_r()` function**

Delete lines 69-159 entirely.

- [ ] **Step 3: Update module docstring**

Remove references to R subprocess requirement.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_deconvolution/ -v -k "not integration"`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add spatialgpu/deconvolution/mudan.py
git commit -m "Remove R subprocess from MUDAN clustering"
```

---

### Task 4: Remove R Subprocess from Other Modules

**Files:**
- Modify: `spatialgpu/deconvolution/spatial_correlation.py`
- Modify: `spatialgpu/deconvolution/extensions.py`
- Modify: `spatialgpu/deconvolution/gene_set_score.py`

- [ ] **Step 1: spatial_correlation.py — remove `_vst_normalize_via_r()`**

Replace `vst_normalize()` entry point to call Python directly. Delete `_vst_normalize_via_r()`.

- [ ] **Step 2: extensions.py — remove `_de_limma_via_r()`**

Replace DE entry point to call Python directly. Delete `_de_limma_via_r()`.

- [ ] **Step 3: gene_set_score.py — remove `_ucell_score_via_r()`**

Replace `gene_set_score()` entry point to call Python directly. Delete `_ucell_score_via_r()`.

- [ ] **Step 4: Verify no remaining R subprocess references**

Run: `grep -r "subprocess\|Rscript\|_via_r" spatialgpu/`
Expected: No matches

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Run lint**

Run: `ruff check spatialgpu/ && black --check spatialgpu/ tests/`

- [ ] **Step 7: Commit**

```bash
git add spatialgpu/deconvolution/spatial_correlation.py spatialgpu/deconvolution/extensions.py spatialgpu/deconvolution/gene_set_score.py
git commit -m "Remove R subprocess from spatial_correlation, extensions, gene_set_score"
```

---

### Task 5: Build End-to-End Validation Script

**Files:**
- Create: `scripts/validate_r_equivalence.py`
- Modify: `tests/test_deconvolution/test_r_equivalence.py`

- [ ] **Step 1: Write validation script for 3 Visium datasets**

```python
"""Validate Python deconvolution against R outputs for vst1, vst2, vst3.

Run: python scripts/validate_r_equivalence.py
"""
import numpy as np
import pandas as pd
import spatialgpu.deconvolution as spacet

RTOL = 1e-10
ATOL = 1e-12
DATASETS = {
    "vst1": {"path": "data/Visium_BC", "cancer": "BRCA"},
    "vst2": {"path": "data/Visium_HCC", "cancer": "LIHC"},
    "vst3": {"path": "data/hiresST_CRC/hiresST_CRC.h5ad", "cancer": "CRC"},
}

for name, cfg in DATASETS.items():
    print(f"\n=== {name} ===")
    # Load data
    if cfg["path"].endswith(".h5ad"):
        import anndata as ad
        adata = ad.read_h5ad(cfg["path"])
    else:
        adata = spacet.create_spacet_object_10x(cfg["path"])
    adata = spacet.quality_control(adata, min_genes=1000)

    # Run Python deconvolution
    adata = spacet.deconvolution(adata, cancer_type=cfg["cancer"], n_jobs=8)

    # Load R reference
    r_propmat = pd.read_csv(f"validation/{name}_propMat.csv", index_col=0)
    r_malprop = pd.read_csv(f"validation/{name}_malProp.csv", index_col=0)

    py_propmat = adata.uns["spacet"]["deconvolution"]["propMat"]

    # Compare
    common_types = sorted(set(r_propmat.index) & set(py_propmat.index))
    common_spots = sorted(set(r_propmat.columns) & set(py_propmat.columns))

    r_sub = r_propmat.loc[common_types, common_spots].values
    py_sub = py_propmat.loc[common_types, common_spots].values

    max_diff = np.max(np.abs(r_sub - py_sub))
    corr = np.corrcoef(r_sub.ravel(), py_sub.ravel())[0, 1]

    print(f"  Common types: {len(common_types)}, spots: {len(common_spots)}")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Correlation: {corr:.10f}")
    print(f"  allclose(rtol={RTOL}): {np.allclose(r_sub, py_sub, rtol=RTOL, atol=ATOL)}")
```

- [ ] **Step 2: Run validation script**

Run: `python scripts/validate_r_equivalence.py`
Observe: Max diff and correlation for each dataset. Iterate on fixes until allclose passes.

- [ ] **Step 3: Update test_r_equivalence.py with full pipeline tests**

Once validation passes, update the pytest tests to run the same comparisons.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_r_equivalence.py tests/test_deconvolution/test_r_equivalence.py
git commit -m "Add end-to-end R equivalence validation for 3 Visium datasets"
```

---

### Task 6: Iterate Until Machine Precision

- [ ] **Step 1: Run validation, identify precision gaps**

Expected gaps: constrOptim convergence, CPM normalization order-of-ops, MUDAN clustering

- [ ] **Step 2: Fix constrOptim if needed**

Compare `constr_optim.py` output against `validation/intermediates/spot*_L1_result.csv` for each diagnostic spot. Adjust convergence parameters if needed.

- [ ] **Step 3: Fix CPM normalization if needed**

Compare Python CPM output against `validation/intermediates/ST_cpm_sample.csv`. Ensure operation ordering matches R's `t(t(counts) * 1e6 / colSums(counts))`.

- [ ] **Step 4: Fix MUDAN clustering if needed**

Compare Python cluster assignments against `validation/mudan_intermediates/`. If GAM produces different overdispersed genes, tune pygam parameters or use scipy spline as alternative.

- [ ] **Step 5: Re-run validation, verify allclose passes for all 3 datasets**

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "Achieve machine-precision R equivalence for all 3 Visium datasets"
```

- [ ] **Step 7: Push**

```bash
git push origin main
```
