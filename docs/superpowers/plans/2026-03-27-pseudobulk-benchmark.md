# Pseudobulk Benchmark (Tutorial T8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pseudobulk benchmarking module and Tutorial T8 that evaluates `deconvolution_bulk` accuracy against known ground truth and exports data for comparison with MuSiC/CIBERSORTx.

**Architecture:** Core functions live in `spatialgpu/benchmarks/pseudobulk.py` — semi-synthetic scRNA-seq generation, pseudobulk mixing (Dirichlet + titration), evaluation metrics, and export/import helpers. The tutorial script `docs/run_full_tutorial_t8_bulk_benchmark.py` orchestrates everything following T1-T7 conventions. Unit tests in `tests/test_benchmarks/test_pseudobulk.py`.

**Tech Stack:** numpy, scipy (negative binomial, Dirichlet, stats), pandas, anndata, matplotlib, spatialgpu.deconvolution (load_comb_ref, get_cancer_signature, deconvolution_bulk)

---

## File Structure

| File | Responsibility |
|---|---|
| `spatialgpu/benchmarks/pseudobulk.py` (create) | All pseudobulk benchmark functions: generation, evaluation, export/import |
| `spatialgpu/benchmarks/__init__.py` (modify) | Re-export new public functions |
| `tests/test_benchmarks/test_pseudobulk.py` (create) | Unit tests for all pseudobulk functions |
| `tests/test_benchmarks/__init__.py` (create) | Empty init |
| `docs/run_full_tutorial_t8_bulk_benchmark.py` (create) | Tutorial script |
| `scripts/slurm_tutorial_t8_gpu.sh` (create) | SLURM submission |

---

### Task 1: Semi-synthetic scRNA-seq generation

**Files:**
- Create: `tests/test_benchmarks/__init__.py`
- Create: `tests/test_benchmarks/test_pseudobulk.py`
- Create: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests for `generate_semi_synthetic_scrna`**

Create `tests/test_benchmarks/__init__.py` (empty file).

Create `tests/test_benchmarks/test_pseudobulk.py`:

```python
"""Tests for pseudobulk benchmark utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialgpu.benchmarks.pseudobulk import generate_semi_synthetic_scrna


class TestGenerateSemiSyntheticScrna:
    """Test semi-synthetic scRNA-seq generation."""

    @pytest.fixture(scope="class")
    def scrna_with_mal(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=True, cancer_type="BRCA", seed=42
        )

    @pytest.fixture(scope="class")
    def scrna_no_mal(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=False, seed=42
        )

    def test_returns_anndata(self, scrna_with_mal):
        import anndata as ad
        assert isinstance(scrna_with_mal, ad.AnnData)

    def test_cell_type_column_exists(self, scrna_with_mal):
        assert "cell_type" in scrna_with_mal.obs.columns

    def test_correct_n_cells_per_type(self, scrna_with_mal):
        counts = scrna_with_mal.obs["cell_type"].value_counts()
        assert all(counts == 50)

    def test_includes_malignant_type(self, scrna_with_mal):
        types = set(scrna_with_mal.obs["cell_type"])
        assert "Malignant_BRCA" in types

    def test_excludes_malignant_when_off(self, scrna_no_mal):
        types = set(scrna_no_mal.obs["cell_type"])
        assert not any("Malignant" in t for t in types)

    def test_has_level1_cell_types(self, scrna_with_mal):
        types = set(scrna_with_mal.obs["cell_type"])
        expected = {"CAF", "Endothelial", "B cell", "T CD4", "T CD8", "Macrophage"}
        assert expected.issubset(types)

    def test_counts_are_nonneg_integers(self, scrna_with_mal):
        X = scrna_with_mal.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert np.all(X >= 0)
        assert np.allclose(X, X.astype(int))

    def test_has_dropout(self, scrna_with_mal):
        """Substantial fraction of entries should be zero (dropout)."""
        X = scrna_with_mal.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        zero_frac = (X == 0).sum() / X.size
        assert zero_frac > 0.3  # expect significant sparsity

    def test_deterministic_with_seed(self):
        a = generate_semi_synthetic_scrna(n_cells_per_type=20, seed=99)
        b = generate_semi_synthetic_scrna(n_cells_per_type=20, seed=99)
        Xa = a.X.toarray() if hasattr(a.X, "toarray") else a.X
        Xb = b.X.toarray() if hasattr(b.X, "toarray") else b.X
        np.testing.assert_array_equal(Xa, Xb)

    def test_gene_names_from_reference(self, scrna_with_mal):
        from spatialgpu.deconvolution.reference import load_comb_ref
        ref = load_comb_ref()
        ref_genes = set(ref["refProfiles"].index)
        scrna_genes = set(scrna_with_mal.var_names)
        # All generated genes should come from the reference
        assert scrna_genes.issubset(ref_genes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGenerateSemiSyntheticScrna -v --no-header 2>&1 | head -30`
Expected: ImportError — `cannot import name 'generate_semi_synthetic_scrna'`

- [ ] **Step 3: Implement `generate_semi_synthetic_scrna`**

Create `spatialgpu/benchmarks/pseudobulk.py`:

```python
"""Pseudobulk benchmark utilities for evaluating deconvolution accuracy.

Generates semi-synthetic scRNA-seq data, creates pseudobulk mixtures with
known cell type proportions, evaluates deconvolution results, and exports
data for comparison with external tools (MuSiC, CIBERSORTx).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse, stats

if TYPE_CHECKING:
    import anndata as ad
    import matplotlib.figure

logger = logging.getLogger(__name__)

# Level 1 cell types used for pseudobulk mixing
_LEVEL1_TYPES = [
    "CAF", "Endothelial", "Plasma", "B cell", "T CD4", "T CD8",
    "NK", "cDC", "pDC", "Macrophage", "Mast", "Neutrophil",
]


def generate_semi_synthetic_scrna(
    n_cells_per_type: int = 500,
    include_malignant: bool = True,
    cancer_type: str = "BRCA",
    seed: int = 42,
) -> ad.AnnData:
    """Generate semi-synthetic scRNA-seq from reference profiles.

    Uses negative binomial noise and logistic dropout to create realistic
    single-cell counts from the bundled SpaCET reference profiles.

    Parameters
    ----------
    n_cells_per_type
        Number of cells to generate per cell type.
    include_malignant
        If True, adds a Malignant_{cancer_type} cell type from the cancer
        signature overlaid on mean expression.
    cancer_type
        Cancer type code for malignant signature (e.g., 'BRCA').
    seed
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Synthetic scRNA-seq with obs['cell_type'] labels and integer counts.
    """
    import anndata as ad

    from spatialgpu.deconvolution.reference import (
        get_cancer_signature,
        load_comb_ref,
    )

    rng = np.random.RandomState(seed)
    ref = load_comb_ref()
    ref_profiles = ref["refProfiles"]  # genes x cell_types, CPM

    # Use Level 1 types only
    type_names = [t for t in _LEVEL1_TYPES if t in ref_profiles.columns]
    profiles = ref_profiles[type_names].copy()

    # Add malignant type from cancer signature
    if include_malignant:
        _, sig = get_cancer_signature(cancer_type)
        if sig is not None and len(sig) > 0:
            mal_name = f"Malignant_{cancer_type}"
            # Start from mean expression, add signature effect
            mean_expr = ref_profiles[type_names].mean(axis=1)
            mal_profile = mean_expr.copy()
            olp = mal_profile.index.intersection(sig.index)
            # Shift expression by signature (positive = upregulated in tumor)
            mal_profile.loc[olp] = mal_profile.loc[olp] + sig.loc[olp] * mean_expr.loc[olp].clip(lower=1)
            mal_profile = mal_profile.clip(lower=0)
            profiles[mal_name] = mal_profile
            type_names.append(mal_name)

    gene_names = np.array(profiles.index)
    n_genes = len(gene_names)

    all_counts = []
    all_labels = []

    for ct in type_names:
        profile = profiles[ct].values.astype(np.float64)
        # Convert CPM to relative probabilities
        profile_prob = profile / (profile.sum() + 1e-10)

        for _ in range(n_cells_per_type):
            # Draw library size from LogNormal
            total_umi = int(rng.lognormal(mean=8.5, sigma=0.5))
            total_umi = max(total_umi, 100)

            # Expected counts per gene
            mu = profile_prob * total_umi

            # Negative binomial: size = dispersion parameter
            # Higher dispersion for lowly expressed genes
            dispersion = np.maximum(0.5, mu / 2)
            # scipy NB: n=size, p=size/(size+mu)
            p = dispersion / (dispersion + mu + 1e-10)
            counts = rng.negative_binomial(n=np.maximum(dispersion, 0.01).astype(np.float64), p=p)

            # Logistic dropout: P(drop) = 1 / (1 + exp(1.5 * log(mu+1) - 2))
            log_mu = np.log(mu + 1)
            drop_prob = 1.0 / (1.0 + np.exp(1.5 * log_mu - 2.0))
            dropout_mask = rng.random(n_genes) < drop_prob
            counts[dropout_mask] = 0

            all_counts.append(counts)
            all_labels.append(ct)

    X = np.vstack(all_counts).astype(np.float64)
    X_sparse = sparse.csr_matrix(X)

    adata = ad.AnnData(
        X=X_sparse,
        obs=pd.DataFrame({"cell_type": all_labels}),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obs_names = [f"Cell_{i:06d}" for i in range(adata.n_obs)]

    return adata
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGenerateSemiSyntheticScrna -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/__init__.py tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add semi-synthetic scRNA-seq generation for pseudobulk benchmark"
```

---

### Task 2: Dirichlet pseudobulk generation

**Files:**
- Modify: `tests/test_benchmarks/test_pseudobulk.py`
- Modify: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests for `generate_pseudobulk_dirichlet`**

Append to `tests/test_benchmarks/test_pseudobulk.py`:

```python
from spatialgpu.benchmarks.pseudobulk import generate_pseudobulk_dirichlet


class TestGeneratePseudobulkDirichlet:
    """Test Dirichlet-based pseudobulk generation."""

    @pytest.fixture(scope="class")
    def scrna(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=True, seed=42
        )

    @pytest.fixture(scope="class")
    def bulk_and_truth(self, scrna):
        return generate_pseudobulk_dirichlet(
            scrna, n_samples=20, n_cells_per_sample=200, alpha=1.0, seed=42
        )

    def test_returns_tuple(self, bulk_and_truth):
        adata_bulk, ground_truth = bulk_and_truth
        import anndata as ad
        assert isinstance(adata_bulk, ad.AnnData)
        assert isinstance(ground_truth, pd.DataFrame)

    def test_bulk_shape(self, bulk_and_truth, scrna):
        adata_bulk, _ = bulk_and_truth
        assert adata_bulk.n_obs == 20
        assert adata_bulk.n_vars == scrna.n_vars

    def test_ground_truth_shape(self, bulk_and_truth):
        _, gt = bulk_and_truth
        assert gt.shape[0] == 20  # n_samples
        assert gt.shape[1] > 5   # n_cell_types

    def test_ground_truth_sums_to_one(self, bulk_and_truth):
        _, gt = bulk_and_truth
        np.testing.assert_allclose(gt.sum(axis=1).values, 1.0, atol=1e-10)

    def test_ground_truth_nonnegative(self, bulk_and_truth):
        _, gt = bulk_and_truth
        assert gt.min().min() >= 0.0

    def test_bulk_counts_nonneg_integers(self, bulk_and_truth):
        adata_bulk, _ = bulk_and_truth
        X = adata_bulk.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        assert np.all(X >= 0)
        assert np.allclose(X, X.astype(int))

    def test_deterministic_with_seed(self, scrna):
        _, gt1 = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=77)
        _, gt2 = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=77)
        np.testing.assert_array_equal(gt1.values, gt2.values)

    def test_cell_type_columns_match_scrna(self, bulk_and_truth, scrna):
        _, gt = bulk_and_truth
        scrna_types = set(scrna.obs["cell_type"].unique())
        gt_types = set(gt.columns)
        assert gt_types == scrna_types
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGeneratePseudobulkDirichlet -v --no-header 2>&1 | head -5`
Expected: ImportError — `cannot import name 'generate_pseudobulk_dirichlet'`

- [ ] **Step 3: Implement `generate_pseudobulk_dirichlet`**

Append to `spatialgpu/benchmarks/pseudobulk.py`:

```python
def generate_pseudobulk_dirichlet(
    scrna_adata: ad.AnnData,
    n_samples: int = 100,
    n_cells_per_sample: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Generate pseudobulk by mixing single cells with Dirichlet proportions.

    Parameters
    ----------
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    n_samples
        Number of pseudobulk samples to generate.
    n_cells_per_sample
        Total cells to sample per pseudobulk mixture.
    alpha
        Dirichlet concentration. 1.0 = uniform; < 1.0 = sparser mixtures.
    seed
        Random seed.

    Returns
    -------
    tuple of (AnnData, DataFrame)
        AnnData: pseudobulk (samples x genes, raw counts).
        DataFrame: ground truth proportions (samples x cell_types, sums to 1.0).
    """
    import anndata as ad

    rng = np.random.RandomState(seed)
    cell_types = sorted(scrna_adata.obs["cell_type"].unique())
    n_types = len(cell_types)

    # Group cell indices by type
    type_indices = {}
    for ct in cell_types:
        type_indices[ct] = np.where(scrna_adata.obs["cell_type"].values == ct)[0]

    X_all = scrna_adata.X
    if sparse.issparse(X_all):
        X_all = X_all.toarray()

    bulk_counts = np.zeros((n_samples, scrna_adata.n_vars), dtype=np.float64)
    proportions = np.zeros((n_samples, n_types), dtype=np.float64)

    alpha_vec = np.full(n_types, alpha)

    for i in range(n_samples):
        # Sample proportions from Dirichlet
        props = rng.dirichlet(alpha_vec)
        proportions[i] = props

        # Draw cell counts per type from Multinomial
        cell_counts = rng.multinomial(n_cells_per_sample, props)

        # Sample and sum cells
        sample_sum = np.zeros(scrna_adata.n_vars, dtype=np.float64)
        for j, ct in enumerate(cell_types):
            if cell_counts[j] == 0:
                continue
            idx = rng.choice(type_indices[ct], size=cell_counts[j], replace=True)
            sample_sum += X_all[idx].sum(axis=0)

        bulk_counts[i] = sample_sum

    adata_bulk = ad.AnnData(
        X=bulk_counts,
        obs=pd.DataFrame(index=[f"Bulk_{i:04d}" for i in range(n_samples)]),
        var=pd.DataFrame(index=scrna_adata.var_names.copy()),
    )

    ground_truth = pd.DataFrame(
        proportions,
        index=adata_bulk.obs_names,
        columns=cell_types,
    )

    return adata_bulk, ground_truth
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGeneratePseudobulkDirichlet -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add Dirichlet pseudobulk generation"
```

---

### Task 3: Titration pseudobulk generation

**Files:**
- Modify: `tests/test_benchmarks/test_pseudobulk.py`
- Modify: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests for `generate_pseudobulk_titration`**

Append to `tests/test_benchmarks/test_pseudobulk.py`:

```python
from spatialgpu.benchmarks.pseudobulk import generate_pseudobulk_titration


class TestGeneratePseudobulkTitration:
    """Test titration-based pseudobulk generation."""

    @pytest.fixture(scope="class")
    def scrna(self):
        return generate_semi_synthetic_scrna(
            n_cells_per_type=50, include_malignant=True, seed=42
        )

    @pytest.fixture(scope="class")
    def titration_result(self, scrna):
        fracs = [0.0, 0.2, 0.5, 0.8]
        return generate_pseudobulk_titration(
            scrna,
            target_type="Malignant_BRCA",
            fractions=fracs,
            n_replicates=3,
            n_cells_per_sample=200,
            seed=42,
        )

    def test_returns_tuple(self, titration_result):
        adata_bulk, gt = titration_result
        import anndata as ad
        assert isinstance(adata_bulk, ad.AnnData)
        assert isinstance(gt, pd.DataFrame)

    def test_correct_sample_count(self, titration_result):
        adata_bulk, gt = titration_result
        # 4 fractions x 3 replicates = 12
        assert adata_bulk.n_obs == 12
        assert gt.shape[0] == 12

    def test_ground_truth_sums_to_one(self, titration_result):
        _, gt = titration_result
        np.testing.assert_allclose(gt.sum(axis=1).values, 1.0, atol=1e-10)

    def test_target_fraction_column(self, titration_result):
        _, gt = titration_result
        assert "target_fraction" in gt.columns

    def test_target_fractions_match_requested(self, titration_result):
        _, gt = titration_result
        expected = [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
        np.testing.assert_allclose(gt["target_fraction"].values, expected, atol=1e-10)

    def test_target_type_proportion_matches(self, titration_result):
        _, gt = titration_result
        # The actual Malignant_BRCA proportion should match target_fraction
        # (they are set explicitly, not sampled)
        mal_col = gt["Malignant_BRCA"]
        np.testing.assert_allclose(
            mal_col.values, gt["target_fraction"].values, atol=1e-10
        )

    def test_other_types_sum_to_remainder(self, titration_result):
        _, gt = titration_result
        non_target = gt.drop(columns=["target_fraction", "Malignant_BRCA"])
        remainder = 1.0 - gt["target_fraction"].values
        np.testing.assert_allclose(
            non_target.sum(axis=1).values, remainder, atol=1e-10
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGeneratePseudobulkTitration -v --no-header 2>&1 | head -5`
Expected: ImportError — `cannot import name 'generate_pseudobulk_titration'`

- [ ] **Step 3: Implement `generate_pseudobulk_titration`**

Append to `spatialgpu/benchmarks/pseudobulk.py`:

```python
def generate_pseudobulk_titration(
    scrna_adata: ad.AnnData,
    target_type: str = "Malignant_BRCA",
    fractions: list[float] | None = None,
    n_replicates: int = 5,
    n_cells_per_sample: int = 1000,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Generate pseudobulk with systematic titration of one cell type.

    Fixes the target type at each specified fraction and distributes
    the remainder across other types via Dirichlet(1.0).

    Parameters
    ----------
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    target_type
        Cell type to titrate (e.g., 'Malignant_BRCA').
    fractions
        Target fractions to sweep. Default: [0.0, 0.1, ..., 0.8].
    n_replicates
        Number of replicates per fraction.
    n_cells_per_sample
        Total cells per pseudobulk sample.
    seed
        Random seed.

    Returns
    -------
    tuple of (AnnData, DataFrame)
        AnnData: pseudobulk (samples x genes).
        DataFrame: ground truth with extra 'target_fraction' column.
    """
    import anndata as ad

    if fractions is None:
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    rng = np.random.RandomState(seed)
    cell_types = sorted(scrna_adata.obs["cell_type"].unique())

    if target_type not in cell_types:
        raise ValueError(
            f"target_type '{target_type}' not in scrna_adata cell types: {cell_types}"
        )

    other_types = [ct for ct in cell_types if ct != target_type]
    n_other = len(other_types)

    type_indices = {}
    for ct in cell_types:
        type_indices[ct] = np.where(scrna_adata.obs["cell_type"].values == ct)[0]

    X_all = scrna_adata.X
    if sparse.issparse(X_all):
        X_all = X_all.toarray()

    all_counts = []
    all_proportions = []
    all_target_fracs = []

    for frac in fractions:
        for _rep in range(n_replicates):
            # Build proportion vector
            props = {}
            props[target_type] = frac

            if frac < 1.0 - 1e-10:
                remainder_props = rng.dirichlet(np.ones(n_other))
                for j, ct in enumerate(other_types):
                    props[ct] = remainder_props[j] * (1.0 - frac)
            else:
                for ct in other_types:
                    props[ct] = 0.0

            # Draw cells and sum
            sample_sum = np.zeros(scrna_adata.n_vars, dtype=np.float64)
            for ct in cell_types:
                n_cells = int(round(props[ct] * n_cells_per_sample))
                if n_cells == 0:
                    continue
                idx = rng.choice(type_indices[ct], size=n_cells, replace=True)
                sample_sum += X_all[idx].sum(axis=0)

            all_counts.append(sample_sum)
            all_proportions.append([props[ct] for ct in cell_types])
            all_target_fracs.append(frac)

    n_total = len(all_counts)
    adata_bulk = ad.AnnData(
        X=np.vstack(all_counts),
        obs=pd.DataFrame(index=[f"Titr_{i:04d}" for i in range(n_total)]),
        var=pd.DataFrame(index=scrna_adata.var_names.copy()),
    )

    ground_truth = pd.DataFrame(
        np.array(all_proportions),
        index=adata_bulk.obs_names,
        columns=cell_types,
    )
    ground_truth["target_fraction"] = all_target_fracs

    return adata_bulk, ground_truth
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestGeneratePseudobulkTitration -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add titration pseudobulk generation"
```

---

### Task 4: Evaluation metrics

**Files:**
- Modify: `tests/test_benchmarks/test_pseudobulk.py`
- Modify: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests for `evaluate_deconvolution`**

Append to `tests/test_benchmarks/test_pseudobulk.py`:

```python
from spatialgpu.benchmarks.pseudobulk import evaluate_deconvolution


class TestEvaluateDeconvolution:
    """Test deconvolution accuracy metrics."""

    def test_perfect_estimation(self):
        """Perfect match should give r=1, RMSE=0."""
        gt = pd.DataFrame(
            {"A": [0.3, 0.5, 0.2], "B": [0.7, 0.5, 0.8]},
            index=["S1", "S2", "S3"],
        )
        result = evaluate_deconvolution(gt.copy(), gt)
        assert result["overall"]["pearson_r"] > 0.999
        assert result["overall"]["rmse"] < 1e-10

    def test_random_estimation(self):
        """Random estimates should give low correlation."""
        rng = np.random.RandomState(42)
        gt = pd.DataFrame(
            rng.dirichlet([1, 1, 1], size=50),
            columns=["A", "B", "C"],
            index=[f"S{i}" for i in range(50)],
        )
        est = pd.DataFrame(
            rng.dirichlet([1, 1, 1], size=50),
            columns=["A", "B", "C"],
            index=[f"S{i}" for i in range(50)],
        )
        result = evaluate_deconvolution(est, gt)
        assert result["overall"]["pearson_r"] < 0.5
        assert result["overall"]["rmse"] > 0.1

    def test_per_type_keys(self):
        gt = pd.DataFrame(
            {"A": [0.3, 0.5], "B": [0.7, 0.5]}, index=["S1", "S2"]
        )
        result = evaluate_deconvolution(gt.copy(), gt)
        assert "per_type" in result
        assert "A" in result["per_type"].index
        assert "B" in result["per_type"].index
        assert "pearson_r" in result["per_type"].columns

    def test_rare_type_mae(self):
        """MAE at low fractions should be computed for truth < 0.05."""
        gt = pd.DataFrame(
            {"A": [0.02, 0.03, 0.8], "B": [0.98, 0.97, 0.2]},
            index=["S1", "S2", "S3"],
        )
        est = pd.DataFrame(
            {"A": [0.05, 0.06, 0.8], "B": [0.95, 0.94, 0.2]},
            index=["S1", "S2", "S3"],
        )
        result = evaluate_deconvolution(est, gt)
        assert "rare_type_mae" in result
        assert result["rare_type_mae"] > 0  # there is error on rare entries

    def test_aligns_on_common_types(self):
        """Should handle mismatched columns gracefully."""
        gt = pd.DataFrame(
            {"A": [0.3, 0.5], "B": [0.7, 0.5]}, index=["S1", "S2"]
        )
        est = pd.DataFrame(
            {"A": [0.3, 0.5], "C": [0.7, 0.5]}, index=["S1", "S2"]
        )
        result = evaluate_deconvolution(est, gt)
        # Only type "A" is shared
        assert len(result["per_type"]) == 1

    def test_spearman_in_overall(self):
        gt = pd.DataFrame(
            {"A": [0.3, 0.5], "B": [0.7, 0.5]}, index=["S1", "S2"]
        )
        result = evaluate_deconvolution(gt.copy(), gt)
        assert "spearman_rho" in result["overall"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestEvaluateDeconvolution -v --no-header 2>&1 | head -5`
Expected: ImportError — `cannot import name 'evaluate_deconvolution'`

- [ ] **Step 3: Implement `evaluate_deconvolution`**

Append to `spatialgpu/benchmarks/pseudobulk.py`:

```python
def _collapse_to_level1(
    prop_mat: pd.DataFrame,
    level1_types: list[str],
) -> pd.DataFrame:
    """Collapse hierarchical propMat to Level 1 types.

    The propMat from deconvolution_bulk has both Level 1 (e.g., 'B cell')
    and Level 2 (e.g., 'B cell naive') rows. For evaluation against
    pseudobulk mixed at Level 1 granularity, we keep only Level 1 rows.

    Level 1 rows already represent the total fraction for that lineage
    (they are not the sum of Level 2 subtypes — they are computed
    independently at Level 1 of the hierarchy). So we just select them.
    """
    available = [t for t in level1_types if t in prop_mat.index]
    return prop_mat.loc[available]


def evaluate_deconvolution(
    estimated: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate deconvolution accuracy against known proportions.

    Parameters
    ----------
    estimated
        Estimated proportions (samples x cell_types) or (cell_types x samples).
        Auto-transposed if needed to match ground_truth orientation.
    ground_truth
        True proportions (samples x cell_types).

    Returns
    -------
    dict with keys:
        overall : dict with 'pearson_r', 'spearman_rho', 'rmse'
        per_type : DataFrame (cell_types x metrics)
        rare_type_mae : float (MAE for entries with truth < 0.05)
    """
    est = estimated.copy()
    gt = ground_truth.copy()

    # Auto-transpose: if estimated looks like cell_types x samples, flip it
    if est.shape[0] != gt.shape[0] and est.shape[1] == gt.shape[0]:
        est = est.T
    if est.shape[1] != gt.shape[1] and est.shape[0] == gt.shape[1]:
        est = est.T

    # Align on common types and samples
    common_types = est.columns.intersection(gt.columns)
    common_samples = est.index.intersection(gt.index)

    if len(common_types) == 0:
        raise ValueError("No common cell types between estimated and ground truth.")

    n_gt_types = len(gt.columns)
    if len(common_types) < 0.8 * n_gt_types:
        logger.warning(
            "evaluate_deconvolution: only %d/%d cell types overlap (%.0f%%).",
            len(common_types), n_gt_types, 100 * len(common_types) / n_gt_types,
        )

    est_aligned = est.loc[common_samples, common_types].values.astype(np.float64)
    gt_aligned = gt.loc[common_samples, common_types].values.astype(np.float64)

    # Overall metrics (flatten all entries)
    est_flat = est_aligned.ravel()
    gt_flat = gt_aligned.ravel()

    pearson_r, _ = stats.pearsonr(est_flat, gt_flat)
    spearman_rho, _ = stats.spearmanr(est_flat, gt_flat)
    rmse = float(np.sqrt(np.mean((est_flat - gt_flat) ** 2)))

    # Per-cell-type metrics
    per_type_rows = []
    for i, ct in enumerate(common_types):
        e = est_aligned[:, i]
        g = gt_aligned[:, i]
        if np.std(g) < 1e-15 or np.std(e) < 1e-15:
            r = np.nan
        else:
            r, _ = stats.pearsonr(e, g)
        ct_rmse = float(np.sqrt(np.mean((e - g) ** 2)))
        per_type_rows.append({
            "cell_type": ct,
            "pearson_r": r,
            "rmse": ct_rmse,
            "n_samples": len(common_samples),
        })

    per_type = pd.DataFrame(per_type_rows).set_index("cell_type")

    # Rare type MAE (truth < 0.05)
    rare_mask = gt_aligned < 0.05
    if rare_mask.sum() > 0:
        rare_mae = float(np.mean(np.abs(est_aligned[rare_mask] - gt_aligned[rare_mask])))
    else:
        rare_mae = 0.0

    return {
        "overall": {
            "pearson_r": float(pearson_r),
            "spearman_rho": float(spearman_rho),
            "rmse": rmse,
        },
        "per_type": per_type,
        "rare_type_mae": rare_mae,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestEvaluateDeconvolution -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add deconvolution evaluation metrics (Pearson, Spearman, RMSE, per-type, rare MAE)"
```

---

### Task 5: Export helpers (MuSiC + CIBERSORTx)

**Files:**
- Modify: `tests/test_benchmarks/test_pseudobulk.py`
- Modify: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests for export functions**

Append to `tests/test_benchmarks/test_pseudobulk.py`:

```python
import os
import tempfile

from spatialgpu.benchmarks.pseudobulk import export_for_cibersortx, export_for_music


class TestExportForMusic:
    """Test MuSiC export file generation."""

    @pytest.fixture(scope="class")
    def export_dir(self):
        scrna = generate_semi_synthetic_scrna(n_cells_per_type=10, seed=42)
        bulk, gt = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=42)
        d = tempfile.mkdtemp()
        export_for_music(bulk, scrna, d, gt)
        return d

    def test_bulk_counts_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "bulk_counts.csv"))

    def test_sc_counts_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "sc_counts.csv"))

    def test_sc_phenodata_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "sc_phenodata.csv"))

    def test_ground_truth_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "ground_truth.csv"))

    def test_run_music_r_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "run_music.R"))

    def test_bulk_counts_shape(self, export_dir):
        df = pd.read_csv(os.path.join(export_dir, "bulk_counts.csv"), index_col=0)
        assert df.shape[1] == 5  # n_samples

    def test_sc_phenodata_has_cell_type(self, export_dir):
        df = pd.read_csv(os.path.join(export_dir, "sc_phenodata.csv"), index_col=0)
        assert "cell_type" in df.columns


class TestExportForCibersortx:
    """Test CIBERSORTx export file generation."""

    @pytest.fixture(scope="class")
    def export_dir(self):
        scrna = generate_semi_synthetic_scrna(n_cells_per_type=10, seed=42)
        bulk, gt = generate_pseudobulk_dirichlet(scrna, n_samples=5, seed=42)
        d = tempfile.mkdtemp()
        export_for_cibersortx(bulk, scrna, d, gt)
        return d

    def test_mixture_txt_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "mixture.txt"))

    def test_sc_reference_txt_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "sc_reference.txt"))

    def test_readme_exists(self, export_dir):
        assert os.path.exists(os.path.join(export_dir, "README_cibersortx.txt"))

    def test_mixture_is_tab_delimited(self, export_dir):
        df = pd.read_csv(os.path.join(export_dir, "mixture.txt"), sep="\t", index_col=0)
        assert df.shape[0] > 100  # genes
        assert df.shape[1] == 5   # samples
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestExportForMusic -v --no-header 2>&1 | head -5`
Expected: ImportError — `cannot import name 'export_for_music'`

- [ ] **Step 3: Implement export functions**

Append to `spatialgpu/benchmarks/pseudobulk.py`:

```python
def export_for_music(
    adata_bulk: ad.AnnData,
    scrna_adata: ad.AnnData,
    output_dir: str,
    ground_truth: pd.DataFrame | None = None,
) -> None:
    """Export pseudobulk data in MuSiC-compatible format.

    Parameters
    ----------
    adata_bulk
        Pseudobulk AnnData (samples x genes, raw counts).
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    output_dir
        Directory to write output files.
    ground_truth
        If provided, saved alongside for evaluation.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Bulk counts: genes x samples
    X_bulk = adata_bulk.X
    if hasattr(X_bulk, "toarray"):
        X_bulk = X_bulk.toarray()
    bulk_df = pd.DataFrame(
        X_bulk.T,
        index=adata_bulk.var_names,
        columns=adata_bulk.obs_names,
    )
    bulk_df.to_csv(os.path.join(output_dir, "bulk_counts.csv"))

    # SC counts: genes x cells
    X_sc = scrna_adata.X
    if hasattr(X_sc, "toarray"):
        X_sc = X_sc.toarray()
    sc_df = pd.DataFrame(
        X_sc.T,
        index=scrna_adata.var_names,
        columns=scrna_adata.obs_names,
    )
    sc_df.to_csv(os.path.join(output_dir, "sc_counts.csv"))

    # Phenodata
    pheno = scrna_adata.obs[["cell_type"]].copy()
    pheno.to_csv(os.path.join(output_dir, "sc_phenodata.csv"))

    # Ground truth
    if ground_truth is not None:
        ground_truth.to_csv(os.path.join(output_dir, "ground_truth.csv"))

    # Ready-to-run R script
    r_script = '''\
library(MuSiC)
library(Biobase)

# Load data
bulk_counts <- as.matrix(read.csv("bulk_counts.csv", row.names = 1, check.names = FALSE))
sc_counts   <- as.matrix(read.csv("sc_counts.csv", row.names = 1, check.names = FALSE))
sc_pheno    <- read.csv("sc_phenodata.csv", row.names = 1)

# Build ExpressionSets
bulk_eset <- ExpressionSet(assayData = bulk_counts)

sc_pheno_df <- new("AnnotatedDataFrame", data = sc_pheno)
sc_eset <- ExpressionSet(assayData = sc_counts, phenoData = sc_pheno_df)

# Run MuSiC
result <- music_prop(
  bulk.eset   = bulk_eset,
  sc.eset     = sc_eset,
  clusters    = "cell_type",
  verbose     = TRUE
)

# Save results
write.csv(result$Est.prop.weighted, "music_results.csv")
cat("MuSiC results saved to music_results.csv\\n")
'''
    with open(os.path.join(output_dir, "run_music.R"), "w") as f:
        f.write(r_script)

    logger.info("MuSiC export written to %s", output_dir)


def export_for_cibersortx(
    adata_bulk: ad.AnnData,
    scrna_adata: ad.AnnData,
    output_dir: str,
    ground_truth: pd.DataFrame | None = None,
) -> None:
    """Export pseudobulk data in CIBERSORTx-compatible format.

    Parameters
    ----------
    adata_bulk
        Pseudobulk AnnData (samples x genes, raw counts).
    scrna_adata
        Semi-synthetic scRNA-seq with obs['cell_type'].
    output_dir
        Directory to write output files.
    ground_truth
        If provided, saved alongside for evaluation.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Mixture file: TPM-normalized, genes x samples, tab-delimited
    X_bulk = adata_bulk.X
    if hasattr(X_bulk, "toarray"):
        X_bulk = X_bulk.toarray()
    # CPM/TPM normalization (column sums to 1e6)
    col_sums = X_bulk.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1
    tpm = X_bulk / col_sums * 1e6

    mixture_df = pd.DataFrame(
        tpm.T,
        index=adata_bulk.var_names,
        columns=adata_bulk.obs_names,
    )
    mixture_df.index.name = "Gene"
    mixture_df.to_csv(os.path.join(output_dir, "mixture.txt"), sep="\t")

    # SC reference: tab-delimited, first row = cell type labels
    X_sc = scrna_adata.X
    if hasattr(X_sc, "toarray"):
        X_sc = X_sc.toarray()
    sc_df = pd.DataFrame(
        X_sc.T,
        index=scrna_adata.var_names,
        columns=scrna_adata.obs["cell_type"].values,
    )
    sc_df.index.name = "Gene"
    sc_df.to_csv(os.path.join(output_dir, "sc_reference.txt"), sep="\t")

    # Ground truth
    if ground_truth is not None:
        ground_truth.to_csv(os.path.join(output_dir, "ground_truth.csv"))

    # Instructions
    readme = """\
CIBERSORTx Export
=================

Files:
  mixture.txt      - TPM-normalized bulk expression (genes x samples, tab-delimited)
  sc_reference.txt - Single-cell reference (genes x cells, tab-delimited, cell type headers)
  ground_truth.csv - True cell type proportions

Steps:
  1. Go to https://cibersortx.stanford.edu/
  2. Create account / sign in
  3. "Create Signature Matrix":
     - Upload sc_reference.txt as "Single Cell Reference"
     - Set "Single Cell" = Yes
  4. "Impute Cell Fractions":
     - Upload mixture.txt as "Mixture File"
     - Select the signature matrix from step 3
     - Run
  5. Download results CSV
  6. Use import_external_results("path/to/results.csv", "CIBERSORTx")
     to load results for comparison.
"""
    with open(os.path.join(output_dir, "README_cibersortx.txt"), "w") as f:
        f.write(readme)

    logger.info("CIBERSORTx export written to %s", output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestExportForMusic tests/test_benchmarks/test_pseudobulk.py::TestExportForCibersortx -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add MuSiC and CIBERSORTx export helpers"
```

---

### Task 6: Import external results and compare methods

**Files:**
- Modify: `tests/test_benchmarks/test_pseudobulk.py`
- Modify: `spatialgpu/benchmarks/pseudobulk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_benchmarks/test_pseudobulk.py`:

```python
from spatialgpu.benchmarks.pseudobulk import compare_methods, import_external_results


class TestImportExternalResults:
    """Test importing results from MuSiC/CIBERSORTx."""

    def test_import_music_csv(self, tmp_path):
        """Should parse MuSiC output CSV (samples x cell_types)."""
        df = pd.DataFrame(
            {"A": [0.3, 0.5], "B": [0.7, 0.5]},
            index=["S1", "S2"],
        )
        path = str(tmp_path / "music_results.csv")
        df.to_csv(path)
        result = import_external_results(path, "MuSiC")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result.values, df.values)

    def test_import_cibersortx_tsv(self, tmp_path):
        """Should parse CIBERSORTx output (tab-delimited with extra columns)."""
        df = pd.DataFrame({
            "A": [0.3, 0.5],
            "B": [0.7, 0.5],
            "P-value": [0.01, 0.02],
            "Correlation": [0.9, 0.8],
            "RMSE": [0.1, 0.2],
        }, index=["S1", "S2"])
        path = str(tmp_path / "CIBERSORTx_Results.txt")
        df.to_csv(path, sep="\t")
        result = import_external_results(path, "CIBERSORTx")
        assert "A" in result.columns
        assert "B" in result.columns
        # Should drop P-value, Correlation, RMSE columns
        assert "P-value" not in result.columns


class TestCompareMethods:
    """Test multi-method comparison."""

    def test_returns_summary_and_figure(self):
        gt = pd.DataFrame(
            {"A": [0.3, 0.5, 0.2], "B": [0.7, 0.5, 0.8]},
            index=["S1", "S2", "S3"],
        )
        est1 = gt.copy()
        rng = np.random.RandomState(42)
        est2 = pd.DataFrame(
            rng.dirichlet([1, 1], size=3),
            columns=["A", "B"],
            index=["S1", "S2", "S3"],
        )
        summary, fig = compare_methods(
            {"perfect": est1, "random": est2}, gt
        )
        assert isinstance(summary, pd.DataFrame)
        assert "perfect" in summary.index
        assert "random" in summary.index
        assert summary.loc["perfect", "pearson_r"] > summary.loc["random", "pearson_r"]
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_method(self):
        gt = pd.DataFrame({"A": [0.5, 0.5]}, index=["S1", "S2"])
        summary, fig = compare_methods({"ours": gt.copy()}, gt)
        assert len(summary) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestImportExternalResults tests/test_benchmarks/test_pseudobulk.py::TestCompareMethods -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 3: Implement `import_external_results` and `compare_methods`**

Append to `spatialgpu/benchmarks/pseudobulk.py`:

```python
def import_external_results(
    results_path: str,
    tool_name: str,
) -> pd.DataFrame:
    """Import deconvolution results from an external tool.

    Parameters
    ----------
    results_path
        Path to the results CSV/TSV file.
    tool_name
        Tool name: 'MuSiC' or 'CIBERSORTx'.

    Returns
    -------
    DataFrame (samples x cell_types) with proportion values.
    """
    if tool_name == "CIBERSORTx":
        df = pd.read_csv(results_path, sep="\t", index_col=0)
        # Drop CIBERSORTx metadata columns
        drop_cols = [
            c for c in df.columns
            if c in ("P-value", "Correlation", "RMSE")
            or c.startswith("P-value")
            or c.startswith("Correlation")
            or c.startswith("RMSE")
        ]
        df = df.drop(columns=drop_cols, errors="ignore")
    else:
        # MuSiC and generic CSV
        df = pd.read_csv(results_path, index_col=0)

    return df


def compare_methods(
    results_dict: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """Compare multiple deconvolution methods against ground truth.

    Parameters
    ----------
    results_dict
        Mapping of method name to estimated proportions DataFrame.
    ground_truth
        True proportions (samples x cell_types).

    Returns
    -------
    tuple of (summary_df, figure)
        summary_df: methods x overall metrics.
        figure: grouped bar chart of per-cell-type Pearson r.
    """
    import matplotlib.pyplot as plt

    summaries = []
    all_per_type = {}

    for method_name, est in results_dict.items():
        metrics = evaluate_deconvolution(est, ground_truth)
        row = {
            "method": method_name,
            **metrics["overall"],
            "rare_type_mae": metrics["rare_type_mae"],
        }
        summaries.append(row)
        all_per_type[method_name] = metrics["per_type"]["pearson_r"]

    summary_df = pd.DataFrame(summaries).set_index("method")

    # Build grouped bar chart
    per_type_df = pd.DataFrame(all_per_type)
    n_types = len(per_type_df)
    n_methods = len(per_type_df.columns)

    fig, ax = plt.subplots(figsize=(max(8, n_types * 0.8), 5))
    x = np.arange(n_types)
    width = 0.8 / max(n_methods, 1)

    for i, method in enumerate(per_type_df.columns):
        offset = (i - n_methods / 2 + 0.5) * width
        vals = per_type_df[method].fillna(0).values
        ax.bar(x + offset, vals, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(per_type_df.index, rotation=45, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-cell-type accuracy by method")
    ax.legend()
    ax.set_ylim(-0.2, 1.1)
    fig.tight_layout()

    return summary_df, fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py::TestImportExternalResults tests/test_benchmarks/test_pseudobulk.py::TestCompareMethods -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_pseudobulk.py spatialgpu/benchmarks/pseudobulk.py
git commit -m "feat: add external results import and multi-method comparison"
```

---

### Task 7: Update benchmarks __init__.py with public exports

**Files:**
- Modify: `spatialgpu/benchmarks/__init__.py`

- [ ] **Step 1: Update exports**

Edit `spatialgpu/benchmarks/__init__.py` to add the new public functions:

```python
"""
Benchmarking utilities for spatial-gpu.

Provides tools to measure and compare performance between CPU and GPU
implementations, and against other libraries like Squidpy.
"""

from spatialgpu.benchmarks.pseudobulk import (
    compare_methods,
    evaluate_deconvolution,
    export_for_cibersortx,
    export_for_music,
    generate_pseudobulk_dirichlet,
    generate_pseudobulk_titration,
    generate_semi_synthetic_scrna,
    import_external_results,
)
from spatialgpu.benchmarks.runner import (
    BenchmarkResult,
    benchmark,
    benchmark_suite,
    compare_backends,
)
from spatialgpu.benchmarks.synthetic import (
    generate_spatial_clusters,
    generate_synthetic_data,
)

__all__ = [
    "benchmark",
    "compare_backends",
    "benchmark_suite",
    "BenchmarkResult",
    "generate_synthetic_data",
    "generate_spatial_clusters",
    # Pseudobulk benchmark
    "generate_semi_synthetic_scrna",
    "generate_pseudobulk_dirichlet",
    "generate_pseudobulk_titration",
    "evaluate_deconvolution",
    "export_for_music",
    "export_for_cibersortx",
    "import_external_results",
    "compare_methods",
]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from spatialgpu.benchmarks import generate_semi_synthetic_scrna, evaluate_deconvolution, compare_methods; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add spatialgpu/benchmarks/__init__.py
git commit -m "feat: export pseudobulk benchmark functions from benchmarks module"
```

---

### Task 8: Tutorial script

**Files:**
- Create: `docs/run_full_tutorial_t8_bulk_benchmark.py`

- [ ] **Step 1: Create the tutorial script**

Create `docs/run_full_tutorial_t8_bulk_benchmark.py`:

```python
"""Run Tutorial 8 — Bulk Deconvolution Pseudobulk Benchmark.

Generates semi-synthetic scRNA-seq, creates pseudobulk with known
proportions, runs deconvolution_bulk, evaluates accuracy, and exports
data for comparison with MuSiC/CIBERSORTx.

Usage: python docs/run_full_tutorial_t8_bulk_benchmark.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def write_output(name, text):
    path = os.path.join(OUTPUTS_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"  Output saved: {path}")


def main():
    import numpy as np
    import pandas as pd

    import spatialgpu.deconvolution as spacet
    from spatialgpu.benchmarks.pseudobulk import (
        _collapse_to_level1,
        _LEVEL1_TYPES,
        evaluate_deconvolution,
        export_for_cibersortx,
        export_for_music,
        generate_pseudobulk_dirichlet,
        generate_pseudobulk_titration,
        generate_semi_synthetic_scrna,
    )

    print("=== Tutorial 8: Bulk Deconvolution Pseudobulk Benchmark ===\n")

    # ---- Step 1: Generate semi-synthetic scRNA-seq ----
    print("1. Generating semi-synthetic scRNA-seq...")
    scrna_brca = generate_semi_synthetic_scrna(
        n_cells_per_type=500, include_malignant=True, cancer_type="BRCA", seed=42
    )
    print(f"   BRCA: {scrna_brca.n_obs} cells, {scrna_brca.n_vars} genes, "
          f"{scrna_brca.obs['cell_type'].nunique()} types")

    scrna_normal = generate_semi_synthetic_scrna(
        n_cells_per_type=500, include_malignant=False, seed=42
    )
    print(f"   Normal: {scrna_normal.n_obs} cells, {scrna_normal.n_vars} genes, "
          f"{scrna_normal.obs['cell_type'].nunique()} types")

    # ---- Step 2: Generate pseudobulk — Dirichlet ----
    print("\n2. Generating pseudobulk (Dirichlet)...")
    bulk_brca, gt_brca = generate_pseudobulk_dirichlet(
        scrna_brca, n_samples=100, n_cells_per_sample=1000, alpha=1.0, seed=42
    )
    print(f"   BRCA: {bulk_brca.n_obs} samples, {bulk_brca.n_vars} genes")

    bulk_normal, gt_normal = generate_pseudobulk_dirichlet(
        scrna_normal, n_samples=100, n_cells_per_sample=1000, alpha=1.0, seed=42
    )
    print(f"   Normal: {bulk_normal.n_obs} samples, {bulk_normal.n_vars} genes")

    # ---- Step 3: Generate pseudobulk — Titration ----
    print("\n3. Generating pseudobulk (Titration)...")
    bulk_titr, gt_titr = generate_pseudobulk_titration(
        scrna_brca,
        target_type="Malignant_BRCA",
        fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        n_replicates=5,
        n_cells_per_sample=1000,
        seed=42,
    )
    print(f"   Titration: {bulk_titr.n_obs} samples")

    # ---- Step 4: Deconvolution — BRCA ----
    print("\n4. Deconvolution — BRCA (100 samples)...")
    spacet.deconvolution_bulk(bulk_brca, cancer_type="BRCA")
    pm_brca = bulk_brca.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_brca.shape[0]} types x {pm_brca.shape[1]} samples")

    # ---- Step 5: Deconvolution — Normal ----
    print("\n5. Deconvolution — Normal (100 samples)...")
    spacet.deconvolution_bulk(bulk_normal, cancer_type="normal")
    pm_normal = bulk_normal.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_normal.shape[0]} types x {pm_normal.shape[1]} samples")

    # ---- Step 6: Deconvolution — Titration ----
    print("\n6. Deconvolution — Titration (45 samples)...")
    spacet.deconvolution_bulk(bulk_titr, cancer_type="BRCA")
    pm_titr = bulk_titr.uns["deconv"]["propMat"]
    print(f"   propMat: {pm_titr.shape[0]} types x {pm_titr.shape[1]} samples")

    # ---- Step 7: Evaluate accuracy ----
    print("\n7. Evaluating accuracy...")

    # Collapse propMat to Level 1 and transpose to samples x types
    est_brca = _collapse_to_level1(pm_brca, _LEVEL1_TYPES + ["Malignant"]).T
    # Map "Malignant" -> "Malignant_BRCA" to match ground truth
    est_brca = est_brca.rename(columns={"Malignant": "Malignant_BRCA"})

    est_normal = _collapse_to_level1(pm_normal, _LEVEL1_TYPES).T

    est_titr = _collapse_to_level1(pm_titr, _LEVEL1_TYPES + ["Malignant"]).T
    est_titr = est_titr.rename(columns={"Malignant": "Malignant_BRCA"})

    # Drop target_fraction from ground truth before evaluation
    gt_titr_eval = gt_titr.drop(columns=["target_fraction"], errors="ignore")

    metrics_brca = evaluate_deconvolution(est_brca, gt_brca)
    metrics_normal = evaluate_deconvolution(est_normal, gt_normal)
    metrics_titr = evaluate_deconvolution(est_titr, gt_titr_eval)

    for name, m in [("BRCA", metrics_brca), ("Normal", metrics_normal), ("Titration", metrics_titr)]:
        print(f"\n   {name}:")
        print(f"     Pearson r:    {m['overall']['pearson_r']:.4f}")
        print(f"     Spearman rho: {m['overall']['spearman_rho']:.4f}")
        print(f"     RMSE:         {m['overall']['rmse']:.4f}")
        print(f"     Rare MAE:     {m['rare_type_mae']:.4f}")

    # Write metrics tables
    for name, m in [("brca", metrics_brca), ("normal", metrics_normal), ("titration", metrics_titr)]:
        lines = [
            f"Overall Pearson r:    {m['overall']['pearson_r']:.4f}",
            f"Overall Spearman rho: {m['overall']['spearman_rho']:.4f}",
            f"Overall RMSE:         {m['overall']['rmse']:.4f}",
            f"Rare type MAE:        {m['rare_type_mae']:.4f}",
            "",
            "Per-cell-type:",
            m["per_type"].to_string(),
        ]
        write_output(f"t8_metrics_{name}.txt", "\n".join(lines))

    # ---- Step 8: Generate figures ----
    print("\n8. Generating figures...")

    # 8a. Scatter: BRCA estimated vs true
    fig, ax = plt.subplots(figsize=(7, 7))
    common_types = est_brca.columns.intersection(gt_brca.columns)
    colors = plt.cm.tab20(np.linspace(0, 1, len(common_types)))
    for i, ct in enumerate(common_types):
        ax.scatter(gt_brca[ct], est_brca.reindex(gt_brca.index)[ct],
                   s=15, alpha=0.6, label=ct, color=colors[i])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Estimated proportion")
    ax.set_title(f"BRCA (r={metrics_brca['overall']['pearson_r']:.3f})")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "benchmark_scatter_brca.png")

    # 8b. Scatter: Normal estimated vs true
    fig, ax = plt.subplots(figsize=(7, 7))
    common_types_n = est_normal.columns.intersection(gt_normal.columns)
    colors_n = plt.cm.tab20(np.linspace(0, 1, len(common_types_n)))
    for i, ct in enumerate(common_types_n):
        ax.scatter(gt_normal[ct], est_normal.reindex(gt_normal.index)[ct],
                   s=15, alpha=0.6, label=ct, color=colors_n[i])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Estimated proportion")
    ax.set_title(f"Normal tissue (r={metrics_normal['overall']['pearson_r']:.3f})")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    save(fig, "benchmark_scatter_normal.png")

    # 8c. Per-cell-type Pearson r bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    brca_pt = metrics_brca["per_type"]["pearson_r"].rename("BRCA")
    normal_pt = metrics_normal["per_type"]["pearson_r"].rename("Normal")
    combined = pd.DataFrame({"BRCA": brca_pt, "Normal": normal_pt}).fillna(0)
    combined.plot(kind="barh", ax=ax)
    ax.set_xlabel("Pearson r")
    ax.set_title("Per-cell-type accuracy")
    ax.axvline(x=0, color="k", linewidth=0.5)
    fig.tight_layout()
    save(fig, "benchmark_per_type_r.png")

    # 8d. Titration: accuracy vs malignant fraction
    titr_fracs = gt_titr["target_fraction"].unique()
    titr_fracs.sort()
    titr_r_values = []
    for frac in titr_fracs:
        mask = gt_titr["target_fraction"] == frac
        sub_est = est_titr.loc[mask]
        sub_gt = gt_titr_eval.loc[mask]
        common = sub_est.columns.intersection(sub_gt.columns)
        e = sub_est[common].values.ravel()
        g = sub_gt[common].values.ravel()
        from scipy.stats import pearsonr
        r, _ = pearsonr(e, g)
        titr_r_values.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(titr_fracs, titr_r_values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Malignant fraction")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("Deconvolution accuracy vs tumor purity")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "benchmark_titration.png")

    # 8e. Rare cell type error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    rare_errors = {}
    common_brca = est_brca.columns.intersection(gt_brca.columns)
    for ct in common_brca:
        mask = gt_brca[ct] < 0.05
        if mask.sum() > 2:
            errors = np.abs(est_brca.reindex(gt_brca.index).loc[mask, ct] - gt_brca.loc[mask, ct])
            rare_errors[ct] = errors.values
    if rare_errors:
        ax.boxplot(rare_errors.values(), labels=rare_errors.keys(), vert=True)
        ax.set_ylabel("Absolute error")
        ax.set_title("Error distribution for rare cell types (true < 5%)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No rare cell type entries", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    save(fig, "benchmark_rare_types.png")

    # ---- Step 9: Export for external tools ----
    print("\n9. Exporting for external tools...")
    music_dir = os.path.join(OUTPUTS_DIR, "t8_export_music")
    cibersortx_dir = os.path.join(OUTPUTS_DIR, "t8_export_cibersortx")

    export_for_music(bulk_brca, scrna_brca, music_dir, gt_brca)
    print(f"   MuSiC export: {music_dir}")

    export_for_cibersortx(bulk_brca, scrna_brca, cibersortx_dir, gt_brca)
    print(f"   CIBERSORTx export: {cibersortx_dir}")

    # ---- Step 10: Session info ----
    print("\n10. Session info...")
    import spatialgpu
    import anndata
    import numpy
    import scipy

    session_lines = [
        f"spatial-gpu version: {spatialgpu.__version__}",
        f"anndata: {anndata.__version__}",
        f"numpy: {numpy.__version__}",
        f"scipy: {scipy.__version__}",
        f"pandas: {pd.__version__}",
        f"matplotlib: {matplotlib.__version__}",
    ]

    try:
        from spatialgpu.core.backend import get_backend
        backend = get_backend()
        session_lines.append(f"GPU available: {backend.is_gpu_available}")
        session_lines.append(f"GPU active: {backend.is_gpu_active}")
    except Exception:
        session_lines.append("GPU: unknown")

    write_output("t8_session_info.txt", "\n".join(session_lines))
    for line in session_lines:
        print(f"   {line}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint check**

Run: `python -m ruff check docs/run_full_tutorial_t8_bulk_benchmark.py`
Expected: No errors (fix any import sorting issues if flagged)

- [ ] **Step 3: Commit**

```bash
git add docs/run_full_tutorial_t8_bulk_benchmark.py
git commit -m "feat: add Tutorial T8 bulk deconvolution benchmark script"
```

---

### Task 9: SLURM submission script

**Files:**
- Create: `scripts/slurm_tutorial_t8_gpu.sh`

- [ ] **Step 1: Create SLURM script**

Create `scripts/slurm_tutorial_t8_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=sgpu_t8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/tutorial_t8_%j.err

set -euo pipefail

# Environment setup
source ~/bin/myconda
conda activate secactpy
module load CUDA/12 cuDNN

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== Tutorial T8: Bulk Deconvolution Benchmark ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo ""

pip install -e . --quiet 2>/dev/null || true

python docs/run_full_tutorial_t8_bulk_benchmark.py

echo ""
echo "End: $(date)"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/slurm_tutorial_t8_gpu.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/slurm_tutorial_t8_gpu.sh
git commit -m "feat: add SLURM script for Tutorial T8 bulk benchmark"
```

---

### Task 10: Run all unit tests and lint

**Files:** None (validation only)

- [ ] **Step 1: Lint all new/modified files**

Run: `python -m ruff check spatialgpu/benchmarks/pseudobulk.py spatialgpu/benchmarks/__init__.py tests/test_benchmarks/test_pseudobulk.py docs/run_full_tutorial_t8_bulk_benchmark.py`
Expected: No errors

- [ ] **Step 2: Run all pseudobulk unit tests (CPU-safe, login node OK)**

Run: `python -m pytest tests/test_benchmarks/test_pseudobulk.py -v --tb=short`
Expected: All tests PASS (these are lightweight — synthetic data with small n_cells_per_type)

- [ ] **Step 3: Submit SLURM job for full tutorial**

Run: `sbatch scripts/slurm_tutorial_t8_gpu.sh`
Expected: Job ID printed. Check results later with `cat validation_results/tutorial_t8_<jobid>.out`

- [ ] **Step 4: Fix any lint or test failures and re-commit**

If needed, fix and commit:
```bash
git add -u
git commit -m "fix: lint and test fixes for pseudobulk benchmark"
```
