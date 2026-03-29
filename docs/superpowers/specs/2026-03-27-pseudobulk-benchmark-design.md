# Pseudobulk Benchmark for Bulk Deconvolution (Tutorial T8)

**Date:** 2026-03-27
**Status:** Draft

## Problem

`deconvolution_bulk` is implemented and tested for correctness (CPU/GPU equivalence, output shape/storage), but there is no accuracy evaluation against known ground truth, no comparison framework for external tools (MuSiC, CIBERSORTx), and no user-facing documentation for the bulk pathway.

## Goals

1. Evaluate `deconvolution_bulk` accuracy on pseudobulk with known cell type proportions.
2. Cover both tumor (BRCA) and normal tissue scenarios.
3. Provide export helpers so users can run the same pseudobulk through MuSiC/CIBERSORTx and compare.
4. Ship as Tutorial T8, following existing T1-T7 conventions.

## Non-goals

- Running MuSiC/CIBERSORTx automatically (they require R / web upload).
- Benchmarking runtime performance (covered by existing `benchmark_comparison.py`).
- Supporting every cancer type — BRCA is the worked example; the functions generalize.

---

## Architecture

### Deliverables

| File | Purpose |
|---|---|
| `spatialgpu/benchmarks/pseudobulk.py` | Core module: pseudobulk generation, semi-synthetic scRNA-seq, evaluation metrics, export/import helpers |
| `docs/run_full_tutorial_t8_bulk_benchmark.py` | Tutorial script: runs BRCA + normal benchmarks, generates figures and output tables |
| `scripts/slurm_tutorial_t8_gpu.sh` | SLURM submission script |

### Data flow

```
load_comb_ref() + get_cancer_signature("BRCA")
        |
        v
generate_semi_synthetic_scrna()          # NB noise + dropout per cell
        |
        v
generate_pseudobulk_dirichlet()          # 100 samples, known proportions
generate_pseudobulk_titration()          # malignant 0-80%, 5 replicates each
        |
        v
deconvolution_bulk(adata, ...)           # our pipeline
        |
        v
evaluate_deconvolution()                 # 5 metrics vs ground truth
        |
        v
export_for_music() / export_for_cibersortx()    # comparison-ready files
import_external_results()                         # load other tools' output
compare_methods()                                 # side-by-side metrics table
```

---

## Module: `spatialgpu/benchmarks/pseudobulk.py`

### `generate_semi_synthetic_scrna`

```python
def generate_semi_synthetic_scrna(
    n_cells_per_type: int = 500,
    include_malignant: bool = True,
    cancer_type: str = "BRCA",
    seed: int = 42,
) -> ad.AnnData:
```

**Algorithm:**

1. Load reference profiles from `load_comb_ref()` (genes x cell types, mean CPM).
2. If `include_malignant`, add a "Malignant_{cancer_type}" column synthesized from `get_cancer_signature(cancer_type)` overlaid on the mean expression profile.
3. For each cell type, generate `n_cells_per_type` cells:
   - Convert CPM profile to relative probabilities per gene.
   - Draw total UMI per cell from `LogNormal(mean=8.5, sd=0.5)` (~2K-10K UMI, mimics real scRNA-seq library sizes).
   - Draw gene counts from `NegativeBinomial(mu=prob * total_umi, size=dispersion)` where `dispersion = max(0.5, mean_expr / 2)` (gene-dependent, higher dispersion for lowly expressed genes).
   - Apply dropout: zero out gene `g` with probability `1 / (1 + exp(a * log_expr_g + b))` where `a=1.5, b=-2` (logistic dropout model, matching empirical scRNA-seq dropout curves).
4. Return AnnData with integer counts, `obs["cell_type"]` labels.

**Cell types:** All non-malignant types from the reference lineage tree (both Level 1 major lineages and Level 2 subtypes are available in the reference; we use Level 1 for pseudobulk mixing since `deconvolution_bulk` reports hierarchical results). Plus `Malignant_BRCA` when `include_malignant=True`.

### `generate_pseudobulk_dirichlet`

```python
def generate_pseudobulk_dirichlet(
    scrna_adata: ad.AnnData,
    n_samples: int = 100,
    n_cells_per_sample: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
```

**Algorithm:**

1. Sample proportions from `Dirichlet(alpha)` for each of `n_samples` pseudobulk samples. `alpha=1.0` gives uniform-ish mixtures; lower alpha gives sparser (more realistic, dominated by fewer types).
2. For each sample:
   - Draw cell counts per type from `Multinomial(n_cells_per_sample, proportions)`.
   - Randomly sample that many cells (with replacement) from `scrna_adata` for each type.
   - Sum counts across sampled cells to produce one bulk expression vector.
3. Return:
   - `adata_bulk`: AnnData (samples x genes, raw integer counts).
   - `ground_truth`: DataFrame (samples x cell_types, true proportions summing to 1.0).

### `generate_pseudobulk_titration`

```python
def generate_pseudobulk_titration(
    scrna_adata: ad.AnnData,
    target_type: str = "Malignant_BRCA",
    fractions: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    n_replicates: int = 5,
    n_cells_per_sample: int = 1000,
    seed: int = 42,
) -> tuple[ad.AnnData, pd.DataFrame]:
```

**Algorithm:**

1. For each `frac` in `fractions`, for each replicate:
   - Fix target type proportion at `frac`.
   - Sample remaining `(1 - frac)` across other types from `Dirichlet(1.0)`.
   - Sample and sum cells as in Dirichlet generation.
2. Return: AnnData + ground truth DataFrame with additional `target_fraction` column.

### `evaluate_deconvolution`

```python
def evaluate_deconvolution(
    estimated: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
```

**Metrics:**

| Metric | Scope | Computation |
|---|---|---|
| Pearson r (overall) | All (type, sample) pairs flattened | `scipy.stats.pearsonr` |
| Spearman rho (overall) | Same | `scipy.stats.spearmanr` |
| RMSE (overall) | Same | `sqrt(mean((est - true)^2))` |
| Per-cell-type Pearson r | One r per cell type across samples | `pearsonr` per row |
| MAE at low fractions | Entries where `truth < 0.05` | `mean(abs(est - true))` |

**Returns:** Dict with keys:
- `overall`: `{"pearson_r", "spearman_rho", "rmse"}`
- `per_type`: DataFrame indexed by cell type with columns `pearson_r`, `rmse`, `n_samples`
- `rare_type_mae`: float

Aligns on common cell types and samples before computing. Since pseudobulk is mixed at Level 1 granularity but `deconvolution_bulk` returns both Level 1 and Level 2 types, evaluation extracts Level 1 rows from the estimated propMat (summing Level 2 subtypes back into their parent lineage where needed). Warns if alignment drops >20% of types.

### `export_for_music`

```python
def export_for_music(
    adata_bulk: ad.AnnData,
    scrna_adata: ad.AnnData,
    output_dir: str,
) -> None:
```

**Outputs:**
- `bulk_counts.csv` — genes x samples (raw counts), MuSiC ExpressionSet-compatible.
- `sc_counts.csv` — genes x cells (raw counts).
- `sc_phenodata.csv` — cell metadata with `cell_type` column.
- `ground_truth.csv` — true proportions (samples x cell_types).
- `run_music.R` — ready-to-run R script:
  ```r
  library(MuSiC)
  library(Biobase)
  # Load bulk and sc data
  # Build ExpressionSets
  # Run MuSiC::music_prop()
  # Save results to music_results.csv
  ```

### `export_for_cibersortx`

```python
def export_for_cibersortx(
    adata_bulk: ad.AnnData,
    scrna_adata: ad.AnnData,
    output_dir: str,
) -> None:
```

**Outputs:**
- `mixture.txt` — tab-delimited, genes x samples (TPM-normalized, as CIBERSORTx expects).
- `sc_reference.txt` — tab-delimited, genes x cells with cell type header row.
- `ground_truth.csv` — true proportions.
- `README_cibersortx.txt` — step-by-step web upload instructions.

### `import_external_results`

```python
def import_external_results(
    results_path: str,
    tool_name: str,
) -> pd.DataFrame:
```

Reads CSV/TSV output from MuSiC (`music_results.csv`) or CIBERSORTx (downloaded results file). Returns standardized DataFrame (samples x cell_types) with cell type names mapped to our naming convention.

### `compare_methods`

```python
def compare_methods(
    results_dict: dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
```

- `results_dict`: e.g., `{"spatial-gpu": df, "MuSiC": df, "CIBERSORTx": df}`
- Runs `evaluate_deconvolution` on each method.
- Returns:
  - Summary DataFrame: methods x metrics.
  - Figure: grouped bar chart of per-cell-type Pearson r, one color per method.

---

## Tutorial: `docs/run_full_tutorial_t8_bulk_benchmark.py`

Follows T1-T7 conventions: headless matplotlib, numbered steps, `save()`/`write_output()` helpers, session info.

### Steps

```
Step 1.  Generate semi-synthetic scRNA-seq
         - BRCA (with malignant): ~6K cells (12 types x 500 cells)
         - Normal (no malignant): ~5.5K cells (11 types x 500 cells)

Step 2.  Generate pseudobulk — Dirichlet (100 samples each)
         - BRCA: 100 samples, alpha=1.0
         - Normal: 100 samples, alpha=1.0

Step 3.  Generate pseudobulk — Malignant titration
         - 9 fractions x 5 replicates = 45 samples

Step 4.  Deconvolution — BRCA Dirichlet samples
         - deconvolution_bulk(adata, cancer_type="BRCA")

Step 5.  Deconvolution — Normal Dirichlet samples
         - deconvolution_bulk(adata, cancer_type="normal")

Step 6.  Deconvolution — Titration series
         - deconvolution_bulk(adata, cancer_type="BRCA")

Step 7.  Evaluate accuracy (5 metrics x 3 scenarios)

Step 8.  Generate figures
         - benchmark_scatter_brca.png
         - benchmark_scatter_normal.png
         - benchmark_per_type_r.png
         - benchmark_titration.png
         - benchmark_rare_types.png

Step 9.  Export for external tools
         - docs/outputs/t8_export_music/
         - docs/outputs/t8_export_cibersortx/

Step 10. Session info
```

### Figures

| Figure | Description |
|---|---|
| `benchmark_scatter_brca.png` | Estimated vs true proportions scatter, colored by cell type. BRCA scenario. |
| `benchmark_scatter_normal.png` | Same for normal tissue. |
| `benchmark_per_type_r.png` | Horizontal bar chart: Pearson r per cell type, grouped by scenario (BRCA/normal). |
| `benchmark_titration.png` | Line plot: overall Pearson r (y) vs malignant fraction (x). Shows accuracy degradation with increasing tumor purity. |
| `benchmark_rare_types.png` | Box plot: absolute error distribution for cell types with true proportion < 5%. |

### Text outputs

| File | Content |
|---|---|
| `t8_metrics_brca.txt` | Full metrics table for BRCA scenario |
| `t8_metrics_normal.txt` | Full metrics table for normal scenario |
| `t8_metrics_titration.txt` | Per-fraction metrics for titration |
| `t8_session_info.txt` | Package versions |

---

## SLURM script: `scripts/slurm_tutorial_t8_gpu.sh`

Standard pattern matching existing scripts:
- Partition: gpu, 1x A100, 32GB, 4 CPUs, 1 hour
- conda activate secactpy, module load CUDA/12 cuDNN
- pip install -e . --quiet
- python docs/run_full_tutorial_t8_bulk_benchmark.py

---

## Testing

The core functions in `pseudobulk.py` should be testable without GPU or heavy compute. The existing `test_bulk.py` (submitted as job 14995127) covers `deconvolution_bulk` correctness. For the benchmark module:

- `test_pseudobulk.py` — unit tests for generation functions (shape, ground truth sums to 1, counts are integers), evaluation metrics (perfect input gives r=1.0, RMSE=0), export file format validation.
- Integration tested via the tutorial script itself (run on SLURM).

---

## What this does NOT do

- Run MuSiC/CIBERSORTx automatically — they require R installation or web access.
- Benchmark runtime — that's `benchmark_comparison.py`.
- Support all cancer types — BRCA is the worked example; functions accept any type.
- Replace proper validation with real matched bulk + scRNA-seq data — this is semi-synthetic. Real validation requires paired datasets.
