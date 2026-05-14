# T8 — Real-bulk deconvolution benchmark (internal)

> **Status: scaffold.** This page is intentionally **not linked** from the
> public tutorial index or README. Internal benchmark only.

## Why this benchmark exists

The earlier T8 used **pseudo-bulk** generated from a BRCA scRNA-seq atlas. Hu
et al. 2026 (`2026.01.14.699304v1`) demonstrate that pseudo-bulk inherits 3′ UMI
capture and dissociation bias on the signature genes that NNLS-based methods
weight most heavily — so the published T8 numbers overstated any method's
lead. This rebuild swaps in **real bulk RNA-seq** from the Tumor Deconvolution
DREAM Challenge (White et al., Nat Commun 2024) so the comparison is honest.

## Methods compared

| Method | `spatial-gpu` API | Description |
|---|---|---|
| **SpaCET-NNLS** | `deconvolution_bulk(..., cross_subject_weighting=False)` | Plain hierarchical constrained NNLS (the `v0_none` weighting variant) |
| **SpaCET-IRWLS** | `deconvolution_bulk(..., cross_subject_weighting=True, subject_col=...)` | The `v3_irwls` weighting variant with per-subject reference reweighting |

Both run on the same reference and the same validation bulks — single lever
swapped (weighting on/off). Optionally compare against DREAM's published top
submissions by lifting their `predictions.csv` from Synapse (see
`scripts/bench_real/README.md`).

## Datasets

- **DREAM in-vitro validation** — 96 admixtures, 4 datasets (DS1–DS4) of
  FACS-sorted PBMC populations spiked with BRCA/CRC cell-line transcripts.
  Ground truth = known mixing ratios. Coarse: 8 cell types. Fine: 14.
- *(Future, blocked)* — Hu et al. 2026 BAL (GSE316545, embargoed to 2027-06-01).

## Metrics

Primary: cross-sample within-cell-type Pearson r, mean across cell types,
bootstrapped n=1000 (DREAM's published scoring rule). Auxiliary:
cross-cell-type within-sample r/rho/RMSE, spillover matrix on purified samples,
limit of detection from in-silico spike-ins.

## How to run

See `scripts/bench_real/README.md`. One-time setup is a Synapse PAT.

## Results

*(Populated after the SLURM run completes — placeholder.)*

| Method | DS1 r [95% CI] | DS2 r | DS3 r | DS4 r | Aggregate |
|---|---|---|---|---|---|
| SpaCET-NNLS  | — | — | — | — | — |
| SpaCET-IRWLS | — | — | — | — | — |

![leaderboard](outputs/bench_real/leaderboard_coarse.png)

## Honest caveats

- SpaCET was built for tumor microenvironments; DREAM bulks are immune-focused
  with cancer cell-line spike-ins as contamination. We set `--no-malignant` to
  exclude the cancer compartment from scoring (matches DREAM evaluation).
- DREAM's published leaderboard fixes a 2019 submission window. Adding SpaCET
  retroactively without re-running through their Synapse scoring container
  means we can only compare *aggregate* scores, not per-attempt ranking with
  Bayes-factor tiebreaks. Our `_common.py` re-implements the published metric.
- This benchmark does **not** support claims about spatial-transcriptomics
  performance. It tests SpaCET's bulk path only.

## References

- White et al., *Nat Commun* 2024 — DREAM Challenge.
  https://www.nature.com/articles/s41467-024-50618-0
- Hu et al., bioRxiv 2026.01.14.699304 — paired bulk/scRNA benchmark motivation.
- Sturm et al., *Bioinformatics* 2019 — `immunedeconv` benchmark.
