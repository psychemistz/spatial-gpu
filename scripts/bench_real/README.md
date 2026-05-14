# Real-bulk deconvolution benchmark (internal, unlinked from public docs)

Targets the **Tumor Deconvolution DREAM Challenge** (White et al., Nat Commun 2024)
validation set to score `spatial-gpu`'s SpaCET-NNLS (`v0_none`) and SpaCET-IRWLS
(`v3_irwls`) bulk-deconvolution variants against the published leaderboard.

GSE316545 (Hu et al. 2026 BAL paired benchmark) is embargoed until 2027-06-01 —
revisit then.

## Why this benchmark replaces the old T8

The retired T8 used pseudo-bulk on BRCA scRNA, which Hu et al. 2026 demonstrate
inherits 3′ UMI capture and dissociation bias on the very signature genes
SpaCET's NNLS weights most. Pseudo-bulk numbers overstate any method's lead.
This benchmark uses **real bulk** (in-vitro FACS-sorted admixtures with known
mixing ratios) — the cleanest available ground truth.

## One-time setup

1. Sign up for a free Synapse account at https://www.synapse.org.
2. Generate a Personal Access Token (PAT) at https://www.synapse.org/settings/personal-access-tokens.
3. Export it before submitting fetch jobs:
   ```bash
   export SYNAPSE_AUTH_TOKEN=<your-PAT>
   ```
   Or write it to `~/.synapseConfig`:
   ```ini
   [authentication]
   authtoken=<your-PAT>
   ```
4. Install the bench extra into the `secactpy` conda env (the working env
   for this repo — verified to have `spatialgpu` installed editably):
   ```bash
   source ~/bin/myconda && conda activate secactpy
   pip install -e '.[bench]'
   ```

## Workflow

```bash
# 1) Fetch validation set + ground truth (~2 hr, light I/O)
sbatch scripts/bench_real/slurm_fetch_dream.sh

# 2) Build a SpaCET-compatible reference from DREAM's training pure profiles.
#    NOTE: this needs the per-dataset metadata (sample -> cell_type, donor) which
#    DREAM provides as part of the training-data bundle. Adjust the --meta path
#    once fetch completes — see scripts/bench_real/fetch_dream.py keys.
python scripts/bench_real/build_dream_reference.py \
    --pure /vf/users/parks34/projects/0sigdiscov/data/dream/pure_profiles \
    --meta /vf/users/parks34/projects/0sigdiscov/data/dream/sample_map/training_metadata.tsv \
    --out /vf/users/parks34/projects/0sigdiscov/data/dream/reference \
    --resolution coarse

# 3) Stress-test ONE config end-to-end before submitting the array (HPC rule):
sbatch --array=0-0 scripts/bench_real/slurm_run_spacet.sh
# After it completes, inspect MaxRSS / Elapsed via `seff <jobid>`. Tune the
# template's --mem and --time, then submit the full array.

sbatch scripts/bench_real/slurm_run_spacet.sh

# 4) Score + plot (cheap, runs in minutes)
sbatch scripts/bench_real/slurm_score.sh
```

## What gets generated

```
/vf/users/parks34/projects/0sigdiscov/
├── data/dream/            # raw + reference (large, scratch)
└── bench_outputs/dream/
    ├── <dataset>_v0_none.csv          # predicted fractions, NNLS
    ├── <dataset>_v3_irwls.csv         # predicted fractions, IRWLS
    └── scores/
        ├── scores_coarse.csv          # per (method, dataset, cell type) Pearson r
        ├── aggregate_coarse.csv       # bootstrap mean + 95% CI
        ├── within_sample_*.csv        # auxiliary per-sample r/rho/RMSE
        └── leaderboard_coarse.png
```

## Tutorial render

The user-facing write-up lives at `docs/t8_real_bulk_benchmark.md` — intentionally
**not** linked from `docs/index.html` or `README.md` (internal benchmark, not
shipped as a tutorial). Re-render by hand when results land.

## Caveats baked into this benchmark

- SpaCET was designed for tumor microenvironments. DREAM's in-vitro admixtures
  spike in cancer cell-line transcripts as contamination — we model this with
  `--no-malignant` so the cancer signal is not scored but is also not explicitly
  fit. A future variant could include a `Malignant` reference and score only
  the immune compartment.
- DREAM evaluated 23 methods with a single submission window in 2019. Adding
  SpaCET retroactively to the leaderboard requires running it on the same
  validation set with their scoring code; we replicate the score function in
  `_common.py` but do not call their Synapse scoring container.
