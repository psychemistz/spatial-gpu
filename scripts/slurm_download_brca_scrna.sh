#!/bin/bash
#SBATCH --job-name=dl_brca
#SBATCH --partition=norm
#SBATCH --mem=64g
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/download_brca_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/download_brca_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

source ~/bin/myconda
conda activate secactpy

echo "=== Download and preprocess Wu et al. 2021 BRCA scRNA-seq (GSE176078) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

DATA_DIR="data/BRCA_scRNA"
mkdir -p "$DATA_DIR"

# ---- Step 1: Download from GEO ----
echo ""
echo "1. Downloading from GEO (GSE176078)..."
GEO_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz"

if [ ! -f "$DATA_DIR/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz" ]; then
    wget -q --show-progress -O "$DATA_DIR/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz" "$GEO_URL"
    echo "   Downloaded: $(du -sh $DATA_DIR/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz | cut -f1)"
else
    echo "   Already downloaded, skipping."
fi

# ---- Step 2: Extract ----
echo ""
echo "2. Extracting..."
if [ ! -f "$DATA_DIR/count_matrix_sparse.mtx" ]; then
    tar -xzf "$DATA_DIR/GSE176078_Wu_etal_2021_BRCA_scRNASeq.tar.gz" -C "$DATA_DIR/"
    # Flatten if extracted into subdirectory
    if [ -d "$DATA_DIR/Wu_etal_2021_BRCA_scRNASeq" ]; then
        mv "$DATA_DIR/Wu_etal_2021_BRCA_scRNASeq/"* "$DATA_DIR/"
        rmdir "$DATA_DIR/Wu_etal_2021_BRCA_scRNASeq"
    fi
    echo "   Extracted files:"
    ls -lh "$DATA_DIR/"
else
    echo "   Already extracted, skipping."
fi

# ---- Step 3: Convert to h5ad and subsample ----
echo ""
echo "3. Converting to h5ad and subsampling..."

python -c "
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy import sparse

data_dir = '$DATA_DIR'

# Read sparse matrix
print('   Loading count matrix...')
X = mmread(f'{data_dir}/count_matrix_sparse.mtx').T.tocsr()  # cells x genes
print(f'   Raw shape: {X.shape}')

# Read barcodes and genes
barcodes = pd.read_csv(f'{data_dir}/count_matrix_barcodes.tsv', header=None)[0].values
genes = pd.read_csv(f'{data_dir}/count_matrix_genes.tsv', header=None)[0].values

# Read metadata
meta = pd.read_csv(f'{data_dir}/metadata.csv', index_col=0)
print(f'   Metadata: {meta.shape[0]} cells, columns: {list(meta.columns)}')

# Build AnnData
adata = sc.AnnData(X=X, obs=meta.loc[barcodes], var=pd.DataFrame(index=genes))
adata.obs_names = barcodes
print(f'   AnnData: {adata.n_obs} cells x {adata.n_vars} genes')

# Check cell type column
ct_col = None
for col in ['celltype_major', 'cell_type', 'celltype']:
    if col in adata.obs.columns:
        ct_col = col
        break
if ct_col is None:
    print(f'   WARNING: No cell type column found. Available: {list(adata.obs.columns)}')
    # Save full adata for inspection
    adata.write(f'{data_dir}/BRCA_scRNA_full.h5ad')
    print(f'   Saved full h5ad for inspection.')
    exit(0)

print(f'   Cell type column: {ct_col}')
print(f'   Cell types:')
print(adata.obs[ct_col].value_counts().to_string())

# Subsample to 500 cells per type for benchmark
print()
print('   Subsampling 500 cells per type...')
rng = np.random.RandomState(42)
sub_idx = []
for ct in adata.obs[ct_col].unique():
    ct_idx = np.where(adata.obs[ct_col] == ct)[0]
    n_sample = min(500, len(ct_idx))
    chosen = rng.choice(ct_idx, n_sample, replace=False)
    sub_idx.extend(chosen)
    print(f'     {ct}: {len(ct_idx)} -> {n_sample}')

adata_sub = adata[sorted(sub_idx)].copy()
# Standardize column name
adata_sub.obs['cell_type'] = adata_sub.obs[ct_col].values
print(f'   Subsampled: {adata_sub.n_obs} cells x {adata_sub.n_vars} genes')

# Save
adata_sub.write(f'{data_dir}/BRCA_scRNA_sub500.h5ad')
print(f'   Saved: {data_dir}/BRCA_scRNA_sub500.h5ad')

# Also save full for reference
adata.write(f'{data_dir}/BRCA_scRNA_full.h5ad')
print(f'   Saved: {data_dir}/BRCA_scRNA_full.h5ad')
" 2>&1

echo ""
echo "Done: $(date)"
