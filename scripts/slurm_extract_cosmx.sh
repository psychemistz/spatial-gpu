#!/bin/bash
#SBATCH --job-name=sgpu_excos
#SBATCH --partition=norm
#SBATCH --mem=32g
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/extract_cosmx_%j.out
#SBATCH --error=/data/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu/validation_results/extract_cosmx_%j.err

set -euo pipefail

cd /vf/users/parks34/projects/0sigdiscov/pkg_dev/spatial-gpu

echo "=== LIHC CosMx Data Extraction ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

# Step 1: Copy rda to local data dir
mkdir -p data/LIHC_CosMx
cp /data/Jiang_Lab/datashare/SecAct_Package/LIHC_CosMx_data.rda data/

# Step 2: Extract with R
module load R/4.3
Rscript scripts/extract_cosmx_data.R

# Step 3: Convert to h5ad
source ~/bin/myconda
conda activate secactpy
pip install -e . --quiet 2>/dev/null || true

python -c "
import os, sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmread
from scipy.sparse import csc_matrix
from pathlib import Path

d = Path('data/LIHC_CosMx')
print('=== Converting LIHC_CosMx to h5ad ===')

# Counts
counts = mmread(str(d / 'counts.mtx'))
counts = csc_matrix(counts)
genes = pd.read_csv(d / 'genes.csv')['gene'].values
cells = pd.read_csv(d / 'cells.csv')['cell'].values
coords = pd.read_csv(d / 'coordinates.csv', index_col=0)
print(f'  Counts: {counts.shape[0]} genes x {counts.shape[1]} cells')
print(f'  Coordinates: {coords.shape[0]} cells')

# Metadata
meta = pd.read_csv(d / 'metadata.csv', index_col=0) if (d / 'metadata.csv').exists() else None

# Build AnnData (cells x genes)
obs_df = pd.DataFrame(index=pd.Index(cells, name='cell'))
obs_df['coordinate_x_um'] = coords.iloc[:, 0].values
obs_df['coordinate_y_um'] = coords.iloc[:, 1].values

if meta is not None:
    for col in meta.columns:
        obs_df[col] = meta[col].values
    print(f'  Metadata columns: {list(meta.columns)}')

adata = ad.AnnData(
    X=counts.T.tocsr(),
    obs=obs_df,
    var=pd.DataFrame(index=pd.Index(genes, name='gene')),
)
adata.obsm['spatial'] = np.column_stack([obs_df['coordinate_x_um'].values, obs_df['coordinate_y_um'].values])
adata.uns['spacet_platform'] = 'CosMx'
adata.uns['spacet'] = {}

out = d / 'LIHC_CosMx.h5ad'
adata.write_h5ad(out)
print(f'  Saved {out}')
print(f'  Final: {adata.shape[0]} cells x {adata.shape[1]} genes')
"

echo ""
echo "--- Verify ---"
ls -lh data/LIHC_CosMx/LIHC_CosMx.h5ad

echo ""
echo "End: $(date)"
echo "=== CosMx extraction complete ==="
