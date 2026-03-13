#!/usr/bin/env python
"""Compare Python intermediate deconvolution values with R's saved intermediates."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import sparse

INT_DIR = "validation/intermediates"
VST_DIR = "/Users/seongyongpark/project/psychemist/sigdiscov/dataset/visium"

# ---- Load VST1 dataset (same as compare_r_vs_python.py) ----
import anndata as ad

counts_file = f"{VST_DIR}/1_counts.tsv"
counts = pd.read_csv(counts_file, sep="\t", index_col=0)
gene_names = np.array(counts.index)
spot_names = np.array(counts.columns)

parts = [s.split("x") for s in spot_names]
array_row = np.array([float(p[0]) for p in parts])
array_col = np.array([float(p[1]) for p in parts])

coord_x_um = array_col * 0.5 * 100.0
coord_y_um = array_row * 0.5 * np.sqrt(3) * 100.0
coord_y_um = coord_y_um.max() - coord_y_um

counts_sparse = sparse.csc_matrix(counts.values.astype(np.float64))

adata = ad.AnnData(
    X=counts_sparse.T.tocsr(),
    obs=pd.DataFrame(
        {"coordinate_x_um": coord_x_um, "coordinate_y_um": coord_y_um},
        index=pd.Index(spot_names),
    ),
    var=pd.DataFrame(index=pd.Index(gene_names)),
)
adata.uns["spacet"] = {}
adata.uns["spacet_platform"] = "Visium"

# ---- Run Python deconvolution step by step ----
from spatialgpu.deconvolution.core import (
    _get_counts_genes_by_spots,
    _cpm_log2_center,
    _infer_mal_cor,
    _spatial_deconv,
)
from spatialgpu.deconvolution.reference import load_comb_ref

# Get counts as genes x spots
ST = _get_counts_genes_by_spots(adata)
py_genes = np.array(adata.var_names)
py_spots = np.array(adata.obs_names)

# Filter zero-sum genes
if sparse.issparse(ST):
    gene_sums = np.asarray(ST.sum(axis=1)).ravel()
else:
    gene_sums = ST.sum(axis=1)
nonzero_mask = gene_sums > 0
ST = ST[nonzero_mask]
py_genes = py_genes[nonzero_mask]

print(f"Python: {len(py_genes)} genes x {len(py_spots)} spots after zero filtering")

# Load reference
ref = load_comb_ref()
reference = ref["refProfiles"].copy()
signature = ref["sigGenes"].copy()
tree = ref["lineageTree"].copy()

print(f"Python Reference: {reference.shape[0]} genes x {reference.shape[1]} cell types")

# Step 1: Intersect genes
olp_genes = np.intersect1d(py_genes, reference.index)
print(f"Python overlapping genes: {len(olp_genes)}")

# Load R's olp genes
r_olp = pd.read_csv(f"{INT_DIR}/olp_genes.csv")["gene"].values
print(f"R overlapping genes: {len(r_olp)}")

# Compare
py_olp_set = set(olp_genes)
r_olp_set = set(r_olp)
print(f"Genes in Python but not R: {len(py_olp_set - r_olp_set)}")
print(f"Genes in R but not Python: {len(r_olp_set - py_olp_set)}")

if py_olp_set != r_olp_set:
    print("WARNING: Gene overlap differs!")
    diff_py = sorted(py_olp_set - r_olp_set)[:5]
    diff_r = sorted(r_olp_set - py_olp_set)[:5]
    print(f"  Python-only (first 5): {diff_py}")
    print(f"  R-only (first 5): {diff_r}")

# Step 2: CPM normalize
gene_idx = np.array([np.where(py_genes == g)[0][0] for g in olp_genes])
ST_sub = ST[gene_idx]

if sparse.issparse(ST_sub):
    col_sums = np.asarray(ST_sub.sum(axis=0)).ravel()
    ST_cpm = ST_sub.toarray().astype(np.float64)
    ST_cpm = ST_cpm / col_sums[np.newaxis, :] * 1e6
else:
    col_sums = ST_sub.sum(axis=0)
    ST_cpm = ST_sub.astype(np.float64) / col_sums[np.newaxis, :] * 1e6

reference_sub = reference.loc[olp_genes]
ref_cpm = reference_sub.values.astype(np.float64)
ref_col_sums = ref_cpm.sum(axis=0)
ref_cpm = ref_cpm / ref_col_sums[np.newaxis, :] * 1e6

# Compare with R's CPM values
r_ST_cpm = pd.read_csv(f"{INT_DIR}/ST_cpm_sample.csv", index_col=0)
r_Ref_cpm = pd.read_csv(f"{INT_DIR}/Ref_cpm_sample.csv", index_col=0)

# Map olp_genes to indices for comparison
olp_gene_to_idx = {g: i for i, g in enumerate(olp_genes)}
r_sample_genes = list(r_ST_cpm.index)

print("\n=== CPM Comparison ===")
# Compare ST CPM for first 20 genes, first 10 spots
py_st_sample = ST_cpm[:20, :10]
r_st_sample = r_ST_cpm.values[:20, :10]
st_cpm_err = np.abs(py_st_sample - r_st_sample)
print(f"ST CPM max abs error (first 20 genes x 10 spots): {st_cpm_err.max():.6e}")
print(f"ST CPM mean abs error: {st_cpm_err.mean():.6e}")

if st_cpm_err.max() > 1e-6:
    print("WARNING: ST CPM values differ significantly!")
    # Show where the errors are largest
    i, j = np.unravel_index(st_cpm_err.argmax(), st_cpm_err.shape)
    print(f"  Largest error at gene={r_sample_genes[i]}, spot_idx={j}")
    print(f"  Python: {py_st_sample[i,j]:.10f}")
    print(f"  R:      {r_st_sample[i,j]:.10f}")

# Compare Ref CPM for first 20 genes
py_ref_sample = ref_cpm[:20, :]
r_ref_sample = r_Ref_cpm.values[:20, :]
ref_cpm_err = np.abs(py_ref_sample - r_ref_sample)
print(f"\nRef CPM max abs error (first 20 genes): {ref_cpm_err.max():.6e}")

# Step 3: malProp comparison
print("\n=== Stage 1: malProp ===")
mal_res = _infer_mal_cor(ST, py_genes, py_spots, "BRCA", None)
py_malProp = mal_res["malProp"]

r_malProp_df = pd.read_csv(f"{INT_DIR}/malProp.csv")
r_malProp = pd.Series(r_malProp_df["malProp"].values, index=r_malProp_df["spot"].values)

# Align
common_spots = py_malProp.index.intersection(r_malProp.index)
mal_err = np.abs(py_malProp[common_spots].values - r_malProp[common_spots].values)
print(f"malProp max abs error: {mal_err.max():.6e}")
print(f"malProp mean abs error: {mal_err.mean():.6e}")
print(f"malProp correlation: {np.corrcoef(py_malProp[common_spots].values, r_malProp[common_spots].values)[0,1]:.6f}")

# Show first 10 spots
print("\nFirst 10 spots malProp comparison:")
for s in list(common_spots[:10]):
    print(f"  {s:12s}  py={py_malProp[s]:.6f}  R={r_malProp[s]:.6f}  err={abs(py_malProp[s]-r_malProp[s]):.6e}")

# Step 4: Level 1 optimization comparison
print("\n=== Level 1 Optimization ===")

# Load R's A matrix and signature genes
r_A_L1 = pd.read_csv(f"{INT_DIR}/A_L1.csv", index_col=0)
r_sig_L1 = pd.read_csv(f"{INT_DIR}/sigGenes_L1.csv")["gene"].values

# Python Level 1 sig genes
level1_types = [t for t in tree.keys() if t in reference.columns]
sig_keys = list(level1_types) + (["T cell"] if "T cell" in signature else [])
sig_genes_l1 = []
for k in sig_keys:
    if k in signature:
        sig_genes_l1.extend(signature[k])
sig_genes_l1 = list(set(sig_genes_l1))
sig_genes_l1 = [g for g in sig_genes_l1 if g in olp_genes]

print(f"Python L1 sig genes: {len(sig_genes_l1)}")
print(f"R L1 sig genes: {len(r_sig_L1)}")

py_sig_set = set(sig_genes_l1)
r_sig_set = set(r_sig_L1)
print(f"Sig genes in Python but not R: {len(py_sig_set - r_sig_set)}")
print(f"Sig genes in R but not Python: {len(r_sig_set - py_sig_set)}")

if py_sig_set == r_sig_set:
    print("Signature genes match!")
else:
    print("WARNING: Signature gene sets differ!")

# Construct Python A_L1 and B_L1
sig_idx = np.array([np.where(olp_genes == g)[0][0] for g in sig_genes_l1])
ref_l1_py = ref_cpm[sig_idx][:, [list(reference_sub.columns).index(t) for t in level1_types]]
mixture_minus_mal_py = ST_cpm.copy()

if mal_res["malProp"].sum() > 0 and mal_res["malRef"] is not None:
    mal_ref_sub = mal_res["malRef"].reindex(olp_genes).values.astype(np.float64)
    if np.isnan(mal_ref_sub).any():
        mal_ref_sub = np.nan_to_num(mal_ref_sub)
    mal_ref_cpm = mal_ref_sub * 1e6 / mal_ref_sub.sum() if mal_ref_sub.sum() > 0 else mal_ref_sub
    mal_prop_arr = py_malProp.reindex(py_spots).values
    mixture_mal = np.outer(mal_ref_cpm, mal_prop_arr)
    mixture_minus_mal_py = ST_cpm - mixture_mal

mix_l1_py = mixture_minus_mal_py[sig_idx]

# Compare A matrices using same gene order as R
r_gene_order = list(r_A_L1.index)
py_gene_to_idx = {g: i for i, g in enumerate(sig_genes_l1)}

if set(r_gene_order) == py_sig_set:
    # Reorder Python A to match R's gene order
    r_idx_in_py = [py_gene_to_idx[g] for g in r_gene_order]
    A_py_reordered = ref_l1_py[r_idx_in_py]
    A_err = np.abs(A_py_reordered - r_A_L1.values)
    print(f"\nA_L1 max abs error (reordered): {A_err.max():.6e}")
    print(f"A_L1 mean abs error: {A_err.mean():.6e}")

    if A_err.max() > 1e-6:
        i, j = np.unravel_index(A_err.argmax(), A_err.shape)
        print(f"  Largest error: gene={r_gene_order[i]}, type={list(r_A_L1.columns)[j]}")
        print(f"  Python: {A_py_reordered[i,j]:.10f}")
        print(f"  R:      {r_A_L1.values[i,j]:.10f}")

# Now run optimization on R's inputs to isolate algorithm difference
print("\n=== Optimization Algorithm Comparison ===")
print("Running Python SLSQP on R's A and B matrices for spot 1...")

from scipy.optimize import minimize

# Load R's spot 1 data
r_spot1 = pd.read_csv(f"{INT_DIR}/spot1_L1_result.csv", index_col=0)
r_spot1_b = pd.read_csv(f"{INT_DIR}/spot1_b_L1.csv")

# Use R's A matrix and b vector
A = r_A_L1.values.astype(np.float64)
b = r_spot1_b["value"].values.astype(np.float64)
n_cell = A.shape[1]

# R spot 1: malProp=0.281518
malProp_spot1 = 0.281518
theta_sum = (1 - malProp_spot1) - 1e-5
theta0 = np.full(n_cell, theta_sum / n_cell)
ppmin = 0.0
ppmax = 1 - malProp_spot1

print(f"theta0: {theta0[0]:.6f}")
print(f"thetaSum: {theta_sum:.6f}")
print(f"ppmin: {ppmin:.6f}, ppmax: {ppmax:.6f}")

# SLSQP optimization (current Python approach)
bounds = [(0, None)] * n_cell
constraints = [
    {"type": "ineq", "fun": lambda x: np.sum(x) - ppmin},
    {"type": "ineq", "fun": lambda x: ppmax - np.sum(x)},
]

def f0(x):
    return np.sum((A @ x - b) ** 2)

res1_slsqp = minimize(f0, theta0, method="SLSQP", bounds=bounds, constraints=constraints)
print(f"\nSLSQP Pass 1 result:")
for i, ct in enumerate(r_A_L1.columns):
    print(f"  {ct:30s} SLSQP={res1_slsqp.x[i]:.6f}  R_pass1={r_spot1.loc[ct, 'pass1']:.6f}  err={abs(res1_slsqp.x[i] - r_spot1.loc[ct, 'pass1']):.6e}")

# Pass 2
bhat = A @ res1_slsqp.x
def f_weighted(x):
    return np.sum((A @ x - b) ** 2 / (bhat + 1))

res2_slsqp = minimize(f_weighted, theta0, method="SLSQP", bounds=bounds, constraints=constraints)
print(f"\nSLSQP Pass 2 result:")
for i, ct in enumerate(r_A_L1.columns):
    print(f"  {ct:30s} SLSQP={res2_slsqp.x[i]:.6f}  R_pass2={r_spot1.loc[ct, 'pass2']:.6f}  err={abs(res2_slsqp.x[i] - r_spot1.loc[ct, 'pass2']):.6e}")

print(f"\nSLSQP Pass 1 obj: {res1_slsqp.fun:.6e}")
print(f"SLSQP Pass 2 obj: {res2_slsqp.fun:.6e}")

# Now try constrOptim (our R-compatible implementation)
print("\n\n=== constrOptim (R-compatible) on same R inputs ===")
from spatialgpu.deconvolution.constr_optim import constr_optim

ui = np.vstack([np.eye(n_cell), np.ones((1, n_cell)), -np.ones((1, n_cell))])
ci = np.concatenate([np.zeros(n_cell), [ppmin], [-ppmax]])

def f0_args(theta, A, b):
    return np.sum((A @ theta - b) ** 2)

try:
    theta_p1, val_p1 = constr_optim(theta0, f0_args, ui, ci, args=(A, b))
    print(f"constrOptim Pass 1 result:")
    for i, ct in enumerate(r_A_L1.columns):
        print(f"  {ct:30s} constr={theta_p1[i]:.6f}  R_pass1={r_spot1.loc[ct, 'pass1']:.6f}  err={abs(theta_p1[i] - r_spot1.loc[ct, 'pass1']):.6e}")
    print(f"constrOptim Pass 1 obj: {val_p1:.6e}")

    # Pass 2
    bhat_co = A @ theta_p1
    def f_weighted_args(theta, A, b):
        return np.sum((A @ theta - b) ** 2 / (bhat_co + 1))

    theta_p2, val_p2 = constr_optim(theta0, f_weighted_args, ui, ci, args=(A, b))
    print(f"\nconstrOptim Pass 2 result:")
    for i, ct in enumerate(r_A_L1.columns):
        print(f"  {ct:30s} constr={theta_p2[i]:.6f}  R_pass2={r_spot1.loc[ct, 'pass2']:.6f}  err={abs(theta_p2[i] - r_spot1.loc[ct, 'pass2']):.6e}")
    print(f"constrOptim Pass 2 obj: {val_p2:.6e}")
except Exception as e:
    print(f"constrOptim failed: {e}")

print("\n=== Summary ===")
print(f"SLSQP vs R max err (Pass2): {max(abs(res2_slsqp.x[i] - r_spot1.iloc[i, 2]) for i in range(n_cell)):.6e}")
try:
    print(f"constrOptim vs R max err (Pass2): {max(abs(theta_p2[i] - r_spot1.iloc[i, 2]) for i in range(n_cell)):.6e}")
except:
    pass
