"""Centralised string constants for AnnData .uns / .obsm keys.

Avoids magic strings scattered across the codebase, makes renaming safe,
and gives IDE auto-complete for key names.
"""

# ── adata.uns root ──────────────────────────────────────────────────────────
UNS_SPACET = "spacet"

# ── adata.uns["spacet"] sub-keys ───────────────────────────────────────────
KEY_DECONV = "deconvolution"
KEY_CCI = "CCI"
KEY_SECACT = "SecAct_output"
KEY_GENESET = "GeneSetScore"
KEY_SPATIAL_CORR = "SpatialCorrelation"

# ── Deconvolution result keys ──────────────────────────────────────────────
KEY_PROPMAT = "propMat"
KEY_PROPMAT_COLS = "propMat_columns"
KEY_MALPROP = "malProp"
KEY_MALREF = "malRef"
KEY_REF = "Ref"

# ── CCI / interaction sub-keys ─────────────────────────────────────────────
KEY_INTERACTION = "interaction"
KEY_TESTRES = "testRes"
KEY_GROUPMAT = "groupMat"

# ── adata.obsm ─────────────────────────────────────────────────────────────
OBSM_SPATIAL = "spatial"

# ── DataFrame column names (CCC / annotations) ────────────────────────────
COL_SENDER = "sender"
COL_RECEIVER = "receiver"
COL_SECRETED_PROTEIN = "secretedProtein"
COL_CELLTYPE = "cellType"
COL_COUNT = "count"

# ── Biological cell type labels ────────────────────────────────────────
LABEL_MALIGNANT = "Malignant"
LABEL_UNIDENTIFIABLE = "Unidentifiable"
LABEL_MACROPHAGE_OTHER = "Macrophage other"
