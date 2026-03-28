"""Centralised string constants for AnnData .uns / .obsm keys.

Avoids magic strings scattered across the codebase, makes renaming safe,
and gives IDE auto-complete for key names.
"""

# ── adata.uns root ──────────────────────────────────────────────────────────
UNS_SPACET = "spacet"
UNS_DECONV = "deconv"

# ── adata.uns top-level metadata ────────────────────────────────────────────
UNS_PLATFORM = "spacet_platform"
UNS_ORGANISM = "spacet_organism"
UNS_IMAGE_PATH = "spacet_image_path"
UNS_TRANSCRIPTS_FILE = "transcripts_file"

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
KEY_MALRES = "malRes"
KEY_REF = "Ref"
KEY_CANCER_TYPE = "cancer_type"

# ── CCI / interaction sub-keys ─────────────────────────────────────────────
KEY_INTERACTION = "interaction"
KEY_TESTRES = "testRes"
KEY_GROUPMAT = "groupMat"
KEY_COLOCALIZATION = "colocalization"
KEY_LR_NETWORK_SCORE = "LRNetworkScore"
KEY_LR_NETWORK_SCORE_COLS = "LRNetworkScore_columns"
KEY_LR_NETWORK_SCORE_IDX = "LRNetworkScore_index"
KEY_INTERFACE = "interface"
KEY_DIST_TO_INTERFACE = "distance_to_interface"

# ── SecAct sub-keys ────────────────────────────────────────────────────────
KEY_SECRETED_PROTEIN_ACTIVITY = "SecretedProteinActivity"
KEY_PATTERN = "pattern"

# ── adata.obsm ─────────────────────────────────────────────────────────────
OBSM_SPATIAL = "spatial"
OBSM_SPACET_PROPMAT = "spacet_propMat"
OBSM_DECONV_PROPMAT = "deconv_propMat"

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
