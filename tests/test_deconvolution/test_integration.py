"""Integration tests: full deconvolution pipeline against R validation data.

Tests run the Python deconvolution on the Visium BC example data and compare
outputs (propMat, malProp) against values produced by the R SpaCET package.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

VAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation")


def _has_validation_data() -> bool:
    return os.path.exists(os.path.join(VAL_DIR, "visium_bc_counts.csv"))


@pytest.fixture(scope="module")
def r_data():
    """Load R validation outputs."""
    if not _has_validation_data():
        pytest.skip("R validation data not available")

    prop_mat = pd.read_csv(os.path.join(VAL_DIR, "visium_bc_propMat.csv"), index_col=0)
    mal_prop = pd.read_csv(os.path.join(VAL_DIR, "visium_bc_malProp.csv"), index_col=0)
    mal_ref = pd.read_csv(os.path.join(VAL_DIR, "visium_bc_malRef.csv"), index_col=0)
    return {"propMat": prop_mat, "malProp": mal_prop, "malRef": mal_ref}


@pytest.fixture(scope="module")
def visium_adata():
    """Create AnnData from Visium BC validation data."""
    if not _has_validation_data():
        pytest.skip("R validation data not available")

    import anndata as ad

    counts = pd.read_csv(os.path.join(VAL_DIR, "visium_bc_counts.csv"), index_col=0)
    coords = pd.read_csv(
        os.path.join(VAL_DIR, "visium_bc_spotCoordinates.csv"), index_col=0
    )

    # counts is genes x spots, AnnData needs spots x genes
    gene_names = np.array(counts.index)
    spot_names = np.array(counts.columns)

    counts_sparse = sparse.csc_matrix(counts.values.astype(np.float64))

    adata = ad.AnnData(
        X=counts_sparse.T.tocsr(),
        obs=pd.DataFrame(coords.values, index=pd.Index(spot_names), columns=coords.columns),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )

    # Add coordinate columns needed for spatial weight computation
    if "coordinate_x_um" in coords.columns:
        adata.obs["coordinate_x_um"] = coords["coordinate_x_um"].values
        adata.obs["coordinate_y_um"] = coords["coordinate_y_um"].values

    adata.uns["spacet"] = {}
    adata.uns["spacet_platform"] = "Visium"

    return adata


class TestRValidationDataFormat:
    """Verify R validation data is properly formatted."""

    def test_propmat_exists(self, r_data):
        assert r_data["propMat"].shape[0] > 0

    def test_propmat_has_cell_types(self, r_data):
        pm = r_data["propMat"]
        assert "Malignant" in pm.index
        assert "CAF" in pm.index

    def test_propmat_fractions_valid(self, r_data):
        pm = r_data["propMat"]
        assert pm.min().min() >= -1e-10
        assert pm.max().max() <= 1 + 1e-10

    def test_malprop_range(self, r_data):
        mp = r_data["malProp"]
        assert mp.min().min() >= -1e-10
        assert mp.max().max() <= 1 + 1e-10

    def test_malref_exists(self, r_data):
        mr = r_data["malRef"]
        assert mr.shape[0] > 0


class TestVisiumBCDeconvolution:
    """Full pipeline deconvolution on Visium BC data."""

    @pytest.fixture(scope="class")
    def deconv_result(self, visium_adata):
        """Run full deconvolution (cached per class)."""
        from spatialgpu.deconvolution.core import deconvolution

        adata = visium_adata.copy()
        result = deconvolution(adata, cancer_type="BRCA")
        return result

    def test_deconvolution_runs(self, deconv_result):
        """Pipeline should complete without errors."""
        assert "spacet" in deconv_result.uns
        assert "deconvolution" in deconv_result.uns["spacet"]

    def test_propmat_exists(self, deconv_result):
        prop_mat = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        assert isinstance(prop_mat, pd.DataFrame)
        assert prop_mat.shape[0] > 10  # >10 cell types
        assert prop_mat.shape[1] > 200  # >200 spots

    def test_malignant_row_present(self, deconv_result):
        prop_mat = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        assert "Malignant" in prop_mat.index

    def test_fractions_nonnegative(self, deconv_result):
        prop_mat = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        assert prop_mat.min().min() >= -1e-10

    def test_fractions_sum_reasonable(self, deconv_result):
        """PropMat sums > 1 because it includes both Level 1 and Level 2 types.

        E.g., 'B cell' + 'B cell naive' + 'B cell memory' + ... are all present.
        The R propMat has sums up to ~1.68. Python should be comparable.
        """
        prop_mat = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        col_sums = prop_mat.sum(axis=0)
        # R has max ~1.68; allow up to 2.5 for our implementation
        assert col_sums.max() <= 2.5

    def test_obsm_propmat_shape(self, deconv_result):
        pm = deconv_result.obsm["spacet_propMat"]
        assert pm.shape[0] == deconv_result.n_obs
        assert pm.shape[1] > 10

    def test_malignant_fraction_correlation_with_r(self, deconv_result, r_data):
        """Python malignant fractions should correlate with R."""
        py_prop = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        r_prop = r_data["propMat"]

        # Align spot names
        common_spots = py_prop.columns.intersection(r_prop.columns)
        if len(common_spots) < 100:
            pytest.skip("Too few common spots for comparison")

        py_mal = py_prop.loc["Malignant", common_spots].values.astype(float)
        r_mal = r_prop.loc["Malignant", common_spots].values.astype(float)

        # Pearson correlation should be high
        from scipy.stats import pearsonr

        corr, _ = pearsonr(py_mal, r_mal)
        assert corr > 0.7, f"Malignant fraction correlation with R: {corr:.3f}"

    def test_cell_type_fractions_correlation_with_r(self, deconv_result, r_data):
        """Average cell type fractions should be reasonably correlated with R."""
        py_prop = deconv_result.uns["spacet"]["deconvolution"]["propMat"]
        r_prop = r_data["propMat"]

        common_types = py_prop.index.intersection(r_prop.index)
        common_spots = py_prop.columns.intersection(r_prop.columns)

        if len(common_types) < 5 or len(common_spots) < 100:
            pytest.skip("Too few common types/spots")

        py_means = py_prop.loc[common_types, common_spots].mean(axis=1)
        r_means = r_prop.loc[common_types, common_spots].mean(axis=1)

        from scipy.stats import spearmanr

        corr, _ = spearmanr(py_means, r_means)
        assert corr > 0.5, f"Cell type mean fraction Spearman with R: {corr:.3f}"


class TestCreateFromValidation:
    """Test creating SpaCET object from validation data."""

    def test_create_object(self, visium_adata):
        assert visium_adata.n_obs > 200
        assert visium_adata.n_vars > 10000

    def test_spot_ids_format(self, visium_adata):
        """Spot IDs should be in 'row x col' format."""
        for sid in visium_adata.obs_names[:10]:
            assert "x" in sid
