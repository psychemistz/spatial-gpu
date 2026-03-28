"""Tests for mouse-to-human gene conversion and solver modes."""

import numpy as np
import pytest
from scipy import sparse


class TestMouse2Human:
    """Tests for mouse2human_mat and ensure_human_genes."""

    def test_mouse2human_basic(self):
        from spatialgpu.deconvolution.reference import (
            _load_mouse2human_map,
            mouse2human_mat,
        )

        m2h = _load_mouse2human_map()
        mouse_genes = m2h["mouse"].values[:10]
        gene_names = np.array(list(mouse_genes) + ["FakeGene1", "FakeGene2"])
        counts = sparse.random(len(gene_names), 5, density=0.5, format="csc")

        out_counts, out_genes = mouse2human_mat(counts, gene_names)

        assert out_counts.shape[1] == 5
        assert len(out_genes) > 0
        assert "FakeGene1" not in out_genes
        assert "FakeGene2" not in out_genes

    def test_mouse2human_no_match(self):
        from spatialgpu.deconvolution.reference import mouse2human_mat

        gene_names = np.array(["ZZZZZ_FAKE1", "ZZZZZ_FAKE2"])
        counts = np.ones((2, 3))

        with pytest.raises(ValueError, match="No mouse genes"):
            mouse2human_mat(counts, gene_names)

    def test_mouse2human_shape_mismatch(self):
        from spatialgpu.deconvolution.reference import mouse2human_mat

        gene_names = np.array(["A", "B", "C"])
        counts = np.ones((5, 3))

        with pytest.raises(ValueError, match="counts has 5 rows"):
            mouse2human_mat(counts, gene_names)

    def test_mouse2human_aggregation(self):
        from collections import Counter

        from spatialgpu.deconvolution.reference import (
            _mouse2human_dict,
            mouse2human_mat,
        )

        m2h = _mouse2human_dict()
        human_counts = Counter(m2h.values())
        dup_human = [h for h, c in human_counts.items() if c >= 2][0]
        dup_mouse = [m for m, h in m2h.items() if h == dup_human][:2]

        gene_names = np.array(dup_mouse)
        counts = np.array([[1.0, 2.0], [3.0, 4.0]])

        out_counts, out_genes = mouse2human_mat(counts, gene_names)

        assert len(out_genes) == 1
        assert out_genes[0] == dup_human
        np.testing.assert_array_almost_equal(np.asarray(out_counts).ravel(), [4.0, 6.0])

    def test_ensure_human_genes_human(self):
        import anndata as ad

        from spatialgpu.deconvolution.reference import ensure_human_genes

        counts = np.ones((10, 5))
        gene_names = np.array([f"GENE{i}" for i in range(10)])
        adata = ad.AnnData(X=np.ones((5, 10)))
        adata.uns["spacet_organism"] = "human"

        out_counts, out_genes = ensure_human_genes(adata, counts, gene_names)
        assert out_counts is counts
        assert out_genes is gene_names

    def test_ensure_human_genes_no_key(self):
        import anndata as ad

        from spatialgpu.deconvolution.reference import ensure_human_genes

        counts = np.ones((10, 5))
        gene_names = np.array([f"GENE{i}" for i in range(10)])
        adata = ad.AnnData(X=np.ones((5, 10)))

        out_counts, out_genes = ensure_human_genes(adata, counts, gene_names)
        assert out_counts is counts


class TestOrganismValidation:

    def test_valid_organisms(self):
        from spatialgpu.deconvolution.io import _validate_organism

        assert _validate_organism("human") == "human"
        assert _validate_organism("mouse") == "mouse"
        assert _validate_organism("Human") == "human"
        assert _validate_organism("MOUSE") == "mouse"
        assert _validate_organism(" mouse ") == "mouse"

    def test_invalid_organism(self):
        from spatialgpu.deconvolution.io import _validate_organism

        with pytest.raises(ValueError, match="Unknown organism"):
            _validate_organism("rat")


class TestSolverParameter:

    def test_solver_auto(self):
        from spatialgpu.deconvolution.core import _solve_constrained_batch

        np.random.seed(42)
        A = np.random.rand(20, 3) + 0.1
        B = np.random.rand(20, 5) + 0.1
        theta_sum = np.full(5, 0.8)
        pp_max = np.full(5, 0.9)
        pp_min = np.zeros(5)

        result = _solve_constrained_batch(
            A, B, 3, theta_sum, pp_min, pp_max, solver="auto"
        )
        assert result.shape == (3, 5)
        assert np.all(result >= -1e-10)

    def test_solver_modes_produce_results(self):
        from spatialgpu.deconvolution.core import _solve_constrained_batch

        np.random.seed(42)
        A = np.random.rand(20, 3) + 0.1
        B = np.random.rand(20, 5) + 0.1
        theta_sum = np.full(5, 0.8)
        pp_min = np.full(5, 0.5)
        pp_max = np.full(5, 0.9)

        for mode in ["auto", "r_compat", "fast"]:
            result = _solve_constrained_batch(
                A, B, 3, theta_sum, pp_min, pp_max, solver=mode
            )
            assert result.shape == (3, 5), f"Failed for solver={mode}"
            assert np.all(np.isfinite(result)), f"Non-finite for solver={mode}"

    def test_small_theta_sum(self):
        from spatialgpu.deconvolution.core import _solve_constrained_batch

        A = np.random.rand(10, 3)
        B = np.random.rand(10, 2)
        theta_sum = np.array([0.005, 0.001])
        pp_min = np.array([0.0, 0.0])
        pp_max = np.array([0.01, 0.01])

        result = _solve_constrained_batch(
            A, B, 3, theta_sum, pp_min, pp_max, solver="auto"
        )
        for i in range(2):
            np.testing.assert_almost_equal(result[:, i].sum(), theta_sum[i], decimal=5)


class TestConstrOptim:

    def test_simple_quadratic(self):
        from spatialgpu.deconvolution.constr_optim import constr_optim

        def f(x):
            return float(x[0] ** 2 + x[1] ** 2)

        ui = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        ci = np.array([0.0, 0.0, 1.0])
        theta0 = np.array([0.6, 0.6])

        result, fval = constr_optim(theta0, f, ui, ci)
        np.testing.assert_almost_equal(result, [0.5, 0.5], decimal=2)
        assert fval < 0.6

    def test_infeasible_start(self):
        from spatialgpu.deconvolution.constr_optim import constr_optim

        def f(x):
            return float(x[0] ** 2)

        ui = np.array([[1.0]])
        ci = np.array([1.0])
        theta0 = np.array([0.5])

        with pytest.raises(ValueError, match="not in the interior"):
            constr_optim(theta0, f, ui, ci)


class TestSparseUtils:

    def test_ensure_dense_sparse(self):
        from spatialgpu.core.array_utils import ensure_dense

        X = sparse.random(10, 5, density=0.3, format="csc")
        result = ensure_dense(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)

    def test_ensure_dense_already_dense(self):
        from spatialgpu.core.array_utils import ensure_dense

        X = np.ones((3, 4))
        result = ensure_dense(X)
        assert result is X or np.array_equal(result, X)

    def test_sparse_sum_axis0(self):
        from spatialgpu.core.array_utils import sparse_sum

        X = sparse.csc_matrix(np.array([[1, 2], [3, 4]]))
        result = sparse_sum(X, axis=0)
        np.testing.assert_array_equal(result, [4, 6])
        assert result.ndim == 1

    def test_sparse_sum_axis1(self):
        from spatialgpu.core.array_utils import sparse_sum

        X = sparse.csc_matrix(np.array([[1, 2], [3, 4]]))
        result = sparse_sum(X, axis=1)
        np.testing.assert_array_equal(result, [3, 7])
        assert result.ndim == 1

    def test_sparse_sum_dense_input(self):
        from spatialgpu.core.array_utils import sparse_sum

        X = np.array([[1, 2], [3, 4]])
        result = sparse_sum(X, axis=0)
        np.testing.assert_array_equal(result, [4, 6])
