"""Tests for spatial graph operations."""

import numpy as np
import pytest
from scipy import sparse

# Skip if dependencies not available
pytest.importorskip("anndata")
pytest.importorskip("scanpy")


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    import anndata as ad

    n_cells = 500
    n_genes = 50

    # Random count matrix
    X = sparse.random(n_cells, n_genes, density=0.3, format="csr")

    # Random spatial coordinates
    coords = np.random.rand(n_cells, 2) * 1000

    # Random clusters
    clusters = np.random.choice(["A", "B", "C", "D"], size=n_cells)

    import pandas as pd

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(clusters)},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
    )
    adata.obsm["spatial"] = coords

    return adata


class TestSpatialNeighbors:
    """Tests for spatial_neighbors function."""

    def test_knn_graph(self, synthetic_adata):
        """Test k-nearest neighbors graph construction."""
        import spatialgpu as sp

        sp.graph.spatial_neighbors(synthetic_adata, n_neighbors=6)

        assert "spatial_connectivities" in synthetic_adata.obsp
        assert "spatial_distances" in synthetic_adata.obsp

        conn = synthetic_adata.obsp["spatial_connectivities"]
        assert conn.shape == (500, 500)
        assert sparse.issparse(conn)

    def test_radius_graph(self, synthetic_adata):
        """Test radius-based graph construction."""
        import spatialgpu as sp

        sp.graph.spatial_neighbors(synthetic_adata, radius=100.0)

        assert "spatial_connectivities" in synthetic_adata.obsp
        conn = synthetic_adata.obsp["spatial_connectivities"]
        assert conn.shape == (500, 500)

    def test_copy_mode(self, synthetic_adata):
        """Test copy=True returns new AnnData."""
        import spatialgpu as sp

        result = sp.graph.spatial_neighbors(synthetic_adata, n_neighbors=6, copy=True)

        assert result is not synthetic_adata
        assert "spatial_connectivities" in result.obsp
        assert "spatial_connectivities" not in synthetic_adata.obsp


class TestKNNGraph:
    """Tests for knn_graph function."""

    def test_basic(self):
        """Test basic kNN graph construction."""
        import spatialgpu as sp

        coords = np.random.rand(100, 2) * 100
        conn, dist = sp.graph.knn_graph(coords, n_neighbors=5)

        assert conn.shape == (100, 100)
        assert dist.shape == (100, 100)

        # Each cell should have approximately n_neighbors connections
        n_connections = np.array(conn.sum(axis=1)).flatten()
        assert np.mean(n_connections) >= 5

    def test_set_diag(self):
        """Test diagonal setting."""
        import spatialgpu as sp

        coords = np.random.rand(50, 2)
        conn, _ = sp.graph.knn_graph(coords, n_neighbors=5, set_diag=True)

        diag = conn.diagonal()
        assert np.all(diag == 1)


class TestRadiusGraph:
    """Tests for radius_graph function."""

    def test_basic(self):
        """Test basic radius graph construction."""
        import spatialgpu as sp

        # Create grid of points
        x = np.arange(10)
        y = np.arange(10)
        xx, yy = np.meshgrid(x, y)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)

        conn, dist = sp.graph.radius_graph(coords, radius=1.5)

        assert conn.shape == (100, 100)
        # Each interior point should have 4 neighbors (up, down, left, right)


class TestDelaunayGraph:
    """Tests for delaunay_graph function."""

    def test_basic(self):
        """Test Delaunay triangulation graph."""
        import spatialgpu as sp

        coords = np.random.rand(50, 2) * 100
        conn, dist = sp.graph.delaunay_graph(coords)

        assert conn.shape == (50, 50)
        assert sparse.issparse(conn)

    def test_requires_2d(self):
        """Test error on 3D coordinates."""
        import spatialgpu as sp

        coords = np.random.rand(50, 3)
        with pytest.raises(ValueError, match="2D coordinates"):
            sp.graph.delaunay_graph(coords)


class TestNhoodEnrichment:
    """Tests for nhood_enrichment function."""

    def test_basic(self, synthetic_adata):
        """Test neighborhood enrichment calculation."""
        import spatialgpu as sp

        sp.graph.spatial_neighbors(synthetic_adata, n_neighbors=6)
        sp.graph.nhood_enrichment(
            synthetic_adata,
            cluster_key="cluster",
            n_perms=50,
            show_progress=False,
        )

        key = "cluster_nhood_enrichment"
        assert key in synthetic_adata.uns
        assert "zscore" in synthetic_adata.uns[key]
        assert "count" in synthetic_adata.uns[key]

        zscore = synthetic_adata.uns[key]["zscore"]
        assert zscore.shape == (4, 4)  # 4 clusters

    def test_copy_mode(self, synthetic_adata):
        """Test copy=True returns results."""
        import spatialgpu as sp

        sp.graph.spatial_neighbors(synthetic_adata, n_neighbors=6)
        zscore, count = sp.graph.nhood_enrichment(
            synthetic_adata,
            cluster_key="cluster",
            n_perms=10,
            copy=True,
            show_progress=False,
        )

        assert zscore.shape == (4, 4)
        assert count.shape == (4, 4)


class TestCoOccurrence:
    """Tests for co_occurrence function."""

    def test_basic(self, synthetic_adata):
        """Test co-occurrence calculation."""
        import spatialgpu as sp

        occurrence, intervals = sp.graph.co_occurrence(
            synthetic_adata,
            cluster_key="cluster",
            n_splits=10,
            copy=True,
            show_progress=False,
        )

        assert occurrence.shape[0] == 4  # clusters
        assert occurrence.shape[1] == 4  # clusters
        assert occurrence.shape[2] == 10  # bins


class TestRipley:
    """Tests for Ripley's statistics."""

    def test_basic(self, synthetic_adata):
        """Test Ripley's L function."""
        import spatialgpu as sp

        result = sp.graph.ripley(
            synthetic_adata,
            mode="L",
            n_simulations=10,
            n_radii=20,
            copy=True,
            show_progress=False,
        )

        assert "radii" in result
        assert "stats" in result
        assert "observed" in result["stats"]
        assert len(result["radii"]) == 20


class TestInteractionMatrix:
    """Tests for interaction_matrix function."""

    def test_basic(self, synthetic_adata):
        """Test interaction matrix calculation."""
        import spatialgpu as sp

        sp.graph.spatial_neighbors(synthetic_adata, n_neighbors=6)
        interaction = sp.graph.interaction_matrix(
            synthetic_adata,
            cluster_key="cluster",
            copy=True,
        )

        assert interaction.shape == (4, 4)
