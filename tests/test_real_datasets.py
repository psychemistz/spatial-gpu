"""Integration tests with real Visium and CosMx datasets."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip if dependencies not available
pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("requests")

# Cache directory for downloaded datasets
CACHE_DIR = Path(tempfile.gettempdir()) / "spatialgpu_test_data"


@pytest.fixture(scope="module")
def visium_adata():
    """Load Visium mouse brain dataset from 10x Genomics."""
    import scanpy as sc

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "visium_mouse_brain.h5ad"

    if cache_file.exists():
        return sc.read_h5ad(cache_file)

    # Download Visium mouse brain section
    # This is a standard 10x Genomics sample dataset
    adata = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")

    # Basic preprocessing
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Select highly variable genes for faster testing
    sc.pp.highly_variable_genes(adata, n_top_genes=500)
    adata = adata[:, adata.var.highly_variable].copy()

    # Save to cache
    adata.write_h5ad(cache_file)

    return adata


@pytest.fixture(scope="module")
def cosmx_adata():
    """Create synthetic CosMx-like dataset for testing.

    Note: Real CosMx data requires NanoString access.
    This creates a representative synthetic dataset with CosMx characteristics.
    """
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    np.random.seed(42)

    # CosMx characteristics:
    # - Single-cell resolution
    # - ~1000 genes per panel
    # - High cell density
    # - XY coordinates in microns

    n_cells = 2000
    n_genes = 500

    # Sparse count matrix (CosMx is count-based)
    X = sparse.random(n_cells, n_genes, density=0.2, format="csr")
    X.data = np.random.poisson(5, size=X.data.shape)

    # Spatial coordinates in microns (typical CosMx FOV ~500x500 um)
    coords = np.random.rand(n_cells, 2) * 500

    # Cell types (typical tissue composition)
    cell_types = np.random.choice(
        ["Epithelial", "Fibroblast", "Immune", "Endothelial", "Other"],
        size=n_cells,
        p=[0.35, 0.25, 0.20, 0.10, 0.10],
    )

    # Create AnnData
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(cell_types),
                "area": np.random.lognormal(5, 0.5, n_cells),
                "fov": np.random.randint(1, 5, n_cells),
            },
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)]),
    )
    adata.obsm["spatial"] = coords

    return adata


class TestVisiumDataset:
    """Tests using real Visium dataset."""

    def test_load_visium(self, visium_adata):
        """Test Visium dataset loads correctly."""
        assert visium_adata is not None
        assert "spatial" in visium_adata.obsm
        assert visium_adata.obsm["spatial"].shape[1] == 2
        assert visium_adata.n_obs > 100

    def test_spatial_neighbors_visium(self, visium_adata):
        """Test spatial neighbor graph on Visium."""
        import spatialgpu as sp

        adata = visium_adata.copy()
        sp.graph.spatial_neighbors(adata, n_neighbors=6)

        assert "spatial_connectivities" in adata.obsp
        assert "spatial_distances" in adata.obsp

        conn = adata.obsp["spatial_connectivities"]
        assert conn.shape[0] == adata.n_obs
        assert conn.nnz > 0

    def test_knn_graph_visium(self, visium_adata):
        """Test kNN graph construction on Visium coordinates."""
        import spatialgpu as sp

        coords = visium_adata.obsm["spatial"]
        conn, dist = sp.graph.knn_graph(coords, n_neighbors=6)

        assert conn.shape == (len(coords), len(coords))
        assert dist.shape == conn.shape

        # Check each spot has neighbors
        n_neighbors_per_spot = np.array(conn.sum(axis=1)).flatten()
        assert np.all(n_neighbors_per_spot >= 6)

    def test_radius_graph_visium(self, visium_adata):
        """Test radius graph on Visium coordinates."""
        import spatialgpu as sp

        coords = visium_adata.obsm["spatial"]

        # Visium spots are ~100um apart, use appropriate radius
        # Coords are in pixel space, estimate spacing
        dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        median_dist = np.median(dists)

        conn, dist = sp.graph.radius_graph(coords, radius=median_dist * 2)

        assert conn.shape == (len(coords), len(coords))
        assert conn.nnz > 0

    def test_delaunay_graph_visium(self, visium_adata):
        """Test Delaunay triangulation on Visium."""
        import spatialgpu as sp

        coords = visium_adata.obsm["spatial"]
        conn, dist = sp.graph.delaunay_graph(coords)

        assert conn.shape == (len(coords), len(coords))
        assert conn.nnz > 0

        # Delaunay should create a connected graph
        n_neighbors = np.array(conn.sum(axis=1)).flatten()
        assert np.all(n_neighbors >= 3)  # Min 3 for Delaunay

    @pytest.mark.slow
    def test_nhood_enrichment_visium(self, visium_adata):
        """Test neighborhood enrichment on Visium with clusters."""
        import pandas as pd
        import scanpy as sc

        import spatialgpu as sp

        adata = visium_adata.copy()

        # Create clusters using simple k-means (no leidenalg dependency)
        if "clusters" not in adata.obs:
            sc.pp.pca(adata)
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            labels = kmeans.fit_predict(adata.obsm["X_pca"][:, :10])
            adata.obs["clusters"] = pd.Categorical([f"C{i}" for i in labels])

        # Build spatial graph
        sp.graph.spatial_neighbors(adata, n_neighbors=6)

        # Run neighborhood enrichment
        sp.graph.nhood_enrichment(
            adata,
            cluster_key="clusters",
            n_perms=50,
            show_progress=False,
        )

        key = "clusters_nhood_enrichment"
        assert key in adata.uns
        assert "zscore" in adata.uns[key]
        assert "count" in adata.uns[key]

        n_clusters = len(adata.obs["clusters"].cat.categories)
        assert adata.uns[key]["zscore"].shape == (n_clusters, n_clusters)


class TestCosMxDataset:
    """Tests using CosMx-like dataset."""

    def test_load_cosmx(self, cosmx_adata):
        """Test CosMx dataset loads correctly."""
        assert cosmx_adata is not None
        assert "spatial" in cosmx_adata.obsm
        assert cosmx_adata.obsm["spatial"].shape[1] == 2
        assert "cell_type" in cosmx_adata.obs

    def test_spatial_neighbors_cosmx(self, cosmx_adata):
        """Test spatial neighbor graph on CosMx."""
        import spatialgpu as sp

        adata = cosmx_adata.copy()
        sp.graph.spatial_neighbors(adata, n_neighbors=10)

        assert "spatial_connectivities" in adata.obsp
        conn = adata.obsp["spatial_connectivities"]
        assert conn.shape[0] == adata.n_obs

    def test_radius_graph_cosmx(self, cosmx_adata):
        """Test radius graph on CosMx (micron scale)."""
        import spatialgpu as sp

        coords = cosmx_adata.obsm["spatial"]

        # CosMx: typical cell diameter ~10-20um, use 50um radius
        conn, dist = sp.graph.radius_graph(coords, radius=50.0)

        assert conn.shape == (len(coords), len(coords))
        assert conn.nnz > 0

    def test_nhood_enrichment_cosmx(self, cosmx_adata):
        """Test neighborhood enrichment on CosMx cell types."""
        import spatialgpu as sp

        adata = cosmx_adata.copy()

        # Build spatial graph
        sp.graph.spatial_neighbors(adata, n_neighbors=10)

        # Run neighborhood enrichment
        sp.graph.nhood_enrichment(
            adata,
            cluster_key="cell_type",
            n_perms=50,
            show_progress=False,
        )

        key = "cell_type_nhood_enrichment"
        assert key in adata.uns

        n_types = len(adata.obs["cell_type"].cat.categories)
        assert adata.uns[key]["zscore"].shape == (n_types, n_types)

    def test_co_occurrence_cosmx(self, cosmx_adata):
        """Test co-occurrence analysis on CosMx."""
        import spatialgpu as sp

        adata = cosmx_adata.copy()

        occurrence, intervals = sp.graph.co_occurrence(
            adata,
            cluster_key="cell_type",
            n_splits=10,
            copy=True,
            show_progress=False,
        )

        n_types = len(adata.obs["cell_type"].cat.categories)
        assert occurrence.shape[0] == n_types
        assert occurrence.shape[1] == n_types
        assert occurrence.shape[2] == 10

    def test_interaction_matrix_cosmx(self, cosmx_adata):
        """Test interaction matrix on CosMx."""
        import spatialgpu as sp

        adata = cosmx_adata.copy()
        sp.graph.spatial_neighbors(adata, n_neighbors=10)

        interaction = sp.graph.interaction_matrix(
            adata,
            cluster_key="cell_type",
            copy=True,
        )

        n_types = len(adata.obs["cell_type"].cat.categories)
        assert interaction.shape == (n_types, n_types)

        # Interaction values should be positive (normalized by expected freq)
        assert np.all(interaction >= 0)
        # Matrix should not be all zeros
        assert interaction.sum() > 0

    def test_ripley_cosmx(self, cosmx_adata):
        """Test Ripley's statistics on CosMx."""
        import spatialgpu as sp

        adata = cosmx_adata.copy()

        result = sp.graph.ripley(
            adata,
            mode="L",
            n_simulations=10,
            n_radii=15,
            copy=True,
            show_progress=False,
        )

        assert "radii" in result
        assert "stats" in result
        assert len(result["radii"]) == 15


class TestVisualization:
    """Test visualization with real datasets."""

    def test_spatial_scatter_visium(self, visium_adata):
        """Test spatial scatter plot on Visium."""
        import matplotlib

        matplotlib.use("Agg")
        import spatialgpu as sp

        adata = visium_adata.copy()

        # Plot without color
        ax = sp.viz.spatial_scatter(adata)
        assert ax is not None

    def test_spatial_scatter_cosmx_celltype(self, cosmx_adata):
        """Test spatial scatter colored by cell type."""
        import matplotlib

        matplotlib.use("Agg")
        import spatialgpu as sp

        adata = cosmx_adata.copy()

        ax = sp.viz.spatial_scatter(adata, color="cell_type")
        assert ax is not None

    def test_nhood_enrichment_plot_cosmx(self, cosmx_adata):
        """Test neighborhood enrichment heatmap."""
        import matplotlib

        matplotlib.use("Agg")
        import spatialgpu as sp

        adata = cosmx_adata.copy()
        sp.graph.spatial_neighbors(adata, n_neighbors=10)
        sp.graph.nhood_enrichment(
            adata,
            cluster_key="cell_type",
            n_perms=20,
            show_progress=False,
        )

        ax = sp.viz.nhood_enrichment_plot(adata, cluster_key="cell_type")
        assert ax is not None


class TestPerformance:
    """Performance benchmarks on real-scale data."""

    @pytest.mark.slow
    def test_large_scale_knn(self):
        """Test kNN on large dataset (10k+ cells)."""
        import time

        import spatialgpu as sp

        n_cells = 10000
        coords = np.random.rand(n_cells, 2) * 1000

        start = time.time()
        conn, dist = sp.graph.knn_graph(coords, n_neighbors=15)
        elapsed = time.time() - start

        assert conn.shape == (n_cells, n_cells)
        print(f"\nkNN on {n_cells} cells: {elapsed:.2f}s")

    @pytest.mark.slow
    def test_large_scale_radius(self):
        """Test radius graph on large dataset."""
        import time

        import spatialgpu as sp

        n_cells = 10000
        coords = np.random.rand(n_cells, 2) * 1000

        start = time.time()
        conn, dist = sp.graph.radius_graph(coords, radius=50.0)
        elapsed = time.time() - start

        assert conn.shape == (n_cells, n_cells)
        print(f"\nRadius graph on {n_cells} cells: {elapsed:.2f}s")
