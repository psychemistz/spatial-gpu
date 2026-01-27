"""Tests for cell segmentation."""

import numpy as np


class TestSegmentationResult:
    """Tests for SegmentationResult class."""

    def test_from_masks(self):
        """Test creating result from masks."""
        from spatialgpu.segmentation.core import SegmentationResult

        # Create simple mask
        masks = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 2],
                [0, 0, 2, 2],
            ]
        )

        result = SegmentationResult.from_masks(masks)

        assert result.n_cells == 2
        assert len(result.cell_ids) == 2
        assert 1 in result.cell_ids
        assert 2 in result.cell_ids
        assert len(result.centroids) == 2
        assert len(result.areas) == 2


class TestSegmentationUtils:
    """Tests for segmentation utility functions."""

    def test_compute_centroids(self):
        """Test centroid computation."""
        from spatialgpu.segmentation.utils import compute_centroids

        masks = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [2, 2, 0, 0],
            ]
        )

        centroids = compute_centroids(masks)

        assert centroids.shape == (2, 2)
        # Cell 1 centroid should be around (0.5, 1.5)
        assert np.allclose(centroids[0], [0.5, 1.5], atol=0.1)

    def test_compute_areas(self):
        """Test area computation."""
        from spatialgpu.segmentation.utils import compute_areas

        masks = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [2, 2, 0, 0],
            ]
        )

        areas = compute_areas(masks)

        assert len(areas) == 2
        assert areas[0] == 4  # Cell 1 has 4 pixels
        assert areas[1] == 2  # Cell 2 has 2 pixels

    def test_compute_boundaries(self):
        """Test boundary computation."""
        from spatialgpu.segmentation.utils import compute_boundaries

        masks = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        boundaries = compute_boundaries(masks)

        # Should have boundaries only at edges of cell 1
        assert boundaries.dtype == bool
        # Interior pixel should not be boundary
        assert not boundaries[2, 2]
        # Edge pixels should be boundaries
        assert boundaries[1, 1] or boundaries[1, 2]

    def test_filter_by_size(self):
        """Test size filtering."""
        from spatialgpu.segmentation.utils import filter_by_size

        masks = np.array(
            [
                [1, 1, 0, 2],
                [1, 1, 0, 0],
                [1, 1, 3, 3],
                [1, 1, 3, 3],
            ]
        )  # Cell 1: 8px, Cell 2: 1px, Cell 3: 4px

        filtered = filter_by_size(masks, min_size=2)

        # Cell 2 (1px) should be removed - its location should now be 0
        assert filtered[0, 3] == 0
        # Two cells should remain (plus background)
        assert len(np.unique(filtered)) == 3  # 0, plus 2 cells
        # Total non-zero pixels should be 8 + 4 = 12 (cells 1 and 3)
        assert np.sum(filtered > 0) == 12

    def test_remove_edge_cells(self):
        """Test edge cell removal."""
        from spatialgpu.segmentation.utils import remove_edge_cells

        masks = np.array(
            [
                [1, 1, 0, 0, 2],
                [1, 1, 3, 3, 2],
                [0, 0, 3, 3, 0],
                [0, 0, 3, 3, 0],
                [4, 4, 0, 0, 5],
            ]
        )

        filtered = remove_edge_cells(masks)

        # Only cell 3 (center) should remain
        unique = np.unique(filtered)
        assert len(unique) == 2  # 0 and cell 3
        assert 3 in unique


class TestSegmentationMetrics:
    """Tests for segmentation evaluation metrics."""

    def test_perfect_match(self):
        """Test metrics for perfect prediction."""
        from spatialgpu.segmentation.evaluation import compute_segmentation_metrics

        masks = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [2, 2, 0, 0],
                [2, 2, 0, 0],
            ]
        )

        metrics = compute_segmentation_metrics(masks, masks)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.iou == 1.0

    def test_no_overlap(self):
        """Test metrics for no overlap."""
        from spatialgpu.segmentation.evaluation import compute_segmentation_metrics

        pred = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        gt = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
            ]
        )

        metrics = compute_segmentation_metrics(pred, gt)

        assert metrics.n_true_positive == 0
        assert metrics.n_false_positive == 1
        assert metrics.n_false_negative == 1

    def test_partial_overlap(self):
        """Test metrics for partial overlap."""
        from spatialgpu.segmentation.evaluation import compute_segmentation_metrics

        pred = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        gt = np.array(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        metrics = compute_segmentation_metrics(pred, gt, iou_threshold=0.5)

        # IoU = 4 / 6 = 0.67, should match
        assert metrics.n_true_positive == 1


class TestMergeTiledMasks:
    """Tests for tiled mask merging."""

    def test_basic_merge(self):
        """Test basic tile merging."""
        from spatialgpu.segmentation.utils import merge_tiled_masks

        # Create two overlapping tiles
        tile1 = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]
        )

        tile2 = np.array(
            [
                [0, 0, 0],
                [0, 2, 2],
                [0, 2, 2],
            ]
        )

        tiles = [
            (tile1, (0, 0)),
            (tile2, (1, 1)),
        ]

        merged = merge_tiled_masks(tiles, output_shape=(4, 4), overlap=1)

        # Should have 2 distinct cells
        unique = np.unique(merged)
        assert len(unique) == 3  # 0 + 2 cells


class TestSyntheticImageGeneration:
    """Tests for synthetic image generation."""

    def test_generate_image(self):
        """Test synthetic image generation."""
        from spatialgpu.benchmarks.synthetic import generate_image_with_cells

        image, masks = generate_image_with_cells(
            n_cells=10,
            image_size=(128, 128),
            seed=42,
        )

        assert image.shape == (128, 128)
        assert masks.shape == (128, 128)
        assert image.dtype == np.float32
        assert masks.dtype == np.int32

        # Should have approximately n_cells
        n_cells = len(np.unique(masks)) - 1  # Exclude 0
        assert 5 <= n_cells <= 10  # Some cells may not fit
