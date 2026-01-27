#!/usr/bin/env python
"""
Cell Segmentation Example
=========================

This example demonstrates GPU-accelerated cell segmentation using
spatial-gpu's unified segmentation interface.
"""

import numpy as np

import spatialgpu as sp
from spatialgpu.segmentation import (
    segment_cells,
    CellSegmenter,
    evaluate_segmentation,
)
from spatialgpu.benchmarks.synthetic import generate_image_with_cells


def main():
    print("=" * 60)
    print("Cell Segmentation Example")
    print("=" * 60)

    # Check GPU availability
    backend = sp.get_backend()
    device = "cuda" if backend.is_gpu_available else "cpu"
    print(f"\nUsing device: {device}")

    # ============================================
    # Generate Synthetic Data
    # ============================================

    print("\n" + "-" * 40)
    print("Generating synthetic image with cells...")
    print("-" * 40)

    image, ground_truth = generate_image_with_cells(
        n_cells=100,
        image_size=(512, 512),
        cell_radius_range=(15, 35),
        noise_level=0.1,
        seed=42,
    )

    n_true_cells = len(np.unique(ground_truth)) - 1
    print(f"Image size: {image.shape}")
    print(f"True number of cells: {n_true_cells}")

    # ============================================
    # Basic Segmentation
    # ============================================

    print("\n" + "-" * 40)
    print("Running cell segmentation...")
    print("-" * 40)

    # Note: This requires Cellpose to be installed
    try:
        result = segment_cells(
            image,
            model="cellpose",
            diameter=30,
            device=device,
        )

        print(f"Detected cells: {result.n_cells}")
        print(f"Cell areas (mean): {np.mean(result.areas):.1f} pixels")
        print(f"Cell areas (range): [{np.min(result.areas):.0f}, {np.max(result.areas):.0f}]")

        # Evaluate against ground truth
        metrics = evaluate_segmentation(result, ground_truth, iou_threshold=0.5)

        print(f"\nEvaluation Metrics:")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1 Score: {metrics.f1:.3f}")
        print(f"  Mean IoU: {metrics.mean_matched_iou:.3f}")
        print(f"  True positives: {metrics.n_true_positive}")
        print(f"  False positives: {metrics.n_false_positive}")
        print(f"  False negatives: {metrics.n_false_negative}")

    except ImportError as e:
        print(f"Segmentation model not available: {e}")
        print("Install with: pip install spatial-gpu[segmentation]")

    # ============================================
    # Tiled Segmentation for Large Images
    # ============================================

    print("\n" + "-" * 40)
    print("Demonstrating tiled segmentation...")
    print("-" * 40)

    # Generate larger image
    large_image, _ = generate_image_with_cells(
        n_cells=500,
        image_size=(2048, 2048),
        cell_radius_range=(15, 35),
        noise_level=0.1,
        seed=123,
    )

    print(f"Large image size: {large_image.shape}")

    try:
        segmenter = CellSegmenter(model="cellpose", device=device)

        result = segmenter.segment_tiled(
            large_image,
            tile_size=512,
            overlap=64,
            diameter=30,
            show_progress=True,
        )

        print(f"Detected cells in large image: {result.n_cells}")

    except ImportError as e:
        print(f"Segmentation model not available: {e}")

    # ============================================
    # Segmentation Quality Metrics
    # ============================================

    print("\n" + "-" * 40)
    print("Computing quality metrics...")
    print("-" * 40)

    from spatialgpu.segmentation.evaluation import compute_quality_metrics

    # Use our synthetic ground truth for demo
    from spatialgpu.segmentation.core import SegmentationResult
    result = SegmentationResult.from_masks(ground_truth)

    quality = compute_quality_metrics(result, image)

    print(f"Number of cells: {quality['n_cells']}")
    print(f"Mean area: {quality['mean_area']:.1f} pixels")
    print(f"Std area: {quality['std_area']:.1f} pixels")
    print(f"Mean circularity: {quality['mean_circularity']:.3f}")
    print(f"Coverage: {quality['coverage']*100:.1f}%")
    print(f"Mean intensity: {quality['mean_intensity']:.3f}")

    # ============================================
    # Save Results
    # ============================================

    print("\n" + "-" * 40)
    print("Example complete!")
    print("-" * 40)

    print("\nTo visualize results:")
    print("  sp.viz.segmentation_overlay(image, result)")
    print("  sp.viz.show_masks(result)")
    print("  sp.viz.show_boundaries(result, image=image)")


if __name__ == "__main__":
    main()
