"""
Segmentation evaluation metrics.

Provides functions to evaluate segmentation quality against ground truth
or using unsupervised quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from spatialgpu.segmentation.core import SegmentationResult


@dataclass
class SegmentationMetrics:
    """
    Segmentation evaluation metrics.

    Attributes
    ----------
    iou : float
        Mean Intersection over Union (Jaccard index).
    precision : float
        Detection precision (correctly detected / total detected).
    recall : float
        Detection recall (correctly detected / total ground truth).
    f1 : float
        F1 score (harmonic mean of precision and recall).
    n_true_positive : int
        Number of correctly detected cells.
    n_false_positive : int
        Number of spuriously detected cells.
    n_false_negative : int
        Number of missed cells.
    mean_matched_iou : float
        Mean IoU for matched cell pairs.
    per_cell_iou : array
        IoU for each matched cell pair.
    """

    iou: float
    precision: float
    recall: float
    f1: float
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    mean_matched_iou: float
    per_cell_iou: Optional[NDArray] = None


def evaluate_segmentation(
    predicted: SegmentationResult | NDArray,
    ground_truth: SegmentationResult | NDArray,
    iou_threshold: float = 0.5,
) -> SegmentationMetrics:
    """
    Evaluate segmentation against ground truth.

    Parameters
    ----------
    predicted
        Predicted segmentation (SegmentationResult or mask array).
    ground_truth
        Ground truth segmentation (SegmentationResult or mask array).
    iou_threshold
        IoU threshold for considering a detection as true positive.

    Returns
    -------
    SegmentationMetrics
        Evaluation metrics.

    Examples
    --------
    >>> metrics = sp.segmentation.evaluate_segmentation(
    ...     predicted_result, ground_truth_masks
    ... )
    >>> print(f"F1 Score: {metrics.f1:.3f}")
    >>> print(f"Mean IoU: {metrics.iou:.3f}")
    """
    # Extract masks if SegmentationResult
    if isinstance(predicted, SegmentationResult):
        pred_masks = predicted.masks
    else:
        pred_masks = predicted

    if isinstance(ground_truth, SegmentationResult):
        gt_masks = ground_truth.masks
    else:
        gt_masks = ground_truth

    return compute_segmentation_metrics(pred_masks, gt_masks, iou_threshold)


def compute_segmentation_metrics(
    pred_masks: NDArray,
    gt_masks: NDArray,
    iou_threshold: float = 0.5,
) -> SegmentationMetrics:
    """
    Compute segmentation metrics from mask arrays.

    Parameters
    ----------
    pred_masks
        Predicted mask array (integer labels).
    gt_masks
        Ground truth mask array (integer labels).
    iou_threshold
        IoU threshold for true positive detection.

    Returns
    -------
    SegmentationMetrics
        Computed metrics.
    """
    from spatialgpu.core.backend import get_backend

    backend = get_backend()

    # Get unique labels
    pred_ids = np.unique(pred_masks)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(gt_masks)
    gt_ids = gt_ids[gt_ids != 0]

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_pred == 0 and n_gt == 0:
        return SegmentationMetrics(
            iou=1.0, precision=1.0, recall=1.0, f1=1.0,
            n_true_positive=0, n_false_positive=0, n_false_negative=0,
            mean_matched_iou=1.0, per_cell_iou=np.array([]),
        )

    if n_pred == 0:
        return SegmentationMetrics(
            iou=0.0, precision=0.0, recall=0.0, f1=0.0,
            n_true_positive=0, n_false_positive=0, n_false_negative=n_gt,
            mean_matched_iou=0.0, per_cell_iou=np.array([]),
        )

    if n_gt == 0:
        return SegmentationMetrics(
            iou=0.0, precision=0.0, recall=0.0, f1=0.0,
            n_true_positive=0, n_false_positive=n_pred, n_false_negative=0,
            mean_matched_iou=0.0, per_cell_iou=np.array([]),
        )

    # Compute IoU matrix
    if backend.is_gpu_active:
        iou_matrix = _compute_iou_matrix_gpu(pred_masks, gt_masks, pred_ids, gt_ids)
    else:
        iou_matrix = _compute_iou_matrix_cpu(pred_masks, gt_masks, pred_ids, gt_ids)

    # Match predicted to ground truth (greedy matching)
    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    # Sort by IoU for greedy matching
    while True:
        # Find best unmatched pair
        best_iou = 0
        best_pred_idx = -1
        best_gt_idx = -1

        for i in range(n_pred):
            if i in matched_pred:
                continue
            for j in range(n_gt):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_pred_idx = i
                    best_gt_idx = j

        if best_iou < iou_threshold:
            break

        matched_pred.add(best_pred_idx)
        matched_gt.add(best_gt_idx)
        matched_ious.append(best_iou)

    # Compute metrics
    n_tp = len(matched_pred)
    n_fp = n_pred - n_tp
    n_fn = n_gt - len(matched_gt)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Mean IoU
    if matched_ious:
        mean_matched_iou = np.mean(matched_ious)
    else:
        mean_matched_iou = 0.0

    # Overall IoU (global)
    pred_binary = pred_masks > 0
    gt_binary = gt_masks > 0
    intersection = np.sum(pred_binary & gt_binary)
    union = np.sum(pred_binary | gt_binary)
    global_iou = intersection / union if union > 0 else 0

    return SegmentationMetrics(
        iou=global_iou,
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_positive=n_tp,
        n_false_positive=n_fp,
        n_false_negative=n_fn,
        mean_matched_iou=mean_matched_iou,
        per_cell_iou=np.array(matched_ious),
    )


def _compute_iou_matrix_cpu(
    pred_masks: NDArray,
    gt_masks: NDArray,
    pred_ids: NDArray,
    gt_ids: NDArray,
) -> NDArray:
    """Compute IoU matrix on CPU."""
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)

    for i, pred_id in enumerate(pred_ids):
        pred_mask = pred_masks == pred_id
        for j, gt_id in enumerate(gt_ids):
            gt_mask = gt_masks == gt_id

            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)

            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix


def _compute_iou_matrix_gpu(
    pred_masks: NDArray,
    gt_masks: NDArray,
    pred_ids: NDArray,
    gt_ids: NDArray,
) -> NDArray:
    """Compute IoU matrix on GPU."""
    from spatialgpu.core.array_utils import to_gpu, to_cpu
    import cupy as cp

    pred_masks_gpu = to_gpu(pred_masks)
    gt_masks_gpu = to_gpu(gt_masks)
    pred_ids_gpu = to_gpu(pred_ids)
    gt_ids_gpu = to_gpu(gt_ids)

    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    iou_matrix = cp.zeros((n_pred, n_gt), dtype=cp.float32)

    for i, pred_id in enumerate(pred_ids_gpu):
        pred_mask = pred_masks_gpu == pred_id
        for j, gt_id in enumerate(gt_ids_gpu):
            gt_mask = gt_masks_gpu == gt_id

            intersection = cp.sum(pred_mask & gt_mask)
            union = cp.sum(pred_mask | gt_mask)

            if union > 0:
                iou_matrix[i, j] = intersection / union

    return to_cpu(iou_matrix)


def compute_quality_metrics(
    segmentation: SegmentationResult,
    image: Optional[NDArray] = None,
) -> dict:
    """
    Compute unsupervised segmentation quality metrics.

    Parameters
    ----------
    segmentation
        Segmentation result.
    image
        Original image (optional, for intensity-based metrics).

    Returns
    -------
    dict
        Quality metrics including:
        - n_cells: Number of cells
        - mean_area: Mean cell area
        - std_area: Std deviation of cell area
        - mean_circularity: Mean cell circularity
        - coverage: Fraction of image covered by cells
    """
    from spatialgpu.segmentation.utils import compute_circularity

    masks = segmentation.masks
    areas = segmentation.areas

    h, w = masks.shape
    total_pixels = h * w

    # Coverage
    coverage = np.sum(masks > 0) / total_pixels

    # Circularity
    circularity = compute_circularity(masks, segmentation.cell_ids)

    metrics = {
        "n_cells": segmentation.n_cells,
        "mean_area": np.mean(areas) if len(areas) > 0 else 0,
        "std_area": np.std(areas) if len(areas) > 0 else 0,
        "min_area": np.min(areas) if len(areas) > 0 else 0,
        "max_area": np.max(areas) if len(areas) > 0 else 0,
        "mean_circularity": np.mean(circularity) if len(circularity) > 0 else 0,
        "coverage": coverage,
    }

    # Add intensity-based metrics if image provided
    if image is not None:
        cell_intensities = []
        for cid in segmentation.cell_ids:
            cell_mask = masks == cid
            cell_intensity = image[cell_mask].mean()
            cell_intensities.append(cell_intensity)

        metrics["mean_intensity"] = np.mean(cell_intensities) if cell_intensities else 0
        metrics["std_intensity"] = np.std(cell_intensities) if cell_intensities else 0

    return metrics
