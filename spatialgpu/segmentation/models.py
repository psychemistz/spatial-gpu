"""
Segmentation model implementations.

Provides GPU-accelerated wrappers for popular segmentation models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from spatialgpu.segmentation.core import BaseSegmentationModel, SegmentationResult


def get_available_models() -> dict[str, bool]:
    """
    Get available segmentation models and their GPU support status.

    Returns
    -------
    dict
        Model name -> GPU support status.
    """
    models = {}

    # Check Cellpose
    try:
        import cellpose
        models["cellpose"] = True
    except ImportError:
        models["cellpose"] = False

    # Check StarDist
    try:
        import stardist
        models["stardist"] = True
    except ImportError:
        models["stardist"] = False

    # Check SAM
    try:
        import segment_anything
        models["sam"] = True
    except ImportError:
        models["sam"] = False

    return models


class CellposeModel(BaseSegmentationModel):
    """
    Cellpose segmentation model wrapper.

    Cellpose is a generalist, deep learning-based segmentation method
    that can segment cells in diverse image types.

    Parameters
    ----------
    device : str
        Device to use ("auto", "cpu", "cuda").
    model_type : str
        Cellpose model type ("cyto", "cyto2", "nuclei", etc.).
    gpu : bool, optional
        Force GPU usage. If None, auto-detect.
    **kwargs
        Additional Cellpose parameters.

    Examples
    --------
    >>> model = CellposeModel(device="cuda", model_type="cyto2")
    >>> result = model.segment(image, diameter=30)
    """

    name = "cellpose"
    supports_gpu = True

    def __init__(
        self,
        device: str = "auto",
        model_type: str = "cyto2",
        gpu: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(device=device)
        self.model_type = model_type
        self.kwargs = kwargs

        # Determine GPU usage
        if gpu is None:
            gpu = self.device.startswith("cuda")

        self._gpu = gpu
        self._model = None

    def _get_model(self):
        """Lazy-load Cellpose model."""
        if self._model is None:
            try:
                from cellpose import models
            except ImportError:
                raise ImportError(
                    "Cellpose not installed. Install with: "
                    "pip install spatial-gpu[segmentation]"
                )

            self._model = models.Cellpose(
                model_type=self.model_type,
                gpu=self._gpu,
            )

        return self._model

    def segment(
        self,
        image: NDArray,
        diameter: Optional[float] = None,
        channels: Optional[list[int]] = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment cells using Cellpose.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        diameter
            Expected cell diameter. If None, Cellpose estimates it.
        channels
            Channel indices [cytoplasm, nucleus]. Default [0, 0] for grayscale.
        flow_threshold
            Flow error threshold for mask filtering.
        cellprob_threshold
            Cell probability threshold.
        **kwargs
            Additional Cellpose.eval() parameters.

        Returns
        -------
        SegmentationResult
            Segmentation results.
        """
        model = self._get_model()

        # Handle channel specification
        if channels is None:
            if image.ndim == 2:
                channels = [0, 0]  # Grayscale
            else:
                channels = [0, 0]  # Use first channel

        # Run Cellpose
        masks, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            **kwargs,
        )

        result = SegmentationResult.from_masks(masks, model_name="cellpose")
        result.flows = flows[0] if flows else None
        result.probabilities = flows[2] if flows and len(flows) > 2 else None
        result.metadata["diameter"] = diams

        return result


class StarDistModel(BaseSegmentationModel):
    """
    StarDist segmentation model wrapper.

    StarDist detects cells with star-convex polygon shapes, especially
    effective for round cells like nuclei.

    Parameters
    ----------
    device : str
        Device to use ("auto", "cpu", "cuda").
    model_name : str
        StarDist model name ("2D_versatile_fluo", "2D_versatile_he", etc.).
    **kwargs
        Additional StarDist parameters.

    Examples
    --------
    >>> model = StarDistModel(model_name="2D_versatile_fluo")
    >>> result = model.segment(image)
    """

    name = "stardist"
    supports_gpu = True

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "2D_versatile_fluo",
        **kwargs,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.kwargs = kwargs
        self._model = None

    def _get_model(self):
        """Lazy-load StarDist model."""
        if self._model is None:
            try:
                from stardist.models import StarDist2D
            except ImportError:
                raise ImportError(
                    "StarDist not installed. Install with: "
                    "pip install spatial-gpu[segmentation]"
                )

            self._model = StarDist2D.from_pretrained(self.model_name)

        return self._model

    def segment(
        self,
        image: NDArray,
        diameter: Optional[float] = None,
        channels: Optional[list[int]] = None,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment cells using StarDist.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        diameter
            Not used by StarDist (included for API consistency).
        channels
            Not used by StarDist (uses all channels).
        prob_thresh
            Probability threshold for cell detection.
        nms_thresh
            Non-maximum suppression threshold.
        **kwargs
            Additional StarDist parameters.

        Returns
        -------
        SegmentationResult
            Segmentation results.
        """
        model = self._get_model()

        # Normalize image
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            if image.max() > 1:
                image = image / image.max()

        # Run StarDist
        labels, details = model.predict_instances(
            image,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            **kwargs,
        )

        result = SegmentationResult.from_masks(labels, model_name="stardist")
        result.probabilities = details.get("prob", None)
        result.metadata["details"] = details

        return result


class EnsembleModel(BaseSegmentationModel):
    """
    Ensemble of multiple segmentation models.

    Combines predictions from multiple models using voting or averaging.

    Parameters
    ----------
    models : list
        List of BaseSegmentationModel instances.
    method : str
        Combination method ("vote", "union", "intersection").

    Examples
    --------
    >>> models = [CellposeModel(), StarDistModel()]
    >>> ensemble = EnsembleModel(models, method="vote")
    >>> result = ensemble.segment(image)
    """

    name = "ensemble"
    supports_gpu = True

    def __init__(
        self,
        models: list[BaseSegmentationModel],
        method: str = "vote",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.models = models
        self.method = method

    def segment(
        self,
        image: NDArray,
        diameter: Optional[float] = None,
        channels: Optional[list[int]] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment using ensemble of models.

        Parameters
        ----------
        image
            Input image.
        diameter
            Expected cell diameter.
        channels
            Channel specification.
        **kwargs
            Additional parameters.

        Returns
        -------
        SegmentationResult
            Combined segmentation results.
        """
        # Get predictions from all models
        results = []
        for model in self.models:
            try:
                result = model.segment(
                    image, diameter=diameter, channels=channels, **kwargs
                )
                results.append(result)
            except Exception as e:
                import warnings
                warnings.warn(f"Model {model.name} failed: {e}")

        if not results:
            raise RuntimeError("All models failed")

        # Combine predictions
        if self.method == "vote":
            combined_masks = self._combine_by_voting(
                [r.masks for r in results]
            )
        elif self.method == "union":
            combined_masks = self._combine_by_union(
                [r.masks for r in results]
            )
        elif self.method == "intersection":
            combined_masks = self._combine_by_intersection(
                [r.masks for r in results]
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return SegmentationResult.from_masks(combined_masks, model_name="ensemble")

    def _combine_by_voting(self, masks_list: list[NDArray]) -> NDArray:
        """Combine masks by voting at each pixel."""
        from scipy import ndimage

        # Create binary masks
        binary = [m > 0 for m in masks_list]
        votes = np.sum(binary, axis=0)

        # Threshold at majority
        threshold = len(masks_list) // 2 + 1
        consensus = votes >= threshold

        # Label connected components
        labeled, _ = ndimage.label(consensus)
        return labeled

    def _combine_by_union(self, masks_list: list[NDArray]) -> NDArray:
        """Combine masks by union (any model detects a cell)."""
        from scipy import ndimage

        union = np.zeros(masks_list[0].shape, dtype=bool)
        for masks in masks_list:
            union |= masks > 0

        labeled, _ = ndimage.label(union)
        return labeled

    def _combine_by_intersection(self, masks_list: list[NDArray]) -> NDArray:
        """Combine masks by intersection (all models agree)."""
        from scipy import ndimage

        intersection = np.ones(masks_list[0].shape, dtype=bool)
        for masks in masks_list:
            intersection &= masks > 0

        labeled, _ = ndimage.label(intersection)
        return labeled
