"""
Core cell segmentation functionality.

Provides a unified interface for cell segmentation with GPU acceleration,
tiled processing, and multi-model support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SegmentationResult:
    """
    Results from cell segmentation.

    Attributes
    ----------
    masks : array
        Integer mask array where each cell has a unique ID (0 = background).
    n_cells : int
        Number of segmented cells.
    cell_ids : array
        Unique cell IDs (excluding background).
    centroids : array
        Cell centroid coordinates, shape (n_cells, 2).
    areas : array
        Cell areas in pixels.
    boundaries : array, optional
        Cell boundary mask (True at boundaries).
    flows : array, optional
        Flow/gradient fields from flow-based models.
    probabilities : array, optional
        Cell probability map.
    """

    masks: NDArray
    n_cells: int
    cell_ids: NDArray
    centroids: NDArray
    areas: NDArray
    boundaries: Optional[NDArray] = None
    flows: Optional[NDArray] = None
    probabilities: Optional[NDArray] = None
    model_name: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_masks(
        cls, masks: NDArray, model_name: str = "unknown"
    ) -> SegmentationResult:
        """Create SegmentationResult from mask array."""
        from spatialgpu.segmentation.utils import (
            compute_areas,
            compute_boundaries,
            compute_centroids,
        )

        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]  # Remove background

        return cls(
            masks=masks,
            n_cells=len(cell_ids),
            cell_ids=cell_ids,
            centroids=compute_centroids(masks, cell_ids),
            areas=compute_areas(masks, cell_ids),
            boundaries=compute_boundaries(masks),
            model_name=model_name,
        )


class BaseSegmentationModel(ABC):
    """
    Abstract base class for segmentation models.

    All segmentation models should inherit from this class and implement
    the `segment` method.
    """

    name: str = "base"
    supports_gpu: bool = False

    def __init__(self, device: str = "auto", **kwargs):
        """
        Initialize segmentation model.

        Parameters
        ----------
        device : str
            Device to use ("auto", "cpu", "cuda", "cuda:0", etc.)
        **kwargs
            Model-specific parameters.
        """
        self.device = self._resolve_device(device)
        self.model = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @abstractmethod
    def segment(
        self,
        image: NDArray,
        diameter: Optional[float] = None,
        channels: Optional[list[int]] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment cells in an image.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        diameter
            Expected cell diameter in pixels.
        channels
            Channel indices to use.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        SegmentationResult
            Segmentation results.
        """
        pass

    def segment_tiled(
        self,
        image: NDArray,
        tile_size: int = 2048,
        overlap: int = 128,
        diameter: Optional[float] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment large image using tiled processing.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        tile_size
            Size of each tile.
        overlap
            Overlap between tiles.
        diameter
            Expected cell diameter in pixels.
        show_progress
            Show progress bar.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        SegmentationResult
            Merged segmentation results.
        """
        from spatialgpu.segmentation.utils import merge_tiled_masks

        h, w = image.shape[:2]

        # Compute tile positions
        tiles = []
        positions = []

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tiles.append((y, x, y_end, x_end))
                positions.append((y, x))

        # Process tiles
        tile_masks = []

        iterator = tiles
        if show_progress:
            iterator = tqdm(tiles, desc="Segmenting tiles")

        for y, x, y_end, x_end in iterator:
            tile = image[y:y_end, x:x_end]
            result = self.segment(tile, diameter=diameter, **kwargs)
            tile_masks.append((result.masks, (y, x)))

        # Merge tiles
        merged_masks = merge_tiled_masks(
            tile_masks,
            output_shape=(h, w),
            overlap=overlap,
        )

        return SegmentationResult.from_masks(merged_masks, model_name=self.name)


class CellSegmenter:
    """
    High-level cell segmentation interface.

    Provides a unified API for cell segmentation with automatic model
    selection, tiled processing, and GPU acceleration.

    Examples
    --------
    >>> from spatialgpu.segmentation import CellSegmenter
    >>> segmenter = CellSegmenter(model="cellpose", device="cuda")
    >>> result = segmenter.segment(image, diameter=30)
    >>> print(f"Found {result.n_cells} cells")

    >>> # Segment large image with tiling
    >>> result = segmenter.segment_tiled(large_image, tile_size=2048)
    """

    def __init__(
        self,
        model: Union[str, BaseSegmentationModel] = "cellpose",
        device: str = "auto",
        **model_kwargs,
    ):
        """
        Initialize cell segmenter.

        Parameters
        ----------
        model : str or BaseSegmentationModel
            Segmentation model to use. Options: "cellpose", "stardist", or
            a custom BaseSegmentationModel instance.
        device : str
            Device to use ("auto", "cpu", "cuda").
        **model_kwargs
            Additional model-specific parameters.
        """
        if isinstance(model, BaseSegmentationModel):
            self.model = model
        else:
            self.model = self._load_model(model, device, **model_kwargs)

    def _load_model(
        self,
        model_name: str,
        device: str,
        **kwargs,
    ) -> BaseSegmentationModel:
        """Load segmentation model by name."""
        from spatialgpu.segmentation.models import (
            CellposeModel,
            StarDistModel,
        )

        models = {
            "cellpose": CellposeModel,
            "stardist": StarDistModel,
        }

        if model_name.lower() not in models:
            available = ", ".join(models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        return models[model_name.lower()](device=device, **kwargs)

    def segment(
        self,
        image: NDArray,
        diameter: Optional[float] = None,
        channels: Optional[list[int]] = None,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment cells in an image.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        diameter
            Expected cell diameter in pixels. If None, model will estimate.
        channels
            Channel indices [cytoplasm, nucleus] for multi-channel images.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        SegmentationResult
            Segmentation results including masks, centroids, areas.
        """
        return self.model.segment(image, diameter=diameter, channels=channels, **kwargs)

    def segment_tiled(
        self,
        image: NDArray,
        tile_size: int = 2048,
        overlap: int = 128,
        diameter: Optional[float] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> SegmentationResult:
        """
        Segment large image using tiled processing.

        Splits image into overlapping tiles, processes each tile, and
        merges results by resolving cells at tile boundaries.

        Parameters
        ----------
        image
            Input image, shape (H, W) or (H, W, C).
        tile_size
            Size of each tile in pixels.
        overlap
            Overlap between adjacent tiles in pixels.
        diameter
            Expected cell diameter in pixels.
        show_progress
            Show progress bar during processing.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        SegmentationResult
            Merged segmentation results.
        """
        return self.model.segment_tiled(
            image,
            tile_size=tile_size,
            overlap=overlap,
            diameter=diameter,
            show_progress=show_progress,
            **kwargs,
        )

    def segment_batch(
        self,
        images: Sequence[NDArray],
        diameter: Optional[float] = None,
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs,
    ) -> list[SegmentationResult]:
        """
        Segment multiple images in batch for GPU efficiency.

        Parameters
        ----------
        images
            List of input images.
        diameter
            Expected cell diameter (applied to all images).
        batch_size
            Number of images to process in parallel on GPU.
        show_progress
            Show progress bar.
        **kwargs
            Model-specific parameters.

        Returns
        -------
        list[SegmentationResult]
            Segmentation results for each image.
        """
        results = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Segmenting batches",
                total=len(images) // batch_size + 1,
            )

        for i in iterator:
            batch = images[i : i + batch_size]
            for img in batch:
                result = self.segment(img, diameter=diameter, **kwargs)
                results.append(result)

        return results


def segment_cells(
    image: NDArray,
    model: str = "cellpose",
    diameter: Optional[float] = None,
    channels: Optional[list[int]] = None,
    device: str = "auto",
    tile_size: Optional[int] = None,
    **kwargs,
) -> SegmentationResult:
    """
    Segment cells from an image.

    Convenience function that creates a CellSegmenter and runs segmentation.

    Parameters
    ----------
    image
        Input image, shape (H, W) or (H, W, C).
    model
        Segmentation model to use ("cellpose", "stardist").
    diameter
        Expected cell diameter in pixels.
    channels
        Channel indices for multi-channel images.
    device
        Device to use ("auto", "cpu", "cuda").
    tile_size
        If set, use tiled processing with this tile size.
    **kwargs
        Additional model-specific parameters.

    Returns
    -------
    SegmentationResult
        Segmentation results.

    Examples
    --------
    >>> import spatialgpu as sp
    >>> from skimage import io
    >>> image = io.imread("cells.tif")
    >>> result = sp.segmentation.segment_cells(image, diameter=30)
    >>> print(f"Segmented {result.n_cells} cells")
    """
    segmenter = CellSegmenter(model=model, device=device)

    if tile_size is not None:
        return segmenter.segment_tiled(
            image, tile_size=tile_size, diameter=diameter, **kwargs
        )

    return segmenter.segment(image, diameter=diameter, channels=channels, **kwargs)


def segment_nuclei(
    image: NDArray,
    model: str = "cellpose",
    diameter: Optional[float] = None,
    device: str = "auto",
    **kwargs,
) -> SegmentationResult:
    """
    Segment nuclei from an image.

    Parameters
    ----------
    image
        Input image (nuclear stain), shape (H, W).
    model
        Segmentation model to use.
    diameter
        Expected nucleus diameter in pixels.
    device
        Device to use.
    **kwargs
        Additional parameters.

    Returns
    -------
    SegmentationResult
        Segmentation results.

    Examples
    --------
    >>> result = sp.segmentation.segment_nuclei(dapi_image, diameter=15)
    """
    segmenter = CellSegmenter(model=model, device=device, model_type="nuclei")
    return segmenter.segment(image, diameter=diameter, **kwargs)
