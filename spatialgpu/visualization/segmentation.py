"""
Segmentation visualization functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from spatialgpu.segmentation.core import SegmentationResult


def segmentation_overlay(
    image: NDArray,
    masks: NDArray | SegmentationResult,
    alpha: float = 0.4,
    colors: Optional[NDArray] = None,
    show_boundaries: bool = True,
    boundary_color: tuple = (1, 1, 1),
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10, 10),
    save: Optional[str] = None,
) -> Axes:
    """
    Overlay segmentation masks on an image.

    Parameters
    ----------
    image
        Input image, shape (H, W) or (H, W, C).
    masks
        Segmentation masks or SegmentationResult.
    alpha
        Transparency of mask overlay.
    colors
        Custom colors for each cell (n_cells, 3).
    show_boundaries
        Whether to show cell boundaries.
    boundary_color
        Color for boundaries (RGB tuple).
    ax
        Matplotlib axes.
    figsize
        Figure size.
    save
        Path to save figure.

    Returns
    -------
    Axes
        Matplotlib axes with overlay.

    Examples
    --------
    >>> result = sp.segmentation.segment_cells(image)
    >>> sp.viz.segmentation_overlay(image, result)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    from spatialgpu.segmentation.core import SegmentationResult

    if isinstance(masks, SegmentationResult):
        masks = masks.masks

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Normalize image for display
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / image.max()

    # Convert grayscale to RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Create mask overlay
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]
    n_cells = len(cell_ids)

    if colors is None:
        # Generate random colors
        hues = np.linspace(0, 1, n_cells, endpoint=False)
        np.random.shuffle(hues)
        colors = np.array([
            hsv_to_rgb([h, 0.8, 0.9])
            for h in hues
        ])

    # Create colored mask
    overlay = image.copy()

    for idx, cell_id in enumerate(cell_ids):
        cell_mask = masks == cell_id
        color = colors[idx % len(colors)]
        for c in range(3):
            overlay[:, :, c] = np.where(
                cell_mask,
                (1 - alpha) * image[:, :, c] + alpha * color[c],
                overlay[:, :, c],
            )

    # Add boundaries
    if show_boundaries:
        from spatialgpu.segmentation.utils import compute_boundaries
        boundaries = compute_boundaries(masks)
        for c in range(3):
            overlay[:, :, c] = np.where(
                boundaries,
                boundary_color[c],
                overlay[:, :, c],
            )

    ax.imshow(np.clip(overlay, 0, 1))
    ax.axis("off")
    ax.set_title(f"Segmentation ({n_cells} cells)")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def show_masks(
    masks: NDArray | SegmentationResult,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "nipy_spectral",
    save: Optional[str] = None,
) -> Axes:
    """
    Display segmentation masks.

    Parameters
    ----------
    masks
        Segmentation masks or SegmentationResult.
    ax
        Matplotlib axes.
    figsize
        Figure size.
    cmap
        Colormap.
    save
        Path to save figure.

    Returns
    -------
    Axes
        Matplotlib axes.

    Examples
    --------
    >>> result = sp.segmentation.segment_cells(image)
    >>> sp.viz.show_masks(result)
    """
    import matplotlib.pyplot as plt

    from spatialgpu.segmentation.core import SegmentationResult

    if isinstance(masks, SegmentationResult):
        n_cells = masks.n_cells
        masks = masks.masks
    else:
        n_cells = len(np.unique(masks)) - 1

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create display array with shuffled colors
    display = np.zeros_like(masks, dtype=np.float32)
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]

    # Shuffle colors
    colors = np.random.permutation(len(cell_ids)) + 1
    for idx, cell_id in enumerate(cell_ids):
        display[masks == cell_id] = colors[idx]

    ax.imshow(display, cmap=cmap)
    ax.axis("off")
    ax.set_title(f"Cell masks ({n_cells} cells)")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def show_boundaries(
    masks: NDArray | SegmentationResult,
    image: Optional[NDArray] = None,
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10, 10),
    boundary_color: str = "red",
    linewidth: float = 1.0,
    save: Optional[str] = None,
) -> Axes:
    """
    Display cell boundaries.

    Parameters
    ----------
    masks
        Segmentation masks or SegmentationResult.
    image
        Background image (optional).
    ax
        Matplotlib axes.
    figsize
        Figure size.
    boundary_color
        Color for boundaries.
    linewidth
        Line width for boundaries.
    save
        Path to save figure.

    Returns
    -------
    Axes
        Matplotlib axes.

    Examples
    --------
    >>> result = sp.segmentation.segment_cells(image)
    >>> sp.viz.show_boundaries(result, image=image)
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    from spatialgpu.segmentation.core import SegmentationResult
    from spatialgpu.segmentation.utils import compute_boundaries

    if isinstance(masks, SegmentationResult):
        masks = masks.masks

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Show image if provided
    if image is not None:
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / image.max()
        ax.imshow(image, cmap="gray")

    # Compute and show boundaries
    boundaries = compute_boundaries(masks)

    # Create RGBA array for boundaries
    color_rgb = mcolors.to_rgb(boundary_color)
    boundary_rgba = np.zeros((*boundaries.shape, 4))
    boundary_rgba[boundaries, :3] = color_rgb
    boundary_rgba[boundaries, 3] = 1.0

    ax.imshow(boundary_rgba)
    ax.axis("off")

    n_cells = len(np.unique(masks)) - 1
    ax.set_title(f"Cell boundaries ({n_cells} cells)")

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return ax


def compare_segmentations(
    image: NDArray,
    segmentations: Sequence[NDArray | SegmentationResult],
    labels: Sequence[str],
    figsize_per_image: tuple[float, float] = (6, 6),
    save: Optional[str] = None,
) -> Figure:
    """
    Compare multiple segmentations side by side.

    Parameters
    ----------
    image
        Input image.
    segmentations
        List of segmentation masks or results.
    labels
        Labels for each segmentation.
    figsize_per_image
        Figure size per subplot.
    save
        Path to save figure.

    Returns
    -------
    Figure
        Matplotlib figure.

    Examples
    --------
    >>> cellpose_result = sp.segmentation.segment_cells(image, model="cellpose")
    >>> stardist_result = sp.segmentation.segment_cells(image, model="stardist")
    >>> sp.viz.compare_segmentations(
    ...     image,
    ...     [cellpose_result, stardist_result],
    ...     labels=["Cellpose", "StarDist"],
    ... )
    """
    import matplotlib.pyplot as plt

    n = len(segmentations)
    figsize = (figsize_per_image[0] * (n + 1), figsize_per_image[1])

    fig, axes = plt.subplots(1, n + 1, figsize=figsize)

    # Show original image
    axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Show each segmentation
    for idx, (seg, label) in enumerate(zip(segmentations, labels)):
        segmentation_overlay(image, seg, ax=axes[idx + 1])
        axes[idx + 1].set_title(label)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")

    return fig
