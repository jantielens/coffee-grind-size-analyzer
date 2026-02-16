"""
Particle segmentation: watershed (primary) and Cellpose (optional comparison).

Takes a preprocessed RGB image and returns an integer label mask where
each connected particle region has a unique ID (0 = background).
"""

from __future__ import annotations

import cv2
import numpy as np

from constants import _relabel_sequential


# ---------------------------------------------------------------------------
# Cellpose (optional, for comparison)
# ---------------------------------------------------------------------------

def segment_cellpose(
    image_rgb: np.ndarray,
    mm_per_px: float,
    expected_diam_mm: float = 0.7,
) -> np.ndarray:
    """Segment particles using Cellpose (cyto3 model) on an inverted image.

    Coffee particles are dark on white paper; Cellpose expects bright on dark.
    We invert the image so particles become bright blobs.

    Args:
        image_rgb: RGB image of the analysis region.
        mm_per_px: Scale factor.
        expected_diam_mm: Diameter hint in mm (default 0.7).

    Returns:
        Integer label mask (0 = background).
    """
    from cellpose import models as cp_models

    # cyto3 model — cpsam (default) finds 0 particles on coffee images.
    model = cp_models.CellposeModel(model_type="cyto3", gpu=True)

    inverted = 255 - image_rgb
    diam_px = expected_diam_mm / mm_per_px

    result = model.eval(
        inverted,
        diameter=diam_px,
        flow_threshold=0.8,
        cellprob_threshold=0.0,
        channels=[0, 0],
    )

    return result[0]


# ---------------------------------------------------------------------------
# Watershed (primary pipeline)
# ---------------------------------------------------------------------------

def segment_watershed(
    image_rgb: np.ndarray,
    mm_per_px: float,
    return_intermediates: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Segment particles using adaptive threshold + watershed.

    Args:
        image_rgb: RGB image of the analysis region.
        mm_per_px: Scale factor.
        return_intermediates: If True, also return intermediate images
            for pipeline visualisation.

    Returns:
        Integer label mask (0 = background), or (labels, intermediates) tuple.
    """
    # Combine inverted lightness + saturation → dark & coloured = particle
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_inv = 255 - lab[:, :, 0]
    sat = hsv[:, :, 1]
    combined = cv2.addWeighted(l_inv, 0.6, sat, 0.4, 0)

    # Adaptive threshold — blockSize=71, C=-8 chosen via sweep (r=0.923)
    binary = cv2.adaptiveThreshold(
        combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=71, C=-8,
    )

    # Morphological clean-up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Sure background via dilation
    sure_bg = cv2.dilate(binary, kernel_med, iterations=3)

    # Distance transform → sure foreground (dist_frac=0.40 via sweep, r=0.953)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.40 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components for markers
    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1       # background = 1
    markers[unknown == 255] = 0  # unknown = 0

    # Watershed
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.watershed(img_bgr, markers)

    # Clean up: boundaries (-1) and background (1) → 0
    result = markers.copy()
    result[result <= 1] = 0
    result[result == -1] = 0

    labels = _relabel_sequential(result).astype(np.int32)

    if return_intermediates:
        return labels, {"binary": binary, "dist_transform": dist, "sure_fg": sure_fg}

    return labels
