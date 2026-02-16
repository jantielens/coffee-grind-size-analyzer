"""
Image preprocessing: perspective warp, marker masking, white-balance.

Takes a raw photo with detected markers and produces a normalised,
marker-free image ready for segmentation.
"""

from __future__ import annotations

import cv2
import numpy as np

from constants import (
    GROUNDS_H_MM,
    GROUNDS_W_MM,
    MARKER_IDS,
    MARKER_RECT_H_MM,
    MARKER_RECT_W_MM,
    MARKER_SIZE_MM,
    WARP_PX_PER_MM,
    _exclusion_rects,
    _marker_top_left_mm,
)


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def warp_to_canonical(
    image_bgr: np.ndarray,
    marker_centers: dict[int, np.ndarray],
) -> tuple[np.ndarray, float]:
    """Warp image so the reference area maps to a canonical rectangle.

    Returns (warped_bgr, mm_per_px).
    """
    half = MARKER_SIZE_MM / 2.0
    px = WARP_PX_PER_MM

    target = {
        0: np.array([half * px, half * px]),
        1: np.array([(half + MARKER_RECT_W_MM) * px, half * px]),
        2: np.array([half * px, (half + MARKER_RECT_H_MM) * px]),
        3: np.array([(half + MARKER_RECT_W_MM) * px, (half + MARKER_RECT_H_MM) * px]),
    }

    src_pts = np.float32([marker_centers[i] for i in MARKER_IDS])
    dst_pts = np.float32([target[i] for i in MARKER_IDS])

    H, _ = cv2.findHomography(src_pts, dst_pts)

    out_w = int(GROUNDS_W_MM * px)
    out_h = int(GROUNDS_H_MM * px)

    warped = cv2.warpPerspective(
        image_bgr, H, (out_w, out_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    mm_per_px = 1.0 / px
    return warped, mm_per_px


# ---------------------------------------------------------------------------
# Marker masking
# ---------------------------------------------------------------------------

def mask_markers(warped: np.ndarray) -> np.ndarray:
    """Paint white rectangles over each marker and the note box.

    Returns a copy with exclusion zones masked.
    """
    img = warped.copy()
    for x0, y0, x1, y1 in _exclusion_rects(warped.shape[:2]):
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return img


def get_analysis_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask (True = analysable region).

    Excludes marker squares, the note box, and padding around each.
    """
    h, w = shape_hw
    mask = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in _exclusion_rects(shape_hw):
        mask[y0:y1, x0:x1] = False
    return mask


# ---------------------------------------------------------------------------
# White-balance normalization
# ---------------------------------------------------------------------------

def normalize_white_balance(warped_bgr: np.ndarray) -> np.ndarray:
    """Normalize image contrast using markers (black) and paper (white).

    Each channel is linearly stretched so that black→0, white→255.
    Compensates for variable exposure / lighting across photos.
    """
    px = WARP_PX_PER_MM
    h, w = warped_bgr.shape[:2]

    # --- Sample black reference from marker interiors (inner 50%) ---
    black_pixels = []
    for mx, my in _marker_top_left_mm():
        inset = MARKER_SIZE_MM * 0.25
        x0 = max(0, int((mx + inset) * px))
        y0 = max(0, int((my + inset) * px))
        x1 = min(w, int((mx + MARKER_SIZE_MM - inset) * px))
        y1 = min(h, int((my + MARKER_SIZE_MM - inset) * px))
        patch = warped_bgr[y0:y1, x0:x1]
        if patch.size > 0:
            black_pixels.append(patch.reshape(-1, 3))

    # --- Sample white reference from paper border strips ---
    white_pixels = []

    # Top strip (between marker 0 and 1, just below marker row)
    strip_y0 = int(MARKER_SIZE_MM * px) + 5
    strip_y1 = strip_y0 + int(5 * px)
    strip_x0 = int(MARKER_SIZE_MM * px) + 5
    strip_x1 = int(MARKER_RECT_W_MM * px) - 5
    if strip_y1 < h and strip_x1 > strip_x0:
        white_pixels.append(warped_bgr[strip_y0:strip_y1, strip_x0:strip_x1].reshape(-1, 3))

    # Bottom strip (between marker 2 and 3, just above marker row)
    strip_y1b = int(MARKER_RECT_H_MM * px) - 5
    strip_y0b = strip_y1b - int(5 * px)
    if strip_y0b > 0 and strip_x1 > strip_x0:
        white_pixels.append(warped_bgr[strip_y0b:strip_y1b, strip_x0:strip_x1].reshape(-1, 3))

    # Left strip (between marker 0 and 2)
    ls_x0 = 0
    ls_x1 = int(2 * px)
    ls_y0 = int(MARKER_SIZE_MM * px) + 5
    ls_y1 = int(MARKER_RECT_H_MM * px) - 5
    if ls_y1 > ls_y0 and ls_x1 < w:
        white_pixels.append(warped_bgr[ls_y0:ls_y1, ls_x0:ls_x1].reshape(-1, 3))

    # Right strip (between marker 1 and 3)
    rs_x0 = int((MARKER_RECT_W_MM + MARKER_SIZE_MM) * px) - int(2 * px)
    rs_x1 = int((MARKER_RECT_W_MM + MARKER_SIZE_MM) * px)
    if ls_y1 > ls_y0 and rs_x0 > 0:
        white_pixels.append(warped_bgr[ls_y0:ls_y1, rs_x0:min(rs_x1, w)].reshape(-1, 3))

    if not black_pixels or not white_pixels:
        return warped_bgr  # fallback: no normalization

    black_ref = np.concatenate(black_pixels, axis=0)
    white_ref = np.concatenate(white_pixels, axis=0)

    # Use only genuinely bright pixels for white (avoid coffee contamination)
    white_brightness = np.mean(white_ref, axis=1)
    bright_mask = white_brightness > np.percentile(white_brightness, 50)
    white_ref_clean = white_ref[bright_mask] if bright_mask.sum() > 100 else white_ref

    black_val = np.percentile(black_ref, 10, axis=0).astype(np.float32)
    white_val = np.percentile(white_ref_clean, 90, axis=0).astype(np.float32)

    # Bilateral filter BEFORE stretch to suppress paper grain while low-amplitude
    smoothed = cv2.bilateralFilter(warped_bgr, d=9, sigmaColor=50, sigmaSpace=50)

    img_f = smoothed.astype(np.float32)
    denom = white_val - black_val
    denom[denom < 1.0] = 1.0

    normalized = (img_f - black_val) / denom * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return normalized
