"""
Shared constants and geometry helpers for the coffee grind analysis pipeline.

Constants must match generate-reference-sheet.py.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reference sheet geometry (must match generate-reference-sheet.py)
# ---------------------------------------------------------------------------

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
MARKER_SIZE_MM = 15
MARKER_RECT_W_MM = 55   # center-to-center horizontal
MARKER_RECT_H_MM = 80   # center-to-center vertical
GROUNDS_W_MM = MARKER_RECT_W_MM + MARKER_SIZE_MM  # 70 mm
GROUNDS_H_MM = MARKER_RECT_H_MM + MARKER_SIZE_MM  # 95 mm

# Warped output resolution
WARP_PX_PER_MM = 20  # 20 px/mm → 50 µm per pixel

# Particle size filters (equivalent diameter in mm)
MIN_PARTICLE_DIAM_MM = 0.05   # ignore dust < 50 µm
MAX_PARTICLE_DIAM_MM = 5.0    # ignore blobs > 5 mm (likely clusters)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Padding (mm) added around markers and note box to avoid edge artefacts
_MASK_PAD_MM = 3


# ---------------------------------------------------------------------------
# Shared geometry helpers (used by masking, white-balance, detection)
# ---------------------------------------------------------------------------

def _marker_top_left_mm() -> list[tuple[float, float]]:
    """Marker top-left corners in mm within the warped grounds zone."""
    return [
        (0, 0),
        (MARKER_RECT_W_MM, 0),
        (0, MARKER_RECT_H_MM),
        (MARKER_RECT_W_MM, MARKER_RECT_H_MM),
    ]


def _exclusion_rects(shape_hw: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) pixel rects for all exclusion zones.

    Covers the four markers and the note box, each padded by _MASK_PAD_MM.
    """
    h, w = shape_hw
    px = WARP_PX_PER_MM
    pad = _MASK_PAD_MM
    rects = []

    for mx, my in _marker_top_left_mm():
        x0 = max(0, int((mx - pad) * px))
        y0 = max(0, int((my - pad) * px))
        x1 = min(w, int((mx + MARKER_SIZE_MM + pad) * px))
        y1 = min(h, int((my + MARKER_SIZE_MM + pad) * px))
        rects.append((x0, y0, x1, y1))

    # Note box: 15×15 mm to the left of marker 3
    nb_x0 = max(0, int((MARKER_RECT_W_MM - MARKER_SIZE_MM - pad) * px))
    nb_y0 = max(0, int((MARKER_RECT_H_MM - pad) * px))
    nb_x1 = min(w, int((MARKER_RECT_W_MM + pad) * px))
    nb_y1 = min(h, int((MARKER_RECT_H_MM + MARKER_SIZE_MM + pad) * px))
    rects.append((nb_x0, nb_y0, nb_x1, nb_y1))

    return rects


def _extract_marker_centers(
    corners: tuple, ids: np.ndarray,
) -> dict[int, np.ndarray] | None:
    """Extract {id: center_xy} from ArUco detection results.

    Returns the dict if all 4 markers found, else None.
    """
    if ids is None:
        return None
    centers: dict[int, np.ndarray] = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in MARKER_IDS:
            centers[int(marker_id)] = corners[i][0].mean(axis=0)
    return centers if len(centers) == 4 else None


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Re-label a mask so IDs run 1, 2, 3, … with no gaps."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    if len(unique) == 0:
        return labels
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        lut[old_id] = new_id
    return lut[labels]


# ---------------------------------------------------------------------------
# Grind-setting estimation (calibration curve)
# ---------------------------------------------------------------------------

# Calibration data from flash-photography batch (4 replicates per setting).
# Each value is the mean of per-replicate median equivalent diameters (mm).
_CAL_SETTINGS = np.array([20.0, 40.0, 60.0, 80.0])
_CAL_MEDIANS = np.array([0.456, 0.675, 0.844, 1.057])


def estimate_grind_setting(median_diameter_mm: float) -> float:
    """Estimate DF54 grind setting from median particle diameter.

    Uses piecewise-linear interpolation of calibration data, with linear
    extrapolation outside the calibrated range, clamped to 0-100.
    """
    d = median_diameter_mm

    if d <= _CAL_MEDIANS[0]:
        # Extrapolate below using first two calibration points
        slope = ((_CAL_SETTINGS[1] - _CAL_SETTINGS[0])
                 / (_CAL_MEDIANS[1] - _CAL_MEDIANS[0]))
        est = _CAL_SETTINGS[0] + slope * (d - _CAL_MEDIANS[0])
    elif d >= _CAL_MEDIANS[-1]:
        # Extrapolate above using last two calibration points
        slope = ((_CAL_SETTINGS[-1] - _CAL_SETTINGS[-2])
                 / (_CAL_MEDIANS[-1] - _CAL_MEDIANS[-2]))
        est = _CAL_SETTINGS[-1] + slope * (d - _CAL_MEDIANS[-1])
    else:
        est = float(np.interp(d, _CAL_MEDIANS, _CAL_SETTINGS))

    return float(np.clip(est, 0, 100))
