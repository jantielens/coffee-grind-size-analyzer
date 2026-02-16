"""
ArUco marker detection with geometric fallback.

Detects the four ArUco markers (IDs 0-3) printed on the reference sheet.
Falls back to a dark-square finder when ArUco decoding fails due to print
quality or grounds covering the marker pattern.
"""

from __future__ import annotations

import cv2
import numpy as np

from constants import (
    ARUCO_DICT_ID,
    MARKER_IDS,
    MARKER_RECT_H_MM,
    MARKER_RECT_W_MM,
    _extract_marker_centers,
)


def detect_aruco_markers(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """Detect ArUco markers and return {id: center_xy} dict.

    Falls back to geometric detection (dark-square finder) when ArUco
    decoding fails — e.g. due to print quality or grounds covering the
    marker pattern.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    # --- Attempt 1: standard ArUco ---
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    result = _extract_marker_centers(corners, ids)
    if result is not None:
        return result

    # --- Attempt 2: ArUco with tuned adaptive-threshold params ---
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector2 = cv2.aruco.ArucoDetector(dictionary, params)
    corners2, ids2, _ = detector2.detectMarkers(gray)
    result2 = _extract_marker_centers(corners2, ids2)
    if result2 is not None:
        return result2

    # --- Attempt 3: geometric fallback (find 4 dark squares) ---
    return _detect_markers_geometric(gray)


def _detect_markers_geometric(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Find 4 dark squares and assign IDs based on rectangle edge lengths.

    The marker rectangle is 55 mm wide × 80 mm tall (aspect ≈ 1.45).
    IDs must satisfy:
        0──1   short edge (55 mm)
        │  │   long edge  (80 mm)
        2──3   short edge (55 mm)

    The algorithm finds the 4 largest dark squares, then uses pairwise
    distances to decide which pairs share a short vs long edge.  This works
    regardless of camera rotation (portrait, landscape, or tilted).
    """
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
        fill = area / (cw * ch) if cw * ch > 0 else 0
        if aspect > 0.65 and fill > 0.55:
            candidates.append({
                "cx": x + cw / 2.0,
                "cy": y + ch / 2.0,
                "area": area,
            })

    if len(candidates) < 4:
        raise ValueError(
            f"Geometric fallback: found only {len(candidates)} square candidates (need 4)"
        )

    # Take the 4 largest candidates
    candidates.sort(key=lambda c: c["area"], reverse=True)
    top4 = candidates[:4]
    centers = np.array([[c["cx"], c["cy"]] for c in top4])

    # Assign IDs using edge lengths.
    # Each corner has 2 edge-neighbours + 1 diagonal-neighbour.
    # 0→1 = short (55 mm), 0→2 = long (80 mm), 0→3 = diagonal.
    dists = np.array([np.linalg.norm(centers[i] - centers[0]) for i in range(1, 4)])
    order = np.argsort(dists)

    d_short = dists[order[0]]
    d_long = dists[order[1]]

    # Sanity: ratio of long/short edge should be close to 80/55 ≈ 1.45
    edge_ratio = d_long / d_short if d_short > 0 else 0
    expected_ratio = MARKER_RECT_H_MM / MARKER_RECT_W_MM
    if abs(edge_ratio - expected_ratio) > 0.35:
        raise ValueError(
            f"Geometric fallback: edge ratio {edge_ratio:.2f} "
            f"(expected ~{expected_ratio:.2f})"
        )

    return {
        0: centers[0],
        1: centers[order[0] + 1],   # nearest  → short edge (55 mm)
        2: centers[order[1] + 1],   # 2nd near → long edge  (80 mm)
        3: centers[order[2] + 1],   # farthest → diagonal
    }
