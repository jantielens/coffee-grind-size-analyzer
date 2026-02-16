#!/usr/bin/env python3
"""
Generate a printable A4 reference sheet for coffee grind size analysis.

Layout:
- A4 paper (210 × 297 mm) at 300 DPI
- 4 reference areas in a 2×2 grid, each containing:
  - 4 ArUco markers (DICT_4X4_50, IDs 0–3, 15 mm each) forming an 80 × 55 mm rectangle
  - Grounds zone: 95 × 70 mm (full area bounded by marker outer edges)
  - Note box: 15 × 15 mm attached to marker 3 for handwritten labels
- Generous spacing between areas so only one set of markers is in frame
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 300
MM_TO_PX = DPI / 25.4  # 1 mm ≈ 11.811 px at 300 DPI

# A4 dimensions in mm
A4_W_MM = 210
A4_H_MM = 297

# A4 in pixels
A4_W_PX = int(round(A4_W_MM * MM_TO_PX))
A4_H_PX = int(round(A4_H_MM * MM_TO_PX))

# ArUco configuration
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
MARKER_SIZE_MM = 15

# Marker rectangle: 55 × 80 mm (center-to-center, portrait orientation)
MARKER_RECT_W_MM = 55
MARKER_RECT_H_MM = 80

# Grounds zone: full area bounded by marker outer edges
GROUNDS_W_MM = MARKER_RECT_W_MM + MARKER_SIZE_MM  # 95 mm
GROUNDS_H_MM = MARKER_RECT_H_MM + MARKER_SIZE_MM  # 70 mm

# Grid layout: 2 × 2
GRID_COLS = 2
GRID_ROWS = 2


def mm_to_px(mm_val: float) -> int:
    """Convert millimeters to pixels at target DPI."""
    return int(round(mm_val * MM_TO_PX))


def generate_aruco_marker(marker_id: int, size_px: int) -> np.ndarray:
    """Generate a single ArUco marker as a grayscale numpy array."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return marker_img


def draw_reference_area(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx_mm: float,
    cy_mm: float,
) -> dict:
    """
    Draw a single reference area centered at (cx_mm, cy_mm).

    Returns the note_box_mm dict for this area.
    """
    marker_size_px = mm_to_px(MARKER_SIZE_MM)
    half_marker = MARKER_SIZE_MM / 2

    # Marker positions (center of each marker)
    marker_positions = {
        0: (cx_mm - MARKER_RECT_W_MM / 2, cy_mm - MARKER_RECT_H_MM / 2),  # top-left
        1: (cx_mm + MARKER_RECT_W_MM / 2, cy_mm - MARKER_RECT_H_MM / 2),  # top-right
        2: (cx_mm - MARKER_RECT_W_MM / 2, cy_mm + MARKER_RECT_H_MM / 2),  # bottom-left
        3: (cx_mm + MARKER_RECT_W_MM / 2, cy_mm + MARKER_RECT_H_MM / 2),  # bottom-right
    }

    # Draw ArUco markers
    for marker_id, (mcx, mcy) in marker_positions.items():
        marker_np = generate_aruco_marker(marker_id, marker_size_px)
        marker_pil = Image.fromarray(marker_np).convert("RGB")
        mx = mm_to_px(mcx) - marker_size_px // 2
        my = mm_to_px(mcy) - marker_size_px // 2
        img.paste(marker_pil, (mx, my))

    # Light grey border around marker outer edges
    border_color = (210, 210, 210)
    bx0 = mm_to_px(cx_mm - MARKER_RECT_W_MM / 2 - half_marker)
    by0 = mm_to_px(cy_mm - MARKER_RECT_H_MM / 2 - half_marker)
    bx1 = mm_to_px(cx_mm + MARKER_RECT_W_MM / 2 + half_marker)
    by1 = mm_to_px(cy_mm + MARKER_RECT_H_MM / 2 + half_marker)
    draw.rectangle([bx0, by0, bx1, by1], outline=border_color, width=1)

    # Note box: 15×15 mm, attached to the left side of marker 3
    m3_cx, m3_cy = marker_positions[3]
    note_box_mm = {
        "x": m3_cx - MARKER_SIZE_MM,
        "y": m3_cy,
        "w": MARKER_SIZE_MM,
        "h": MARKER_SIZE_MM,
    }
    nb_left = mm_to_px(note_box_mm["x"] - MARKER_SIZE_MM / 2)
    nb_top = mm_to_px(note_box_mm["y"] - MARKER_SIZE_MM / 2)
    nb_right = mm_to_px(note_box_mm["x"] + MARKER_SIZE_MM / 2)
    nb_bottom = mm_to_px(note_box_mm["y"] + MARKER_SIZE_MM / 2)

    nb_color = (180, 180, 180)
    draw.rectangle([nb_left, nb_top, nb_right, nb_bottom], outline=nb_color, width=2)
    try:
        font_notes = ImageFont.truetype("arial.ttf", mm_to_px(3))
    except (OSError, IOError):
        font_notes = ImageFont.load_default()
    draw.text(
        (nb_left + mm_to_px(1), nb_top + mm_to_px(0.5)),
        "notes",
        fill=(200, 200, 200),
        font=font_notes,
        anchor="lt",
    )

    return note_box_mm


def create_reference_sheet(output_path: str = "reference_sheet.pdf") -> None:
    """Create the A4 reference sheet with a 2×2 grid of reference areas."""

    # Create white A4 canvas (RGB)
    img = Image.new("RGB", (A4_W_PX, A4_H_PX), "white")
    draw = ImageDraw.Draw(img)

    # Compute grid centers with maximum spacing
    # Horizontal: 2 areas of 70mm across 210mm page
    # 10mm margin is safe for virtually all printers (inkjet & laser)
    h_margin = 10.0  # mm from page edge to area outer edge
    col_centers = [
        h_margin + GROUNDS_W_MM / 2,                    # left column
        A4_W_MM - h_margin - GROUNDS_W_MM / 2,          # right column
    ]
    h_gap = col_centers[1] - col_centers[0] - GROUNDS_W_MM  # gap between areas

    # Vertical: 2 areas of 70mm across 297mm page — lots of room
    v_margin = 15.0  # mm from page edge to area outer edge
    row_centers = [
        v_margin + GROUNDS_H_MM / 2,                    # top row
        A4_H_MM - v_margin - GROUNDS_H_MM / 2,          # bottom row
    ]
    v_gap = row_centers[1] - row_centers[0] - GROUNDS_H_MM  # gap between areas

    # Draw each reference area — capture the first area's note box for metadata
    note_box_mm = None
    for row_idx, ry in enumerate(row_centers):
        for col_idx, cx in enumerate(col_centers):
            area_note_box = draw_reference_area(img, draw, cx, ry)
            if note_box_mm is None:
                note_box_mm = area_note_box

    # Metadata uses the first area's center (top-left) as the canonical reference
    # (all areas are identical—the analysis pipeline processes one photo at a time)
    cx_mm = col_centers[0]
    cy_mm = row_centers[0]

    marker_positions_mm = {
        0: (cx_mm - MARKER_RECT_W_MM / 2, cy_mm - MARKER_RECT_H_MM / 2),
        1: (cx_mm + MARKER_RECT_W_MM / 2, cy_mm - MARKER_RECT_H_MM / 2),
        2: (cx_mm - MARKER_RECT_W_MM / 2, cy_mm + MARKER_RECT_H_MM / 2),
        3: (cx_mm + MARKER_RECT_W_MM / 2, cy_mm + MARKER_RECT_H_MM / 2),
    }

    metadata = {
        "marker_dictionary": "DICT_4X4_50",
        "marker_ids": MARKER_IDS,
        "marker_size_mm": MARKER_SIZE_MM,
        "marker_centers_mm": {
            str(k): {"x": v[0], "y": v[1]} for k, v in marker_positions_mm.items()
        },
        "marker_rect_mm": {"w": MARKER_RECT_W_MM, "h": MARKER_RECT_H_MM},
        "grounds_zone_mm": {"w": GROUNDS_W_MM, "h": GROUNDS_H_MM},
        "grounds_zone_center_mm": {"x": cx_mm, "y": cy_mm},
        "note_box_mm": note_box_mm,
    }

    # Save PDF
    output = Path(output_path)
    img.save(str(output), "PDF", resolution=DPI)
    print(f"✓ Reference sheet saved to: {output.resolve()}")
    print(f"  Paper: A4 ({A4_W_MM}×{A4_H_MM} mm) at {DPI} DPI")
    print(f"  Layout: {GRID_COLS}×{GRID_ROWS} = {GRID_COLS * GRID_ROWS} reference areas")
    print(f"  Each area: {GROUNDS_W_MM}×{GROUNDS_H_MM} mm")
    print(f"  Gaps: {h_gap:.0f} mm horizontal, {v_gap:.0f} mm vertical")
    print(f"  Markers: {MARKER_SIZE_MM} mm, IDs {MARKER_IDS}")

    # Save metadata JSON
    meta_path = output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {meta_path.resolve()}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate coffee grind analysis reference sheet")
    parser.add_argument(
        "-o", "--output",
        default="reference_sheet.pdf",
        help="Output PDF path (default: reference_sheet.pdf)",
    )
    args = parser.parse_args()
    create_reference_sheet(args.output)


if __name__ == "__main__":
    main()
