"""
Particle measurement: extract size and shape metrics from a label mask.

Filters out particles that overlap with exclusion zones, are outside
the diameter range, or are too bright (paper grain artefacts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

from constants import MAX_PARTICLE_DIAM_MM, MIN_PARTICLE_DIAM_MM


def measure_particles(
    label_mask: np.ndarray,
    mm_per_px: float,
    analysis_mask: np.ndarray | None = None,
    image_gray: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure properties of segmented particles.

    Filters out particles that overlap with exclusion zones, are outside
    the diameter range, or are too bright (paper grain artefacts).
    """
    props = measure.regionprops(label_mask, intensity_image=image_gray)

    records = []
    for p in props:
        # Reject particles overlapping marker/note-box zones
        if analysis_mask is not None:
            region_mask = label_mask[p.slice] == p.label
            region_analysis = analysis_mask[p.slice]
            if np.sum(region_mask & ~region_analysis) > 0:
                continue

        area_mm2 = p.area * (mm_per_px ** 2)
        equiv_diam = p.equivalent_diameter_area * mm_per_px
        major = p.axis_major_length * mm_per_px
        minor = p.axis_minor_length * mm_per_px

        # Size filter
        if equiv_diam < MIN_PARTICLE_DIAM_MM or equiv_diam > MAX_PARTICLE_DIAM_MM:
            continue

        # Intensity filter: reject paper grain (bright pixels ≠ coffee)
        mean_intensity = None
        if image_gray is not None and hasattr(p, "intensity_mean"):
            mean_intensity = p.intensity_mean
            if mean_intensity > 160:
                continue

        records.append({
            "label": p.label,
            "area_mm2": round(area_mm2, 6),
            "equiv_diameter_mm": round(equiv_diam, 4),
            "major_axis_mm": round(major, 4),
            "minor_axis_mm": round(minor, 4),
            "eccentricity": round(p.eccentricity, 4),
            "solidity": round(p.solidity, 4),
            "mean_intensity": round(float(mean_intensity), 1) if mean_intensity is not None else None,
            "centroid_y_px": p.centroid[0],
            "centroid_x_px": p.centroid[1],
        })

    return pd.DataFrame(records)


def compute_distribution_stats(df: pd.DataFrame) -> dict:
    """Compute industry-standard particle size distribution statistics.

    Returns a dict with:
        D10, D50, D90     – diameter percentiles (mm)
        span               – (D90 − D10) / D50
        fines_pct          – % of particles < 0.200 mm
        boulders_pct       – % of particles > 1.000 mm
        D_4_3              – volume-weighted mean diameter (De Brouckere)
        D_3_2              – surface-weighted mean diameter (Sauter)
    """
    if df.empty:
        return {k: None for k in (
            "D10", "D50", "D90", "span",
            "fines_pct", "boulders_pct", "D_4_3", "D_3_2",
        )}

    d = df["equiv_diameter_mm"].values

    D10, D50, D90 = np.percentile(d, [10, 50, 90])
    span = (D90 - D10) / D50 if D50 > 0 else None

    fines_pct = float((d < 0.200).sum()) / len(d) * 100
    boulders_pct = float((d > 1.000).sum()) / len(d) * 100

    # Volume-weighted (De Brouckere) mean diameter  D[4,3] = Σd⁴ / Σd³
    D_4_3 = (d ** 4).sum() / (d ** 3).sum()
    # Surface-weighted (Sauter) mean diameter  D[3,2] = Σd³ / Σd²
    D_3_2 = (d ** 3).sum() / (d ** 2).sum()

    return {
        "D10": round(D10, 4),
        "D50": round(D50, 4),
        "D90": round(D90, 4),
        "span": round(span, 3) if span is not None else None,
        "fines_pct": round(fines_pct, 1),
        "boulders_pct": round(boulders_pct, 1),
        "D_4_3": round(D_4_3, 4),
        "D_3_2": round(D_3_2, 4),
    }
