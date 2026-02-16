#!/usr/bin/env python3
"""
MCP server for coffee grind size analysis.

Exposes the analysis pipeline as MCP tools that any compatible AI assistant
(Claude, Copilot, etc.) can call directly.

Tools:
    analyze_grind_photo   — Full pipeline: detect markers → warp → segment → measure
    estimate_grind_setting — Quick lookup: median diameter → estimated DF54 setting
    get_brew_recommendation — Suggest a brew method based on median particle diameter

Resources:
    reference://instructions — How to take a photo for analysis

Usage (stdio transport — local):
    python server.py

Configuration (Claude Desktop / VS Code):
    See README.md for setup instructions.
"""

from __future__ import annotations

import asyncio
import base64
import random
import shutil
import sys
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Add the src/ directory to sys.path so we can import the pipeline modules
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from analyze import preprocess_image, segment_and_measure, save_model_results
from constants import estimate_grind_setting as _estimate_grind_setting

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "coffee-grind-analyzer",
    instructions=(
        "Analyse coffee grind size from photos taken on an ArUco reference sheet. "
        "Segments particles using watershed segmentation and returns size "
        "distribution statistics."
    ),
)

# ---------------------------------------------------------------------------
# Brew recommendations (same data as the Telegram bot)
# ---------------------------------------------------------------------------

BREW_RECOMMENDATIONS: list[tuple[float, float, str, list[str]]] = [
    (0.00, 0.30, "Turkish", [
        "Ultra-fine! Perfect for a rich, thick Turkish coffee.",
        "This is Turkish-grind territory — time to break out the cezve!",
        "Ground to dust! Ideal for an authentic Turkish brew.",
    ]),
    (0.30, 0.50, "Espresso", [
        "Dialed in for a punchy espresso shot!",
        "This grind is screaming espresso — pull that shot!",
        "Espresso-ready. May the puck prep gods be with you.",
        "Looking like a solid espresso grind. Bottomless portafilter time!",
    ]),
    (0.50, 0.70, "Moka Pot / AeroPress", [
        "Great for a Moka pot or AeroPress — solid middle ground!",
        "Moka pot vibes! This will make a strong, full-bodied cup.",
        "AeroPress sweet spot — get experimenting with recipes!",
        "Between espresso and pour-over — perfect for a Moka pot.",
    ]),
    (0.70, 0.90, "Pour-over (V60 / Chemex)", [
        "Looks great for a delicious V60!",
        "Chemex or V60 — this grind is in the sweet spot for pour-over.",
        "Pour-over perfection. Time to make James Hoffmann proud.",
        "This is V60 territory — spiral pour, bloom, enjoy!",
    ]),
    (0.90, 1.10, "Drip / Batch Brew", [
        "Well suited for drip or batch brew — set it and sip it!",
        "Classic drip-brew grind. Your coffee maker will thank you.",
        "Batch brew ready — make a pot and share with friends!",
        "Right in the drip-brew zone. Consistent cups incoming.",
    ]),
    (1.10, 1.40, "French Press", [
        "Coarse enough for a French press — plunge away!",
        "French press grind detected! 4 minutes and you're golden.",
        "This coarseness is begging for a French press. Steep it!",
        "French press vibes. Bold, full-bodied, no filter needed (well, mesh).",
    ]),
    (1.40, 5.00, "Cold Brew / Cupping", [
        "This coarse? Perfect for cold brew — steep it overnight!",
        "Cold brew grind! 12-24 hours in the fridge and you'll have liquid gold.",
        "Extra coarse — ideal for cold brew or a classic cupping session.",
        "Beach-day cold brew grind. Get a mason jar and some patience.",
    ]),
]


def _get_brew_recommendation(median_diameter_mm: float) -> dict:
    """Return brew method and recommendation for a given median diameter."""
    for low, high, method, phrases in BREW_RECOMMENDATIONS:
        if low <= median_diameter_mm < high:
            return {"method": method, "recommendation": random.choice(phrases)}
    if median_diameter_mm < 0:
        return {"method": "Turkish", "recommendation": random.choice(BREW_RECOMMENDATIONS[0][3])}
    return {"method": "Cold Brew / Cupping", "recommendation": random.choice(BREW_RECOMMENDATIONS[-1][3])}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _run_pipeline(image_path: Path, cleanup_dir: Path | None = None) -> dict:
    """Run the full analysis pipeline synchronously.

    Args:
        image_path: Path to the image file on disk.
        cleanup_dir: If set, this temp directory will be removed when done.

    Returns a dict with all statistics, or raises on failure.
    """
    tmp_dir = cleanup_dir or Path(tempfile.mkdtemp(prefix="coffee_mcp_"))
    try:
        output_dir = tmp_dir / "results"
        output_dir.mkdir(exist_ok=True)

        prep = preprocess_image(image_path, grind_setting=None, output_dir=output_dir)
        if prep is None:
            raise ValueError(
                "Could not detect all 4 ArUco markers. "
                "Make sure the reference sheet is fully visible in the photo."
            )

        result = segment_and_measure(prep, model_name="watershed", expected_diam_mm=0.7)
        if result is None:
            raise ValueError("Segmentation failed. Try a different photo.")

        s = result["summary"]
        median_d = s["median_diameter_mm"]

        # Estimate grind setting
        if median_d is not None:
            est = _estimate_grind_setting(median_d)
            est_rounded = round(est)
        else:
            est = None
            est_rounded = None

        # Build response
        response = {
            "n_particles": s["n_particles"],
            "median_diameter_mm": median_d,
            "mean_diameter_mm": s.get("mean_diameter_mm"),
            "std_diameter_mm": s.get("std_diameter_mm"),
            "D10_mm": s.get("D10"),
            "D50_mm": s.get("D50"),
            "D90_mm": s.get("D90"),
            "span": s.get("span"),
            "fines_pct": s.get("fines_pct"),
            "boulders_pct": s.get("boulders_pct"),
            "D_4_3_mm": s.get("D_4_3"),
            "D_3_2_mm": s.get("D_3_2"),
            "estimated_df54_setting": est_rounded,
            "estimated_df54_setting_precise": round(est, 1) if est is not None else None,
        }

        # Add brew recommendation
        if median_d is not None:
            brew = _get_brew_recommendation(median_d)
            response["brew_method"] = brew["method"]
            response["brew_recommendation"] = brew["recommendation"]

        return response

    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


@mcp.tool()
async def analyze_grind_photo(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict:
    """Analyse a coffee grind photo and return particle size statistics.

    The photo must show coffee grounds spread on the printed ArUco reference
    sheet with all four corner markers fully visible.

    Provide exactly one of the two arguments.

    Args:
        image_path: Absolute path to an image file on disk (preferred).
        image_base64: The photo as a base64-encoded string (JPEG or PNG).
                      Use this only when you don't have a file on disk.

    Returns:
        A dict with particle count, size distribution (median, mean, D10/D50/D90,
        fines/boulders percentages, span), estimated DF54 grind setting, and a
        brew method recommendation.
    """
    if image_path:
        p = Path(image_path)
        if not p.is_file():
            raise ValueError(f"File not found: {image_path}")
        return await asyncio.to_thread(_run_pipeline, p)
    elif image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            raise ValueError("Invalid base64 image data.")
        tmp_dir = Path(tempfile.mkdtemp(prefix="coffee_mcp_"))
        tmp_path = tmp_dir / "photo.jpg"
        tmp_path.write_bytes(image_bytes)
        return await asyncio.to_thread(_run_pipeline, tmp_path, tmp_dir)
    else:
        raise ValueError(
            "Provide either image_path (path to a file on disk) "
            "or image_base64 (base64-encoded image data)."
        )


@mcp.tool()
async def estimate_grind_setting(median_diameter_mm: float) -> dict:
    """Estimate the DF54 grind setting from a median particle diameter.

    Uses piecewise-linear interpolation of calibration data (flash-photography
    batch, 4 replicates at settings 20/40/60/80).

    Args:
        median_diameter_mm: Median equivalent diameter in millimetres.

    Returns:
        A dict with the estimated setting (rounded and precise) plus the
        recommended brew method.
    """
    est = _estimate_grind_setting(median_diameter_mm)
    brew = _get_brew_recommendation(median_diameter_mm)
    return {
        "median_diameter_mm": median_diameter_mm,
        "estimated_df54_setting": round(est),
        "estimated_df54_setting_precise": round(est, 1),
        "brew_method": brew["method"],
        "brew_recommendation": brew["recommendation"],
    }


@mcp.tool()
async def get_brew_recommendation(median_diameter_mm: float) -> dict:
    """Suggest a brew method based on the median particle diameter.

    Args:
        median_diameter_mm: Median equivalent diameter in millimetres.

    Returns:
        A dict with the recommended brew method and a descriptive suggestion.
    """
    return _get_brew_recommendation(median_diameter_mm)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("reference://instructions")
def reference_instructions() -> str:
    """How to take a photo for coffee grind analysis."""
    return (
        "## How to take a photo for analysis\n\n"
        "1. **Print the reference sheet** — download from:\n"
        "   https://github.com/jantielens/coffee-grind-size-analyzer/blob/main/reference-sheet.pdf\n\n"
        "2. **Spread a small amount of coffee grounds** on the white area of the sheet.\n"
        "   Less is more — particles should not touch each other.\n\n"
        "3. **Make sure all 4 ArUco markers** (corner squares) are fully visible.\n\n"
        "4. **Use flash** for consistent lighting.\n\n"
        "5. **Take the photo from directly above** to minimise perspective distortion.\n\n"
        "6. Use the highest resolution available (send as a file, not a compressed photo).\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
