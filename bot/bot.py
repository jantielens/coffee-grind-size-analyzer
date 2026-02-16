#!/usr/bin/env python3
"""
Telegram bot for coffee grind size analysis.

Flow:
    1. User starts a chat with the bot (/start)
    2. User sends a photo of coffee grounds on the reference sheet
    3. Bot runs the analysis pipeline (watershed segmentation)
    4. Bot returns the summary.png with full pipeline visualisation

Setup:
    1. Create a bot via @BotFather on Telegram → get your BOT_TOKEN
    2. Set the token:  export COFFEE_BOT_TOKEN="123456:ABC-DEF..."
    3. pip install -r requirements.txt
    4. python bot.py
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the src/ directory to sys.path so we can import the pipeline modules
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Pipeline imports (from src/)
from analyze import preprocess_image, segment_and_measure, save_model_results
from constants import estimate_grind_setting

# ---------------------------------------------------------------------------
# Brew method recommendations (keyed by median diameter range)
# ---------------------------------------------------------------------------

BREW_RECOMMENDATIONS: list[tuple[float, float, str, list[str]]] = [
    (0.00, 0.30, "Turkish", [
        "Ultra-fine! Perfect for a rich, thick Turkish coffee ☕",
        "This is Turkish-grind territory — time to break out the cezve!",
        "Ground to dust! Ideal for an authentic Turkish brew.",
    ]),
    (0.30, 0.50, "Espresso", [
        "Dialed in for a punchy espresso shot! ☕",
        "This grind is screaming espresso — pull that shot!",
        "Espresso-ready. May the puck prep gods be with you.",
        "Looking like a solid espresso grind. Bottomless portafilter time!",
    ]),
    (0.50, 0.70, "Moka Pot / AeroPress", [
        "Great for a Moka pot or AeroPress — solid middle ground! 🫖",
        "Moka pot vibes! This will make a strong, full-bodied cup.",
        "AeroPress sweet spot — get experimenting with recipes!",
        "Between espresso and pour-over — perfect for a Moka pot.",
    ]),
    (0.70, 0.90, "Pour-over (V60 / Chemex)", [
        "Looks great for a delicious V60! ☕✨",
        "Chemex or V60 — this grind is in the sweet spot for pour-over.",
        "Pour-over perfection. Time to make James Hoffmann proud.",
        "This is V60 territory — spiral pour, bloom, enjoy! 🌀",
    ]),
    (0.90, 1.10, "Drip / Batch Brew", [
        "Well suited for drip or batch brew — set it and sip it! ☕",
        "Classic drip-brew grind. Your coffee maker will thank you.",
        "Batch brew ready — make a pot and share with friends!",
        "Right in the drip-brew zone. Consistent cups incoming.",
    ]),
    (1.10, 1.40, "French Press", [
        "Coarse enough for a French press — plunge away! 🫖",
        "French press grind detected! 4 minutes and you're golden.",
        "This coarseness is begging for a French press. Steep it!",
        "French press vibes. Bold, full-bodied, no filter needed (well, mesh).",
    ]),
    (1.40, 5.00, "Cold Brew / Cupping", [
        "This coarse? Perfect for cold brew — steep it overnight! 🧊",
        "Cold brew grind! 12-24 hours in the fridge and you'll have liquid gold.",
        "Extra coarse — ideal for cold brew or a classic cupping session.",
        "Beach-day cold brew grind. Get a mason jar and some patience. 🏖️",
    ]),
]


def get_brew_recommendation(median_diameter_mm: float) -> str:
    """Return a brew method recommendation based on median particle diameter."""
    for low, high, method, phrases in BREW_RECOMMENDATIONS:
        if low <= median_diameter_mm < high:
            return f"*{method}:* {random.choice(phrases)}"
    # Fallback for anything outside the ranges
    if median_diameter_mm < 0:
        return random.choice(BREW_RECOMMENDATIONS[0][3])
    return random.choice(BREW_RECOMMENDATIONS[-1][3])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot token
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ.get("COFFEE_BOT_TOKEN", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_tag(update: Update) -> str:
    """Return a readable user identifier for logs."""
    user = update.effective_user
    if not user:
        return "unknown"
    name = user.username or user.full_name or str(user.id)
    return f"{name} (id:{user.id})"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — greet the user."""
    logger.info("[%s] /start", _user_tag(update))
    await update.message.reply_text(
        "☕ *Coffee Grind Size Analyser*\n\n"
        "Send me a photo of your coffee grounds on the "
        "[reference sheet](https://github.com/jantielens/coffee-grind-size-analyzer/blob/main/reference-sheet.pdf) "
        "and I'll analyse the particle size distribution\\!\n\n"
        "📋 *Tips for great results:*\n"
        "• Use your phone's *flash* for consistent lighting\n"
        "• Keep all 4 ArUco markers fully visible\n"
        "• Less is more — spread a *tiny* amount of grounds so particles "
        "don't touch each other 🫘\n"
        "• Send your photo as a *file* \\(📎\\) instead of as a photo — "
        "Telegram compresses photos and you lose detail\!",
        parse_mode="MarkdownV2",
        disable_web_page_preview=True,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help."""
    logger.info("[%s] /help", _user_tag(update))
    await update.message.reply_text(
        "📷 Just send a photo!\n\n"
        "The photo should show coffee grounds spread on the printed "
        "reference sheet with all four corner markers visible.\n\n"
        "I'll detect the markers, correct perspective, segment the "
        "particles, and send you back a full analysis summary.",
    )


IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}


# ---------------------------------------------------------------------------
# Gauge renderers — produce compact chart PNGs for inline display
# ---------------------------------------------------------------------------
def _render_spread_gauge(df_diams, summary: dict) -> BytesIO:
    """Render the D10–D50–D90 range bar as a standalone PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    D10 = summary["D10"]
    D50 = summary["D50"]
    D90 = summary["D90"]
    d_min, d_max = df_diams.min(), df_diams.max()
    q25, q75 = df_diams.quantile(0.25), df_diams.quantile(0.75)

    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.22)
    ax.set_title("Particle Size Spread", fontsize=13, fontweight="bold", pad=10)

    # Background (full data range)
    ax.barh(0, d_max - d_min, left=d_min, height=0.3,
            color="#e0e0e0", edgecolor="none", zorder=1)
    # D10-D90 range
    ax.barh(0, D90 - D10, left=D10, height=0.3,
            color="#5B9BD5", alpha=0.5, edgecolor="none", zorder=2)
    # IQR
    ax.barh(0, q75 - q25, left=q25, height=0.3,
            color="#2E75B6", alpha=0.6, edgecolor="none", zorder=3)

    mk = dict(zorder=5, clip_on=False)
    ax.plot(D10, 0, "v", color="#e67e22", markersize=14, **mk)
    ax.plot(D50, 0, "D", color="#c0392b", markersize=13, **mk)
    ax.plot(D90, 0, "^", color="#e67e22", markersize=14, **mk)

    ax.text(D10, -0.30, f"D10\n{D10:.3f}", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#e67e22")
    ax.text(D50, 0.30, f"D50\n{D50:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#c0392b")
    ax.text(D90, -0.30, f"D90\n{D90:.3f}", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#e67e22")

    ax.set_xlim(max(0, d_min - 0.05), d_max + 0.05)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xlabel("Diameter (mm)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    legend_elements = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e67e22",
               markersize=8, label="D10 / D90"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b",
               markersize=8, label="D50 (Median)"),
        plt.Rectangle((0, 0), 1, 1, fc="#5B9BD5", alpha=0.5, label="D10\u2013D90 range"),
        plt.Rectangle((0, 0), 1, 1, fc="#2E75B6", alpha=0.6, label="IQR (25th\u201375th)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", framealpha=0.9)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_fines_boulders_gauge(summary: dict) -> BytesIO:
    """Render fines/boulders percentage bars as a standalone PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fines = summary["fines_pct"]
    boulders = summary["boulders_pct"]

    def _pct_color(pct: float) -> str:
        if pct < 5:
            return "#27ae60"
        elif pct < 15:
            return "#f39c12"
        return "#e74c3c"

    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.subplots_adjust(left=0.28, right=0.88, top=0.82, bottom=0.15)
    ax.set_title("Fines & Boulders", fontsize=13, fontweight="bold", pad=10)

    labels = ["Fines\n(<0.2 mm)", "Boulders\n(>1.0 mm)"]
    values = [fines, boulders]
    colors = [_pct_color(fines), _pct_color(boulders)]

    # Background track
    ax.barh([1, 0], [100, 100], height=0.45,
            color="#f0f0f0", edgecolor="none", zorder=1)
    bars = ax.barh([1, 0], values, height=0.45,
                   color=colors, edgecolor="white", zorder=3)

    for bar, val in zip(bars, values):
        x_pos = max(val + 1, 5)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold")

    ax.set_xlim(0, max(max(values) * 1.5, 25))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("%", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_histogram(df_diams, summary: dict) -> BytesIO:
    """Render the particle size distribution histogram as a standalone PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    median_d = summary.get("median_diameter_mm", df_diams.median())
    mean_d = summary.get("mean_diameter_mm", df_diams.mean())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.15)
    ax.set_title("Particle Size Distribution", fontsize=13, fontweight="bold", pad=10)

    ax.hist(df_diams, bins=50, edgecolor="black", alpha=0.7, color="#5B9BD5")
    ax.axvline(median_d, color="red", linestyle="--", linewidth=2,
               label=f"Median: {median_d:.3f} mm")
    ax.axvline(mean_d, color="orange", linestyle=":", linewidth=2,
               label=f"Mean: {mean_d:.3f} mm")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlabel("Equivalent Diameter (mm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


async def _process_and_reply(
    msg, user: str, file, file_desc: str,
) -> None:
    """Shared analysis pipeline: download → analyse → reply with summary.png."""
    status = await msg.reply_text("⏳ Analysing your coffee grounds…")
    t_start = time.perf_counter()

    tmp_dir = Path(tempfile.mkdtemp(prefix="coffee_bot_"))
    try:
        img_path = tmp_dir / "photo.jpg"
        await file.download_to_drive(str(img_path))

        output_dir = tmp_dir / "results"
        output_dir.mkdir()

        prep = preprocess_image(img_path, grind_setting=None, output_dir=output_dir)
        if prep is None:
            logger.warning("[%s] marker detection failed", user)
            await status.edit_text(
                "❌ Could not detect all 4 ArUco markers in your photo.\n\n"
                "Make sure the reference sheet is fully visible and try again."
            )
            return

        result = segment_and_measure(prep, model_name="watershed", expected_diam_mm=0.7)
        if result is None:
            logger.warning("[%s] segmentation failed", user)
            await status.edit_text("❌ Segmentation failed. Please try a different photo.")
            return

        median_d = result["summary"]["median_diameter_mm"]
        if median_d is not None:
            est = estimate_grind_setting(median_d)
            est_rounded = round(est)
        else:
            est = None
            est_rounded = None

        prep["grind_setting"] = est_rounded
        prep["grind_setting_estimated"] = True
        result["summary"]["grind_setting"] = est_rounded
        result["summary"]["estimated_grind_setting"] = (
            round(est, 1) if est is not None else None
        )

        save_model_results(prep, result)

        s = result["summary"]
        lines = ["\u2615 *Analysis Complete*\n"]
        if est_rounded is not None:
            lines.append(f"Estimated DF54 setting: *~{est_rounded}*")
        lines.append(f"Particles detected: *{s['n_particles']}*")
        if median_d is not None:
            lines.append(f"Median diameter: *{median_d:.3f} mm*")
        if s.get("D10") is not None:
            lines.append(f"D10 / D50 / D90: *{s['D10']:.3f}* / *{s['D50']:.3f}* / *{s['D90']:.3f}* mm")
        if s.get("fines_pct") is not None:
            lines.append(f"Fines (<0.2 mm): *{s['fines_pct']:.1f}%*")
        if s.get("boulders_pct") is not None:
            lines.append(f"Boulders (>1.0 mm): *{s['boulders_pct']:.1f}%*")
        if median_d is not None:
            lines.append(f"\n\U0001FAD8 {get_brew_recommendation(median_d)}")
        caption = "\n".join(lines)

        await status.delete()

        # --- Send gauge charts as inline photos ---
        if result.get("df") is not None and not result["df"].empty:
            df_diams = result["df"]["equiv_diameter_mm"]
            hist_buf = _render_histogram(df_diams, s)
            await msg.reply_photo(photo=hist_buf)
        if s.get("D10") is not None and result.get("df") is not None:
            df_diams = result["df"]["equiv_diameter_mm"]
            spread_buf = _render_spread_gauge(df_diams, s)
            await msg.reply_photo(photo=spread_buf)
        if s.get("fines_pct") is not None and s.get("boulders_pct") is not None:
            fb_buf = _render_fines_boulders_gauge(s)
            await msg.reply_photo(photo=fb_buf)

        # --- Send text summary ---
        await msg.reply_text(caption, parse_mode="Markdown")

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[%s] analysis complete in %.1fs (%s) — "
            "particles=%d, median=%.3f mm, est_setting=~%s, "
            "D10=%.3f, D50=%.3f, D90=%.3f, fines=%.1f%%, boulders=%.1f%%",
            user, elapsed, file_desc,
            s["n_particles"],
            median_d or 0,
            est_rounded or "?",
            s.get("D10", 0) or 0,
            s.get("D50", 0) or 0,
            s.get("D90", 0) or 0,
            s.get("fines_pct", 0) or 0,
            s.get("boulders_pct", 0) or 0,
        )

    except Exception:
        logger.exception("[%s] error processing photo", user)
        await status.edit_text("❌ An unexpected error occurred. Please try again.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle compressed photo uploads — analyse but suggest sending as file."""
    msg = update.message
    user = _user_tag(update)

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_size_kb = (file.file_size or 0) / 1024
    logger.info("[%s] uploaded photo (%.1f KB, %dx%d px)",
                user, file_size_kb, photo.width, photo.height)

    await msg.reply_text(
        "💡 *Tip:* Telegram compresses photos and reduces detail, which can "
        "affect the analysis\. For better results, send your image as a "
        "*file* instead \\(tap 📎 → File → pick your photo\\)\."
        "\n\nI'll do my best with this one anyway\! ☕",
        parse_mode="MarkdownV2",
    )

    await _process_and_reply(msg, user, file,
                             f"photo {photo.width}x{photo.height} {file_size_kb:.0f}KB")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle images sent as files (uncompressed)."""
    msg = update.message
    user = _user_tag(update)
    doc = msg.document

    mime = doc.mime_type or ""
    if mime not in IMAGE_MIME_TYPES:
        logger.info("[%s] sent non-image document: %s (%s)", user, doc.file_name, mime)
        await msg.reply_text(
            "📷 That doesn't look like an image file. "
            "Send a photo (JPG, PNG) of your coffee grounds on the reference sheet!",
        )
        return

    file = await context.bot.get_file(doc.file_id)
    file_size_kb = (file.file_size or 0) / 1024
    logger.info("[%s] uploaded file: %s (%.1f KB, %s)",
                user, doc.file_name, file_size_kb, mime)

    await _process_and_reply(msg, user, file,
                             f"file {doc.file_name} {file_size_kb:.0f}KB")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages — log and reply with a hint."""
    text = update.message.text or ""
    logger.info("[%s] messaged: \"%s\"", _user_tag(update), text[:200])
    await update.message.reply_text(
        "📷 Send me a photo of your coffee grounds on the reference sheet "
        "and I'll analyse it!\n\n"
        "Type /help for more info.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BOT_TOKEN:
        print(
            "ERROR: Set your Telegram bot token:\n"
            "  export COFFEE_BOT_TOKEN=\"123456:ABC-DEF...\"\n"
            "Get one from @BotFather on Telegram."
        )
        sys.exit(1)

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started — polling for updates…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
