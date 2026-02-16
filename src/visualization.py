"""
Visualization: overlays, distribution plots, correlation charts, pipeline
summary PNGs, and model comparison images.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_overlay(
    image_rgb: np.ndarray,
    label_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Create RGB overlay with coloured particle masks and contours."""
    overlay = image_rgb.copy()
    n_labels = label_mask.max()
    if n_labels == 0:
        return overlay

    rng = np.random.default_rng(42)
    colors = rng.integers(60, 240, size=(n_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    colored = colors[label_mask]
    fg = label_mask > 0

    overlay[fg] = (
        (1 - alpha) * overlay[fg].astype(np.float32)
        + alpha * colored[fg].astype(np.float32)
    ).astype(np.uint8)

    for lbl in range(1, n_labels + 1):
        binary = (label_mask == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

    return overlay


def plot_distribution(
    df: pd.DataFrame,
    model_name: str,
    grind_setting: int | None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot histogram of equivalent particle diameters."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = None

    if df.empty:
        ax.text(0.5, 0.5, "No particles detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
    else:
        ax.hist(df["equiv_diameter_mm"], bins=50, edgecolor="black", alpha=0.7)
        median_d = df["equiv_diameter_mm"].median()
        ax.axvline(median_d, color="red", linestyle="--", label=f"Median: {median_d:.3f} mm")
        ax.legend()

    title = f"{model_name}"
    if grind_setting is not None:
        title += f"  —  Grind setting {grind_setting}"
    ax.set_title(title)
    ax.set_xlabel("Equivalent diameter (mm)")
    ax.set_ylabel("Count")

    return fig


def plot_correlation(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot median particle size vs. grind setting for each model."""
    models = summary_df["model"].unique()
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in sorted(models):
        sub = summary_df[summary_df["model"] == model_name].sort_values("grind_setting")
        ax.plot(
            sub["grind_setting"],
            sub["median_diameter_mm"],
            "o-",
            label=model_name,
            markersize=6,
        )
        if "q25_diameter_mm" in sub.columns and "q75_diameter_mm" in sub.columns:
            ax.fill_between(
                sub["grind_setting"],
                sub["q25_diameter_mm"],
                sub["q75_diameter_mm"],
                alpha=0.15,
            )

    ax.set_xlabel("Grind setting (DF54 scale, 0–100)")
    ax.set_ylabel("Median equivalent diameter (mm)")
    ax.set_title("Particle size vs. grind setting")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "correlation.png", dpi=200)
    plt.close(fig)
    print(f"  Correlation plot → {output_dir / 'correlation.png'}")


def save_pipeline_summary(prep: dict, result: dict) -> None:
    """Create a single pipeline-summary PNG showing all stages.

    Layout (10 rows):
        Row 0 (colspan 2): Histogram
        Row 1: Spacer
        Row 2 (colspan 2): Visual gauges (D10–D90 range, fines/boulders bars, span badge)
        Row 3: Spacer
        Row 4 (colspan 2): Distribution statistics table
        Row 5: Original photo  |  Warped
        Row 6: Normalised      |  Binary threshold
        Row 7: Distance transform (heatmap)  |  Watershed markers (sure_fg)
        Row 8–9 (colspan 2): Overlay (filtered particles)
    """
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    image_path: Path = prep["image_path"]
    grind_setting = prep["grind_setting"]
    grind_estimated = prep.get("grind_setting_estimated", False)
    img_out_dir: Path = prep["img_out_dir"]
    df = result["df"]
    overlay = result["overlay"]
    intermediates = result.get("intermediates", {})
    summary = result.get("summary", {})

    def _ensure_landscape(img: np.ndarray) -> np.ndarray:
        """Rotate image 90° CW if it is in portrait orientation."""
        h, w = img.shape[:2]
        if h > w:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    # --- Load original photo (downscaled for display) ---
    original_bgr = cv2.imread(str(image_path))
    if original_bgr is not None:
        original_bgr = _ensure_landscape(original_bgr)
        scale = 700 / max(original_bgr.shape[1], 1)
        if scale < 1:
            original_bgr = cv2.resize(
                original_bgr, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    # --- Load warped image ---
    warped_path = img_out_dir / "warped.jpg"
    if warped_path.exists():
        warped_rgb = cv2.cvtColor(
            cv2.imread(str(warped_path)), cv2.COLOR_BGR2RGB,
        )
        warped_rgb = _ensure_landscape(warped_rgb)
    else:
        warped_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    normed_rgb = _ensure_landscape(prep["normed_rgb"].copy())

    # --- Intermediates ---
    binary = intermediates.get("binary")
    dist_transform = intermediates.get("dist_transform")
    sure_fg = intermediates.get("sure_fg")

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 42))
    gs = GridSpec(
        10, 2, figure=fig,
        height_ratios=[0.8, 0.1, 0.45, 0.1, 1.1, 1.0, 1.0, 1.0, 1.3, 1.3],
        hspace=0.18, wspace=0.05,
        left=0.08, right=0.95,
        top=0.97, bottom=0.02,
    )

    title = f"{image_path.name}"
    if grind_setting is not None:
        if grind_estimated:
            title += f"  \u2014  Estimated DF54 Setting: ~{grind_setting}"
        else:
            title += f"  \u2014  Grind Setting {grind_setting}"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)

    n_particles = len(df)

    # ---------------------------------------------------------------
    # Row 0 (colspan 2): Histogram
    # ---------------------------------------------------------------
    ax_hist = fig.add_subplot(gs[0, :])
    if not df.empty:
        diams = df["equiv_diameter_mm"]
        ax_hist.hist(diams, bins=50, edgecolor="black", alpha=0.7, color="#5B9BD5")
        median_d = diams.median()
        mean_d = diams.mean()
        ax_hist.axvline(median_d, color="red", linestyle="--", linewidth=2,
                        label=f"Median: {median_d:.3f} mm")
        ax_hist.axvline(mean_d, color="orange", linestyle=":", linewidth=2,
                        label=f"Mean: {mean_d:.3f} mm")
        ax_hist.legend(fontsize=10, loc="upper left")
    else:
        ax_hist.text(0.5, 0.5, "No particles detected", transform=ax_hist.transAxes,
                     ha="center", va="center", fontsize=14)
    ax_hist.set_xlabel("Equivalent Diameter (mm)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Particle Size Distribution", fontsize=12, fontweight="bold")

    # Row 1: Spacer
    fig.add_subplot(gs[1, :]).axis("off")

    # --- Use pre-computed stats from result dict ---
    if not df.empty:
        diams = df["equiv_diameter_mm"]
        median_d = summary.get("median_diameter_mm", diams.median())
        mean_d = summary.get("mean_diameter_mm", diams.mean())
        std_d = summary.get("std_diameter_mm", diams.std())
        D10 = summary.get("D10", diams.quantile(0.10))
        D50 = summary.get("D50", median_d)
        D90 = summary.get("D90", diams.quantile(0.90))
        span_val = summary.get("span", (D90 - D10) / D50 if D50 > 0 else 0)
        fines = summary.get("fines_pct", (diams < 0.200).mean() * 100)
        boulders = summary.get("boulders_pct", (diams > 1.000).mean() * 100)
        D_4_3 = summary.get("D_4_3")
        D_3_2 = summary.get("D_3_2")
        if D_4_3 is None:
            d_vals = diams.values
            D_4_3 = (d_vals ** 4).sum() / (d_vals ** 3).sum()
        if D_3_2 is None:
            d_vals = diams.values
            D_3_2 = (d_vals ** 3).sum() / (d_vals ** 2).sum()

        # ---------------------------------------------------------------
        # Row 2: Visual gauges — D10–D90 range bar | Fines/Boulders | Span
        # ---------------------------------------------------------------
        from matplotlib.patches import FancyBboxPatch

        gs_gauges = gs[2, :].subgridspec(1, 3, width_ratios=[2.0, 1.0, 0.6],
                                          wspace=0.35)

        # --- D10–D50–D90 range bar ---
        ax_range = fig.add_subplot(gs_gauges[0, 0])
        ax_range.set_title("Particle Size Spread", fontsize=11,
                           fontweight="bold", pad=8)

        # Draw background range (full data extent)
        d_min, d_max = diams.min(), diams.max()
        ax_range.barh(0, d_max - d_min, left=d_min, height=0.3,
                      color="#e0e0e0", edgecolor="none", zorder=1)
        # D10–D90 range (colored bar)
        ax_range.barh(0, D90 - D10, left=D10, height=0.3,
                      color="#5B9BD5", alpha=0.5, edgecolor="none", zorder=2)
        # IQR (darker band)
        q25, q75 = diams.quantile(0.25), diams.quantile(0.75)
        ax_range.barh(0, q75 - q25, left=q25, height=0.3,
                      color="#2E75B6", alpha=0.6, edgecolor="none", zorder=3)

        # Markers
        marker_kw = dict(zorder=5, clip_on=False)
        ax_range.plot(D10, 0, "v", color="#e67e22", markersize=12, **marker_kw)
        ax_range.plot(D50, 0, "D", color="#c0392b", markersize=11, **marker_kw)
        ax_range.plot(D90, 0, "^", color="#e67e22", markersize=12, **marker_kw)

        # Labels
        ax_range.text(D10, -0.28, f"D10\n{D10:.3f}", ha="center", va="top",
                      fontsize=8, fontweight="bold", color="#e67e22")
        ax_range.text(D50, 0.28, f"D50\n{D50:.3f}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold", color="#c0392b")
        ax_range.text(D90, -0.28, f"D90\n{D90:.3f}", ha="center", va="top",
                      fontsize=8, fontweight="bold", color="#e67e22")

        ax_range.set_xlim(max(0, d_min - 0.05), d_max + 0.05)
        ax_range.set_ylim(-0.6, 0.6)
        ax_range.set_xlabel("Diameter (mm)", fontsize=9)
        ax_range.spines["top"].set_visible(False)
        ax_range.spines["right"].set_visible(False)
        ax_range.spines["left"].set_visible(False)
        ax_range.set_yticks([])

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#e67e22",
                   markersize=8, label="D10 / D90"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#c0392b",
                   markersize=8, label="D50 (Median)"),
            plt.Rectangle((0, 0), 1, 1, fc="#5B9BD5", alpha=0.5, label="D10–D90 range"),
            plt.Rectangle((0, 0), 1, 1, fc="#2E75B6", alpha=0.6, label="IQR (25th–75th)"),
        ]
        ax_range.legend(handles=legend_elements, fontsize=7, loc="upper right",
                        framealpha=0.9)

        # --- Fines / Boulders percentage bars ---
        ax_bars = fig.add_subplot(gs_gauges[0, 1])
        ax_bars.set_title("Fines & Boulders", fontsize=11,
                          fontweight="bold", pad=8)

        def _pct_color(pct: float) -> str:
            """Green for low %, yellow for moderate, red for high."""
            if pct < 5:
                return "#27ae60"
            elif pct < 15:
                return "#f39c12"
            else:
                return "#e74c3c"

        bar_labels = ["Fines\n(<0.2 mm)", "Boulders\n(>1.0 mm)"]
        bar_values = [fines, boulders]
        bar_colors = [_pct_color(fines), _pct_color(boulders)]

        bars = ax_bars.barh([1, 0], bar_values, height=0.45,
                            color=bar_colors, edgecolor="white", zorder=3)
        # Background track
        ax_bars.barh([1, 0], [100, 100], height=0.45,
                     color="#f0f0f0", edgecolor="none", zorder=1)

        for bar, val in zip(bars, bar_values):
            x_pos = max(val + 1, 5)
            ax_bars.text(x_pos, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", ha="left",
                         fontsize=10, fontweight="bold")

        ax_bars.set_xlim(0, max(max(bar_values) * 1.5, 20))
        ax_bars.set_yticks([0, 1])
        ax_bars.set_yticklabels(bar_labels, fontsize=9)
        ax_bars.set_xlabel("%", fontsize=9)
        ax_bars.spines["top"].set_visible(False)
        ax_bars.spines["right"].set_visible(False)

        # --- Span badge ---
        ax_span = fig.add_subplot(gs_gauges[0, 2])
        ax_span.axis("off")
        ax_span.set_title("Span", fontsize=11, fontweight="bold", pad=8)

        if span_val < 1.0:
            span_color = "#27ae60"   # green — tight
            span_label = "Tight"
        elif span_val < 1.5:
            span_color = "#f39c12"   # yellow — moderate
            span_label = "Moderate"
        else:
            span_color = "#e74c3c"   # red — wide
            span_label = "Wide"

        badge = FancyBboxPatch(
            (0.2, 0.25), 0.6, 0.45,
            boxstyle="round,pad=0.08",
            facecolor=span_color, edgecolor="white", linewidth=2,
            transform=ax_span.transAxes, zorder=3,
        )
        ax_span.add_patch(badge)
        ax_span.text(0.5, 0.55, f"{span_val:.2f}", transform=ax_span.transAxes,
                     ha="center", va="center", fontsize=22, fontweight="bold",
                     color="white", zorder=4)
        ax_span.text(0.5, 0.35, span_label, transform=ax_span.transAxes,
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color="white", zorder=4)
        ax_span.text(0.5, 0.12, "(D90−D10) / D50", transform=ax_span.transAxes,
                     ha="center", va="center", fontsize=7, color="#666666")

    # Row 3: Spacer between gauges and table
    fig.add_subplot(gs[3, :]).axis("off")

    # ---------------------------------------------------------------
    # Row 4 (colspan 2): Distribution statistics table
    # ---------------------------------------------------------------
    ax_stats = fig.add_subplot(gs[4, :])
    ax_stats.axis("off")
    if not df.empty:
        stats_rows = [
            ("Particles",  f"{n_particles}",              "Total particles after filtering"),
            ("Median",     f"{median_d:.3f} mm",          "Middle value of the size distribution"),
            ("Mean",       f"{mean_d:.3f} mm",            "Arithmetic average diameter"),
            ("Std Dev",    f"{std_d:.3f} mm",             "Standard deviation of diameters"),
            ("IQR",        f"[{diams.quantile(0.25):.3f}, {diams.quantile(0.75):.3f}] mm",
                                                          "Inter-quartile range (25th–75th percentile)"),
            ("D10",        f"{D10:.3f} mm",               "10th percentile — 10% of particles are smaller"),
            ("D50",        f"{D50:.3f} mm",               "50th percentile (median diameter)"),
            ("D90",        f"{D90:.3f} mm",               "90th percentile — 90% of particles are smaller"),
            ("Span",       f"{span_val:.2f}",             "(D90 − D10) / D50 — distribution width"),
            ("Fines",      f"{fines:.1f}%",               "Fraction of particles < 0.200 mm"),
            ("Boulders",   f"{boulders:.1f}%",            "Fraction of particles > 1.000 mm"),
            ("D[4,3]",     f"{D_4_3:.3f} mm",             "Volume-weighted mean (De Brouckere mean)"),
            ("D[3,2]",     f"{D_3_2:.3f} mm",             "Surface-weighted mean (Sauter mean)"),
        ]
        if grind_estimated and grind_setting is not None:
            stats_rows.append(
                ("Est. DF54", f"~{grind_setting}", "Estimated grind setting from calibration curve"),
            )

        col_labels = ["Metric", "Value", "Description"]
        table_data = [[r[0], r[1], r[2]] for r in stats_rows]

        tbl = ax_stats.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="upper center",
            cellLoc="left",
            colWidths=[0.10, 0.18, 0.40],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.25)

        # Style header row
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")

        # Style data rows with alternating colours
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                cell = tbl[i, j]
                cell.set_edgecolor("#dddddd")
                if i % 2 == 0:
                    cell.set_facecolor("#f0f4fa")
                else:
                    cell.set_facecolor("white")

        # Place title between gauges and table using figure coordinates
        stats_box = ax_stats.get_position()
        fig.text(
            stats_box.x0 + stats_box.width / 2, stats_box.y1 + 0.005,
            "Distribution Statistics", fontsize=12, fontweight="bold",
            ha="center", va="bottom",
        )

    # ---------------------------------------------------------------
    # Row 5: Original | Warped
    # ---------------------------------------------------------------
    ax_orig = fig.add_subplot(gs[5, 0])
    ax_orig.imshow(original_rgb)
    ax_orig.set_title("Original Photo", fontsize=12, fontweight="bold")
    ax_orig.axis("off")

    ax_warp = fig.add_subplot(gs[5, 1])
    ax_warp.imshow(warped_rgb)
    ax_warp.set_title("Warped (Perspective Corrected)", fontsize=12, fontweight="bold")
    ax_warp.axis("off")

    # Row 6: Normalised | Binary
    ax_norm = fig.add_subplot(gs[6, 0])
    ax_norm.imshow(normed_rgb)
    ax_norm.set_title("White-Balance Normalised", fontsize=12, fontweight="bold")
    ax_norm.axis("off")

    ax_bin = fig.add_subplot(gs[6, 1])
    if binary is not None:
        ax_bin.imshow(_ensure_landscape(binary), cmap="gray")
    else:
        ax_bin.text(0.5, 0.5, "N/A", transform=ax_bin.transAxes,
                    ha="center", va="center", fontsize=14)
    ax_bin.set_title("Adaptive Threshold", fontsize=12, fontweight="bold")
    ax_bin.axis("off")

    # Row 7: Distance transform | Sure foreground markers
    ax_dist = fig.add_subplot(gs[7, 0])
    if dist_transform is not None:
        im = ax_dist.imshow(_ensure_landscape(dist_transform), cmap="inferno")
        cax = inset_axes(ax_dist, width="3%", height="60%", loc="center right",
                         borderpad=-3.5)
        fig.colorbar(im, cax=cax, label="px")
        cax.yaxis.set_tick_params(labelsize=8)
        cax.yaxis.label.set_size(8)
    else:
        ax_dist.text(0.5, 0.5, "N/A", transform=ax_dist.transAxes,
                     ha="center", va="center", fontsize=14)
    ax_dist.set_title("Distance Transform", fontsize=12, fontweight="bold")
    ax_dist.axis("off")

    ax_markers = fig.add_subplot(gs[7, 1])
    if sure_fg is not None:
        if binary is not None:
            bg = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            peaks_colored = bg.copy()
            peaks_colored[sure_fg > 0] = [0, 255, 255]
            ax_markers.imshow(_ensure_landscape(peaks_colored))
        else:
            ax_markers.imshow(_ensure_landscape(sure_fg), cmap="gray")
    else:
        ax_markers.text(0.5, 0.5, "N/A", transform=ax_markers.transAxes,
                        ha="center", va="center", fontsize=14)
    ax_markers.set_title("Watershed Seeds (Sure Foreground)", fontsize=12, fontweight="bold")
    ax_markers.axis("off")

    # Row 8–9 (colspan 2): Overlay
    ax_overlay = fig.add_subplot(gs[8:10, :])
    ax_overlay.imshow(_ensure_landscape(overlay))
    ax_overlay.set_title(
        f"Segmentation Overlay  —  {n_particles} particles detected",
        fontsize=14, fontweight="bold",
    )
    ax_overlay.axis("off")

    summary_path = img_out_dir / "summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Pipeline summary → {summary_path}")


def save_comparison(
    img_out_dir: Path,
    warped_rgb: np.ndarray,
    models: list[str],
    grind_setting: int | None,
) -> None:
    """Save a side-by-side comparison image of model overlays."""
    available: list[tuple[str, np.ndarray]] = []
    for m in models:
        path = img_out_dir / f"overlay_{m}.jpg"
        if path.exists():
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            available.append((m, img))

    if len(available) >= 2:
        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, available):
            ax.imshow(img)
            ax.set_title(name, fontsize=14)
            ax.axis("off")
        title = "Model comparison"
        if grind_setting is not None:
            title += f"  —  Grind setting {grind_setting}"
        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(img_out_dir / "comparison.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Comparison → {img_out_dir / 'comparison.png'}")
