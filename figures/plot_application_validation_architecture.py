from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "application_release"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2C73B9"
ORANGE = "#E48118"
GREEN = "#31845B"
NAVY = "#173A5E"
INK = "#26323D"
GRID = "#D9DEE4"


def fit_line(x, y):
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return slope, intercept


def scatter_panel(ax, x, era, phy, corrected, limits, xlabel, ylabel):
    lo, hi = limits
    xx = np.linspace(lo, hi, 240)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_facecolor("#FBFCFD")
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.76, zorder=0)
    ax.plot(xx, xx, color="#22272E", linestyle=(0, (5, 4)), linewidth=1.55, zorder=1)

    # Identical symbol sizes and strokes in both validation panels.
    ax.scatter(x, era, s=34, marker="o", facecolor="white", edgecolor=BLUE,
               linewidth=1.15, alpha=1.0, zorder=2)
    ax.scatter(x, phy, s=34, marker="D", facecolor=ORANGE, edgecolor="white",
               linewidth=0.55, alpha=1.0, zorder=3)
    ax.scatter(x, corrected, s=34, marker="^", facecolor=GREEN, edgecolor="white",
               linewidth=0.55, alpha=1.0, zorder=4)

    for values, color in [(era, BLUE), (phy, ORANGE), (corrected, GREEN)]:
        slope, intercept = fit_line(x, values)
        ax.plot(xx, intercept + slope * xx, color=color, linewidth=2.25, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color("#65717D")
        spine.set_linewidth(0.85)


def add_box(ax, xy, wh, title, lines, edge=NAVY, face="#F7FAFC", title_color=NAVY):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.25, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(x + 0.5 * w, y + h - 0.24 * h, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=12.1, fontweight="bold",
            color=title_color)
    ax.text(x + 0.5 * w, y + 0.35 * h, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=10.2, color=INK, linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#65717D", lw=1.55):
    ax.annotate("", xy=end, xytext=start, xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=4, shrinkB=4, mutation_scale=12))


def main():
    event = pd.read_csv(OUT.parent / "ibtracs_event_peak_trial" / "event_peak_pairs.csv")
    cal = pd.read_csv(OUT.parent / "ibtracs_event_peak_trial" / "calibration_candidates_loo.csv")
    event = event.merge(
        cal[["storm_id", "three_band_pressure_residual_1"]],
        on="storm_id", how="inner", validate="one_to_one",
    )
    station = pd.read_csv(OUT.parent / "hourly_postprocess_screen" /
                          "pressure_only_station_peak_predictions.csv")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12.0,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.2,
        "ytick.labelsize": 11.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(12.0, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.56], hspace=0.28, wspace=0.25,
                          left=0.075, right=0.985, bottom=0.055, top=0.925)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    scatter_panel(
        ax_a,
        event["ib_peak_domain_10min_ms"], event["era5_event_max_ms"],
        event["phytopo_event_max_ms"], event["three_band_pressure_residual_1"],
        (8, 68),
        "IBTrACS event-peak wind speed (m/s)",
        "Gridded event-peak wind speed (m/s)",
    )
    scatter_panel(
        ax_b,
        station["obs"], station["era"], station["phy"],
        station["pressure_only_spread_tail15_w16"],
        (4, 53),
        "Observed station-event peak (m/s)",
        "Reconstructed station-event peak (m/s)",
    )
    ax_a.text(-0.14, 1.045, "A", transform=ax_a.transAxes, fontsize=17,
              fontweight="bold", color=INK)
    ax_b.text(-0.14, 1.045, "B", transform=ax_b.transAxes, fontsize=17,
              fontweight="bold", color=INK)

    handles = [
        Line2D([0], [0], marker="o", color=BLUE, markerfacecolor="white",
               markeredgecolor=BLUE, lw=2.0, markersize=6.4, label="ERA5-Interp."),
        Line2D([0], [0], marker="D", color=ORANGE, markerfacecolor=ORANGE,
               markeredgecolor="white", lw=2.0, markersize=6.0, label="PhyTopo"),
        Line2D([0], [0], marker="^", color=GREEN, markerfacecolor=GREEN,
               markeredgecolor="white", lw=2.0, markersize=6.4,
               label="Pressure-constrained"),
        Line2D([0], [0], color="#22272E", lw=1.4, linestyle=(0, (5, 4)),
               label="1:1 line"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.985),
               ncol=4, frameon=False, columnspacing=1.45, handlelength=2.2,
               fontsize=12.2)

    ax_c.set_axis_off()
    ax_c.text(-0.005, 1.03, "C", transform=ax_c.transAxes, fontsize=17,
              fontweight="bold", color=INK)
    ax_c.text(0.035, 1.03, "Reproducible inference and product-generation workflow",
              transform=ax_c.transAxes, fontsize=13.4, fontweight="bold",
              color=INK, va="center")

    add_box(ax_c, (0.005, 0.42), (0.185, 0.40), "Dynamic forcing",
            ["ERA5 hourly fields", "U10, V10, MSL, T2,", "SST, BLH and FSR"],
            edge=BLUE, face="#F3F8FC", title_color=BLUE)
    add_box(ax_c, (0.005, 0.02), (0.185, 0.27), "Static predictors",
            ["3-km terrain, slope,", "land mask and coordinates"],
            edge="#7A8793", face="#F7F8F9", title_color="#4D5964")
    add_box(ax_c, (0.265, 0.27), (0.235, 0.48), "PhyTopo-Net",
            ["dual encoders  →  cross-attention", "residual backbone  →  dual heads",
             "overlapping-patch inference with", "Hann-window blending"],
            edge=NAVY, face="#F3F6FA", title_color=NAVY)
    add_box(ax_c, (0.575, 0.50), (0.190, 0.30), "Direct reconstruction",
            ["3-km U10, V10 and MSL", "hourly NetCDF product"],
            edge=ORANGE, face="#FFF8EF", title_color="#A85A00")
    add_box(ax_c, (0.575, 0.05), (0.190, 0.30), "Full-field postprocessing",
            ["regional and track-aware", "hourly corrected U10, V10", "and wind speed NetCDF"],
            edge=GREEN, face="#F3FAF6", title_color=GREEN)
    add_box(ax_c, (0.835, 0.27), (0.155, 0.48), "Open release",
            ["117-event catalogue", "direct + corrected NetCDF", "CPU / GPU inference",
             "versioned provenance"],
            edge=NAVY, face="#F7FAFC", title_color=NAVY)

    arrow(ax_c, (0.190, 0.61), (0.265, 0.61), BLUE)
    arrow(ax_c, (0.190, 0.16), (0.265, 0.39), "#7A8793")
    arrow(ax_c, (0.500, 0.56), (0.575, 0.65), ORANGE)
    arrow(ax_c, (0.500, 0.40), (0.575, 0.20), GREEN)
    arrow(ax_c, (0.765, 0.65), (0.835, 0.62), ORANGE)
    arrow(ax_c, (0.765, 0.20), (0.835, 0.39), GREEN)
    arrow(ax_c, (0.670, 0.50), (0.670, 0.35), GREEN)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"Figure_application_validation_architecture.{ext}",
                    dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(OUT / "Figure_application_validation_architecture.png")


if __name__ == "__main__":
    main()
