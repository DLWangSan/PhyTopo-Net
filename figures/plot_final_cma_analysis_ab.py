"""Plot actual 117-event and CMA-station validation of the final analysis archive."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
EVENT_REFERENCE = ROOT / "outputs" / "ibtracs_event_peak_trial" / "calibration_candidates_loo.csv"
EVENT_ANALYSIS = ROOT / "outputs" / "cma_residual_assimilation" / "cma_analysis_event_peaks.csv"
STATION_ANALYSIS = ROOT / "outputs" / "cma_residual_assimilation" / "cma_analysis_archive_station_predictions.csv"
STATION_SOURCE = ROOT / "outputs" / "hourly_postprocess_screen" / "cma_hourly_with_ibtrack.csv.gz"
OUT = ROOT / "output" / "pdf"

BLUE = "#397CB8"
ORANGE = "#E48016"
GREEN = "#34875D"
INK = "#202A33"
GRID = "#D7DDE3"


def metrics(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (np.mean(np.abs(y - x)), np.sqrt(np.mean((y - x) ** 2)),
            np.corrcoef(x, y)[0, 1])


def panel(ax, x, series, limits, xlabel, ylabel, letter):
    lo, hi = limits
    line = np.linspace(lo, hi, 300)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
    ax.set_facecolor("#FBFCFD")
    ax.grid(color=GRID, lw=0.75, alpha=0.9, zorder=0)
    ax.plot(line, line, color="#282D31", lw=1.55, ls=(0, (5, 4)), zorder=1)
    markers = ["o", "D", "^"]
    colors = [BLUE, ORANGE, GREEN]
    text_rows = []
    for (label, values), marker, color in zip(series, markers, colors):
        face = "white" if label == "ERA5-Interp." else color
        size = 31 if len(x) < 300 else 20
        alpha = 0.88 if len(x) < 300 else 0.60
        ax.scatter(x, values, s=size, marker=marker, facecolor=face,
                   edgecolor=color if label == "ERA5-Interp." else "white",
                   linewidth=0.75, alpha=alpha, zorder=2)
        slope, intercept = np.polyfit(x, values, 1)
        ax.plot(line, intercept + slope * line, color=color, lw=2.45, zorder=3)
        mae, rmse, corr = metrics(x, values)
        text_rows.append(f"{label:<25} MAE={mae:4.2f}  r={corr:.2f}")
    ax.text(0.035, 0.965, "\n".join(text_rows), transform=ax.transAxes,
            va="top", ha="left", fontsize=11.8, color=INK, linespacing=1.38,
            bbox=dict(boxstyle="round,pad=0.44", fc="white", ec="#AAB4BD", alpha=0.95),
            family="DejaVu Sans Mono")
    ax.text(-0.125, 1.03, letter, transform=ax.transAxes, fontsize=22,
            fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    for spine in ax.spines.values():
        spine.set_color("#65717C"); spine.set_linewidth(0.95)


def load_station_peaks():
    pred = pd.read_csv(STATION_ANALYSIS).rename(columns={"obs": "obs_speed", "phy": "v2_speed"})
    src = pd.read_csv(STATION_SOURCE, usecols=[
        "event_id", "station_id", "time_utc", "obs_speed", "v2_speed", "era5_bilinear_speed"
    ])
    keys = ["event_id", "station_id", "time_utc", "obs_speed", "v2_speed"]
    pred["_duplicate"] = pred.groupby(keys).cumcount()
    src["_duplicate"] = src.groupby(keys).cumcount()
    merged = pred.merge(src, on=keys + ["_duplicate"], how="left", validate="one_to_one")
    if merged.era5_bilinear_speed.isna().any():
        raise RuntimeError("ERA5 station matching failed")
    return merged.groupby(["event_id", "station_id"], as_index=False).agg(
        obs=("obs_speed", "max"), era=("era5_bilinear_speed", "max"),
        phy=("v2_speed", "max"), analysis=("analysis", "max")
    )


def main():
    event = pd.read_csv(EVENT_REFERENCE).merge(
        pd.read_csv(EVENT_ANALYSIS), left_on="storm_id", right_on="event_id", validate="one_to_one"
    )
    station = load_station_peaks()
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 15, "axes.labelsize": 16.5, "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.25), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.988, bottom=0.13, top=0.84, wspace=0.27)
    panel(
        axes[0], event.ib_peak_domain_10min_ms,
        [("ERA5-Interp.", event.era5_event_max_ms),
         ("PhyTopo", event.phytopo_event_max_ms),
         ("CMA-constrained analysis", event.analysis_event_max_ms)],
        (8, 68), "IBTrACS event-peak wind speed (m/s)",
        "Regional event-maximum wind speed (m/s)", "A",
    )
    panel(
        axes[1], station.obs,
        [("ERA5-Interp.", station.era),
         ("PhyTopo", station.phy),
         ("CMA-constrained analysis", station.analysis)],
        (0, 54), "Observed station-event peak (m/s)",
        "Analyzed station-event peak (m/s)", "B",
    )
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, markerfacecolor="white",
               markeredgecolor=BLUE, lw=2.3, markersize=7, label="ERA5-Interp."),
        Line2D([0], [0], marker="D", color=ORANGE, markerfacecolor=ORANGE,
               markeredgecolor="white", lw=2.3, markersize=6.6, label="PhyTopo"),
        Line2D([0], [0], marker="^", color=GREEN, markerfacecolor=GREEN,
               markeredgecolor="white", lw=2.3, markersize=7,
               label="CMA-constrained analysis"),
        Line2D([0], [0], color="#282D31", lw=1.55, ls=(0, (5, 4)), label="1:1 line"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.975),
               frameon=False, ncol=4, fontsize=14.0, handlelength=2.2, columnspacing=1.45)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "Figure_application_validation_AB_final_archive.pdf"
    png = OUT / "Figure_application_validation_AB_final_archive.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(pdf); print(png)


if __name__ == "__main__":
    main()
