from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from build_section34_mechanism_figure_v3 import (
    HOURLY,
    load_dem,
    sector_polygon,
    site_table,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/cma_validation_prd_all_events/terrain_glo30_v2_only/directional_physics_audit/exposure_skill_all_strong_background.csv.gz"
OUT = ROOT / "outputs/discussion_cn"

INK = "#18242E"
MUTED = "#66737E"
BLUE = "#2878B5"
GREEN = "#4D9553"
LAND = "#DDD7C6"
SEA = "#EAF3F6"


def smooth_skill_surface(d, gx, wy, hg=0.060, hw=1.10, min_neff=28):
    g = d.exposure_gap.to_numpy(float)
    w = d.era_speed_mean.to_numpy(float)
    era = d.era_err.to_numpy(float)
    gain = (d.era_err - d.v2_err).to_numpy(float)
    z = np.full((len(wy), len(gx)), np.nan)
    neff = np.zeros_like(z)
    for iy, yy in enumerate(wy):
        dw = ((w - yy) / hw) ** 2
        for ix, xx in enumerate(gx):
            weights = np.exp(-0.5 * (dw + ((g - xx) / hg) ** 2))
            sw = weights.sum()
            if sw <= 0:
                continue
            neff[iy, ix] = sw * sw / np.square(weights).sum()
            denom = np.sum(weights * era)
            if denom > 0:
                z[iy, ix] = 100 * np.sum(weights * gain) / denom
    if min_neff > 0:
        z[neff < min_neff] = np.nan
    return z, neff


def load_actual_fingerprints():
    """Return three observed station-pair hours spanning low to high G."""
    sites, meta = site_table()
    key_to_site = dict(zip(sites.station_key.astype(str), sites.site))
    selected = [
        ("28|62", "2008_2008229N13147", "2008-08-22 06:00:00", "Similar upstream pathways"),
        ("6|35", "2012_2012201N15129", "2012-07-23 02:00:00", "Partly differentiated pathways"),
        ("2|38", "2023_2023271N14144", "2023-10-07 21:00:00", "Marine–sheltered split"),
    ]
    h = pd.read_csv(
        HOURLY,
        usecols=["station_key", "event_id", "time_utc", "source_direction_deg", "ocean_fraction_upwind"],
    )
    h["site"] = h.station_key.astype(str).map(key_to_site)
    h["time_utc"] = h.time_utc.astype(str)
    cases = []
    for pair_id, event_id, time_utc, title in selected:
        a, b = [int(v) for v in pair_id.split("|")]
        q = h[(h.event_id == event_id) & (h.time_utc == time_utc) & h.site.isin([a, b])]
        q = q.groupby("site", as_index=True).agg(
            source=("source_direction_deg", "mean"),
            ocean=("ocean_fraction_upwind", "mean"),
        )
        if not {a, b}.issubset(q.index):
            raise RuntimeError(f"Missing station-pair hour for {pair_id} at {time_utc}")
        cases.append({
            "pair_id": pair_id,
            "title": title,
            "a": meta.loc[a],
            "b": meta.loc[b],
            "source": float(q.loc[[a, b], "source"].mean()),
            "ea": float(q.loc[a, "ocean"]),
            "eb": float(q.loc[b, "ocean"]),
        })
    return cases


def terrain_glyph(ax, case):
    """Draw an actual DEM/coastline fingerprint for one observed station-pair hour."""
    a, b = case["a"], case["b"]
    source = case["source"]
    polygons = [
        sector_polygon(a.lon, a.lat, source, radius_km=16, half_angle=15),
        sector_polygon(b.lon, b.lat, source, radius_km=16, half_angle=15),
    ]
    allx = np.concatenate([[a.lon, b.lon], polygons[0][:, 0], polygons[1][:, 0]])
    ally = np.concatenate([[a.lat, b.lat], polygons[0][:, 1], polygons[1][:, 1]])
    dx = max(allx.max() - allx.min(), .16)
    dy = max(ally.max() - ally.min(), .12)
    bounds = (allx.min() - .09 * dx, allx.max() + .09 * dx,
              ally.min() - .13 * dy, ally.max() + .13 * dy)
    # Expand, rather than stretch, each real map to the same wide viewport.
    # This preserves geographic proportions while letting the three examples
    # use the full horizontal width allocated to them.
    xmin, xmax, ymin, ymax = bounds
    mean_lat = .5 * (ymin + ymax)
    target_ratio = 1.68
    projected_ratio = (xmax - xmin) * np.cos(np.deg2rad(mean_lat)) / (ymax - ymin)
    if projected_ratio < target_ratio:
        wanted_dx = target_ratio * (ymax - ymin) / np.cos(np.deg2rad(mean_lat))
        extra = .5 * (wanted_dx - (xmax - xmin))
        xmin, xmax = xmin - extra, xmax + extra
    else:
        wanted_dy = (xmax - xmin) * np.cos(np.deg2rad(mean_lat)) / target_ratio
        extra = .5 * (wanted_dy - (ymax - ymin))
        ymin, ymax = ymin - extra, ymax + extra
    bounds = (xmin, xmax, ymin, ymax)
    xs, ys, dem = load_dem(bounds, max_size=260)
    ax.set_facecolor(SEA)
    land = np.ma.masked_where(dem <= .8, dem)
    terrain = LinearSegmentedColormap.from_list(
        "fingerprint_terrain", ["#EEECE3", "#D8DCCA", "#B9C3A9", "#899B7E"]
    )
    ax.contourf(xs, ys, land, levels=[1, 20, 60, 120, 220, 380, 650, 1000],
                cmap=terrain, extend="max", antialiased=True)
    ax.contour(xs, ys, dem, levels=[.8], colors="#73868A", linewidths=.65)
    ax.contour(xs, ys, np.ma.masked_where(dem < 80, dem), levels=[100, 250, 500],
               colors="#7C8579", linewidths=.25, alpha=.45)
    for row, poly, color in [(a, polygons[0], BLUE), (b, polygons[1], GREEN)]:
        ax.add_patch(Polygon(poly, closed=True, fc=color, ec=color, lw=1.25,
                             ls=(0, (4, 2)), alpha=.20, zorder=5))
        ax.scatter([row.lon], [row.lat], s=58, fc="white", ec=color, lw=1.8, zorder=8)
    g = abs(case["ea"] - case["eb"])
    ax.text(.035, .95, f"Observed pair   $G$ = {g:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.7, weight="bold", color=INK,
            bbox=dict(fc="white", ec="none", alpha=.76, pad=1.5), zorder=10)
    ax.text(.5, -.10, case["title"], ha="center", va="top", fontsize=10.5,
            weight="bold", color=INK, transform=ax.transAxes)
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(bounds[2:]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#A6B0B6")
        spine.set_linewidth(.8)


def make_figure(variant="integrated", min_neff=0):
    d = pd.read_csv(SRC)
    gx = np.linspace(0.0, 0.58, 160)
    wy = np.linspace(8.0, 16.0, 145)
    z, neff = smooth_skill_surface(d, gx, wy, min_neff=min_neff)

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12.5,
        "axes.labelsize": 14,
        "axes.edgecolor": "#66727C",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(13.4, 8.8), facecolor="white")
    ax = fig.add_axes([.095, .285, .725, .625])
    cax = fig.add_axes([.855, .325, .027, .545])

    cmap = LinearSegmentedColormap.from_list(
        "skill", ["#2F6597", "#A7CBE0", "#F7F7F3", "#F2C86E", "#CB593A"]
    )
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=45)
    mesh = ax.pcolormesh(gx, wy, z, cmap=cmap, norm=norm,
                         shading="auto", rasterized=False)

    # A deterministic subsample prevents opaque vertical stacks while retaining data support.
    sample = d.sample(min(700, len(d)), random_state=20260916)
    ax.scatter(sample.exposure_gap, sample.era_speed_mean, s=7,
               facecolors="white", edgecolors="#42515C", linewidths=.22,
               alpha=.26, zorder=3)

    finite = np.isfinite(z)
    levels = [0, 20, 35]
    cs = ax.contour(gx, wy, np.where(finite, z, np.nan), levels=levels,
                    colors=["#4F616D", "#A76921", "#812F25"],
                    linewidths=[1.15, 1.55, 1.75], zorder=4)
    ax.clabel(cs, fmt={0: "0%", 20: "20%", 35: "35%"}, inline=True,
              fontsize=10, colors=["#4F616D", "#8B571D", "#75291F"])

    # The line is descriptive, not a hard threshold; the wording is deliberately cautious.
    ax.axvline(.188, color="#4D5962", lw=1.1, ls=(0, (5, 4)), zorder=5)
    ax.text(.196, 15.72, "onset of the consistently positive-gain region",
            ha="left", va="top", fontsize=10.2, color="#3B474F")
    ax.text(.055, 14.55, "Higher wind speed alone\ndoes not produce stable gain",
            ha="center", va="center", fontsize=11.4, weight="bold",
            color="#244C69")
    ax.text(.350, 14.70, "Directional exposure contrast establishes the gain;\n"
                         "stronger flow enlarges it",
            ha="center", va="center", fontsize=11.8, weight="bold",
            color="#7B3424")

    anchors = [
        (.36, 8.75, "+24.3%", "95% CI +10.6 to +36.7"),
        (.36, 9.85, "+35.9%", "95% CI +14.5 to +46.5"),
        (.36, 12.25, "+37.3%", "95% CI +21.8 to +45.2"),
    ]
    for x, y, value, ci in anchors:
        ax.scatter([x], [y], s=105, fc="white", ec="#8B3E28", lw=1.55, zorder=7)
        ax.text(x+.018, y+.02, value, ha="left", va="center", fontsize=12.8,
                weight="bold", color="#7C321F", zorder=8)
        ax.text(x+.018, y-.30, ci, ha="left", va="top", fontsize=8.8,
                color="#6D473D", zorder=8)

    ax.set_xlim(0, .58)
    ax.set_ylim(8, 16)
    ax.set_xlabel("Directional terrain-exposure contrast, $G$", labelpad=9, weight="bold")
    ax.set_ylabel("ERA5 background wind speed (m/s)", weight="bold")
    ax.grid(color="white", lw=.55, alpha=.35)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Reduction in within-cell contrast MAE (%)", weight="bold", labelpad=12)
    cbar.ax.tick_params(labelsize=11)

    # A small marginal histogram documents where the samples actually lie without becoming a panel.
    if variant == "integrated":
        hist_ax = fig.add_axes([.095, .918, .725, .045], sharex=ax)
        bins = np.linspace(0, .58, 30)
        xk = np.linspace(0, .58, 400)
        values = d.exposure_gap.to_numpy(float)
        bw = .018
        # Kernel estimate expressed as an equivalent number of pair-hours in
        # a 0.02-wide G interval, so the small vertical scale remains quantitative.
        density = np.exp(-.5 * ((xk[:, None] - values[None, :]) / bw) ** 2).sum(axis=1)
        density = density / (np.sqrt(2*np.pi) * bw) * .02
        ymax = float(density.max())
        upper_tick = max(20, int(np.ceil(ymax / 20) * 20))
        lower_tick = int(upper_tick / 2)
        hist_ax.fill_between(xk, 0, density, color="#87959F", alpha=.24)
        hist_ax.plot(xk, density, color="#65737D", lw=1.35)
        hist_ax.set_ylim(0, upper_tick * 1.08)
        hist_ax.set_yticks([lower_tick, upper_tick])
        hist_ax.set_ylabel("Pair-hours\nper 0.02 $G$", fontsize=8.0,
                           rotation=0, labelpad=27, va="center", color=MUTED)
        hist_ax.tick_params(axis="y", labelsize=8.0, colors=MUTED,
                            length=3, width=.7, pad=2)
        hist_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for name, spine in hist_ax.spines.items():
            spine.set_visible(name == "left")
            spine.set_color("#7B8790")
            spine.set_linewidth(.7)

    # Three terrain fingerprints explain the physical meaning of the x axis.
    fig.text(.095, .222, "Observed terrain fingerprints spanning the $G$ continuum",
             ha="left", va="center", fontsize=10.2, color=MUTED, style="italic")
    positions = [.095, .340, .585]
    cases = load_actual_fingerprints()
    for x, case in zip(positions, cases):
        gax = fig.add_axes([x, .052, .235, .140])
        terrain_glyph(gax, case)

    fig.text(.817, .927, "1,466 pair-hours; 36 typhoons",
             ha="right", va="center", fontsize=9.2, color=MUTED)

    base = "integrated" if variant == "integrated" else "minimal"
    suffix = base if min_neff == 0 else f"{base}_neff{min_neff:g}"
    png = OUT / f"Figure11_directional_terrain_skill_landscape_{suffix}.png"
    pdf = OUT / f"Figure11_directional_terrain_skill_landscape_{suffix}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["integrated", "minimal"], default="integrated")
    parser.add_argument("--min-neff", type=float, default=0)
    args = parser.parse_args()
    make_figure(args.variant, args.min_neff)
