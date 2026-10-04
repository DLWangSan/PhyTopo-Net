#!/usr/bin/env python3
"""Plot native ERA5, interpolated ERA5, and PhyTopo instantaneous winds."""

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/talim_spatial_case"
ERA_FILE = DATA / "2023_2023194N16123_ERA5_sfc.nc"
PHY_FILE = DATA / "2023_2023194N16123_phytopo_recon.nc"
DEM_FILE = ROOT / "data/copernicus_dem_glo30_prd/copernicus_glo30_prd_buffered_111p8_115E_21p5_23p8N.tif"
HOURLY = ROOT / "outputs/cma_validation_prd_all_events/recon_v2_comparison/hourly_old_v2.csv.gz"
OUT = ROOT / "outputs/cma_validation_prd_all_events/terrain_glo30_v2_only/section34_mechanism_figure/spatial_wind_comparison"

CASE_TIME = pd.Timestamp("2023-07-17 00:00:00")
EVENT = "2023_2023194N16123"
EXTENT = (113.70, 114.35, 22.00, 22.50)
CELL = (113.875, 114.125, 22.125, 22.375)
STATIONS = {
    "A": {"lon": 113.91, "lat": 22.30, "color": "#2878B5"},
    "B": {"lon": 114.0267, "lat": 22.2011, "color": "#4E8C4A"},
}


def regular_interp(lat, lon, field, target_lat, target_lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    z = np.asarray(field, float)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    f = RegularGridInterpolator((lat, lon), z, bounds_error=False, fill_value=np.nan)
    return f(np.c_[target_lat.ravel(), target_lon.ravel()]).reshape(target_lat.shape)


def dem_coastline(extent, max_size=500):
    xmin, xmax, ymin, ymax = extent
    with rasterio.open(DEM_FILE) as ds:
        win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()
        scale = max(win.width, win.height) / max_size
        oh = max(50, int(win.height / max(scale, 1)))
        ow = max(50, int(win.width / max(scale, 1)))
        z = ds.read(1, window=win, out_shape=(oh, ow)).astype(float)
        tr = ds.window_transform(win) * ds.transform.scale(win.width / ow, win.height / oh)
        xs = tr.c + (np.arange(ow) + .5) * tr.a
        ys = tr.f + (np.arange(oh) + .5) * tr.e
    return xs, ys, z


def station_case_values():
    h = pd.read_csv(HOURLY, parse_dates=["time_utc"])
    q = h[(h.event_id == EVENT) & (h.time_utc == CASE_TIME)].copy()
    rows = {}
    for label, spec in STATIONS.items():
        d2 = (q.lon - spec["lon"]) ** 2 + (q.lat - spec["lat"]) ** 2
        r = q.loc[d2.idxmin()]
        rows[label] = {
            "obs": float(r.obs_speed),
            "era": float(r.era_bilinear_speed_uv),
            "phy": float(r.v2_speed),
        }
    return rows


def add_common_layers(ax, coast, station_values, value_key=None):
    xs, ys, dem = coast
    ax.contour(xs, ys, dem, levels=[0.8], colors="#26343D", linewidths=.75, zorder=8)
    ax.add_patch(Rectangle((CELL[0], CELL[2]), CELL[1] - CELL[0], CELL[3] - CELL[2],
                           fill=False, ec="#111820", lw=1.55, ls=(0, (5, 3)), zorder=10))
    for label, spec in STATIONS.items():
        ax.scatter(spec["lon"], spec["lat"], s=105, facecolor="white",
                   edgecolor=spec["color"], linewidth=2.2, zorder=12)
        ax.text(spec["lon"], spec["lat"], label, ha="center", va="center",
                color=spec["color"], weight="bold", fontsize=10, zorder=13)
        if value_key:
            model_name = "ERA5" if value_key == "era" else "PhyTopo"
            value_text = (f"Obs {station_values[label]['obs']:.1f}\n"
                          f"{model_name} {station_values[label][value_key]:.1f}")
        else:
            value_text = f"Obs {station_values[label]['obs']:.1f}"
        ax.text(spec["lon"] + .014, spec["lat"] + .012, value_text,
                color=spec["color"], fontsize=7.8, weight="bold", zorder=13,
                linespacing=1.12,
                bbox=dict(fc="white", ec=spec["color"], lw=.45,
                          alpha=.82, pad=1.0))
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
    ax.set_aspect(1 / math.cos(math.radians(np.mean(EXTENT[2:]))))
    ax.set_xlabel("Longitude (deg E)")
    ax.grid(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(ERA_FILE) as era, xr.open_dataset(PHY_FILE) as phy:
        eu = era.u10.sel(valid_time=CASE_TIME).values
        ev = era.v10.sel(valid_time=CASE_TIME).values
        elat = era.latitude.values
        elon = era.longitude.values
        pu = phy.u10.sel(time=CASE_TIME).values
        pv = phy.v10.sel(time=CASE_TIME).values
        plat = phy.XLAT.values
        plon = phy.XLONG.values

    era_speed = np.hypot(eu, ev)
    iu = regular_interp(elat, elon, eu, plat, plon)
    iv = regular_interp(elat, elon, ev, plat, plon)
    interp_speed = np.hypot(iu, iv)
    phy_speed = np.hypot(pu, pv)

    mask_phy = ((plon >= EXTENT[0]) & (plon <= EXTENT[1]) &
                (plat >= EXTENT[2]) & (plat <= EXTENT[3]))
    combined = np.r_[interp_speed[mask_phy], phy_speed[mask_phy]]
    lo = max(0, np.floor(np.nanpercentile(combined, 1)))
    hi = np.ceil(np.nanpercentile(combined, 99))
    levels = np.linspace(lo, hi, 15)

    cmap = LinearSegmentedColormap.from_list(
        "wind", ["#E8F2F6", "#A7D1DB", "#55A6B7", "#F0C36A", "#D76A3A", "#8C2431"]
    )
    coast = dem_coastline(EXTENT)
    station_values = station_case_values()

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 9.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.edgecolor": "#6F7B83", "axes.linewidth": .8,
    })
    fig, axs = plt.subplots(1, 3, figsize=(14.7, 5.5), sharex=True, sharey=True)

    # Native cell edges for the 0.25-degree ERA5 grid.
    lon_edges = np.r_[elon - .125, elon[-1] + .125]
    lat_asc = elat[::-1]
    era_asc = era_speed[::-1, :]
    lat_edges = np.r_[lat_asc - .125, lat_asc[-1] + .125]
    m0 = axs[0].pcolormesh(lon_edges, lat_edges, era_asc, cmap=cmap,
                           vmin=lo, vmax=hi, shading="flat")
    axs[0].set_title("a  Native ERA5 wind field (0.25 deg)", loc="left", weight="bold")
    add_common_layers(axs[0], coast, station_values)

    m1 = axs[1].contourf(plon, plat, interp_speed, levels=levels, cmap=cmap, extend="both")
    axs[1].set_title("b  ERA5 bilinear interpolation (3-km grid)", loc="left", weight="bold")
    add_common_layers(axs[1], coast, station_values, "era")

    m2 = axs[2].contourf(plon, plat, phy_speed, levels=levels, cmap=cmap, extend="both")
    axs[2].set_title("c  PhyTopo reconstruction (3-km grid)", loc="left", weight="bold")
    add_common_layers(axs[2], coast, station_values, "phy")

    # Sparse arrows show that differences involve both speed and local vector structure.
    for ax, uu, vv in [(axs[1], iu, iv), (axs[2], pu, pv)]:
        sub = (slice(None, None, 8), slice(None, None, 8))
        inside = ((plon[sub] >= EXTENT[0]) & (plon[sub] <= EXTENT[1]) &
                  (plat[sub] >= EXTENT[2]) & (plat[sub] <= EXTENT[3]))
        ax.quiver(plon[sub][inside], plat[sub][inside], uu[sub][inside], vv[sub][inside],
                  color="#24343D", alpha=.62, scale=360, width=.0030,
                  headwidth=3.2, headlength=4.0, zorder=9)

    axs[0].set_ylabel("Latitude (deg N)")
    fig.suptitle("Instantaneous 10-m wind field within and around the selected ERA5 cell\n"
                 "Typhoon Talim (2023), 2023-07-17 00:00 UTC",
                 fontsize=14.2, weight="bold", color="#143D63", y=.99)
    cbar = fig.colorbar(m2, ax=axs, orientation="horizontal", fraction=.060,
                        pad=.22, aspect=48)
    cbar.set_label("10-m wind speed (m s$^{-1}$)")
    fig.subplots_adjust(left=.055, right=.985, top=.82, bottom=.29, wspace=.08)

    pdf = OUT / "talim_era5_interpolation_phytopo_wind_fields.pdf"
    png = OUT / "talim_era5_interpolation_phytopo_wind_fields.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=400, facecolor="white")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
