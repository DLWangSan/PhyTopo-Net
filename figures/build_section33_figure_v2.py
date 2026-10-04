from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PHYTOPO_DATA_ROOT", ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("PHYTOPO_OUTPUT_ROOT", ROOT / "outputs"))
ERA5_PATH = DATA_ROOT / "2018_2018250N12170_ERA5_sfc.nc"
PHY_PATH = DATA_ROOT / "2018_2018250N12170_phytopo_recon_v2.nc"
GEO_PATH = Path(os.environ.get("PHYTOPO_GEO_FILE", ROOT / "assets" / "geo_em.d03.nc"))
COAST_DEM_PATH = Path(os.environ.get("PHYTOPO_DEM_PATH", DATA_ROOT / "coastal_dem.tif"))
DATA_PATH = OUTPUT_ROOT / "section33_v2" / "hko_hourly_merged_v2.csv"
AGG_PATH = OUTPUT_ROOT / "section33_v2" / "hko_aggregate_old_v2.csv"
STATION_PATH = OUTPUT_ROOT / "section33_v2" / "hko_per_station_old_v2.csv"
CMA_PEAK_PATH = (
    OUTPUT_ROOT
    / "cma_validation_prd_all_events"
    / "recon_v2_comparison"
    / "station_event_peaks_old_v2.csv"
)
OUT = OUTPUT_ROOT / "section33_v2"

SNAPSHOT = pd.Timestamp("2018-09-16 05:00:00")
MAP_EXTENT = (110.8, 116.8, 19.5, 24.8)
PEAK_WINDOW = (pd.Timestamp("2018-09-15 18:00:00"), pd.Timestamp("2018-09-16 13:00:00"))

COL_OBS = "#20252b"
COL_ERA = "#3b78b4"
COL_PHY = "#df7f18"
COL_GRID = "#d9e0e6"


def panel_label(ax: plt.Axes, text: str, x: float = -0.06, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=16.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111820",
        clip_on=False,
    )


def format_map(ax: plt.Axes, left: bool) -> None:
    ax.set_xlim(MAP_EXTENT[:2])
    ax.set_ylim(MAP_EXTENT[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=9.5, length=3, width=0.7, color="#66727c")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}°E"))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.0f}°N"))
    ax.set_xlabel("Longitude", fontsize=10.2, labelpad=3)
    if left:
        ax.set_ylabel("Latitude", fontsize=10.2, labelpad=3)
    else:
        ax.tick_params(labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#66727c")
        spine.set_linewidth(0.75)


def interp_era5_to_target(
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]
    interpolator = RegularGridInterpolator(
        (lat, lon), field, method=method, bounds_error=False, fill_value=np.nan
    )
    points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    return interpolator(points).reshape(target_lat.shape)


def add_station_markers(ax: plt.Axes, stations: pd.DataFrame, annotate: bool = False) -> None:
    offsets = {"WGL": (6, -6), "CCH": (-29, -8), "TO": (-23, 7), "LFS": (-10, 9)}
    for row in stations.itertuples(index=False):
        ax.scatter(
            row.lon,
            row.lat,
            s=28,
            marker="^",
            facecolor="white",
            edgecolor="#17212b",
            linewidth=0.8,
            zorder=8,
        )
        if annotate:
            dx, dy = offsets.get(row.station, (4, 4))
            ax.annotate(
                row.station,
                (row.lon, row.lat),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.1,
                fontweight="bold",
                color="#17212b",
                path_effects=[pe.withStroke(linewidth=2.1, foreground="white")],
                zorder=9,
            )


def draw_maps(fig: plt.Figure, axes: list[plt.Axes], data: pd.DataFrame):
    with xr.open_dataset(ERA5_PATH) as ds_e, xr.open_dataset(PHY_PATH) as ds_p, xr.open_dataset(
        GEO_PATH
    ) as ds_g:
        time_e = pd.DatetimeIndex(pd.to_datetime(ds_e["valid_time"].values))
        time_p = pd.DatetimeIndex(pd.to_datetime(ds_p["time"].values))
        ie = int(np.where(time_e == SNAPSHOT)[0][0])
        ip = int(np.where(time_p == SNAPSHOT)[0][0])

        lat_e = ds_e["latitude"].values.astype(float)
        lon_e = ds_e["longitude"].values.astype(float)
        ue = ds_e["u10"].isel(valid_time=ie).values.astype(float)
        ve = ds_e["v10"].isel(valid_time=ie).values.astype(float)
        ws_e = np.hypot(ue, ve)

        lat_p = ds_p["XLAT"].values.astype(float)
        lon_p = ds_p["XLONG"].values.astype(float)
        up = ds_p["u10"].isel(time=ip).values.astype(float)
        vp = ds_p["v10"].isel(time=ip).values.astype(float)
        ws_p = np.hypot(up, vp)
        ui = interp_era5_to_target(lat_e, lon_e, ue, lat_p, lon_p)
        vi = interp_era5_to_target(lat_e, lon_e, ve, lat_p, lon_p)
        ws_i = np.hypot(ui, vi)

        # Illustrator can misread rasterized QuadMesh objects written by the
        # Matplotlib PDF backend (the mesh may appear as one flat colour even
        # though Acrobat/Poppler renders it correctly).  Build a regular,
        # display-only lon/lat raster for the interpolated ERA5 panel so it can
        # be embedded as one opaque RGB image instead of a PDF mesh object.
        display_lon = np.linspace(MAP_EXTENT[0] - 0.15, MAP_EXTENT[1] + 0.15, 1000)
        display_lat = np.linspace(MAP_EXTENT[2] - 0.15, MAP_EXTENT[3] + 0.15, 884)
        display_lon2d, display_lat2d = np.meshgrid(display_lon, display_lat)
        ws_e_display = interp_era5_to_target(
            lat_e, lon_e, ws_e, display_lat2d, display_lon2d, method="nearest"
        )
        ui_display = interp_era5_to_target(lat_e, lon_e, ue, display_lat2d, display_lon2d)
        vi_display = interp_era5_to_target(lat_e, lon_e, ve, display_lat2d, display_lon2d)
        ws_i_display = np.hypot(ui_display, vi_display)

        lat_g = ds_g["XLAT_M"].isel(Time=0).values.astype(float)
        lon_g = ds_g["XLONG_M"].isel(Time=0).values.astype(float)
        land = ds_g["LANDMASK"].isel(Time=0).values.astype(float)

    # Crop before plotting. This keeps the PDF genuinely vector-editable without
    # carrying half a million off-frame polygons into the output file.
    pad = 0.15
    lon0, lon1, lat0, lat1 = MAP_EXTENT
    keep_p = (lon_p >= lon0 - pad) & (lon_p <= lon1 + pad) & (lat_p >= lat0 - pad) & (lat_p <= lat1 + pad)
    yp, xp = np.where(keep_p)
    sy = slice(yp.min(), yp.max() + 1)
    sx = slice(xp.min(), xp.max() + 1)
    lon_p, lat_p, ws_i, ws_p = lon_p[sy, sx], lat_p[sy, sx], ws_i[sy, sx], ws_p[sy, sx]

    keep_g = (lon_g >= lon0 - pad) & (lon_g <= lon1 + pad) & (lat_g >= lat0 - pad) & (lat_g <= lat1 + pad)
    yg, xg = np.where(keep_g)
    sg_y = slice(yg.min(), yg.max() + 1)
    sg_x = slice(xg.min(), xg.max() + 1)
    lon_g, lat_g, land = lon_g[sg_y, sg_x], lat_g[sg_y, sg_x], land[sg_y, sg_x]

    xkeep = (lon_e >= lon0 - 0.5) & (lon_e <= lon1 + 0.5)
    ykeep = (lat_e >= lat0 - 0.5) & (lat_e <= lat1 + 0.5)
    lon_e, lat_e, ws_e = lon_e[xkeep], lat_e[ykeep], ws_e[np.ix_(ykeep, xkeep)]

    cmap = mpl.colormaps["turbo"]
    norm = mpl.colors.BoundaryNorm(np.arange(0, 47.5, 2.5), cmap.N, extend="max")
    stations = data[["station", "lat", "lon"]].drop_duplicates("station")
    stations = stations[stations.station.isin(["WGL", "CCH"])]
    fields = [
        (display_lon, display_lat, ws_e_display, "Native ERA5 (0.25°)", "regular_nearest"),
        (display_lon, display_lat, ws_i_display, "ERA5-Int. (3 km)", "regular"),
        (lon_p, lat_p, ws_p, "PhyTopo (3 km)", "mesh"),
    ]
    image = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for i, (ax, (lon, lat, wind, title, render_mode)) in enumerate(zip(axes[:3], fields)):
        if render_mode in {"regular_nearest", "regular"}:
            # Pre-colour the field into an opaque RGB raster.  Avoiding an alpha
            # mask is important because Illustrator has also been inconsistent
            # when importing clipped PDF image masks.
            rgb = np.asarray(np.clip(cmap(norm(wind))[..., :3] * 255, 0, 255), dtype=np.uint8)
            dx = float(np.median(np.abs(np.diff(lon))))
            dy = float(np.median(np.abs(np.diff(lat))))
            extent = (
                float(np.nanmin(lon) - dx / 2),
                float(np.nanmax(lon) + dx / 2),
                float(np.nanmin(lat) - dy / 2),
                float(np.nanmax(lat) + dy / 2),
            )
            origin = "upper" if lat[0] > lat[-1] else "lower"
            ax.imshow(
                rgb,
                extent=extent,
                origin=origin,
                # Use the same interpolated PDF image flag for both opaque RGB
                # layers. Illustrator 2023/2025 can corrupt an imported image
                # whose PDF /Interpolate flag is false. The native ERA5 field
                # was already expanded with nearest-neighbour sampling above,
                # so this only anti-aliases sub-pixel cell edges.
                interpolation="bilinear",
                aspect="auto",
                zorder=0,
            )
        else:
            ax.pcolormesh(
                lon,
                lat,
                wind,
                cmap=cmap,
                norm=norm,
                shading="auto",
                # Rasterize only the dense geophysical field. Coastlines,
                # stations, axes, text, and statistics remain vector objects.
                rasterized=True,
                edgecolors="none",
                linewidth=0,
                antialiased=False,
            )
        ax.contour(lon_g, lat_g, land, levels=[0.5], colors="#26343d", linewidths=0.55, alpha=0.9)
        add_station_markers(ax, stations, annotate=False)
        format_map(ax, left=i == 0)
        ax.set_title(title, fontsize=13.0, fontweight="bold", pad=5, color="#18232d")

    # A geographically explicit Hong Kong enlargement, matching the structure
    # of the earlier manuscript figure while using the updated v2 field.
    axz = axes[3]
    # Smooth only the visual rendering of the 3-km field. The station samples
    # and all quantitative calculations continue to use the native values.
    smooth_levels = np.linspace(0, 45, 91)
    zoom_field = axz.contourf(
        lon_p,
        lat_p,
        ws_p,
        levels=smooth_levels,
        cmap=cmap,
        norm=mpl.colors.Normalize(0, 45),
        extend="max",
        antialiased=True,
    )
    for collection in zoom_field.collections:
        collection.set_rasterized(True)
    # A broader PRD/Hong Kong context lowers the enlargement ratio while the
    # physical panel size remains identical to the other three map panels.
    zoom_extent = (113.25, 114.95, 21.45, 22.95)
    # The Arc/Info DEM is about 0.0042 degrees, substantially finer than the
    # WRF 3-km LANDMASK. Its zero-elevation contour gives the zoom panel a
    # cleaner, geographically more detailed coastline.
    if COAST_DEM_PATH.exists():
        with rasterio.open(COAST_DEM_PATH) as dem:
            window = rasterio.windows.from_bounds(
                zoom_extent[0], zoom_extent[2], zoom_extent[1], zoom_extent[3], transform=dem.transform
            )
            z = dem.read(1, window=window, masked=True)
            transform = dem.window_transform(window)
            coast_lon = transform.c + (np.arange(z.shape[1]) + 0.5) * transform.a
            coast_lat = transform.f + (np.arange(z.shape[0]) + 0.5) * transform.e
            axz.contour(
                coast_lon,
                coast_lat,
                z.astype(float).filled(np.nan),
                levels=[0.0],
                colors="#26343d",
                linewidths=0.58,
                alpha=0.96,
                zorder=6,
            )
    else:
        axz.contour(lon_g, lat_g, land, levels=[0.5], colors="#26343d", linewidths=0.65, alpha=0.95)
    axz.set_xlim(zoom_extent[:2])
    axz.set_ylim(zoom_extent[2:])
    axz.set_aspect("equal", adjustable="box")
    axz.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    axz.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    axz.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}°E"))
    axz.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.1f}°N"))
    axz.tick_params(labelsize=9.1, length=2.5, width=0.7, color="#66727c")
    axz.set_xlabel("Longitude", fontsize=10.2, labelpad=3)
    axz.set_title("Hong Kong-PRD zoom (PhyTopo)", fontsize=13.0, fontweight="bold", pad=5, color="#18232d")
    for spine in axz.spines.values():
        spine.set_color("#66727c")
        spine.set_linewidth(0.75)

    snap = data[data.datetime == SNAPSHOT].set_index("station")
    offsets = {"WGL": (8, -6), "CCH": (-42, -12)}
    for row in stations.itertuples(index=False):
        if row.station not in snap.index:
            continue
        obs = float(snap.loc[row.station, "obs_wspd"])
        phy = float(snap.loc[row.station, "v2_recon_wspd"])
        axz.scatter(row.lon, row.lat, s=39, marker="^", facecolor="white", edgecolor="#17212b", linewidth=0.9, zorder=8)
        dx, dy = offsets.get(row.station, (5, 5))
        axz.annotate(
            f"{row.station}  {obs:.1f}/{phy:.1f}",
            (row.lon, row.lat),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            fontweight="bold",
            color="#17212b",
            arrowprops=dict(arrowstyle="-", color="#38444d", lw=0.55),
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            zorder=9,
        )
    axz.text(
        0.02,
        0.02,
        "labels: observed / PhyTopo (m/s)",
        transform=axz.transAxes,
        fontsize=8.2,
        color="#2f3b45",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#9aa5ae", lw=0.55, alpha=0.88),
        zorder=10,
    )

    axes[2].add_patch(
        Rectangle(
            (zoom_extent[0], zoom_extent[2]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[2],
            fill=False,
            ec="white",
            lw=1.1,
            ls=(0, (4, 2)),
            zorder=10,
        )
    )
    cbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.055,
        pad=0.065,
        aspect=45,
        ticks=np.arange(0, 46, 5),
        extend="max",
    )
    cbar.set_label("10-m wind speed (m/s)", fontsize=10.8, labelpad=2)
    cbar.ax.tick_params(labelsize=9.5, length=2.5)
    cbar.outline.set_linewidth(0.6)


def draw_time_series(ax_top: plt.Axes, ax_bottom: plt.Axes, data: pd.DataFrame) -> None:
    stations = [("WGL", "Waglan Island"), ("CCH", "Cheung Chau")]
    for idx, (ax, (code, name)) in enumerate(zip([ax_top, ax_bottom], stations)):
        g = data[(data.station == code) & data.datetime.between(*PEAK_WINDOW)].copy()
        ax.plot(g.datetime, g.obs_wspd, color=COL_OBS, lw=2.15, marker="o", ms=3.4, mfc="white", mew=0.8, label="Observed", zorder=4)
        ax.plot(g.datetime, g.era5_wspd, color=COL_ERA, lw=1.55, marker="s", ms=3.0, markevery=2, label="ERA5", zorder=3)
        ax.plot(g.datetime, g.v2_recon_wspd, color=COL_PHY, lw=1.75, marker="^", ms=3.2, markevery=2, label="PhyTopo", zorder=3)
        ax.fill_between(g.datetime, 25, 48, color="#f7e9dc", alpha=0.45, zorder=0)
        ax.axhline(25, color="#bf7b42", lw=0.8, ls=(0, (3, 3)), alpha=0.8)
        ax.set_ylim(0, 48)
        ax.set_xlim(*PEAK_WINDOW)
        ax.grid(axis="y", color=COL_GRID, lw=0.7)
        ax.text(0.018, 0.90, f"{code}  {name}", transform=ax.transAxes, ha="left", va="top", fontsize=10.8, fontweight="bold", color="#17212b")
        ax.tick_params(labelsize=9.4, length=3, color="#73808a")
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#83909a")
            ax.spines[spine].set_linewidth(0.7)
        if idx == 0:
            panel_label(ax, "B", x=-0.13, y=1.10)
            ax.set_title("Peak-impact time series at the selected high-wind stations", loc="left", fontsize=12.0, fontweight="bold", pad=7)
            ax.legend(loc="upper center", bbox_to_anchor=(0.61, 1.015), ncol=3, frameon=False, fontsize=9.3, handlelength=2.2, columnspacing=1.0)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("UTC time (September 2018)", fontsize=10.0, labelpad=3)
        ax.set_ylabel("Wind speed (m/s)", fontsize=10.0, labelpad=3)


def draw_station_distributions(axes: list[plt.Axes], data: pd.DataFrame) -> None:
    stations = [
        ("WGL", "Waglan Island"),
        ("CCH", "Cheung Chau"),
    ]
    series = [
        ("obs_wspd", "Observed", COL_OBS),
        ("era5_wspd", "ERA5", COL_ERA),
        ("v2_recon_wspd", "PhyTopo", COL_PHY),
    ]
    xgrid = np.linspace(0, 48, 360)
    density_cache: dict[tuple[str, str], np.ndarray] = {}
    global_density_max = 0.0
    for code, _ in stations:
        g = data[data.station == code].dropna(subset=[s[0] for s in series]).copy()
        for column, _, _ in series:
            values = g[column].to_numpy(float)
            density = gaussian_kde(values, bw_method=0.38)(xgrid)
            density_cache[(code, column)] = density
            global_density_max = max(global_density_max, float(np.nanmax(density)))

    for i, (ax, (code, name)) in enumerate(zip(axes, stations)):
        g = data[data.station == code].dropna(subset=[s[0] for s in series]).copy()
        ax.axvspan(25, 50, color="#f7e9dc", alpha=0.62, zorder=0)
        for column, label, color in series:
            density = density_cache[(code, column)]
            ax.plot(
                xgrid,
                density,
                color=color,
                lw=1.75 if column != "obs_wspd" else 2.05,
                label=label,
                zorder=3,
            )
            ax.fill_between(xgrid, 0, density, color=color, alpha=0.055 if column == "obs_wspd" else 0.075, zorder=2)
            peak = float(g[column].max())
            ax.plot([peak, peak], [0, global_density_max * 0.055], color=color, lw=1.35, zorder=4)
        ax.set_xlim(0, 48)
        ax.set_ylim(0, global_density_max * 1.12)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.grid(axis="y", color=COL_GRID, lw=0.62)
        ax.tick_params(labelsize=7.2, length=2.5, color="#73808a")
        ax.text(
            0.03,
            0.92,
            f"{code}  {name}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color="#17212b",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#83909a")
            ax.spines[spine].set_linewidth(0.65)
        if i == 0:
            ax.set_ylabel("Probability\ndensity", fontsize=7.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("10-m wind speed (m/s)", fontsize=7.5)
        if i == 0:
            panel_label(ax, "B", x=-0.26, y=1.16)
            ax.set_title(
                "Smoothed within-station wind-speed distributions",
                loc="left",
                fontsize=10.2,
                fontweight="bold",
                pad=19,
            )
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(1.08, 1.17),
                ncol=3,
                frameon=False,
                fontsize=7.1,
                handlelength=1.8,
                columnspacing=0.8,
            )
    axes[-1].text(
        0.98,
        0.05,
        "shading: upper tail ≥25 m/s",
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#78501e",
    )


def draw_peak_comparison(ax: plt.Axes, station_metrics: pd.DataFrame) -> None:
    d = station_metrics[station_metrics.peak_valid.astype(str).str.lower().isin(["true", "1"])].copy()
    order = ["WGL", "CCH", "TO", "LFS"]
    d["station"] = pd.Categorical(d.station, categories=order, ordered=True)
    d = d.sort_values("station")
    y = np.arange(len(d))[::-1]

    for yi, row in zip(y, d.itertuples(index=False)):
        severe = row.obs_peak >= 40.0
        if severe:
            ax.axhspan(yi - 0.38, yi + 0.38, color="#fff1df", alpha=0.72, zorder=0)
        ax.plot([row.era5_peak, row.obs_peak], [yi + 0.13, yi + 0.13], color=COL_ERA, lw=1.1, alpha=0.65)
        ax.plot([row.v2_peak, row.obs_peak], [yi - 0.13, yi - 0.13], color=COL_PHY, lw=1.1, alpha=0.75)
        ax.scatter(row.era5_peak, yi + 0.13, s=39, color=COL_ERA, zorder=4)
        ax.scatter(row.v2_peak, yi - 0.13, s=43, color=COL_PHY, zorder=4)
        ax.scatter(row.obs_peak, yi, s=45, color=COL_OBS, marker="|", linewidths=2.4, zorder=5)
        ax.text(row.obs_peak + 0.55, yi, f"{row.obs_peak:.1f}", va="center", fontsize=7.7, color=COL_OBS)

    ax.set_yticks(y, d.station.astype(str))
    ax.set_xlim(15, 47)
    ax.set_ylim(-0.65, len(d) - 0.35)
    ax.grid(axis="x", color=COL_GRID, lw=0.7)
    ax.set_xlabel("Event-maximum 10-m wind speed (m/s)", fontsize=8.5)
    ax.tick_params(labelsize=8, length=3, color="#73808a")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#83909a")
        ax.spines[spine].set_linewidth(0.7)
    panel_label(ax, "C", x=-0.16, y=1.05)
    ax.set_title("Station-wise peak recovery", loc="left", fontsize=10.2, fontweight="bold", pad=8)
    handles = [
        Line2D([0], [0], marker="|", lw=0, color=COL_OBS, markersize=10, markeredgewidth=2.2, label="Observed"),
        Line2D([0], [0], marker="o", lw=0, color=COL_ERA, markersize=5, label="ERA5"),
        Line2D([0], [0], marker="o", lw=0, color=COL_PHY, markersize=5, label="PhyTopo"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.87), frameon=False, fontsize=7.1, handletextpad=0.3)
    ax.text(
        0.02,
        0.04,
        "All four: 11.82 → 6.99 (−40.9%)\nObserved ≥40 m/s: 18.64 → 3.81 (−79.6%)",
        transform=ax.transAxes,
        fontsize=6.9,
        color="#5d421d",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff4df", ec="#e0b66d", lw=0.7),
    )
    ax.text(0.02, 0.93, "shading: observed peak ≥ 40 m/s", transform=ax.transAxes, fontsize=7.4, color="#66727c")


def draw_threshold_errors(ax: plt.Axes, aggregate: pd.DataFrame) -> None:
    labels = ["All matched\nhours", "Observed ≥25\nm/s", "Observed ≥30\nm/s"]
    x = np.arange(3)
    era = aggregate.ERA5_rmse.to_numpy(float)
    phy = aggregate.v2_rmse.to_numpy(float)
    reduction = aggregate.v2_rmse_reduction_pct.to_numpy(float)
    n = aggregate.n.to_numpy(int)
    width = 0.33
    bars1 = ax.bar(x - width / 2, era, width, color=COL_ERA, label="ERA5", zorder=3)
    bars2 = ax.bar(x + width / 2, phy, width, color=COL_PHY, label="PhyTopo", zorder=3)
    ax.set_ylim(0, 17.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (m/s)", fontsize=10.2, labelpad=3)
    ax.grid(axis="y", color=COL_GRID, lw=0.7, zorder=0)
    ax.tick_params(labelsize=9.3, length=3, color="#73808a")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#83909a")
        ax.spines[spine].set_linewidth(0.7)
    panel_label(ax, "C", x=-0.14, y=1.05)
    ax.set_title("Error reduction strengthens toward the upper tail", loc="left", fontsize=11.8, fontweight="bold", pad=6)
    ax.legend(frameon=False, fontsize=9.2, ncol=2, loc="upper left")
    for b, value in zip(bars1, era):
        ax.text(b.get_x() + b.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=8.7, color=COL_ERA)
    for i, (b, value, red, ni) in enumerate(zip(bars2, phy, reduction, n)):
        ax.text(b.get_x() + b.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=8.7, color=COL_PHY)
        y = max(era[i], value) + 1.55
        ax.annotate(
            f"−{red:.1f}%\nN={ni}",
            xy=(x[i], y - 0.25),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#6f4815",
        )


def draw_cma_peak_distribution(ax: plt.Axes, cma: pd.DataFrame) -> None:
    d = cma[cma.event_id == "2018_2018250N12170"].copy()
    d = d[np.isfinite(d[["obs", "era", "v2"]]).all(axis=1)]
    high = d.obs >= 25.0
    lim = 53
    ax.plot([0, lim], [0, lim], color="#4e5963", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvspan(25, lim, color="#fff0dd", alpha=0.65, zorder=0)
    ax.scatter(d.obs, d.era, s=31, marker="o", color=COL_ERA, edgecolor="white", linewidth=0.45, alpha=0.76, label="ERA5", zorder=3)
    ax.scatter(d.obs, d.v2, s=34, marker="D", color=COL_PHY, edgecolor="white", linewidth=0.45, alpha=0.76, label="PhyTopo", zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=COL_GRID, lw=0.65, zorder=0)
    ax.set_xlabel("Observed station peak (m/s)", fontsize=10.2, labelpad=3)
    ax.set_ylabel("Reconstructed station peak (m/s)", fontsize=10.2, labelpad=3)
    ax.tick_params(labelsize=9.3, length=3, color="#73808a")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#83909a")
        ax.spines[spine].set_linewidth(0.7)
    panel_label(ax, "D", x=-0.16, y=1.05)
    ax.set_title("Peak-wind distribution across CMA stations", loc="left", fontsize=11.8, fontweight="bold", pad=6)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left", ncol=2, columnspacing=0.9, handletextpad=0.4)
    era_mae = np.mean(np.abs(d.loc[high, "era"] - d.loc[high, "obs"]))
    phy_mae = np.mean(np.abs(d.loc[high, "v2"] - d.loc[high, "obs"]))
    gain = 100 * (era_mae - phy_mae) / era_mae
    ax.text(
        0.98,
        0.04,
        f"Mangkhut station peaks (N={len(d)})\nupper tail ≥25 m/s: N={high.sum()}, MAE {era_mae:.2f} → {phy_mae:.2f} m/s (−{gain:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        color="#654414",
        bbox=dict(boxstyle="round,pad=0.28", fc="#fff6e7", ec="#e3b363", lw=0.65),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    station_metrics = pd.read_csv(STATION_PATH)
    aggregate = pd.read_csv(AGG_PATH)
    cma = pd.read_csv(CMA_PEAK_PATH)

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.0,
            "mathtext.fontset": "stix",
            "axes.titlecolor": "#17212b",
            "axes.labelcolor": "#17212b",
            "xtick.color": "#27333d",
            "ytick.color": "#27333d",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(19.2, 9.25), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.03, 0.97], hspace=0.18, left=0.048, right=0.988, top=0.925, bottom=0.072)
    maps = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0], wspace=0.045)
    map_axes = [fig.add_subplot(maps[0, i]) for i in range(4)]
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.72, 0.88, 1.08], wspace=0.25)
    time_grid = bottom[0].subgridspec(2, 1, hspace=0.055)
    ax_ts1 = fig.add_subplot(time_grid[0])
    ax_ts2 = fig.add_subplot(time_grid[1], sharex=ax_ts1)
    ax_err = fig.add_subplot(bottom[1])
    ax_cma = fig.add_subplot(bottom[2])

    draw_maps(fig, map_axes, data)
    draw_time_series(ax_ts1, ax_ts2, data)
    draw_threshold_errors(ax_err, aggregate)
    draw_cma_peak_distribution(ax_cma, cma)

    fig.text(
        0.048,
        0.968,
        "A   Wind-field structure at peak impact (16 September 2018, 05:00 UTC)",
        ha="left",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#16395c",
    )

    png = OUT / "Figure6_Mangkhut_upper_tail_v12_Illustrator.png"
    pdf = OUT / "Figure6_Mangkhut_upper_tail_v12_Illustrator.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
