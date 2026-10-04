"""
Saola (2023): station-model extreme wind sanity check pipeline.

Interpretation:
    - OBS `Date time` is treated as Hong Kong civil time (+08, no DST) → UTC.
    - Hourly aggregation: mean 10-min sustained wind; gust = hourly max (m/s).
    - Era5 / PhyTopo: WS = hypot(u10, v10).
    - ERA5 / PhyTopo / bilinear-on-d03: at each AWS, take nearest grid cell as the
      centre of a (2N+1)^2 stencil (--nc-region-half N), nan-mean u and v there, then
      WS = hypot(ū,v̄) (single-cell NN when N = 0).

Outputs in analysis/outputs/ (default), outputs_saola_hadisd/, outputs_yagi/, outputs_yagi_hadisd/,
outputs_tc2016288_hadisd/, outputs_mangkhut_cowin/, outputs_mangkhut_hadisd/.

Reconstructed 2026-05 from ``analysis/_disasm_run_saola.txt`` after accidental source loss.
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.stats import gaussian_kde, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data"
STATIONS_CSV = Path(__file__).resolve().parent / "stations_wgs84.csv"
EVENT_START_UTC = pd.Timestamp("2023-08-31 00:00:00", tz="UTC")
EVENT_END_UTC = pd.Timestamp("2023-09-03 23:00:00", tz="UTC")
BASE_UTC = pd.Timestamp("2023-08-22 00:00:00", tz="UTC")
KEY_STATIONS = [
    "Waglan Island",
    "Tate's Cairn",
    "Cheung Chau",
    "Tap Mun",
    "Lau Fau Shan",
]
KEY_STATIONS_MANGKHUT_COWIN = [
    "Waglan Island (CoWIN 6002)",
    "Tate's Cairn area (CoWIN 6061)",
    "Cheung Chau area (CoWIN 6088)",
    "Tap Mun area (CoWIN 6003)",
    "Lau Fau Shan area (CoWIN 6092)",
]

# Hourly WS columns shared by KDE / tail / QQ figures
_WS_DIST_SERIES: tuple[tuple[str, str, str], ...] = (
    ("obs_ws", "#111111", "Observation hourly mean"),
    ("era5_ws", "#1f77b4", "ERA5 (NN station cell)"),
    ("bilinear_ws", "#2ca02c", "ERA5 bilinear→d03 NN"),
    ("phytopo_ws", "#d62728", "PhyTopo-Net NN"),
)
HADISD_AUTO_KEY_STATIONS = "__auto_top_obs__"
TRACK_TZ = "Asia/Shanghai"
WIND_CSV_DAY_PREFIX = re.compile(r"^(\d{8})")


def infer_phy_base_utc_from_nc(era5_nc: Path, phy_nc: Path) -> pd.Timestamp | None:
    """First ERA5 UTC time minus first PhyTopo `time` (hours offset), matching load_bundle indexing."""
    if not era5_nc.is_file() or not phy_nc.is_file():
        return None
    ds_e = ds_p = None
    try:
        ds_e = nc.Dataset(era5_nc)
        ds_p = nc.Dataset(phy_nc)
        era0 = pd.to_datetime(
            ds_e.variables["valid_time"][0].astype(np.int64),
            unit="s",
            utc=True,
        )
        t0_h = float(ds_p.variables["time"][0])
        return pd.Timestamp(era0) - pd.Timedelta(hours=t0_h)
    finally:
        if ds_e is not None:
            ds_e.close()
        if ds_p is not None:
            ds_p.close()


def parse_hkt_walltime(dt_raw: object) -> pd.Timestamp:
    s = str(dt_raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        ts = pd.to_datetime(s, format="%Y%m%d%H%M")
    else:
        ts = pd.to_datetime(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ts = ts.tz_localize("Asia/Hong_Kong", ambiguous=True, nonexistent="shift_forward")
    return ts.tz_convert("UTC")


def load_stations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["station"] = df["station"].astype(str).str.strip()
    return df.set_index("station")


def geo_em_spacing_m(geo_nc: Path) -> float:
    g = nc.Dataset(geo_nc)
    try:
        return float(getattr(g, "DX"))
    finally:
        g.close()


def scan_obs_hourly(
    csv_dir: Path,
    *,
    utc_clip: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    filter_paths_pre_read: bool = True,
) -> pd.DataFrame:
    paths = sorted(csv_dir.glob("*-latest_10min_wind.csv"))
    if not paths:
        paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV under {csv_dir}")
    if utc_clip is not None and filter_paths_pre_read:
        paths = filter_wind_paths_by_calendar_overlap(
            paths, utc_clip[0], utc_clip[1], 48.0
        )
        if not paths:
            raise FileNotFoundError(
                "No HK wind CSV survived date filtering for the requested UTC window "
                "(check filenames / folder)."
            )
    parts: list[pd.DataFrame] = []
    for p in paths:
        try:
            part = pd.read_csv(p)
        except Exception:
            continue
        if part.empty:
            continue
        part.columns = [str(c).strip() for c in part.columns]
        spd = pd.to_numeric(part["10-Minute Mean Speed(km/hour)"], errors="coerce")
        gust = pd.to_numeric(part["10-Minute Maximum Gust(km/hour)"], errors="coerce")
        parts.append(
            pd.DataFrame(
                {
                    "time_utc": part["Date time"].astype(str).map(parse_hkt_walltime),
                    "station": part["Automatic Weather Station"].astype(str),
                    "ws10_ms": spd / 3.6,
                    "gust10_ms": gust / 3.6,
                }
            )
        )
    df = pd.concat(parts, ignore_index=True).dropna(subset=["time_utc"])
    df["hour_floor"] = df["time_utc"].dt.floor("h")
    hourly = (
        df.groupby(["hour_floor", "station"])
        .agg(obs_ws=("ws10_ms", "mean"), obs_gust=("gust10_ms", "max"))
        .reset_index()
        .rename(columns={"hour_floor": "time"})
    )
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
    if utc_clip is not None:
        utc_lo, utc_hi = utc_clip
        utc_lo = (
            pd.Timestamp(utc_lo).tz_convert("UTC")
            if utc_lo.tzinfo
            else utc_lo.tz_localize("UTC")
        )
        utc_hi = (
            pd.Timestamp(utc_hi).tz_convert("UTC")
            if utc_hi.tzinfo
            else utc_hi.tz_localize("UTC")
        )
        hourly = hourly.loc[
            (hourly["time"] >= utc_lo) & (hourly["time"] <= utc_hi)
        ].reset_index(drop=True)
    return hourly


def load_track(csv_path: Path) -> pd.DataFrame:
    tr = pd.read_csv(csv_path).rename(columns={"lng": "lon"})
    t = pd.to_datetime(tr["time"])
    t = t.dt.tz_localize(TRACK_TZ, ambiguous=True, nonexistent="shift_forward")
    out = tr.assign(time=t.dt.tz_convert("UTC")).copy()
    out["lon"] = out["lon"].astype(float)
    out["lat"] = out["lat"].astype(float)
    return out


def hav_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    r = 6371.0088
    p1 = np.radians(lat1)
    l1 = np.radians(lon1)
    p2 = math.radians(lat2)
    l2 = math.radians(lon2)
    dphi = p1 - p2
    dlmb = np.radians(np.mod(lon1 - lon2 + 180.0, 360.0) - 180.0)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(p2) * np.cos(p1) * np.sin(dlmb / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_track_peak_window(
    track: pd.DataFrame,
    lat0: float,
    lon0: float,
    *,
    half_hours: float = 18.0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dd = hav_km(
        track["lat"].to_numpy(),
        track["lon"].to_numpy(),
        lat0,
        lon0,
    )
    imin = int(np.argmin(dd))
    anchor = pd.Timestamp(track["time"].iloc[imin])
    h = pd.Timedelta(hours=half_hours)
    return anchor - h, anchor + h


def track_anchor_utc(track: pd.DataFrame, lat0: float, lon0: float) -> pd.Timestamp:
    dd = hav_km(
        track["lat"].to_numpy(),
        track["lon"].to_numpy(),
        lat0,
        lon0,
    )
    imin = int(np.argmin(dd))
    return pd.Timestamp(track["time"].iloc[imin])


def obs_time_window_around_anchor(
    anchor_utc: pd.Timestamp,
    *,
    peak_span_days: float = 3.0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    half_days = peak_span_days / 2.0
    half = pd.Timedelta(days=half_days)
    return anchor_utc - half, anchor_utc + half


def filter_wind_paths_by_calendar_overlap(
    paths: list[Path],
    utc_lo: pd.Timestamp,
    utc_hi: pd.Timestamp,
    pad_hours: float = 48.0,
) -> list[Path]:
    lo_ts = utc_lo.tz_convert("UTC") if utc_lo.tzinfo else utc_lo.tz_localize("UTC")
    hi_ts = utc_hi.tz_convert("UTC") if utc_hi.tzinfo else utc_hi.tz_localize("UTC")
    lo = lo_ts - pd.Timedelta(hours=pad_hours)
    hi = hi_ts + pd.Timedelta(hours=pad_hours)
    hk_tz = "Asia/Hong_Kong"
    kept: list[Path] = []
    for p in paths:
        m = WIND_CSV_DAY_PREFIX.match(p.name)
        if not m:
            kept.append(p)
            continue
        s = m.group(1)
        day_start_hk = pd.Timestamp(
            f"{s[:4]}-{s[4:6]}-{s[6:8]}",
            tz=hk_tz,
        )
        day_end_hk = day_start_hk + pd.Timedelta(days=1)
        day_start = day_start_hk.tz_convert("UTC")
        day_end = day_end_hk.tz_convert("UTC")
        if day_end <= lo or day_start >= hi:
            continue
        kept.append(p)
    return kept


def terrain_table(
    hgt_m: np.ndarray,
    landmask: np.ndarray,
    stations: list[str],
    pj: dict[str, int],
    pi: dict[str, int],
    geo_nc: Path,
) -> pd.DataFrame:
    dx = geo_em_spacing_m(geo_nc)
    dy = dx
    gy, gx = np.gradient(hgt_m.astype(float), dy, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(np.square(gx) + np.square(gy))))
    sea = landmask <= 0.5
    land_cells = (~sea).astype(np.uint8)
    dt_land_to_sea_km = distance_transform_edt(
        land_cells, sampling=(dx / 1000.0, dy / 1000.0)
    )
    rows: list[dict[str, float | str]] = []
    for s in stations:
        jj, ii = pj[s], pi[s]
        if sea[jj, ii]:
            dist_coast_km = math.nan
        else:
            dist_coast_km = float(dt_land_to_sea_km[jj, ii])
        rows.append(
            {
                "station": s,
                "elev_m": float(hgt_m[jj, ii]),
                "slope_deg": float(slope_deg[jj, ii]),
                "dist_coast_km": dist_coast_km,
            }
        )
    return pd.DataFrame(rows).set_index("station")


def load_bundle(
    era5_nc: Path,
    phy_nc: Path,
    geo_nc: Path,
    stations_csv: Path,
    *,
    base_utc: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> dict:
    ds_e = nc.Dataset(era5_nc)
    ds_p = nc.Dataset(phy_nc)
    geo = nc.Dataset(geo_nc)
    try:
        lat_desc = ds_e.variables["latitude"][:].astype(float)
        lon_asc = ds_e.variables["longitude"][:].astype(float)
        lat_asc = lat_desc[::-1].copy()
        era_time = pd.to_datetime(
            ds_e.variables["valid_time"][:].astype(np.int64),
            unit="s",
            utc=True,
        )
        phyt_time = base_utc + pd.to_timedelta(
            ds_p.variables["time"][:].astype(float), unit="h"
        )
        if not era_time.equals(phyt_time):
            raise RuntimeError("ERA5 vs PhyTopo time mismatch.")
        xlat = ds_p.variables["XLAT"][:].astype(float)
        if "XLONG" in ds_p.variables:
            xlong = ds_p.variables["XLONG"][:].astype(float)
        else:
            xlong = ds_p.variables["LONG"][:].astype(float)
        hgt = np.ma.getdata(geo.variables["HGT_M"][0]).astype(float)
        lmsk = np.ma.getdata(geo.variables["LANDMASK"][0]).astype(float)
        st_df = load_stations(stations_csv)
        phy_j: dict[str, int] = {}
        phy_i: dict[str, int] = {}
        era_j: dict[str, int] = {}
        era_i: dict[str, int] = {}
        for st, row in st_df.iterrows():
            sla, slo = float(row["lat"]), float(row["lon"])
            ij = int(np.argmin((xlat - sla) ** 2 + (xlong - slo) ** 2))
            jj, ii = np.unravel_index(ij, xlat.shape)
            phy_j[st], phy_i[st] = jj, ii
            era_j[st] = int(np.argmin(np.abs(lat_asc - sla)))
            era_i[st] = int(np.argmin(np.abs(lon_asc - slo)))
        idx0 = int((event_start - base_utc) / pd.Timedelta(hours=1))
        n_hours = int((event_end - event_start) / pd.Timedelta(hours=1)) + 1
        n_time = int(ds_e.variables["u10"].shape[0])
        idx1 = min(max(0, idx0 + n_hours), n_time)
        idx0_clipped = max(0, idx0)
        if idx1 <= idx0_clipped:
            raise ValueError(
                f"ERA5 slice empty: idx0={idx0} n_hours={n_hours} n_time={n_time}"
            )
        sl = slice(idx0_clipped, idx1)
        u_raw = np.array(ds_e.variables["u10"][sl])
        v_raw = np.array(ds_e.variables["v10"][sl])
        if u_raw.ndim == 4 and u_raw.shape[1] == 1:
            u_raw = u_raw[:, 0, :, :]
            v_raw = v_raw[:, 0, :, :]
        if u_raw.ndim == 4:
            u_e = np.ascontiguousarray(u_raw[:, :, ::-1, :])
            v_e = np.ascontiguousarray(v_raw[:, :, ::-1, :])
        elif u_raw.ndim == 3:
            u_e = np.ascontiguousarray(u_raw[:, ::-1, :])
            v_e = np.ascontiguousarray(v_raw[:, ::-1, :])
        else:
            raise ValueError(f"Unexpected ERA5 u10 shape {u_raw.shape}")
        u_p = np.array(ds_p.variables["u10"][sl])
        v_p = np.array(ds_p.variables["v10"][sl])
        terr = terrain_table(hgt, lmsk, list(st_df.index), phy_j, phy_i, geo_nc)
        return {
            "time": pd.DatetimeIndex(era_time[sl]),
            "lat_asc": lat_asc,
            "lon_asc": lon_asc,
            "u_e": u_e,
            "v_e": v_e,
            "u_p": u_p,
            "v_p": v_p,
            "XLAT": xlat,
            "XLONG": xlong,
            "phy_j": phy_j,
            "phy_i": phy_i,
            "era_j": era_j,
            "era_i": era_i,
            "hgt": hgt,
            "terrain_station": terr,
        }
    finally:
        ds_e.close()
        ds_p.close()
        geo.close()


def _stencil_mean_uv_hypot_ws(
    u: np.ndarray, v: np.ndarray, j: int, i: int, half: int
) -> tuple[float, float, float]:
    """Stencil nan-mean of u,v about (j,i), clamped at domain; WS = hypot(ū, v̄)."""
    nj, ni = int(u.shape[0]), int(u.shape[1])
    if v.shape != (nj, ni):
        raise ValueError("u/v shape mismatch in stencil mean")
    if half <= 0:
        uc = float(np.asarray(u[j, i], dtype=float))
        vc = float(np.asarray(v[j, i], dtype=float))
        if not (math.isfinite(uc) and math.isfinite(vc)):
            return float("nan"), float("nan"), float("nan")
        return float(np.hypot(uc, vc)), uc, vc
    j0, j1 = max(0, j - half), min(nj, j + half + 1)
    i0, i1 = max(0, i - half), min(ni, i + half + 1)
    ub = np.asarray(u[j0:j1, i0:i1], dtype=float)
    vb = np.asarray(v[j0:j1, i0:i1], dtype=float)
    m = np.isfinite(ub) & np.isfinite(vb)
    if not m.any():
        return float("nan"), float("nan"), float("nan")
    uc = float(np.mean(ub[m]))
    vc = float(np.mean(vb[m]))
    return float(np.hypot(uc, vc)), uc, vc


def build_merged(
    bundle: dict,
    obs_hourly: pd.DataFrame,
    stations_csv: Path,
    *,
    nc_region_half: int = 2,
) -> pd.DataFrame:
    st_df = load_stations(stations_csv)
    sts = list(st_df.index)
    pi, pj = bundle["phy_i"], bundle["phy_j"]
    ei, ej = bundle["era_i"], bundle["era_j"]
    ue, ve = bundle["u_e"], bundle["v_e"]
    up, vp = bundle["u_p"], bundle["v_p"]
    pts = np.column_stack([bundle["XLAT"].ravel(), bundle["XLONG"].ravel()])
    rows: list[dict[str, float | pd.Timestamp | str]] = []
    for k in range(ue.shape[0]):
        t = bundle["time"][k]
        u = np.ascontiguousarray(ue[k])
        v = np.ascontiguousarray(ve[k])
        iu = RegularGridInterpolator(
            (bundle["lat_asc"], bundle["lon_asc"]),
            u,
            bounds_error=False,
            fill_value=np.nan,
        )
        iv = RegularGridInterpolator(
            (bundle["lat_asc"], bundle["lon_asc"]),
            v,
            bounds_error=False,
            fill_value=np.nan,
        )
        uf = iu(pts).reshape(bundle["XLAT"].shape)
        vf = iv(pts).reshape(bundle["XLAT"].shape)
        for s in sts:
            jj, ii = pj[s], pi[s]
            era5_ws, _, _ = _stencil_mean_uv_hypot_ws(
                ue[k], ve[k], ej[s], ei[s], nc_region_half
            )
            bil_ws, bu, bv = _stencil_mean_uv_hypot_ws(
                uf, vf, jj, ii, nc_region_half
            )
            phy_ws, _, _ = _stencil_mean_uv_hypot_ws(
                up[k], vp[k], jj, ii, nc_region_half
            )
            rows.append(
                {
                    "time": t,
                    "station": s,
                    "era5_ws": era5_ws,
                    "bilinear_ws": bil_ws,
                    "phytopo_ws": phy_ws,
                    "bilin_u_grid": bu,
                    "bilin_v_grid": bv,
                }
            )
    mod = pd.DataFrame(rows)
    merged = pd.merge(
        obs_hourly, mod, on=["time", "station"], how="inner"
    ).sort_values(["station", "time"])
    return merged.reset_index(drop=True)


def pick_top_obs_stations(merged: pd.DataFrame, n: int) -> list[str]:
    g = merged.groupby("station")["obs_ws"].max().sort_values(ascending=False)
    return list(g.head(int(n)).index)


def plot_fig1(
    path: Path,
    merged: pd.DataFrame,
    shade0: pd.Timestamp,
    shade1: pd.Timestamp,
    suptitle: str,
    *,
    key_stations: list[str] | None = None,
) -> None:
    ks = key_stations if key_stations is not None else KEY_STATIONS
    n = len(ks)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 10.8), sharex=True, constrained_layout=True
    )
    series = [
        ("obs_ws", "#111111", "Observation hourly mean"),
        ("era5_ws", "#1f77b4", "ERA5 (NN station cell)"),
        ("bilinear_ws", "#2ca02c", "ERA5 bilinear→d03 NN"),
        ("phytopo_ws", "#d62728", "PhyTopo-Net NN"),
    ]
    axes = np.atleast_1d(axes)
    shade0n = pd.Timestamp(shade0)
    shade1n = pd.Timestamp(shade1)
    if shade0n.tzinfo:
        shade0n = shade0n.tz_convert(None)
    if shade1n.tzinfo:
        shade1n = shade1n.tz_convert(None)
    for ax, st in zip(axes, ks):
        df = merged.loc[merged["station"] == st].sort_values("time")
        tser = pd.to_datetime(df["time"])
        if getattr(tser.dt, "tz", None) is None:
            tser = tser.dt.tz_localize("UTC")
        ts = tser.dt.tz_convert(None)
        ax.axvspan(
            mdates.date2num(shade0n),
            mdates.date2num(shade1n),
            color="#ffb347",
            alpha=0.2,
            lw=0,
            zorder=0,
        )
        for col, colr, lbl in series:
            y = pd.to_numeric(df[col], errors="coerce")
            if not y.notna().any():
                continue
            lw = 2.5 if col == "obs_ws" else 1.5
            ax.plot(ts, y, color=colr, lw=lw, label=lbl, alpha=0.9)
            j = int(np.nanargmax(y.to_numpy(dtype=float)))
            short = col.split("_")[0][:4]
            ax.annotate(
                f"{short} max={float(y.iloc[j]):.2f}",
                xy=(mdates.date2num(ts.iloc[j]), float(y.iloc[j])),
                xytext=(8, -10 if col == "obs_ws" else 12),
                textcoords="offset points",
                fontsize=8,
                color=colr,
            )
        ax.set_title(st)
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.3)
        if st == ks[0]:
            ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("UTC time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %HZ"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.suptitle(suptitle, fontsize=12, y=1.008)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def peak_metric_table(merged: pd.DataFrame) -> pd.DataFrame:
    gcols = {
        "obs_ws": "obs_peak",
        "era5_ws": "era_peak",
        "bilinear_ws": "bilin_peak",
        "phytopo_ws": "phy_peak",
    }
    grp = (
        merged.groupby("station")[list(gcols.keys())]
        .max()
        .rename(columns=gcols)
    )
    grp["era_bias"] = grp["era_peak"] - grp["obs_peak"]
    grp["bilinear_bias"] = grp["bilin_peak"] - grp["obs_peak"]
    grp["phy_bias"] = grp["phy_peak"] - grp["obs_peak"]
    return grp.reset_index()


def peak_stats(peaks_obs: pd.Series, peaks_pred: pd.Series) -> tuple[float, float, float, int]:
    df = pd.DataFrame({"obs": peaks_obs, "pred": peaks_pred}).dropna()
    x = df["obs"].to_numpy(dtype=float)
    y = df["pred"].to_numpy(dtype=float)
    if len(df) < 2:
        return math.nan, math.nan, math.nan, len(df)
    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    r = float(pearsonr(x, y)[0])
    return bias, rmse, r, len(df)


def plot_fig_station_wind_pdf(
    path_pdf: Path | None,
    merged: pd.DataFrame,
    suptitle: str,
    *,
    key_stations: list[str] | None = None,
    path_png: Path | None = None,
) -> None:
    """Kernel density of hourly 10-m wind speed at key stations (obs + three grids)."""
    ks = key_stations if key_stations is not None else KEY_STATIONS
    fig, axes = plt.subplots(
        len(ks), 1, figsize=(11, 10.8), sharex=False, constrained_layout=True
    )
    series = list(_WS_DIST_SERIES)
    axes = np.atleast_1d(axes)
    xmax_global = 0.0
    for st in ks:
        df = merged.loc[merged["station"] == st]
        if df.empty:
            continue
        sub = df[[c for c, _, _ in series]]
        m = float(np.nanmax(sub.to_numpy(dtype=float)))
        if np.isfinite(m):
            xmax_global = max(xmax_global, float(m))
    if xmax_global <= 0:
        xmax_global = 25.0
    else:
        xmax_global = min(45.0, xmax_global * 1.08 + 1.0)
    xs = np.linspace(0.0, xmax_global, 400)
    for ax, st in zip(axes, ks):
        df = merged.loc[merged["station"] == st].sort_values("time")
        for col, color, lbl in series:
            y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            y = y[np.isfinite(y) & (y >= 0)]
            if y.size < 2:
                continue
            try:
                kde = gaussian_kde(y)
                dens = kde(xs)
                dens = np.clip(dens, 0.0, None)
                ax.plot(xs, dens, color=color, lw=1.5, label=lbl)
            except (np.linalg.LinAlgError, ValueError):
                continue
        ax.set_title(st)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.0, xmax_global)
        if st == ks[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=1)
    axes[-1].set_xlabel("10-m wind speed (m/s)")
    fig.suptitle(suptitle, fontsize=12, y=1.008)
    if path_png is not None:
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, dpi=180, bbox_inches="tight")
    if path_pdf is not None:
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _xmax_station_winds(merged: pd.DataFrame, ks: list[str]) -> float:
    series = list(_WS_DIST_SERIES)
    xmax = 0.0
    for st in ks:
        df = merged.loc[merged["station"] == st]
        if df.empty:
            continue
        sub = df[[c for c, _, _ in series]]
        m = float(np.nanmax(sub.to_numpy(dtype=float)))
        if np.isfinite(m):
            xmax = max(xmax, m)
    if xmax <= 0:
        xmax = 25.0
    else:
        xmax = min(45.0, xmax * 1.08 + 1.0)
    return float(xmax)


def plot_fig_station_wind_kde_logdensity(
    path_png: Path,
    merged: pd.DataFrame,
    suptitle: str,
    *,
    key_stations: list[str] | None = None,
) -> None:
    """KDE density with log-scaled ρ so low-density tails are visible."""
    ks = key_stations if key_stations is not None else KEY_STATIONS
    series = list(_WS_DIST_SERIES)
    fig, axes = plt.subplots(
        len(ks), 1, figsize=(11, 10.8), sharex=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    xmax_global = _xmax_station_winds(merged, ks)
    xs = np.linspace(0.0, xmax_global, 400)
    floor = 1e-8
    for ax, st in zip(axes, ks):
        df = merged.loc[merged["station"] == st].sort_values("time")
        for col, color, lbl in series:
            y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            y = y[np.isfinite(y) & (y >= 0)]
            if y.size < 2:
                continue
            try:
                kde = gaussian_kde(y)
                dens = np.clip(np.asarray(kde(xs), dtype=float), floor, None)
                ax.plot(xs, dens, color=color, lw=1.5, label=lbl)
            except (np.linalg.LinAlgError, ValueError):
                continue
        ax.set_title(st)
        ax.set_ylabel("Density (log scale)")
        ax.set_yscale("log")
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0.12)
        ax.set_xlim(0.0, xmax_global)
        if st == ks[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=1)
    axes[-1].set_xlabel("10-m wind speed (m/s)")
    fig.suptitle(f"{suptitle} — KDE density (log y-axis)", fontsize=12, y=1.008)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig_station_wind_exceedance(
    path_png: Path,
    merged: pd.DataFrame,
    suptitle: str,
    *,
    key_stations: list[str] | None = None,
) -> None:
    """Empirical P(W>x); log y emphasizes rare high-wind regime."""
    ks = key_stations if key_stations is not None else KEY_STATIONS
    series = list(_WS_DIST_SERIES)
    fig, axes = plt.subplots(
        len(ks), 1, figsize=(11, 10.8), sharex=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    xmax_global = _xmax_station_winds(merged, ks)
    thr = np.linspace(0.0, xmax_global, 120)
    y_floor = 1e-5
    for ax, st in zip(axes, ks):
        df = merged.loc[merged["station"] == st]
        if df.empty:
            continue
        for col, color, lbl in series:
            s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            if s.size < 3:
                continue
            frac = np.array([np.mean(s > float(t)) for t in thr], dtype=float)
            frac = np.clip(frac, y_floor, 1.0)
            ax.semilogy(thr, frac, color=color, lw=1.4, label=lbl)
        ax.set_title(st)
        ax.set_ylabel("P(W > x) empirical")
        ax.set_yscale("log")
        ax.set_ylim(bottom=y_floor, top=2.0)
        ax.grid(which="major", alpha=0.35)
        ax.grid(which="minor", alpha=0.12)
        ax.set_xlim(0.0, xmax_global)
        if st == ks[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=1)
    axes[-1].set_xlabel("Threshold x (10-m wind speed, m/s)")
    fig.suptitle(f"{suptitle} — exceedance (log-scale tail)", fontsize=12, y=1.008)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig_station_wind_qq_vs_obs(
    path_png: Path,
    merged: pd.DataFrame,
    suptitle: str,
    *,
    key_stations: list[str] | None = None,
) -> None:
    """Marginal QQ: sorted obs hourly winds vs sorted model hourly (same paired rows)."""
    ks = key_stations if key_stations is not None else KEY_STATIONS
    model_cols = [t for t in _WS_DIST_SERIES if t[0] != "obs_ws"]
    fig, axes = plt.subplots(
        len(ks), 1, figsize=(11, 10.8), sharex=False, sharey=False, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, ks):
        df = (
            merged.loc[merged["station"] == st, ["obs_ws"] + [c for c, _, _ in model_cols]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        if df.shape[0] < 10:
            ax.set_title(f"{st} (insufficient overlap)")
            ax.grid(alpha=0.3)
            continue
        o = np.sort(df["obs_ws"].to_numpy(dtype=float))
        vmax = float(np.nanmax(o)) * 1.03
        for col, color, lbl in model_cols:
            m = np.sort(df[col].to_numpy(dtype=float))
            if m.shape[0] != o.shape[0]:
                continue
            ax.plot(o, m, ".", markersize=2.5, alpha=0.72, color=color, label=lbl)
        xy = np.linspace(0.0, max(np.nanmax(o), vmax, 25.0), 50)
        ref_lbl = "1:1" if st == ks[0] else None
        ax.plot(xy, xy, "k--", lw=0.95, alpha=0.55, label=ref_lbl)
        ax.set_title(st)
        ax.set_aspect("equal")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        mn = float(min(ax.get_xlim()[0], ax.get_ylim()[0]))
        mx = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_ylabel("Model quantiles (sorted hourly, m/s)")
        ax.grid(alpha=0.3)
        if st == ks[0]:
            ax.legend(loc="lower right", fontsize=7, ncol=2)

    axes[-1].set_xlabel("Obs quantiles (sorted hourly, m/s)")
    fig.suptitle(f"{suptitle} — QQ (marginal hourly pairs)", fontsize=12, y=1.008)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(path: Path, peaks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    sets = [
        ("era_peak", "ERA5 (NN peak)"),
        ("bilin_peak", "Bilinear→d03 NN peak"),
        ("phy_peak", "PhyTopo peak"),
    ]
    xmax = float(peaks["obs_peak"].max())
    ymax = float(
        max(
            peaks[c].max() for c in ("era_peak", "bilin_peak", "phy_peak")
        )
    )
    lim = float(np.nanmax([xmax, ymax]) * 1.05)
    x = peaks["obs_peak"].to_numpy(dtype=float)
    for ax, (ycol, ttl) in zip(axes, sets):
        yv = peaks[ycol].to_numpy(dtype=float)
        ax.scatter(
            x,
            yv,
            ec="k",
            linewidths=0.35,
            fc="#6baed6",
            s=42,
            alpha=0.82,
        )
        ax.plot([0, lim], [0, lim], "k--", lw=1.0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        ax.set_title(ttl)
        ax.set_xlabel("Obs peak (m/s)")
        ax.set_ylabel("Model peak (m/s)")
        b, rmse, r, n = peak_stats(peaks["obs_peak"], peaks[ycol])
        ax.text(
            0.04,
            0.96,
            f"bias={b:+.2f}\nRMSE={rmse:.2f}\nr={r:.2f}\nn={int(n)}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_fig3_terrain(fig_path: Path, tbl: pd.DataFrame) -> None:
    xcols = ["elev_m", "slope_deg", "dist_coast_km"]
    ycols = [
        ("era_bias", "ERA5 peak bias"),
        ("phy_bias", "PhyTopo peak bias"),
    ]
    fig, axes = plt.subplots(
        len(xcols),
        len(ycols),
        figsize=(9.8, 7.9),
        constrained_layout=True,
    )
    for i, xv in enumerate(xcols):
        for j, (yv, ylab) in enumerate(ycols):
            ax = axes[i, j]
            ax.scatter(
                tbl[xv],
                tbl[yv],
                edgecolors="k",
                linewidths=0.35,
            )
            ax.set_xlabel(xv)
            ax.set_ylabel(f"{yv} (m/s)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(ylab)
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)


def peak_hour_phy_index(bundle: dict) -> int:
    lat = bundle["XLAT"]
    lon = bundle["XLONG"]
    up = bundle["u_p"]
    vp = bundle["v_p"]
    ws = np.hypot(up, vp)
    box = (
        (lat >= 21.92)
        & (lat <= 22.6)
        & (lon >= 113.9)
        & (lon <= 114.5)
        & np.isfinite(lat)
        & np.isfinite(lon)
    )
    sub = ws[:, box].reshape(ws.shape[0], -1)
    agg = np.nanmean(sub, axis=1)
    return int(np.nanargmax(agg))


def plot_fig4_delta_u(
    path: Path,
    bundle: dict,
    k: int,
    track: pd.DataFrame | None,
    *,
    geo_nc: Path,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    track_legend: str,
    stations_csv: Path,
) -> None:
    xlat = bundle["XLAT"]
    xlong = bundle["XLONG"]
    lat_asc, lon_asc = bundle["lat_asc"], bundle["lon_asc"]
    ue = bundle["u_e"][k]
    ve = bundle["v_e"][k]
    pts = np.column_stack([xlat.ravel(), xlong.ravel()])
    iu = RegularGridInterpolator(
        (lat_asc, lon_asc),
        ue,
        bounds_error=False,
        fill_value=np.nan,
    )
    iv = RegularGridInterpolator(
        (lat_asc, lon_asc),
        np.ascontiguousarray(bundle["v_e"][k]),
        bounds_error=False,
        fill_value=np.nan,
    )
    ue_b = iu(pts).reshape(xlat.shape)
    du = bundle["u_p"][k] - ue_b
    dx_km = geo_em_spacing_m(geo_nc) / 1000.0
    ls = LightSource(azdeg=315, altdeg=45)
    zs = np.where(np.isfinite(bundle["hgt"]), bundle["hgt"], 0.0)
    shade = ls.hillshade(zs, dx=dx_km / 111.0, dy=dx_km / 111.0, vert_exag=0.00012)
    fig, ax = plt.subplots(figsize=(9.2, 8.8), constrained_layout=True)
    ext = [
        float(xlong.min()),
        float(xlong.max()),
        float(xlat.min()),
        float(xlat.max()),
    ]
    ax.imshow(shade, cmap="gist_earth", origin="upper", extent=ext, aspect="auto")
    vmin, vmax = -8.0, 8.0
    im = ax.imshow(
        du,
        origin="upper",
        extent=ext,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        interpolation="nearest",
    )
    plt.colorbar(
        im,
        ax=ax,
        shrink=0.69,
        pad=0.02,
        label="ΔU PhyTopo − bilinear ERA5 (m/s)",
    )
    st = load_stations(stations_csv)
    ax.scatter(
        st["lon"],
        st["lat"],
        s=52,
        fc="cyan",
        ec="k",
        lw=0.35,
        marker="v",
        label="stations",
    )
    if track is not None and not track.empty:
        tk = track.loc[
            (track["time"] >= event_start) & (track["time"] <= event_end)
        ]
        if not tk.empty:
            ax.plot(
                tk["lon"],
                tk["lat"],
                lw=2.6,
                alpha=0.85,
                label=track_legend,
            )
    ax.legend(loc="lower left", fontsize=8)
    tm = bundle["time"][k]
    ttl_ts = pd.Timestamp(tm).strftime("%Y-%m-%d %HZ")
    ax.set_title(f"Peak-hour ΔU (HK box, {ttl_ts} UTC)")
    ax.set_xlabel("Lon °E")
    ax.set_ylabel("Lat °N")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def resolve_case(preset: str, data: Path) -> dict:
    """Paths and time anchors for `--preset`; aligns with MATLAB setProjectPaths case windows."""
    key = preset.strip().lower()
    geo_nc = data / "geo_em.d03.nc"
    _here = Path(__file__).resolve().parent
    mangkhut_cowin_stations = _here / "stations_wgs84_mangkhut_cowin.csv"
    hadisd_saola_stations = _here / "outputs_hadisd_saola" / "stations_wgs84_hadisd_saola.csv"
    hadisd_mangkhut_stations = _here / "outputs_hadisd_mangkhut" / "stations_wgs84_hadisd_mangkhut.csv"
    hadisd_yagi_stations = _here / "outputs_hadisd_yagi" / "stations_wgs84_hadisd_yagi.csv"
    hadisd_tc2016288_stations = (
        _here / "outputs_hadisd_tc2016288" / "stations_wgs84_hadisd_tc2016288.csv"
    )
    if key == "yagi":
        return {
            "geo_nc": geo_nc,
            "base_utc": pd.Timestamp("2024-08-31 00:00:00", tz="UTC"),
            "event_start": pd.Timestamp("2024-09-04 01:00:00", tz="UTC"),
            "event_end": pd.Timestamp("2024-09-07 01:00:00", tz="UTC"),
            "era5_nc": data / "ERA5" / "2024_2024244N09137_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2024_2024244N09137_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2024Yagi",
            "track_csv": data / "routes" / "Yagi.csv",
            "fig1_suptitle": "Typhoon Yagi (2024 Sep) — HK AWS vs grids",
            "track_legend": "Yagi track (UTC)",
            "stations_csv": None,
            "key_stations": None,
            "track_clip_like_yagi": True,
            "infer_phy_base": False,
        }
    if key == "yagi_hadisd":
        return {
            "geo_nc": geo_nc,
            "base_utc": pd.Timestamp("2024-09-04 01:00:00", tz="UTC"),
            "event_start": pd.Timestamp("2024-09-04 01:00:00", tz="UTC"),
            "event_end": pd.Timestamp("2024-09-07 01:00:00", tz="UTC"),
            "era5_nc": data / "ERA5" / "2024_2024244N09137_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2024_2024244N09137_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2024Yagi_hadisd",
            "track_csv": data / "routes" / "Yagi.csv",
            "fig1_suptitle": "Typhoon Yagi (2024 Sep) — HadISD vs grids (regional ISD subset)",
            "track_legend": "Yagi track (UTC)",
            "stations_csv": hadisd_yagi_stations,
            "key_stations": HADISD_AUTO_KEY_STATIONS,
            "track_clip_like_yagi": True,
            "infer_phy_base": True,
        }
    if key == "tc2016288_hadisd":
        return {
            "geo_nc": geo_nc,
            "base_utc": pd.Timestamp("2016-10-19 00:00:00", tz="UTC"),
            "event_start": pd.Timestamp("2016-10-19 00:00:00", tz="UTC"),
            "event_end": pd.Timestamp("2016-10-21 23:00:00", tz="UTC"),
            "era5_nc": data / "ERA5" / "2016_2016288N06146_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2016_2016288N06146_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2016_tc288_hadisd",
            "track_csv": data / "routes" / "TC2016288_eratrack.csv",
            "fig1_suptitle": "TC2016288 (2016 Sep field, ERA5 2016288N06146) — HadISD vs grids",
            "track_legend": "TC track from ERA5 MSLP minimum (hourly)",
            "stations_csv": hadisd_tc2016288_stations,
            "key_stations": HADISD_AUTO_KEY_STATIONS,
            "track_clip_like_yagi": True,
            "infer_phy_base": True,
        }
    if key == "mangkhut_hadisd":
        return {
            "geo_nc": geo_nc,
            "base_utc": pd.Timestamp("2018-09-14 00:00:00", tz="UTC"),
            "event_start": pd.Timestamp("2018-09-14 00:00:00", tz="UTC"),
            "event_end": pd.Timestamp("2018-09-17 23:00:00", tz="UTC"),
            "era5_nc": data / "ERA5" / "2018_2018250N12170_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2018_2018250N12170_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2018Mangkhut_hadisd",
            "track_csv": data / "routes" / "Mangkhut.csv",
            "fig1_suptitle": "Typhoon Mangkhut (Sep 2018) — HadISD vs grids (regional ISD subset)",
            "track_legend": "Mangkhut track (UTC)",
            "stations_csv": hadisd_mangkhut_stations,
            "key_stations": HADISD_AUTO_KEY_STATIONS,
            "track_clip_like_yagi": True,
            "infer_phy_base": True,
        }
    if key == "saola_hadisd":
        return {
            "geo_nc": geo_nc,
            "base_utc": BASE_UTC,
            "event_start": EVENT_START_UTC,
            "event_end": EVENT_END_UTC,
            "era5_nc": data / "ERA5" / "2023_2023234N18128_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2023_2023234N18128_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2023Saola_hadisd",
            "track_csv": data / "routes" / "Saola.csv",
            "fig1_suptitle": "Typhoon Saola (202309) — HadISD vs grids (regional ISD subset)",
            "track_legend": "Saola track (UTC)",
            "stations_csv": hadisd_saola_stations,
            "key_stations": HADISD_AUTO_KEY_STATIONS,
            "track_clip_like_yagi": False,
            "infer_phy_base": False,
        }
    if key == "mangkhut_cowin":
        return {
            "geo_nc": geo_nc,
            "base_utc": pd.Timestamp("2018-09-14 00:00:00", tz="UTC"),
            "event_start": pd.Timestamp("2018-09-14 00:00:00", tz="UTC"),
            "event_end": pd.Timestamp("2018-09-17 23:00:00", tz="UTC"),
            "era5_nc": data / "ERA5" / "2018_2018250N12170_ERA5_sfc.nc",
            "phy_nc": data / "PhyTopo" / "2018_2018250N12170_phytopo_recon.nc",
            "hk_dir": data / "hk" / "hk_2018Mangkhut_cowin",
            "track_csv": data / "routes" / "Mangkhut.csv",
            "fig1_suptitle": "Typhoon Mangkhut (20180914–917 UTC) — CoWIN hourly vs grids",
            "track_legend": "Mangkhut track (UTC)",
            "stations_csv": mangkhut_cowin_stations,
            "key_stations": KEY_STATIONS_MANGKHUT_COWIN,
            "track_clip_like_yagi": True,
            "infer_phy_base": True,
        }
    if key != "saola":
        raise ValueError(f"Unknown preset '{preset}'.")
    return {
        "geo_nc": geo_nc,
        "base_utc": BASE_UTC,
        "event_start": EVENT_START_UTC,
        "event_end": EVENT_END_UTC,
        "era5_nc": data / "ERA5" / "2023_2023234N18128_ERA5_sfc.nc",
        "phy_nc": data / "PhyTopo" / "2023_2023234N18128_phytopo_recon.nc",
        "hk_dir": data / "hk" / "hk_2023Saola",
        "track_csv": data / "routes" / "Saola.csv",
        "fig1_suptitle": "Typhoon Saola (20230831–903 UTC) — HK AWS vs grids",
        "track_legend": "Saola track (UTC)",
        "stations_csv": None,
        "key_stations": None,
        "track_clip_like_yagi": False,
        "infer_phy_base": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: analysis/outputs[_yagi|_mangkhut_*]",
    )
    p.add_argument(
        "--preset",
        choices=(
            "saola",
            "saola_hadisd",
            "yagi",
            "yagi_hadisd",
            "tc2016288_hadisd",
            "mangkhut_cowin",
            "mangkhut_hadisd",
        ),
        default="saola",
    )
    p.add_argument(
        "--stations-csv",
        type=Path,
        default=None,
        help="Override station lat/lon table (default: analysis/stations_wgs84.csv "
        "or preset-specific CoWIN list).",
    )
    p.add_argument(
        "--base-utc",
        type=str,
        default=None,
        help=(
            "PhyTopo hour-0 (ISO UTC). Default: infer from ERA5+Phy NC for mangkhut_* presets."
        ),
    )
    p.add_argument(
        "--nc-region-half",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Neighbourhood half-width N in grid cells: sample (2N+1)^2 centred on "
            "the nearest ERA5 / d03 raster cell with nan-mean u,v then hypot (vector wind). "
            "Use 0 for single-cell (nearest-point) sampling."
        ),
    )
    p.add_argument(
        "--save-station-wind-pdf",
        action="store_true",
        help="Also save fig_station_wind_pdf.pdf (default: PNG KDE only).",
    )
    args = p.parse_args(argv)
    if args.nc_region_half < 0:
        p.error("--nc-region-half must be >= 0")
    cfg = resolve_case(args.preset, args.data)
    geo_nc = cfg["geo_nc"]
    stations_csv = args.stations_csv or cfg.get("stations_csv") or STATIONS_CSV
    key_stations_cfg = cfg.get("key_stations")
    if args.base_utc:
        ts = pd.Timestamp(args.base_utc)
        cfg["base_utc"] = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    elif cfg.get("infer_phy_base"):
        inferred = infer_phy_base_utc_from_nc(cfg["era5_nc"], cfg["phy_nc"])
        if inferred is not None:
            cfg["base_utc"] = inferred
        else:
            raise FileNotFoundError(
                f"{args.preset}: place ERA5/PhyTopo NC under data/ or pass "
                "--base-utc (PhyTopo hour-0)."
            )
    base = cfg["base_utc"]
    if getattr(base, "tzinfo", None) is None:
        cfg["base_utc"] = pd.Timestamp(base).tz_localize("UTC")
    else:
        cfg["base_utc"] = pd.Timestamp(base).tz_convert("UTC")

    out = args.out
    if out is None:
        sub = {
            "saola_hadisd": "outputs_saola_hadisd",
            "yagi": "outputs_yagi",
            "yagi_hadisd": "outputs_yagi_hadisd",
            "tc2016288_hadisd": "outputs_tc2016288_hadisd",
            "mangkhut_cowin": "outputs_mangkhut_cowin",
            "mangkhut_hadisd": "outputs_mangkhut_hadisd",
        }.get(args.preset, "outputs")
        out = Path(__file__).resolve().parent / sub

    out.mkdir(parents=True, exist_ok=True)
    ref_lat, ref_lon = 22.3, 114.05
    track_csv: Path = cfg["track_csv"]
    if track_csv.exists():
        track = load_track(track_csv)
    else:
        track = None
    utc_clip: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if cfg.get("track_clip_like_yagi"):
        if track is not None:
            anchor_t = track_anchor_utc(track, ref_lat, ref_lon)
            utc_clip = obs_time_window_around_anchor(anchor_t, peak_span_days=3.0)
            print(
                f"{args.preset}: OBS clip UTC {utc_clip[0]} → {utc_clip[1]} "
                f"(track anchor UTC {anchor_t})"
            )
        else:
            utc_clip = (cfg["event_start"], cfg["event_end"])
            warnings.warn(
                f"{args.preset}: track CSV missing; clipping OBS to preset "
                "event_start/event_end.",
                RuntimeWarning,
                stacklevel=1,
            )
    obs_hourly = scan_obs_hourly(cfg["hk_dir"], utc_clip=utc_clip)
    bundle = load_bundle(
        cfg["era5_nc"],
        cfg["phy_nc"],
        geo_nc,
        stations_csv,
        base_utc=cfg["base_utc"],
        event_start=cfg["event_start"],
        event_end=cfg["event_end"],
    )
    merged = build_merged(
        bundle, obs_hourly, stations_csv, nc_region_half=args.nc_region_half
    )
    if key_stations_cfg is None:
        key_stations = list(KEY_STATIONS)
    elif isinstance(key_stations_cfg, str) and key_stations_cfg == HADISD_AUTO_KEY_STATIONS:
        key_stations = pick_top_obs_stations(merged, 8)
        if not key_stations and not merged.empty:
            key_stations = list(
                merged["station"].drop_duplicates().iloc[:8]
            )
    else:
        key_stations = list(key_stations_cfg)

    merged.to_csv(out / "station_event_hourly.csv", index=False)
    tbl_peaks = peak_metric_table(merged)
    tbl_peaks.to_csv(out / "table1_peak_winds.csv", index=False)

    if track is not None:
        shade0, shade1 = nearest_track_peak_window(
            track, ref_lat, ref_lon, half_hours=18.0
        )
    else:
        shade0 = cfg["event_start"] + pd.Timedelta(hours=24)
        shade1 = cfg["event_end"] - pd.Timedelta(hours=24)

    plot_fig1(
        out / "fig1_timeseries_key5.png",
        merged,
        shade0,
        shade1,
        cfg["fig1_suptitle"],
        key_stations=key_stations,
    )
    kde_title = f"{cfg['fig1_suptitle']} Hourly wind PDF (KDE)"
    wind_pdf_out = (
        out / "fig_station_wind_pdf.pdf" if args.save_station_wind_pdf else None
    )
    plot_fig_station_wind_pdf(
        wind_pdf_out,
        merged,
        kde_title,
        key_stations=key_stations,
        path_png=out / "fig_station_wind_pdf.png",
    )
    plot_fig_station_wind_kde_logdensity(
        out / "fig_station_wind_pdf_logdensity.png",
        merged,
        kde_title,
        key_stations=key_stations,
    )
    plot_fig_station_wind_exceedance(
        out / "fig_station_wind_exceedance_logy.png",
        merged,
        kde_title,
        key_stations=key_stations,
    )
    plot_fig_station_wind_qq_vs_obs(
        out / "fig_station_wind_qq_vs_obs.png",
        merged,
        kde_title,
        key_stations=key_stations,
    )
    plot_fig2(out / "fig2_peak_scatter.png", tbl_peaks)
    tbl_terr = bundle["terrain_station"]
    tbl_join = tbl_peaks.set_index("station").join(tbl_terr, how="left").reset_index()
    tbl_join.to_csv(out / "terrain_peaks_joined.csv", index=False)
    plot_fig3_terrain(out / "fig3_terrain_vs_bias.png", tbl_join.reset_index())
    k_peak = peak_hour_phy_index(bundle)
    plot_fig4_delta_u(
        out / "fig4_deltaU_peak.png",
        bundle,
        k_peak,
        track,
        geo_nc=geo_nc,
        event_start=cfg["event_start"],
        event_end=cfg["event_end"],
        track_legend=cfg["track_legend"],
        stations_csv=stations_csv,
    )
    print("Wrote outputs to", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
