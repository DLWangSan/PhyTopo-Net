#!/usr/bin/env python3
"""
Multi-event ERA5 vs PhyTopo-Net peak-wind analysis for Section 3.4 (Results).

Reads paired files:
  {season}_{sid}_ERA5_sfc.nc
  {season}_{sid}_phytopo_recon.nc

Optionally filters by event_list.csv. Extracts IBTrACS peak wind in the analysis
window (knots -> m s^-1). Interpolates raw ERA5 u,v onto the PhyTopo lat/lon grid,
computes spatial maps and exports summary CSV + multi_event_maps.mat for MATLAB Fig. 5.

Dependencies: numpy, pandas, xarray, scipy, netcdf4 (optional engines), tqdm optional.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.io import savemat
from scipy.stats import pearsonr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass(frozen=True)
class CropHelper:
    """
    Spatial slices shared by all events (built from first available pair so stacked
    maps align on identical WRF/ERA subset indices).
    """

    era_time: str
    era_lat_name: str
    era_lon_name: str
    era_lat_slice: slice
    era_lon_slice: slice
    phy_time: str
    phy_sn_slice: slice
    phy_we_slice: slice
    flip_era_lat: bool  # ERA5 latitude often descending in file → flip uv after crop

# ---------------------------------------------------------------------------
# Paths & constants (edit these for your machine; CLI overrides defaults)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_ERA5_DIR = DEFAULT_DATA_ROOT / "ERA5"
DEFAULT_PHY_DIR = DEFAULT_DATA_ROOT / "PhyTopo"
DEFAULT_IBTRACS = DEFAULT_DATA_ROOT / "IBTrACS" / "IBTrACS.WP.v04r01.nc"

DEFAULT_REGION_LON = (113.0, 117.0)
DEFAULT_REGION_LAT = (21.0, 24.0)

HOTSPOT_THRESHOLD_DEFAULT = 5.0
AGG_CHOICES = ("p90", "median", "max")
KNOTS_TO_MS = 0.514444

# IBTrACS sustained wind (USA best track) in knots
DEFAULT_IBTRACS_WIND_VAR = "usa_wind"

# If True, require track point inside REGION for IBTrACS peak; else time-only
DEFAULT_IBTRACS_USE_REGION_CLIP = True


def _decode_bytes(val: object) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    if isinstance(val, np.bytes_):
        return val.astype(str).item().strip()
    return str(val).strip()


def _normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    """Longitude in degrees east, range (-180, 180]."""
    x = lon.astype(np.float64)
    return ((x + 180.0) % 360.0) - 180.0


def load_event_list(
    path: Path | None,
    *,
    era5_dir: Path,
    phy_dir: Path,
) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      event_id, event_name, start_time, end_time, era5_path, phy_path
    If path is None, discover all pairs under era5_dir / phy_dir.
    """
    pat = re.compile(r"^(\d{4})_(.+)_ERA5_sfc\.nc$")

    def discover_pairs() -> list[dict]:
        rows: list[dict] = []
        if not era5_dir.is_dir():
            raise FileNotFoundError(f"ERA5 directory not found: {era5_dir}")
        if not phy_dir.is_dir():
            raise FileNotFoundError(f"PhyTopo directory not found: {phy_dir}")
        for era in sorted(era5_dir.glob("*_ERA5_sfc.nc")):
            m = pat.match(era.name)
            if not m:
                warnings.warn(f"Skip ERA5 file with unexpected name: {era.name}", RuntimeWarning)
                continue
            season, sid = m.group(1), m.group(2)
            stem = era.name.replace("_ERA5_sfc.nc", "")
            phy = phy_dir / f"{stem}_phytopo_recon.nc"
            event_id = sid
            if not phy.is_file():
                warnings.warn(
                    f"Missing PhyTopo twin for {era.name}: expected {phy.name}",
                    RuntimeWarning,
                )
                continue
            rows.append(
                {
                    "event_id": event_id,
                    "event_name": "",
                    "era5_path": era,
                    "phy_path": phy,
                    "season": int(season),
                }
            )
        return rows

    discovered = pd.DataFrame(discover_pairs())
    if discovered.empty:
        raise FileNotFoundError(
            f"No paired ERA5 / PhyTopo files found under {era5_dir} and {phy_dir}."
        )

    if path is None:
        discovered["start_time"] = pd.NaT
        discovered["end_time"] = pd.NaT
        print(f"[load_event_list] Discover mode: {len(discovered)} paired events.")
        return discovered

    if not path.is_file():
        raise FileNotFoundError(f"Event list CSV not found: {path}")
    tbl = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in tbl.columns}
    for req in ["event_id", "start_time", "end_time"]:
        if req not in cols:
            raise ValueError(
                f"Event list CSV must contain column '{req}'. Found {list(tbl.columns)}"
            )
    eid_c = cols["event_id"]
    st_c = cols["start_time"]
    et_c = cols["end_time"]
    name_c = cols.get("event_name")

    tbl = tbl.copy()
    tbl["_event_id"] = tbl[eid_c].astype(str).str.strip()
    tbl["_start"] = pd.to_datetime(tbl[st_c], utc=True, errors="coerce")
    tbl["_end"] = pd.to_datetime(tbl[et_c], utc=True, errors="coerce")
    tbl["_event_name"] = tbl[name_c].astype(str) if name_c else ""

    merged = tbl.merge(
        discovered,
        left_on="_event_id",
        right_on="event_id",
        how="inner",
        suffixes=("", "_disk"),
    )
    if merged.empty:
        raise ValueError(
            "Event list CSV has no rows matching discovered *_ERA5_sfc.nc / phytopo_recon.nc pairs."
        )
    merged["start_time"] = merged["_start"]
    merged["end_time"] = merged["_end"]
    merged["event_name"] = merged["_event_name"].where(
        merged["_event_name"].notna(), merged.get("event_name", "")
    )
    # Prefer explicit CSV times; drop helper columns
    out = merged[
        ["event_id", "event_name", "start_time", "end_time", "era5_path", "phy_path", "season"]
    ].reset_index(drop=True)
    lost = len(tbl) - len(out)
    if lost:
        warnings.warn(
            f"{lost} event_list rows did not match a file pair on disk.",
            RuntimeWarning,
        )
    print(f"[load_event_list] CSV + disk: {len(out)} paired events.")
    return out


def build_ibtracs_sid_index(ib_path: Path) -> tuple[xr.Dataset, dict[str, int]]:
    if not ib_path.is_file():
        raise FileNotFoundError(f"IBTrACS NetCDF not found: {ib_path}")
    ds = xr.open_dataset(ib_path, decode_times=True, mask_and_scale=True)
    sids_raw = ds["sid"].values
    index: dict[str, int] = {}
    for i in range(ds.sizes["storm"]):
        index[_decode_bytes(sids_raw[i])] = i
    return ds, index


def load_ibtracs_peak_for_event(
    ds_ib: xr.Dataset,
    sid_index: dict[str, int],
    event_id: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    region_lon: tuple[float, float],
    region_lat: tuple[float, float],
    wind_var: str,
    *,
    use_region_clip: bool,
    knots_to_ms: float,
) -> tuple[float | float, bool]:
    """Return (peak_ms, clipped_to_region_fallback_used)."""
    if event_id not in sid_index:
        return np.nan, False
    si = sid_index[event_id]
    numobs = int(float(ds_ib["numobs"].isel(storm=si).values))
    numobs = max(0, min(numobs, ds_ib.sizes["date_time"]))

    t = ds_ib["time"].isel(storm=si, date_time=slice(0, numobs)).values
    lat = ds_ib["lat"].isel(storm=si, date_time=slice(0, numobs)).values.astype(np.float64)
    lon_raw = ds_ib["lon"].isel(storm=si, date_time=slice(0, numobs)).values.astype(np.float64)
    lon = _normalize_lon_180(lon_raw)

    try:
        wkt = ds_ib[wind_var].isel(storm=si, date_time=slice(0, numobs)).values.astype(np.float64)
    except KeyError as e:
        raise KeyError(f"IBTrACS wind variable '{wind_var}' not in dataset.") from e

    t_series = pd.to_datetime(t.ravel(), utc=True, errors="coerce")
    mtime = t_series.notna() & (t_series >= t0) & (t_series <= t1)

    if use_region_clip:
        mreg = (
            (lat >= region_lat[0])
            & (lat <= region_lat[1])
            & (lon >= region_lon[0])
            & (lon <= region_lon[1])
        )
        mask = mtime & mreg & np.isfinite(wkt)
        if not np.any(mask):
            mask = mtime & np.isfinite(wkt)
            fallback = True
        else:
            fallback = False
    else:
        mask = mtime & np.isfinite(wkt)
        fallback = False

    if not np.any(mask):
        return np.nan, fallback
    peak_kt = float(np.nanmax(wkt[mask]))
    return peak_kt * knots_to_ms, fallback


def _pick_time_coord(ds: xr.Dataset) -> str:
    for c in ("valid_time", "time", "Time", "datetime"):
        if c in ds.variables or c in ds.coords:
            return c
    raise ValueError(f"Cannot find time coordinate in {list(ds.variables)}")


def _pick_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    for la, lo in (("latitude", "longitude"), ("lat", "lon")):
        if la in ds.variables and lo in ds.variables:
            return la, lo
    raise ValueError(f"Cannot find lat/lon in {list(ds.variables)}")


def _wind_speed_from_dataset(ds: xr.Dataset) -> tuple[xr.DataArray, str]:
    """
    Return (ws_da, mode) where mode is 'scalar' or 'uv'.
    ws_da has dims including time, y, x (best effort).
    """
    lower = {k.lower(): k for k in ds.data_vars}
    for cand in ("ws", "wind_speed", "ws10", "sfcwind", "wind"):
        if cand in lower:
            return ds[lower[cand]], "scalar"
    u_key = v_key = None
    for u_c in ("u10", "U10", "UU", "u"):
        if u_c in ds.data_vars:
            u_key = u_c
            break
    for v_c in ("v10", "V10", "VV", "v"):
        if v_c in ds.data_vars:
            v_key = v_c
            break
    if u_key is None or v_key is None:
        raise ValueError(
            "No wind-speed or u/v pair found. Expected ws/wind_speed or u10/v10."
        )
    u = ds[u_key]
    v = ds[v_key]
    return np.hypot(u.astype(np.float32), v.astype(np.float32)), "uv"



def _pick_phy_time_name(ds: xr.Dataset) -> str:
    for c in ("time", "Time"):
        if c in ds.dims:
            return c
    raise ValueError(f"PhyTopo dataset has no time/Time dimension. Got dims={tuple(ds.dims)}.")


def build_crop_helper(
    era_path: Path,
    phy_path: Path,
    region_lon: tuple[float, float],
    region_lat: tuple[float, float],
    *,
    pad_e: int = 2,
    pad_p: int = 2,
) -> CropHelper:
    """
    Crop indices from the first ERA5 / PhyTopo pair (prd130 grids are uniform).
    Keeps stacked delta maps on identical phy grid shape across events.
    """
    ds_e = xr.open_dataset(era_path)
    ds_p = xr.open_dataset(phy_path)
    try:
        te = _pick_time_coord(ds_e)
        lt_n, ln_n = _pick_lat_lon_names(ds_e)
        lat_vals = ds_e[lt_n].values.astype(np.float64)
        lon_vals = ds_e[ln_n].values.astype(np.float64)
        lon_geo = _normalize_lon_180(lon_vals.copy()) if float(np.nanmax(lon_vals)) > 180.0 else lon_vals

        if "XLAT" not in ds_p.variables:
            raise ValueError(f"PhyTopo file missing XLAT: {phy_path}")
        xlat = ds_p["XLAT"].values.astype(np.float64)
        if xlat.ndim == 3:
            xlat = xlat[0]
        xlname = "XLONG" if "XLONG" in ds_p.variables else "LONG"
        if xlname not in ds_p.variables:
            raise ValueError(f"PhyTopo file missing XLONG/LONG: {phy_path}")
        xlong = ds_p[xlname].values.astype(np.float64)
        if xlong.ndim == 3:
            xlong = xlong[0]
        xlong = _normalize_lon_180(xlong)

        msk = (
            (xlat >= region_lat[0] - 0.5)
            & (xlat <= region_lat[1] + 0.5)
            & (xlong >= region_lon[0] - 0.5)
            & (xlong <= region_lon[1] + 0.5)
        )
        if not np.any(msk):
            raise ValueError("PhyTopo grid has no land/sea points within buffered REGION box.")
        jj, ii = np.where(msk)
        j0p, j1p = int(jj.min()), int(jj.max())
        i0p, i1p = int(ii.min()), int(ii.max())
        j0p = max(j0p - pad_p, 0)
        j1p = min(j1p + pad_p, xlat.shape[0] - 1)
        i0p = max(i0p - pad_p, 0)
        i1p = min(i1p + pad_p, xlat.shape[1] - 1)

        xlat_crop = xlat[j0p : j1p + 1, i0p : i1p + 1]
        xlong_crop = xlong[j0p : j1p + 1, i0p : i1p + 1]
        buf = 0.75
        lat_min = float(np.nanmin(xlat_crop)) - buf
        lat_max = float(np.nanmax(xlat_crop)) + buf
        lon_min = float(np.nanmin(xlong_crop)) - buf
        lon_max = float(np.nanmax(xlong_crop)) + buf

        mlat = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        if not np.any(mlat):
            raise ValueError(
                "ERA5 latitude range does not cover PhyTopo extent; check regional ERA5 box."
            )
        jb_lo = int(np.flatnonzero(mlat)[0])
        jb_hi = int(np.flatnonzero(mlat)[-1])
        jb_lo = max(jb_lo - pad_e, 0)
        jb_hi = min(jb_hi + pad_e, lat_vals.size - 1)

        mlon = (lon_geo >= lon_min) & (lon_geo <= lon_max)
        if not np.any(mlon):
            raise ValueError(
                "ERA5 longitude range does not cover PhyTopo extent; check regional ERA5 box."
            )
        ib_lo = int(np.flatnonzero(mlon)[0])
        ib_hi = int(np.flatnonzero(mlon)[-1])
        ib_lo = max(ib_lo - pad_e, 0)
        ib_hi = min(ib_hi + pad_e, lon_vals.size - 1)

        lat_sel = lat_vals[jb_lo : jb_hi + 1]
        flip_era_lat = lat_sel.size > 1 and float(lat_sel[0]) > float(lat_sel[-1])

        tphy = _pick_phy_time_name(ds_p)

        print(
            f"[build_crop_helper] ERA5 slice lat[{jb_lo}:{jb_hi+1}] lon[{ib_lo}:{ib_hi+1}] | "
            f"Phy ({j0p}:{j1p+1}, {i0p}:{i1p+1}) flip_era_lat={flip_era_lat}"
        )
        return CropHelper(
            era_time=te,
            era_lat_name=lt_n,
            era_lon_name=ln_n,
            era_lat_slice=slice(jb_lo, jb_hi + 1),
            era_lon_slice=slice(ib_lo, ib_hi + 1),
            phy_time=tphy,
            phy_sn_slice=slice(j0p, j1p + 1),
            phy_we_slice=slice(i0p, i1p + 1),
            flip_era_lat=flip_era_lat,
        )
    finally:
        ds_e.close()
        ds_p.close()


def interpolate_era5_uv_to_phy(
    u_e: np.ndarray,
    v_e: np.ndarray,
    lat_e: np.ndarray,
    lon_e: np.ndarray,
    xlat: np.ndarray,
    xlon: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """
    u_e,v_e shaped (time, ny, nx) on ERA5 grid (lat_e ascending, lon_e ascending).
    Interpolate bilinear onto WRF-like 2D lat/lon.
    """
    nt, ny, nx = u_e.shape
    if not (lat_e.ndim == 1 and lon_e.ndim == 1 and lat_e.size == ny and lon_e.size == nx):
        raise ValueError(
            f"ERA5 slice shape mismatch: u {u_e.shape}, lat_e {lat_e.shape}, lon_e {lon_e.shape}"
        )
    pj, pi = xlat.flatten(), xlon.flatten()
    u_out = np.empty((nt, xlat.shape[0], xlat.shape[1]), dtype=np.float32)
    v_out = np.empty_like(u_out)

    points = np.column_stack([pj, pi])

    for it in range(nt):
        interp_u = RegularGridInterpolator(
            (lat_e, lon_e),
            np.asarray(u_e[it], dtype=np.float64),
            bounds_error=False,
            fill_value=fill_value,
            method="linear",
        )
        interp_v = RegularGridInterpolator(
            (lat_e, lon_e),
            np.asarray(v_e[it], dtype=np.float64),
            bounds_error=False,
            fill_value=fill_value,
            method="linear",
        )
        u_out[it] = interp_u(points).reshape(xlat.shape).astype(np.float32)
        v_out[it] = interp_v(points).reshape(xlat.shape).astype(np.float32)

    return u_out, v_out


def load_event_model_fields(
    era_path: Path,
    phy_path: Path,
    region_lon: tuple[float, float],
    region_lat: tuple[float, float],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    crop: CropHelper,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
]:
    """
    Returns ((u_phy, v_phy), (u_era_on_phy_grid), XLAT, XLONG, aligned_times).

    ERA5 raw u,v are bilinear-interpolated onto the cropped PhyTopo lat/lon.
    Times are taken from ERA5 `valid_time` (required to match PhyTopo `time` 1:1 for prd130).
    """
    del region_lon, region_lat  # fixed by shared crop; kept in API for clarity

    ds_e = xr.open_dataset(era_path)
    ds_p = xr.open_dataset(phy_path)
    try:
        te = crop.era_time
        tphy = crop.phy_time
        lt_n, ln_n = crop.era_lat_name, crop.era_lon_name

        te_arr = pd.DatetimeIndex(pd.to_datetime(ds_e[te].values, utc=True, errors="coerce"))
        tp_arr = pd.DatetimeIndex(pd.to_datetime(ds_p[tphy].values, utc=True, errors="coerce"))
        if te_arr.size != tp_arr.size:
            raise ValueError(
                f"ERA5 n_time={te_arr.size} vs PhyTopo n_time={tp_arr.size} in {phy_path.name}"
            )
        dt_sec = np.abs(te_arr.asi8.astype(np.float64) - tp_arr.asi8.astype(np.float64)) / 1e9
        if np.nanmax(dt_sec) > 1.0:
            raise ValueError(
                f"ERA5 vs PhyTopo timestamp mismatch (>1 s) for {phy_path.name}. "
                "Check recon driver NC alignment."
            )

        t0_cmp = pd.Timestamp(t0)
        t1_cmp = pd.Timestamp(t1)
        if t0_cmp.tzinfo is None:
            t0_cmp = t0_cmp.tz_localize("UTC")
        else:
            t0_cmp = t0_cmp.tz_convert("UTC")
        if t1_cmp.tzinfo is None:
            t1_cmp = t1_cmp.tz_localize("UTC")
        else:
            t1_cmp = t1_cmp.tz_convert("UTC")

        idx = np.flatnonzero((te_arr >= t0_cmp) & (te_arr <= t1_cmp))
        if idx.size == 0:
            raise ValueError(f"No ERA5 times in [{t0}, {t1}] for {era_path.name}")


        u_e_raw = (
            ds_e["u10"]
            .isel(
                {
                    te: idx,
                    lt_n: crop.era_lat_slice,
                    ln_n: crop.era_lon_slice,
                }
            )
            .values.astype(np.float32)
        )
        v_e_raw = (
            ds_e["v10"]
            .isel(
                {
                    te: idx,
                    lt_n: crop.era_lat_slice,
                    ln_n: crop.era_lon_slice,
                }
            )
            .values.astype(np.float32)
        )
        if u_e_raw.ndim == 4 and u_e_raw.shape[1] == 1:
            u_e_raw = u_e_raw[:, 0, :, :]
            v_e_raw = v_e_raw[:, 0, :, :]
        if u_e_raw.ndim != 3:
            raise ValueError(f"Unexpected ERA5 u10 shape {u_e_raw.shape} for {era_path.name}")

        if crop.flip_era_lat:
            u_e_raw = u_e_raw[:, ::-1, :]
            v_e_raw = v_e_raw[:, ::-1, :]

        lat_e = ds_e[lt_n].isel({lt_n: crop.era_lat_slice}).values.astype(np.float64)
        if crop.flip_era_lat:
            lat_e = lat_e[::-1]

        lon_e = ds_e[ln_n].isel({ln_n: crop.era_lon_slice}).values.astype(np.float64)
        if float(np.nanmax(lon_e)) > 180.0:
            lon_e = _normalize_lon_180(lon_e)
        if lon_e.size > 1 and np.any(np.diff(lon_e) < 0):
            order = np.argsort(lon_e)
            lon_e = lon_e[order]
            u_e_raw = u_e_raw[:, :, order]
            v_e_raw = v_e_raw[:, :, order]

        up = (
            ds_p["u10"]
            .isel(
                {
                    tphy: idx,
                    "south_north": crop.phy_sn_slice,
                    "west_east": crop.phy_we_slice,
                }
            )
            .values.astype(np.float32)
        )
        vp = (
            ds_p["v10"]
            .isel(
                {
                    tphy: idx,
                    "south_north": crop.phy_sn_slice,
                    "west_east": crop.phy_we_slice,
                }
            )
            .values.astype(np.float32)
        )

        xlname = "XLONG" if "XLONG" in ds_p.variables else "LONG"
        xlat = (
            ds_p["XLAT"]
            .isel(south_north=crop.phy_sn_slice, west_east=crop.phy_we_slice)
            .values.astype(np.float64)
        )
        xlong = (
            ds_p[xlname]
            .isel(south_north=crop.phy_sn_slice, west_east=crop.phy_we_slice)
            .values.astype(np.float64)
        )
        xlong = _normalize_lon_180(xlong)

        u_p_int, v_p_int = interpolate_era5_uv_to_phy(u_e_raw, v_e_raw, lat_e, lon_e, xlat, xlong)

        times = pd.DatetimeIndex(te_arr[idx])
        return (up, vp), (u_p_int, v_p_int), xlat.astype(np.float32), xlong.astype(np.float32), times
    finally:
        ds_e.close()
        ds_p.close()


def compute_event_peak_fields(
    u_p: np.ndarray,
    v_p: np.ndarray,
    u_e_on_p: np.ndarray,
    v_e_on_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-gridcell nanmax over time."""
    ws_p = np.hypot(u_p.astype(np.float64), v_p.astype(np.float64))
    ws_e = np.hypot(u_e_on_p.astype(np.float64), v_e_on_p.astype(np.float64))
    peak_p = np.nanmax(ws_p, axis=0)
    peak_e = np.nanmax(ws_e, axis=0)
    delta = peak_p - peak_e
    return peak_e, peak_p, delta


def compute_event_domain_max(peak_e: np.ndarray, peak_p: np.ndarray) -> tuple[float, float]:
    return float(np.nanmax(peak_e)), float(np.nanmax(peak_p))


def aggregate_delta_stack(stack: np.ndarray, method: str) -> np.ndarray:
    """stack shape (n_event, ny, nx)."""
    if method == "p90":
        return np.nanpercentile(stack, 90.0, axis=0)
    if method == "median":
        return np.nanmedian(stack, axis=0)
    if method == "max":
        return np.nanmax(stack, axis=0)
    raise ValueError(method)


def infer_event_window_from_era(era_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    ds = xr.open_dataset(era_path)
    try:
        te = _pick_time_coord(ds)
        era_times = pd.to_datetime(ds[te].values, utc=True, errors="coerce")
        era_times = era_times[era_times.notna()]
        if len(era_times) == 0:
            raise ValueError("No valid ERA5 times.")
        return pd.Timestamp(era_times.min()), pd.Timestamp(era_times.max())
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--era5-dir", type=Path, default=None)
    p.add_argument("--phy-dir", type=Path, default=None)
    p.add_argument("--ibtracs", type=Path, default=DEFAULT_IBTRACS)
    p.add_argument("--event-list", type=Path, default=None, help="Optional CSV filter / times.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis" / "outputs_multi_event")
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--region-lon", type=float, nargs=2, default=list(DEFAULT_REGION_LON))
    p.add_argument("--region-lat", type=float, nargs=2, default=list(DEFAULT_REGION_LAT))
    p.add_argument("--hotspot-threshold", type=float, default=HOTSPOT_THRESHOLD_DEFAULT)
    p.add_argument("--agg-method", choices=AGG_CHOICES, default="p90")
    p.add_argument("--ibtracs-wind-var", type=str, default=DEFAULT_IBTRACS_WIND_VAR)
    p.add_argument(
        "--no-ibtracs-region-clip",
        action="store_true",
        help="Use time mask only for IBTrACS peak (ignore lat/lon filter).",
    )
    p.add_argument(
        "--save-delta-stack",
        action="store_true",
        help="Also write delta_u_peak_events (n_event,ny,nx) into the .mat (large).",
    )
    p.add_argument("--save-netcdf", action="store_true")
    p.add_argument("--max-events", type=int, default=0, help="0 = all (debug cap).")
    args = p.parse_args(argv)

    era5_dir = args.era5_dir or (args.data_root / "ERA5")
    phy_dir = args.phy_dir or (args.data_root / "PhyTopo")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    region_lon = (float(args.region_lon[0]), float(args.region_lon[1]))
    region_lat = (float(args.region_lat[0]), float(args.region_lat[1]))

    events = load_event_list(args.event_list, era5_dir=era5_dir, phy_dir=phy_dir)
    if args.max_events and args.max_events > 0:
        events = events.iloc[: args.max_events].copy()

    crop_helper = build_crop_helper(
        Path(events.iloc[0]["era5_path"]),
        Path(events.iloc[0]["phy_path"]),
        region_lon,
        region_lat,
    )

    ds_ib, sid_index = build_ibtracs_sid_index(args.ibtracs)

    delta_list: list[np.ndarray] = []
    era5_peak_list: list[np.ndarray] = []
    phy_peak_list: list[np.ndarray] = []
    rows: list[dict] = []

    iterator = range(len(events))
    if tqdm is not None:
        iterator = tqdm(iterator, desc="events", unit="evt")

    n_skip = 0
    ib_fallback_count = 0

    for k in iterator:
        r = events.iloc[k]
        eid = str(r["event_id"])
        name_csv = str(r.get("event_name") or "")
        era_path = Path(r["era5_path"])
        phy_path = Path(r["phy_path"])

        if pd.isna(r.get("start_time")) or pd.isna(r.get("end_time")):
            t0, t1 = infer_event_window_from_era(era_path)
        else:
            t0 = pd.Timestamp(r["start_time"])
            t1 = pd.Timestamp(r["end_time"])
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            if t1.tzinfo is None:
                t1 = t1.tz_localize("UTC")

        try:
            (u_p, v_p), (u_ei, v_ei), xlat, xlong, _times = load_event_model_fields(
                era_path,
                phy_path,
                region_lon,
                region_lat,
                t0,
                t1,
                crop_helper,
            )
            peak_e, peak_p, delta = compute_event_peak_fields(u_p, v_p, u_ei, v_ei)
            dmax_e, dmax_p = compute_event_domain_max(peak_e, peak_p)

            ib_peak, fb = load_ibtracs_peak_for_event(
                ds_ib,
                sid_index,
                eid,
                t0,
                t1,
                region_lon,
                region_lat,
                args.ibtracs_wind_var,
                use_region_clip=not args.no_ibtracs_region_clip,
                knots_to_ms=KNOTS_TO_MS,
            )
            if fb:
                ib_fallback_count += 1

            ename = name_csv
            if not ename.strip():
                si = sid_index.get(eid)
                if si is not None:
                    raw = ds_ib["name"].isel(storm=si).values
                    ename = _decode_bytes(raw)

            rows.append(
                {
                    "event_id": eid,
                    "event_name": ename,
                    "start_time": t0.isoformat(),
                    "end_time": t1.isoformat(),
                    "ibtracs_peak_ms": ib_peak,
                    "era5_domain_max_ms": dmax_e,
                    "phytopo_domain_max_ms": dmax_p,
                }
            )
            delta_list.append(delta.astype(np.float32))
            era5_peak_list.append(peak_e.astype(np.float32))
            phy_peak_list.append(peak_p.astype(np.float32))
        except Exception as ex:  # pragma: no cover
            n_skip += 1
            warnings.warn(f"Skip event {eid} ({era_path.name}): {ex}", RuntimeWarning)

    ds_ib.close()

    if not rows:
        print("No events processed successfully.", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "event_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}  rows={len(summary)}")

    stack = np.stack(delta_list, axis=0)
    delta_u_map = aggregate_delta_stack(stack, args.agg_method).astype(np.float32)
    hotspot_frequency = np.mean(stack > float(args.hotspot_threshold), axis=0).astype(np.float32)
    era5_peak_p90 = np.nanpercentile(np.stack(era5_peak_list, axis=0), 90, axis=0).astype(np.float32)
    phy_peak_p90 = np.nanpercentile(np.stack(phy_peak_list, axis=0), 90, axis=0).astype(np.float32)

    mat_payload: dict = {
        "lon": xlong.astype(np.float32),
        "lat": xlat.astype(np.float32),
        "delta_u_map": delta_u_map,
        "hotspot_frequency": hotspot_frequency,
        "era5_peak_p90": era5_peak_p90,
        "phytopo_peak_p90": phy_peak_p90,
        "era5_domain_max_ms": summary["era5_domain_max_ms"].values.astype(np.float64),
        "phytopo_domain_max_ms": summary["phytopo_domain_max_ms"].values.astype(np.float64),
        "ibtracs_peak_ms": summary["ibtracs_peak_ms"].values.astype(np.float64),
        "event_id": np.array(summary["event_id"].tolist(), dtype=object),
        "event_name": np.array(summary["event_name"].tolist(), dtype=object),
        "agg_method": np.array(args.agg_method),
        "hotspot_threshold": np.float64(args.hotspot_threshold),
        "region_lon": np.array(region_lon, dtype=np.float64),
        "region_lat": np.array(region_lat, dtype=np.float64),
        "map_extent_lon": np.array(region_lon, dtype=np.float64),
        "map_extent_lat": np.array(region_lat, dtype=np.float64),
    }
    if args.save_delta_stack:
        mat_payload["delta_u_peak_events"] = stack.astype(np.float32)

    mat_path = out_dir / "multi_event_maps.mat"
    savemat(mat_path, mat_payload, do_compression=True)
    print(f"[write] {mat_path}")

    if args.save_netcdf:
        ds_out = xr.Dataset(
            data_vars={
                "delta_u_map": (("south_north", "west_east"), delta_u_map),
                "hotspot_frequency": (("south_north", "west_east"), hotspot_frequency),
                "era5_peak_p90": (("south_north", "west_east"), era5_peak_p90),
                "phytopo_peak_p90": (("south_north", "west_east"), phy_peak_p90),
            },
            coords={
                "XLAT": (("south_north", "west_east"), xlat),
                "XLONG": (("south_north", "west_east"), xlong),
            },
        )
        nc_path = out_dir / "multi_event_maps.nc"
        ds_out.to_netcdf(nc_path)
        print(f"[write] {nc_path}")

    s = summary
    valid_ib = s["ibtracs_peak_ms"].replace([np.inf, -np.inf], np.nan).dropna()
    print("\n=== Summary ===")
    print(f"Events listed: {len(events)}")
    print(f"Events used:   {len(summary)}  skipped: {n_skip}")
    print(f"IBTrACS time-only fallback (no point in region): {ib_fallback_count}")
    if len(valid_ib):
        print(f"Mean IBTrACS peak (used events):   {float(valid_ib.mean()):.3f} m s^-1")
    print(f"Median ERA5 domain max:    {float(np.nanmedian(s['era5_domain_max_ms'])):.3f} m s^-1")
    print(f"Median PhyTopo domain max: {float(np.nanmedian(s['phytopo_domain_max_ms'])):.3f} m s^-1")
    delta_dom = s["phytopo_domain_max_ms"] - s["era5_domain_max_ms"]
    print(f"Mean (PhyTopo-ERA5) domain max: {float(np.nanmean(delta_dom)):.3f} m s^-1")
    mx = np.isfinite(s["ibtracs_peak_ms"]) & np.isfinite(s["era5_domain_max_ms"])
    mx2 = np.isfinite(s["ibtracs_peak_ms"]) & np.isfinite(s["phytopo_domain_max_ms"])
    if mx.sum() >= 5:
        r_e, pe = pearsonr(s.loc[mx, "ibtracs_peak_ms"], s.loc[mx, "era5_domain_max_ms"])
        print(f"Pearson r (IBTrACS vs ERA5 domain max):   r={r_e:.3f}, p={pe:.3g}")
    if mx2.sum() >= 5:
        r_p, pp = pearsonr(s.loc[mx2, "ibtracs_peak_ms"], s.loc[mx2, "phytopo_domain_max_ms"])
        print(f"Pearson r (IBTrACS vs PhyTopo domain max): r={r_p:.3f}, p={pp:.3g}")

    print(f"Outputs: {csv_path}, {mat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
