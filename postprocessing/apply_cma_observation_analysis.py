#!/usr/bin/env python3
"""Create an observation-constrained hourly wind analysis for one typhoon.

Stage 1 applies an IBTrACS-pressure-based event-intensity anchor to the direct
PhyTopo field. Stage 2 interpolates simultaneous CMA wind-speed residuals with
a regularised Gaussian RBF. Wind direction is retained from PhyTopo.
"""

import argparse
import calendar
import csv
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


def decode_chars(row):
    arr = np.asarray(row)
    if arr.dtype.kind == "S":
        return b"".join(arr.tolist()).decode("ascii", errors="ignore").strip("\x00 ")
    return "".join(str(x) for x in arr).strip("\x00 ")


def datetime_seconds(values, var):
    dates = num2date(values, var.units, calendar=getattr(var, "calendar", "standard"),
                    only_use_cftime_datetimes=True)
    return np.array([calendar.timegm((x.year, x.month, x.day, x.hour, x.minute,
                                      int(x.second))) for x in dates], float)


def interpolate_pressure(ib_path, sid, target_seconds):
    with Dataset(ib_path) as ds:
        sids = [decode_chars(x) for x in ds.variables["sid"][:]]
        i = sids.index(sid)
        tv = ds.variables["time"]
        raw = tv[i, :]
        valid = ~np.ma.getmaskarray(raw)
        tt = datetime_seconds(np.asarray(raw)[valid], tv)
        wmo = np.ma.asarray(ds.variables["wmo_pres"][i, valid], dtype=float).filled(np.nan)
        usa = np.ma.asarray(ds.variables["usa_pres"][i, valid], dtype=float).filled(np.nan)
        pressure = np.where(np.isfinite(wmo), wmo, usa)
    ok = np.isfinite(tt) & np.isfinite(pressure)
    return np.interp(target_seconds, tt[ok], pressure[ok])


def regular_bilinear(field, lat, lon, target_lat, target_lon):
    iy = np.searchsorted(lat, target_lat, side="right") - 1
    ix = np.searchsorted(lon, target_lon, side="right") - 1
    iy = np.clip(iy, 0, len(lat) - 2)
    ix = np.clip(ix, 0, len(lon) - 2)
    y0, y1 = lat[iy], lat[iy + 1]
    x0, x1 = lon[ix], lon[ix + 1]
    wy = np.divide(target_lat - y0, y1 - y0, out=np.zeros_like(target_lat), where=y1 != y0)
    wx = np.divide(target_lon - x0, x1 - x0, out=np.zeros_like(target_lon), where=x1 != x0)
    return ((1 - wy) * ((1 - wx) * field[iy, ix] + wx * field[iy, ix + 1])
            + wy * ((1 - wx) * field[iy + 1, ix] + wx * field[iy + 1, ix + 1]))


def pressure_anchor(speed, event_ratio, threshold=20.0, gamma=3.0, fraction=0.5):
    activation = 1.0 / (1.0 + np.exp(-(speed - threshold) / gamma))
    return np.maximum(speed * (1.0 + fraction * (event_ratio - 1.0) * activation), 0.0)


def rbf_increment(lat2d, lon2d, station_lat, station_lon, residual,
                  length_km=30.0, nugget=0.005):
    ref = float(np.nanmean(station_lat))
    lon_scale = 111.0 * np.cos(np.deg2rad(ref))
    sx = station_lon * lon_scale
    sy = station_lat * 111.0
    # Construct the basis on the actual output grid and pass it through the
    # same bilinear observation operator used in validation.  This prevents
    # a constraint fitted in continuous coordinates from weakening when it is
    # written to, and sampled back from, the 3-km grid.
    lat1, lon1 = lat2d[:, 0], lon2d[0, :]
    kernels = []
    nearest = np.full_like(lat2d, np.inf, dtype=float)
    for slat, slon in zip(station_lat, station_lon):
        d = np.hypot((lon2d - slon) * lon_scale, (lat2d - slat) * 111.0)
        kernels.append(np.exp(-0.5 * (d / length_km) ** 2))
        nearest = np.minimum(nearest, d)
    operator = np.empty((len(station_lat), len(station_lat)), float)
    for j, basis in enumerate(kernels):
        for i, (slat, slon) in enumerate(zip(station_lat, station_lon)):
            operator[i, j] = regular_bilinear(basis, lat1, lon1, slat, slon)
    try:
        alpha = np.linalg.solve(operator + nugget * np.eye(len(residual)), residual)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(operator + nugget * np.eye(len(residual)), residual, rcond=None)[0]
    increment = np.zeros_like(lat2d, dtype=float)
    for a, basis in zip(alpha, kernels):
        increment += a * basis
    influence = np.exp(-0.5 * (nearest / length_km) ** 2)
    # Retain large observation-supported residuals at extreme stations.  The
    # final analysed speed is still bounded to the physically plausible
    # 0--80 m/s range below.
    return np.clip(increment, -30.0, 40.0), influence


def load_cma_event(path, storm):
    """Return {epoch_seconds: [(station_id, lat, lon, speed), ...]}."""
    opener = gzip.open if str(path).endswith(".gz") else open
    raw = {}
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event_id"] != storm:
                continue
            stamp = datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
            seconds = calendar.timegm(stamp.utctimetuple())
            # Preserve documented coordinate variants as distinct observation
            # anchors; collapsing them to the physical station ID degraded the
            # written field near relocated or differently reported sites.
            key = (seconds, row.get("station_key", row["station_id"]))
            raw.setdefault(key, []).append((float(row["lat"]), float(row["lon"]),
                                             float(row["obs_speed"])))
    by_time = {}
    for (seconds, station_id), values in raw.items():
        arr = np.asarray(values, float)
        by_time.setdefault(seconds, []).append((
            station_id, float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
            float(np.median(arr[:, 2]))
        ))
    return by_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("storm_id")
    ap.add_argument("--era-root", required=True)
    ap.add_argument("--recon-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--cma-hourly", required=True)
    ap.add_argument("--global-model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ibtracs", required=True)
    args = ap.parse_args()

    storm = args.storm_id
    era_path = Path(args.era_root) / f"{storm}_ERA5_sfc.nc"
    recon_path = Path(args.recon_root) / f"{storm}_phytopo_recon.nc"
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{storm}_phytopo_cma_analysis.nc"
    with open(args.pairs, newline="", encoding="utf-8") as handle:
        sid_map = {row["storm_id"]: row["sid"] for row in csv.DictReader(handle)}
    model = np.load(args.global_model, allow_pickle=False)
    cma_by_time = load_cma_event(args.cma_hourly, storm)

    with Dataset(recon_path) as rr, Dataset(era_path) as ee:
        lat = np.asarray(rr.variables["XLAT"][:], float)
        lon = np.asarray(rr.variables["XLONG"][:], float)
        lat1, lon1 = lat[:, 0], lon[0, :]
        et = ee.variables["valid_time"]
        seconds = datetime_seconds(et[:], et)
        pressure = interpolate_pressure(args.ibtracs, sid_map[storm], seconds)
        nt, ny, nx = len(seconds), *lat.shape

        direct_peak = 0.0
        for t in range(nt):
            u = np.asarray(rr.variables["u10"][t], float)
            v = np.asarray(rr.variables["v10"][t], float)
            direct_peak = max(direct_peak, float(np.nanmax(np.hypot(u, v))))
        deficit = max(1010.0 - float(np.nanmin(pressure)), 0.0)
        global_target = float(model["global_anchor_intercept"]
                              + model["global_anchor_coef"][0] * direct_peak
                              + model["global_anchor_coef"][1] * np.sqrt(deficit))
        global_target = float(np.clip(global_target, 0.0, 80.0))
        event_ratio = float(np.clip(global_target / max(direct_peak, 0.5), 0.70, 1.65))

        with Dataset(out_path, "w", format="NETCDF4") as oo:
            oo.createDimension("time", nt)
            oo.createDimension("south_north", ny)
            oo.createDimension("west_east", nx)
            tv = oo.createVariable("time", "f8", ("time",))
            tv[:] = et[:]
            tv.units = et.units
            tv.calendar = getattr(et, "calendar", "standard")
            yv = oo.createVariable("XLAT", "f4", ("south_north", "west_east"), zlib=True)
            xv = oo.createVariable("XLONG", "f4", ("south_north", "west_east"), zlib=True)
            yv[:], xv[:] = lat, lon
            yv.units, xv.units = "degrees_north", "degrees_east"
            kwargs = dict(zlib=True, complevel=4, shuffle=True, chunksizes=(1, ny, nx))
            uo = oo.createVariable("u10_analysis", "f4", ("time", "south_north", "west_east"), **kwargs)
            vo = oo.createVariable("v10_analysis", "f4", ("time", "south_north", "west_east"), **kwargs)
            wo = oo.createVariable("wspd10_analysis", "f4", ("time", "south_north", "west_east"), **kwargs)
            io = oo.createVariable("analysis_increment", "f4", ("time", "south_north", "west_east"), **kwargs)
            qo = oo.createVariable("cma_influence_weight", "f4", ("time", "south_north", "west_east"), **kwargs)
            uo.units = vo.units = wo.units = io.units = "m s-1"
            qo.units = "1"
            pv = oo.createVariable("ib_center_pressure", "f4", ("time",))
            pv[:] = pressure
            pv.units = "hPa"
            no = oo.createVariable("assimilated_station_count", "i2", ("time",))
            oo.title = "PhyTopo pressure-anchored and CMA-constrained typhoon wind analysis"
            oo.product_type = "observation-constrained historical analysis"
            oo.direct_product_reference = str(recon_path)
            oo.method = ("IBTrACS central-pressure event anchor followed by regularised "
                         "Gaussian-RBF interpolation of simultaneous CMA wind-speed residuals")
            oo.rbf_length_scale_km = 30.0
            oo.rbf_nugget = 0.005
            oo.observation_operator = "grid-consistent bilinear sampling"
            oo.direct_phytopo_event_peak_ms = direct_peak
            oo.pressure_anchor_event_peak_ms = global_target
            oo.pressure_anchor_ratio = event_ratio
            oo.event_intensity_ceiling_ms = global_target
            oo.validation_note = ("Independent skill is assessed with station-wise holdout; "
                                  "this archive is refitted using all available CMA stations")

            cma_row_count = 0
            for t, stamp in enumerate(seconds.astype(np.int64)):
                u = np.asarray(rr.variables["u10"][t], float)
                v = np.asarray(rr.variables["v10"][t], float)
                speed = np.hypot(u, v)
                anchored = pressure_anchor(speed, event_ratio)
                gt = cma_by_time.get(int(stamp), [])
                if gt:
                    slat = np.asarray([row[1] for row in gt], float)
                    slon = np.asarray([row[2] for row in gt], float)
                    obs = np.asarray([row[3] for row in gt], float)
                    sampled = regular_bilinear(anchored, lat1, lon1, slat, slon)
                    residual = obs - sampled
                    increment, influence = rbf_increment(lat, lon, slat, slon, residual)
                    cma_row_count += len(gt)
                else:
                    increment = np.zeros_like(anchored)
                    influence = np.zeros_like(anchored)
                # Central pressure constrains the event-scale intensity ceiling,
                # while CMA residuals redistribute wind locally below it.
                analysis = np.clip(anchored + increment, 0.0, global_target)
                scale = np.divide(analysis, speed, out=np.zeros_like(analysis), where=speed > 0.05)
                scale = np.clip(scale, 0.0, 4.0)
                uo[t] = (u * scale).astype("f4")
                vo[t] = (v * scale).astype("f4")
                wo[t] = analysis.astype("f4")
                io[t] = (analysis - speed).astype("f4")
                qo[t] = influence.astype("f4")
                no[t] = len(gt)
    print(out_path)
    print({"direct_peak": direct_peak, "pressure_target": global_target,
           "cma_rows": int(sum(len(v) for v in cma_by_time.values()))})


if __name__ == "__main__":
    main()
