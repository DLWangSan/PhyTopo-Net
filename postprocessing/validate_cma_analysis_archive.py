#!/usr/bin/env python3
"""Validate the completed CMA-constrained NetCDF archive without pandas."""

import argparse
import calendar
import csv
import gzip
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


def datetime_seconds(values, var):
    dates = num2date(values, var.units, calendar=getattr(var, "calendar", "standard"),
                    only_use_cftime_datetimes=True)
    return np.array([calendar.timegm((x.year, x.month, x.day, x.hour, x.minute,
                                      int(x.second))) for x in dates], np.int64)


def regular_bilinear(field, lat, lon, target_lat, target_lon):
    iy = np.searchsorted(lat, target_lat, side="right") - 1
    ix = np.searchsorted(lon, target_lon, side="right") - 1
    iy = np.clip(iy, 0, len(lat) - 2)
    ix = np.clip(ix, 0, len(lon) - 2)
    y0, y1 = lat[iy], lat[iy + 1]
    x0, x1 = lon[ix], lon[ix + 1]
    wy = 0.0 if y1 == y0 else (target_lat - y0) / (y1 - y0)
    wx = 0.0 if x1 == x0 else (target_lon - x0) / (x1 - x0)
    return ((1 - wy) * ((1 - wx) * field[iy, ix] + wx * field[iy, ix + 1])
            + wy * ((1 - wx) * field[iy + 1, ix] + wx * field[iy + 1, ix + 1]))


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), bias=float(np.mean(p-y)), mae=float(np.mean(np.abs(p-y))),
                rmse=float(np.sqrt(np.mean((p-y)**2))), r=float(np.corrcoef(y,p)[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--cma-hourly", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.archive)
    files = sorted(root.glob("*_phytopo_cma_analysis.nc"))
    required = {"time", "XLAT", "XLONG", "u10_analysis", "v10_analysis",
                "wspd10_analysis", "analysis_increment", "cma_influence_weight",
                "assimilated_station_count"}
    structural = []
    for path in files:
        with Dataset(path) as ds:
            missing = sorted(required - set(ds.variables))
            w = ds.variables["wspd10_analysis"]
            sample = np.asarray(w[::max(1, w.shape[0] // 6)], float)
            q = np.asarray(ds.variables["cma_influence_weight"][::max(1, w.shape[0] // 6)], float)
            structural.append((path.name, w.shape[0], missing, float(np.nanmin(sample)),
                               float(np.nanmax(sample)), float(np.nanmin(q)), float(np.nanmax(q))))

    by_event = {}
    with gzip.open(args.cma_hourly, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_event.setdefault(row["event_id"], []).append(row)
    predictions = []
    for path in files:
        event = path.name.split("_phytopo_cma_analysis.nc")[0]
        rows = by_event.get(event, [])
        if not rows:
            continue
        with Dataset(path) as ds:
            lat2 = np.asarray(ds.variables["XLAT"][:], float)
            lon2 = np.asarray(ds.variables["XLONG"][:], float)
            lat, lon = lat2[:, 0], lon2[0, :]
            tv = ds.variables["time"]
            seconds = datetime_seconds(tv[:], tv)
            index = {int(v): i for i, v in enumerate(seconds)}
            cache = {}
            for row in rows:
                from datetime import datetime
                stamp = datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00"))
                sec = calendar.timegm(stamp.utctimetuple())
                if sec not in index:
                    continue
                t = index[sec]
                if t not in cache:
                    cache[t] = np.asarray(ds.variables["wspd10_analysis"][t], float)
                value = regular_bilinear(cache[t], lat, lon, float(row["lat"]), float(row["lon"]))
                predictions.append({
                    "event_id": event, "station_id": row["station_id"], "time_utc": row["time_utc"],
                    "obs": float(row["obs_speed"]), "phy": float(row["v2_speed"]),
                    "analysis": float(value),
                })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "station_id", "time_utc", "obs", "phy", "analysis"])
        writer.writeheader(); writer.writerows(predictions)

    y = np.array([x["obs"] for x in predictions])
    phy = np.array([x["phy"] for x in predictions])
    ana = np.array([x["analysis"] for x in predictions])
    peaks = {}
    for x in predictions:
        key = (x["event_id"], x["station_id"])
        current = peaks.setdefault(key, {"obs": -np.inf, "phy": -np.inf, "analysis": -np.inf})
        for name in current:
            current[name] = max(current[name], x[name])
    py = np.array([x["obs"] for x in peaks.values()])
    pp = np.array([x["phy"] for x in peaks.values()])
    pa = np.array([x["analysis"] for x in peaks.values()])
    print("files", len(files), "structural_failures", sum(bool(x[2]) for x in structural))
    print("sample_wind_range", min(x[3] for x in structural), max(x[4] for x in structural))
    print("sample_influence_range", min(x[5] for x in structural), max(x[6] for x in structural))
    print("hourly_phy", score(y, phy))
    print("hourly_analysis", score(y, ana))
    print("peak_phy", score(py, pp))
    print("peak_analysis", score(py, pa))
    mask = py >= 25
    print("peak25_phy", score(py[mask], pp[mask]))
    print("peak25_analysis", score(py[mask], pa[mask]))
    print("predictions", out)


if __name__ == "__main__":
    main()
