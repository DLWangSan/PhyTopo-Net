#!/usr/bin/env python3
"""Extract actual domain maxima from the completed CMA-analysis archive."""

import argparse
import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(Path(args.archive).glob("*_phytopo_cma_analysis.nc")):
        event_id = path.name.split("_phytopo_cma_analysis.nc")[0]
        with Dataset(path) as ds:
            var = ds.variables["wspd10_analysis"]
            maximum = -np.inf
            for t in range(var.shape[0]):
                maximum = max(maximum, float(np.nanmax(np.asarray(var[t], float))))
            rows.append({
                "event_id": event_id,
                "analysis_event_max_ms": maximum,
                "direct_event_max_ms": float(getattr(ds, "direct_phytopo_event_peak_ms")),
                "pressure_anchor_event_peak_ms": float(getattr(ds, "pressure_anchor_event_peak_ms")),
                "assimilated_station_hours": int(np.sum(ds.variables["assimilated_station_count"][:])),
            })
    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(args.output, len(rows))


if __name__ == "__main__":
    main()
