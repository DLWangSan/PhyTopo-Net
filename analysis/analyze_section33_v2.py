import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PHYTOPO_DATA_ROOT", ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("PHYTOPO_OUTPUT_ROOT", ROOT / "outputs"))
OLD_MERGED = DATA_ROOT / "hko_hourly_merged.csv"
OLD_NC = DATA_ROOT / "2018_2018250N12170_phytopo_recon.nc"
V2_NC = DATA_ROOT / "2018_2018250N12170_phytopo_recon_v2.nc"
OUT = OUTPUT_ROOT / "section33_v2"


def nearest_indices(lat2d: np.ndarray, lon2d: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    # Match the original validation pipeline exactly.
    distance2 = (lat2d - lat) ** 2 + (lon2d - lon) ** 2
    flat = int(np.nanargmin(distance2))
    return tuple(int(v) for v in np.unravel_index(flat, distance2.shape))


def sample_nc(
    path: Path,
    stations: pd.DataFrame,
    times: pd.DatetimeIndex,
    half_width: int = 2,
) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        lat2d = ds["XLAT"].values
        lon2d = ds["XLONG"].values
        nc_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        time_lookup = {t: i for i, t in enumerate(nc_times)}
        result: dict[str, np.ndarray] = {}
        for row in stations.itertuples(index=False):
            iy, ix = nearest_indices(lat2d, lon2d, float(row.lat), float(row.lon))
            y0, y1 = max(0, iy - half_width), min(lat2d.shape[0], iy + half_width + 1)
            x0, x1 = max(0, ix - half_width), min(lat2d.shape[1], ix + half_width + 1)
            u_block = ds["u10"][:, y0:y1, x0:x1].values.astype(float)
            v_block = ds["v10"][:, y0:y1, x0:x1].values.astype(float)
            valid = np.isfinite(u_block) & np.isfinite(v_block)
            u_sum = np.where(valid, u_block, 0.0).sum(axis=(1, 2))
            v_sum = np.where(valid, v_block, 0.0).sum(axis=(1, 2))
            count = valid.sum(axis=(1, 2))
            u_mean = np.divide(u_sum, count, out=np.full_like(u_sum, np.nan), where=count > 0)
            v_mean = np.divide(v_sum, count, out=np.full_like(v_sum, np.nan), where=count > 0)
            ws = np.hypot(u_mean, v_mean)
            result[row.station] = np.array(
                [ws[time_lookup[t]] if t in time_lookup else np.nan for t in times], dtype=float
            )
    return result


def sample_nc_interpolated(
    path: Path,
    stations: pd.DataFrame,
    times: pd.DatetimeIndex,
    neighborhood: int = 3,
    interpolate_speed: bool = False,
) -> dict[str, np.ndarray]:
    """Local linear interpolation on the curvilinear target grid."""
    with xr.open_dataset(path) as ds:
        lat2d = ds["XLAT"].values
        lon2d = ds["XLONG"].values
        nc_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        time_lookup = {t: i for i, t in enumerate(nc_times)}
        result: dict[str, np.ndarray] = {}
        for row in stations.itertuples(index=False):
            iy, ix = nearest_indices(lat2d, lon2d, float(row.lat), float(row.lon))
            y0, y1 = max(0, iy - neighborhood), min(lat2d.shape[0], iy + neighborhood + 1)
            x0, x1 = max(0, ix - neighborhood), min(lat2d.shape[1], ix + neighborhood + 1)
            points = np.column_stack(
                [lat2d[y0:y1, x0:x1].reshape(-1), lon2d[y0:y1, x0:x1].reshape(-1)]
            )
            u_block = ds["u10"][:, y0:y1, x0:x1].values.astype(float).reshape(ds.sizes["time"], -1)
            v_block = ds["v10"][:, y0:y1, x0:x1].values.astype(float).reshape(ds.sizes["time"], -1)
            target = np.array([[float(row.lat), float(row.lon)]])
            ws = np.full(ds.sizes["time"], np.nan)
            for k in range(ds.sizes["time"]):
                valid = np.isfinite(u_block[k]) & np.isfinite(v_block[k]) & np.isfinite(points).all(axis=1)
                if valid.sum() < 3:
                    continue
                if interpolate_speed:
                    speed = np.hypot(u_block[k, valid], v_block[k, valid])
                    ws[k] = float(griddata(points[valid], speed, target, method="linear")[0])
                else:
                    ui = float(griddata(points[valid], u_block[k, valid], target, method="linear")[0])
                    vi = float(griddata(points[valid], v_block[k, valid], target, method="linear")[0])
                    ws[k] = np.hypot(ui, vi)
            result[row.station] = np.array(
                [ws[time_lookup[t]] if t in time_lookup else np.nan for t in times], dtype=float
            )
    return result


def sample_nc_local_max(
    path: Path,
    stations: pd.DataFrame,
    times: pd.DatetimeIndex,
    half_width: int,
) -> dict[str, np.ndarray]:
    with xr.open_dataset(path) as ds:
        lat2d = ds["XLAT"].values
        lon2d = ds["XLONG"].values
        nc_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        time_lookup = {t: i for i, t in enumerate(nc_times)}
        result: dict[str, np.ndarray] = {}
        for row in stations.itertuples(index=False):
            iy, ix = nearest_indices(lat2d, lon2d, float(row.lat), float(row.lon))
            y0, y1 = max(0, iy - half_width), min(lat2d.shape[0], iy + half_width + 1)
            x0, x1 = max(0, ix - half_width), min(lat2d.shape[1], ix + half_width + 1)
            u = ds["u10"][:, y0:y1, x0:x1].values.astype(float)
            v = ds["v10"][:, y0:y1, x0:x1].values.astype(float)
            ws = np.nanmax(np.hypot(u, v), axis=(1, 2))
            result[row.station] = np.array(
                [ws[time_lookup[t]] if t in time_lookup else np.nan for t in times], dtype=float
            )
    return result


def metrics(obs: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    valid = obs.notna() & pred.notna()
    error = pred[valid] - obs[valid]
    return {
        "n": int(valid.sum()),
        "bias": float(error.mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OLD_MERGED, parse_dates=["datetime"])
    stations = data[["station", "label", "lat", "lon"]].drop_duplicates("station")
    all_times = pd.DatetimeIndex(sorted(data["datetime"].unique()))
    time_pos = {t: i for i, t in enumerate(all_times)}
    diagnostics = []
    candidates = {}
    for half_width in range(0, 6):
        sampled = sample_nc(OLD_NC, stations, all_times, half_width=half_width)
        values = np.array(
            [sampled[s][time_pos[t]] for s, t in zip(data["station"], data["datetime"])]
        )
        difference = np.abs(data["recon_wspd"].to_numpy() - values)
        diagnostics.append((half_width, float(np.nanmax(difference)), float(np.nanmean(difference))))
        candidates[half_width] = values
    for label, interpolate_speed in [("linear_uv", False), ("linear_speed", True)]:
        sampled = sample_nc_interpolated(
            OLD_NC, stations, all_times, neighborhood=3, interpolate_speed=interpolate_speed
        )
        values = np.array(
            [sampled[s][time_pos[t]] for s, t in zip(data["station"], data["datetime"])]
        )
        difference = np.abs(data["recon_wspd"].to_numpy() - values)
        diagnostics.append((label, float(np.nanmax(difference)), float(np.nanmean(difference))))
        candidates[label] = values
    for half_width in range(1, 7):
        label = f"max_{half_width}"
        sampled = sample_nc_local_max(OLD_NC, stations, all_times, half_width=half_width)
        values = np.array(
            [sampled[s][time_pos[t]] for s, t in zip(data["station"], data["datetime"])]
        )
        difference = np.abs(data["recon_wspd"].to_numpy() - values)
        diagnostics.append((label, float(np.nanmax(difference)), float(np.nanmean(difference))))
        candidates[label] = values
    print("Stencil diagnostics (half-width, max abs diff, mean abs diff):", diagnostics)
    # Section 3.3 states that target-grid winds are sampled at the nearest grid
    # cell, so the auditable v2 update must use half-width 0 even though the
    # undocumented July series does not exactly reproduce any tested sampler.
    best_half = 0
    print("Selected method for v2: nearest target-grid cell (half-width 0)")
    data["old_recon_resampled"] = candidates[best_half]
    if isinstance(best_half, str):
        if best_half.startswith("max_"):
            v2_sample = sample_nc_local_max(
                V2_NC, stations, all_times, half_width=int(best_half.split("_")[1])
            )
        else:
            v2_sample = sample_nc_interpolated(
                V2_NC,
                stations,
                all_times,
                neighborhood=3,
                interpolate_speed=best_half == "linear_speed",
            )
    else:
        v2_sample = sample_nc(V2_NC, stations, all_times, half_width=best_half)
    data["v2_recon_wspd"] = [
        v2_sample[s][time_pos[t]] for s, t in zip(data["station"], data["datetime"])
    ]

    old_difference = np.nanmax(np.abs(data["recon_wspd"] - data["old_recon_resampled"]))
    print(f"Maximum old merged-vs-resampled difference: {old_difference:.9f} m/s")

    per_station_rows = []
    for station, group in data.groupby("station", sort=False):
        obs = group["obs_wspd"]
        row = {
            "station": station,
            "label": group["label"].iloc[0],
            "n": int(obs.notna().sum()),
            "peak_valid": station != "TME",
            "obs_peak": float(obs.max()),
            "era5_peak": float(group["era5_wspd"].max()),
            "old_peak": float(group["recon_wspd"].max()),
            "v2_peak": float(group["v2_recon_wspd"].max()),
        }
        for label, column in [
            ("era5", "era5_wspd"),
            ("old", "recon_wspd"),
            ("v2", "v2_recon_wspd"),
        ]:
            stat = metrics(obs, group[column])
            row[f"{label}_bias"] = stat["bias"]
            row[f"{label}_mae"] = stat["mae"]
            row[f"{label}_rmse"] = stat["rmse"]
            row[f"{label}_peak_abs_error"] = abs(row[f"{label}_peak"] - row["obs_peak"])
        per_station_rows.append(row)
    per_station = pd.DataFrame(per_station_rows)

    aggregate_rows = []
    subsets = [
        ("All matched station hours", data["obs_wspd"].notna()),
        ("Observed wind speed >= 25 m/s", data["obs_wspd"] >= 25.0),
        ("Observed wind speed >= 30 m/s", data["obs_wspd"] >= 30.0),
    ]
    for subset, mask in subsets:
        mask = mask & data[["era5_wspd", "recon_wspd", "v2_recon_wspd"]].notna().all(axis=1)
        row: dict[str, float | int | str] = {"subset": subset}
        for label, column in [
            ("ERA5", "era5_wspd"),
            ("old", "recon_wspd"),
            ("v2", "v2_recon_wspd"),
        ]:
            stat = metrics(data.loc[mask, "obs_wspd"], data.loc[mask, column])
            row["n"] = stat["n"]
            row[f"{label}_bias"] = stat["bias"]
            row[f"{label}_mae"] = stat["mae"]
            row[f"{label}_rmse"] = stat["rmse"]
        row["old_rmse_reduction_pct"] = 100 * (
            row["ERA5_rmse"] - row["old_rmse"]
        ) / row["ERA5_rmse"]
        row["v2_rmse_reduction_pct"] = 100 * (
            row["ERA5_rmse"] - row["v2_rmse"]
        ) / row["ERA5_rmse"]
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    data.to_csv(OUT / "hko_hourly_merged_v2.csv", index=False)
    per_station.to_csv(OUT / "hko_per_station_old_v2.csv", index=False)
    aggregate.to_csv(OUT / "hko_aggregate_old_v2.csv", index=False)
    print("\nPer-station comparison")
    print(per_station.round(3).to_string(index=False))
    print("\nAggregate comparison")
    print(aggregate.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
