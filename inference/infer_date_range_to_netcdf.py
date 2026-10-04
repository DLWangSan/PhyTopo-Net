#!/usr/bin/env python3
"""
按日期范围在指定 geo_em 网格上做 ERA5 + PhyTopo-Net 推理，输出 NetCDF（多时次 + 经纬裁剪）。
默认仅写出模型推理场：u10、v10、msl（不含 ERA5 对照或差值）。

日期：``--date-start`` / ``--date-end`` 若为纯日期 ``YYYY-MM-DD``，则 **end 含该日全天**（逐小时时含当日 00–23 点）；若带时刻则 end 为精确截止。

南海低纬 geo（如 lgj_geo_em.d03.nc）默认使用放宽的 LAT/LON 归一化；与训练域不一致，属外推。

示例:
  python inference/infer_date_range_to_netcdf.py \\
    --geo-file /path/to/geo_em.d03.nc \\
    --date-start 2025-07-10 --date-end 2025-07-31 \\
    --lat-min 8.0 --lat-max 11.25 --lon-min 114.25 --lon-max 119.75 \\
    --step-hours 6 \\
    --output-nc outputs/infer_lgj_uv_202507.nc
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predict_full_field as pf  # noqa: E402
import predict_full_field_blend_adv as pfa  # noqa: E402
from model import PhyTopoNet  # noqa: E402


def _strip_tz(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


_DATE_ONLY = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*$")


def parse_date_range_bounds(date_start: str, date_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """解析 --date-start / --date-end。

    - 若仅为 ``YYYY-MM-DD``（无时刻）：**start** 取该日 00:00；**end** 取该日 **自然日结束**（含 23:00 等整点），与 ERA5 日界一致。
    - 若含时刻（如 ``2020-03-31 18:00``）：按该瞬时作为起/止，不再扩展。
    """
    ds = date_start.strip()
    de = date_end.strip()

    t0 = _strip_tz(pd.to_datetime(ds, utc=True))
    if _DATE_ONLY.match(ds):
        t0 = t0.normalize()

    t1 = _strip_tz(pd.to_datetime(de, utc=True))
    if _DATE_ONLY.match(de):
        # 含该日历日全天：结束时刻取当日最后一秒的上一整点之前均可被 date_range 覆盖
        t1 = t1.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return t0, t1


def time_steps(t0: pd.Timestamp, t1: pd.Timestamp, step_h: float) -> list[pd.Timestamp]:
    """生成 [t0, t1] 内按 step_h 对齐的整点时刻（两端均包含，与 ERA5 小时层一致）。"""
    a = _strip_tz(pd.Timestamp(t0)).floor("h")
    b = _strip_tz(pd.Timestamp(t1))
    if b < a:
        a, b = b, a
    if step_h <= 0:
        step_h = 1.0
    idx = pd.date_range(
        start=a,
        end=b,
        freq=pd.Timedelta(hours=step_h),
        inclusive="both",
    )
    return [pd.Timestamp(x) for x in idx]


def bbox_crop_slices(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[slice, slice]:
    m = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not np.any(m):
        return slice(0, 0), slice(0, 0)
    ii, jj = np.where(m)
    return slice(int(ii.min()), int(ii.max()) + 1), slice(int(jj.min()), int(jj.max()) + 1)


def load_static_data_lowlat(geo_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """地形/坡度同原版；LAT/LON 用 0–25°N、100–130°E 线性归一化并 clip 到 [0,1]。"""
    from netCDF4 import Dataset
    from scipy.ndimage import gaussian_filter, sobel

    nc = Dataset(geo_file)
    hgt = np.array(nc.variables["HGT_M"][0, :, :])
    landmask = np.array(nc.variables["LANDMASK"][0, :, :])
    lat = np.array(nc.variables["XLAT_M"][0, :, :])
    lon = np.array(nc.variables["XLONG_M"][0, :, :])
    nc.close()

    hgt_smooth = gaussian_filter(hgt, sigma=1.0)
    grad_y = sobel(hgt_smooth, axis=0)
    grad_x = sobel(hgt_smooth, axis=1)
    dx, dy = 3000.0, 3000.0
    slope = np.arctan(np.sqrt((grad_x / dx) ** 2 + (grad_y / dy) ** 2)) * 180.0 / np.pi

    hgt_norm = pf.normalize(hgt, "HGT")
    slope_norm = pf.normalize(slope, "SLOPE")
    lat_norm = np.clip((lat - 0.0) / 25.0, 0.0, 1.0).astype(np.float32)
    lon_norm = np.clip((lon - 100.0) / 30.0, 0.0, 1.0).astype(np.float32)
    landmask_norm = landmask.astype(np.float32)

    static_tensor = np.stack([hgt_norm, slope_norm, lat_norm, lon_norm, landmask_norm], axis=-1)
    return static_tensor, lat, lon


def main() -> int:
    ap = argparse.ArgumentParser(description="日期范围推理 → NetCDF")
    ap.add_argument("--geo-file", type=str, default=os.path.join(_ROOT, "assets", "geo_em.d03.nc"))
    ap.add_argument(
        "--era5-dir",
        type=str,
        default=os.environ.get("ERA5_DIR", os.path.join(_ROOT, "..", "era5_download", "era5_nn_nc_2020_2025")),
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(_ROOT, "models", "best_model.pth"),
    )
    ap.add_argument("--date-start", type=str, required=True)
    ap.add_argument("--date-end", type=str, required=True)
    ap.add_argument("--step-hours", type=float, default=6.0)
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--use-training-latlon-norm", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--blend-window", choices=["hann", "cosine"], default="hann")
    ap.add_argument("--no-dual-grid", action="store_true")
    ap.add_argument("--no-triple-grid", action="store_true")
    ap.add_argument("--norm-uv-smooth-sigma", type=float, default=0.35)
    ap.add_argument("--output-nc", type=str, required=True)
    ap.add_argument("--max-times", type=int, default=0)
    ap.add_argument(
        "--no-msl",
        action="store_true",
        help="不写出海平面气压 msl（默认写出，单位 hPa）",
    )
    ap.add_argument(
        "--include-era5-in-output",
        action="store_true",
        help="在 NetCDF 中额外写出 ERA5 插值到网格的 u10/v10/msl（默认不写，仅推理场）",
    )
    args = ap.parse_args()
    export_msl = not args.no_msl
    include_era5_out = args.include_era5_in_output

    geo_file = os.path.abspath(args.geo_file)
    era5_dir = os.path.abspath(args.era5_dir)
    model_path = os.path.abspath(args.model_path)
    out_nc = os.path.abspath(args.output_nc)

    if not os.path.isfile(geo_file):
        print(f"找不到 geo: {geo_file}", file=sys.stderr)
        return 1
    if not os.path.isfile(model_path):
        print(f"找不到模型: {model_path}", file=sys.stderr)
        return 1

    if args.use_training_latlon_norm:
        static_tensor, wrf_lat, wrf_lon = pf.load_static_data(geo_file)
    else:
        static_tensor, wrf_lat, wrf_lon = load_static_data_lowlat(geo_file)

    si, sj = bbox_crop_slices(
        wrf_lat, wrf_lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    if si.stop <= si.start or sj.stop <= sj.start:
        print(
            "错误: 裁剪框与网格无交集。\n"
            f"  纬度 [{float(np.nanmin(wrf_lat)):.4f}, {float(np.nanmax(wrf_lat)):.4f}]\n"
            f"  经度 [{float(np.nanmin(wrf_lon)):.4f}, {float(np.nanmax(wrf_lon)):.4f}]",
            file=sys.stderr,
        )
        return 2

    t0, t1 = parse_date_range_bounds(args.date_start, args.date_end)
    times = time_steps(t0, t1, args.step_hours)
    if args.max_times > 0:
        times = times[: args.max_times]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    use_residual = checkpoint.get("residual", False) if isinstance(checkpoint, dict) else False
    model = PhyTopoNet(residual=use_residual)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    preds_u: list[np.ndarray] = []
    preds_v: list[np.ndarray] = []
    preds_slp: list[np.ndarray] = []
    era5_u: list[np.ndarray] = []
    era5_v: list[np.ndarray] = []
    era5_slp: list[np.ndarray] = []
    ok_times: list[pd.Timestamp] = []

    for tt in times:
        era5_file = pf.find_era5_file_for_time(era5_dir, tt)
        if era5_file is None:
            print(f"  跳过(无 ERA5): {tt}")
            continue
        result = pf.load_era5_for_time(era5_file, tt, wrf_lat, wrf_lon)
        era5_vars = result[0] if isinstance(result, tuple) else result
        if not era5_vars or "U10" not in era5_vars or "V10" not in era5_vars:
            print(f"  跳过(ERA5 空): {tt}")
            continue
        if export_msl and "SLP" not in era5_vars:
            print(f"  跳过(无 ERA5 SLP): {tt}")
            continue

        pred_norm = pfa.predict_full_field_advanced(
            model,
            era5_vars,
            static_tensor,
            device,
            patch_size=args.patch_size,
            stride=args.stride,
            blend=args.blend_window,
            dual_grid=not args.no_dual_grid,
            triple_grid=not args.no_triple_grid,
            norm_uv_smooth_sigma=args.norm_uv_smooth_sigma,
        )
        era5_order = ["U10", "V10", "SLP", "T2"]
        pred_phys = np.zeros((pred_norm.shape[0], pred_norm.shape[1], 4), dtype=np.float32)
        for i, var in enumerate(era5_order):
            pred_phys[:, :, i] = pf.denormalize(pred_norm[:, :, i], var)

        pu = pred_phys[:, :, 0][si, sj]
        pv = pred_phys[:, :, 1][si, sj]
        eu = np.asarray(era5_vars["U10"], dtype=np.float32)[si, sj]
        ev = np.asarray(era5_vars["V10"], dtype=np.float32)[si, sj]

        preds_u.append(pu)
        preds_v.append(pv)
        if include_era5_out:
            era5_u.append(eu)
            era5_v.append(ev)
        if export_msl:
            ps = pred_phys[:, :, 2][si, sj]
            preds_slp.append(ps)
            if include_era5_out:
                es = np.asarray(era5_vars["SLP"], dtype=np.float32)[si, sj]
                era5_slp.append(es)
        ok_times.append(tt)
        print(f"  ✓ {tt}  crop {pu.shape}")

    if not ok_times:
        print("错误: 无成功时刻", file=sys.stderr)
        return 3

    lat_c = np.asarray(wrf_lat[si, sj], dtype=np.float32)
    lon_c = np.asarray(wrf_lon[si, sj], dtype=np.float32)
    Pu = np.stack(preds_u, axis=0)
    Pv = np.stack(preds_v, axis=0)

    tcoord = pd.to_datetime(ok_times)
    # 仅推理场：u10, v10, msl（与 ERA5 差值或对照默认不写）
    data_vars: dict = {
        "u10": (
            ["time", "south_north", "west_east"],
            Pu,
            {"long_name": "10m u-wind (model)", "units": "m s-1"},
        ),
        "v10": (
            ["time", "south_north", "west_east"],
            Pv,
            {"long_name": "10m v-wind (model)", "units": "m s-1"},
        ),
    }
    if export_msl:
        Ps = np.stack(preds_slp, axis=0)
        data_vars["msl"] = (
            ["time", "south_north", "west_east"],
            Ps,
            {"long_name": "Mean sea level pressure (model)", "units": "hPa"},
        )
    if include_era5_out:
        Eu = np.stack(era5_u, axis=0)
        Ev = np.stack(era5_v, axis=0)
        data_vars["era5_u10"] = (["time", "south_north", "west_east"], Eu)
        data_vars["era5_v10"] = (["time", "south_north", "west_east"], Ev)
        if export_msl:
            Es = np.stack(era5_slp, axis=0)
            data_vars["era5_msl"] = (
                ["time", "south_north", "west_east"],
                Es,
                {"long_name": "Sea level pressure (ERA5 on WRF grid)", "units": "hPa"},
            )

    ds = xr.Dataset(
        data_vars=data_vars,  # type: ignore[arg-type]
        coords={
            "time": ("time", tcoord),
            "XLAT": (["south_north", "west_east"], lat_c),
            "XLONG": (["south_north", "west_east"], lon_c),
        },
        attrs={
            "title": "PhyTopo-Net inference (cropped)",
            "geo_file": geo_file,
            "model_path": model_path,
            "checkpoint_residual": str(use_residual),
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "static_latlon_norm": "training" if args.use_training_latlon_norm else "wide_lowlat",
            "history": json.dumps(
                {
                    "date_start": args.date_start,
                    "date_end": args.date_end,
                    "step_hours": args.step_hours,
                    "patch_size": args.patch_size,
                    "stride": args.stride,
                    "n_time_ok": len(ok_times),
                    "export_msl_hPa": export_msl,
                    "include_era5_in_output": include_era5_out,
                }
            ),
        },
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
    ds.to_netcdf(out_nc, encoding=encoding)
    vars_note = "u10,v10" + (",msl" if export_msl else "")
    print(f"✓ NetCDF → {out_nc}  vars=({vars_note}) time={len(ok_times)} ny={Pu.shape[1]} nx={Pu.shape[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
