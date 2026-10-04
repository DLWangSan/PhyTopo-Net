#!/usr/bin/env python3
"""
将 IBTrACS prd130 目录下每场台风的 ``*_ERA5_sfc.nc``（多时次 valid_time）逐时做 PhyTopo-Net 推理，
写出 NetCDF：仅 u10, v10, msl（hPa），不含 ERA5 对照或 LANDMASK。

用法（仓库根）:
  python visualization/batch_prd130_era5_to_recon_nc.py \\
    --era5-root /path/to/cee_ERA5_IBTrACS_prd130 \\
    --out-dir /path/to/recon_out \\
    --geo-file wrf/data/geo_em.d03.nc \\
    --model-path experiments/phy_topo_stage3_semisup/best_model.pth

单时刻 ERA5 vs 重构风场图：``visualization/plot_era5_vs_recon_wind_snapshot.py``（``--cursor-dataset-root`` + ``--storm-id``）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "visualization"))

import predict_full_field as pf  # noqa: E402
from model import PhyTopoNet  # noqa: E402


def process_one_file(
    era5_path: str,
    geo_file: str,
    model: torch.nn.Module,
    device: str,
    out_nc: str,
    patch_size: int,
    stride: int,
    min_file_bytes: int,
    overwrite: bool,
    *,
    predictor: str = "legacy",
    blend_window: str = "hann",
    dual_grid: bool = True,
    triple_grid: bool = True,
    norm_uv_smooth_sigma: float = 0.35,
) -> str:
    if os.path.isfile(out_nc) and not overwrite:
        return "skipped_exists"
    if not os.path.isfile(era5_path):
        return "missing_input"
    if os.path.getsize(era5_path) < min_file_bytes:
        return "input_too_small"

    static_tensor, wrf_lat, wrf_lon = pf.load_static_data(geo_file)
    ds_probe = xr.open_dataset(era5_path)
    if "valid_time" in ds_probe.coords and "time" not in ds_probe.coords:
        ds_probe = ds_probe.rename({"valid_time": "time"})
    if "time" not in ds_probe.coords:
        ds_probe.close()
        return "no_time_coord"
    times = [pd.Timestamp(t).round("h") for t in pd.to_datetime(ds_probe.time.values)]
    ds_probe.close()

    preds_u: list[np.ndarray] = []
    preds_v: list[np.ndarray] = []
    preds_slp: list[np.ndarray] = []
    ok_times: list[pd.Timestamp] = []

    for tt in times:
        result = pf.load_era5_for_time(era5_path, tt, wrf_lat, wrf_lon, return_raw=False)
        if result is None:
            continue
        era5_vars = result[0] if isinstance(result, tuple) else result
        need = ("U10", "V10", "SLP", "T2", "SST", "BLH", "FSR")
        if not era5_vars or not all(k in era5_vars for k in need):
            continue
        if predictor == "blend_adv":
            import predict_full_field_blend_adv as pfa

            pred_norm = pfa.predict_full_field_advanced(
                model,
                era5_vars,
                static_tensor,
                device,
                patch_size=patch_size,
                stride=stride,
                blend=blend_window,
                dual_grid=dual_grid,
                triple_grid=triple_grid,
                norm_uv_smooth_sigma=float(norm_uv_smooth_sigma),
            )
        else:
            pred_norm = pf.predict_full_field(
                model, era5_vars, static_tensor, device, patch_size=patch_size, stride=stride
            )
        era5_order = ["U10", "V10", "SLP", "T2"]
        pred_phys = np.zeros((pred_norm.shape[0], pred_norm.shape[1], 4), dtype=np.float32)
        for i, var in enumerate(era5_order):
            pred_phys[:, :, i] = pf.denormalize(pred_norm[:, :, i], var)
        preds_u.append(pred_phys[:, :, 0].astype(np.float32))
        preds_v.append(pred_phys[:, :, 1].astype(np.float32))
        preds_slp.append(pred_phys[:, :, 2].astype(np.float32))
        ok_times.append(tt)

    if not ok_times:
        return "no_valid_timesteps"

    lat_c = np.asarray(wrf_lat, dtype=np.float32)
    lon_c = np.asarray(wrf_lon, dtype=np.float32)
    Pu = np.stack(preds_u, axis=0)
    Pv = np.stack(preds_v, axis=0)
    Ps = np.stack(preds_slp, axis=0)
    tcoord = pd.to_datetime(ok_times)

    data_vars: dict = {
        "u10": (
            ["time", "south_north", "west_east"],
            Pu,
            {"long_name": "10m u-wind (PhyTopo-Net)", "units": "m s-1"},
        ),
        "v10": (
            ["time", "south_north", "west_east"],
            Pv,
            {"long_name": "10m v-wind (PhyTopo-Net)", "units": "m s-1"},
        ),
        "msl": (
            ["time", "south_north", "west_east"],
            Ps,
            {"long_name": "Mean sea level pressure (PhyTopo-Net)", "units": "hPa"},
        ),
    }
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", tcoord),
            "XLAT": (["south_north", "west_east"], lat_c),
            "XLONG": (["south_north", "west_east"], lon_c),
        },
        attrs={
            "title": "PhyTopo-Net reconstruction (prd130 ERA5_sfc driver)",
            "source_era5_nc": era5_path,
            "geo_file": geo_file,
            "n_time_written": len(ok_times),
            "n_time_in_file": len(times),
            "static_latlon_norm": "training",
            "history": json.dumps(
                {
                    "predictor": predictor,
                    "blend_window": blend_window if predictor == "blend_adv" else None,
                    "dual_grid": dual_grid if predictor == "blend_adv" else None,
                    "triple_grid": triple_grid if predictor == "blend_adv" else None,
                    "norm_uv_smooth_sigma": float(norm_uv_smooth_sigma)
                    if predictor == "blend_adv"
                    else None,
                    "patch_size": patch_size,
                    "stride": stride,
                }
            ),
        },
    )
    os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_nc, encoding=enc)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="prd130 ERA5_sfc → PhyTopo-Net 多时次 recon NetCDF")
    ap.add_argument("--era5-root", type=str, required=True, help="含 *_ERA5_sfc.nc 的目录")
    ap.add_argument("--out-dir", type=str, required=True, help="输出目录")
    ap.add_argument("--geo-file", type=str, default=os.path.join(_ROOT, "wrf", "data", "geo_em.d03.nc"))
    ap.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(_ROOT, "experiments", "phy_topo_stage3_semisup", "best_model.pth"),
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument(
        "--predictor",
        choices=["legacy", "blend_adv"],
        default="legacy",
        help="legacy：原 patch 拼接；blend_adv：与 yhk/infer_date_range Hann 融合一致，减轻接缝网格纹",
    )
    ap.add_argument("--blend-window", choices=["hann", "cosine"], default="hann")
    ap.add_argument(
        "--no-dual-grid",
        action="store_true",
        help="blend_adv：仅单套 patch 起点（更平滑背景、略损细节）",
    )
    ap.add_argument(
        "--no-triple-grid",
        action="store_true",
        help="blend_adv：禁用第三套错位网格",
    )
    ap.add_argument(
        "--norm-uv-smooth-sigma",
        type=float,
        default=0.35,
        help="blend_adv：归一化 U/V 高斯平滑 sigma；0 关闭",
    )
    ap.add_argument(
        "--only-filename",
        type=str,
        default="",
        help="仅处理 era5-root 下该文件名（例如 2023_2023234N18128_ERA5_sfc.nc）",
    )
    ap.add_argument("--min-file-mb", type=float, default=5.0, help="跳过小于该大小的输入（失败占位文件）")
    ap.add_argument("--max-files", type=int, default=0, help="0=不限制（调试用）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    era5_root = os.path.abspath(args.era5_root)
    out_dir = os.path.abspath(args.out_dir)
    geo_file = os.path.abspath(args.geo_file)
    model_path = os.path.abspath(args.model_path)
    min_bytes = int(args.min_file_mb * 1024 * 1024)

    if not os.path.isdir(era5_root):
        print(f"错误: 无目录 {era5_root}", file=sys.stderr)
        return 1
    if not os.path.isfile(geo_file):
        print(f"错误: 无 geo {geo_file}", file=sys.stderr)
        return 1
    if not os.path.isfile(model_path):
        print(f"错误: 无模型 {model_path}", file=sys.stderr)
        return 1

    files = sorted(
        f
        for f in os.listdir(era5_root)
        if f.endswith("_ERA5_sfc.nc") and os.path.isfile(os.path.join(era5_root, f))
    )
    only_fn = (args.only_filename or "").strip()
    if only_fn:
        files = [f for f in files if f == only_fn]
        if not files:
            print(f"错误: --only-filename={only_fn!r} 在目录中未找到", file=sys.stderr)
            return 1
    if args.max_files > 0:
        files = files[: args.max_files]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU", file=sys.stderr)
        device = "cpu"

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    use_residual = ckpt.get("residual", False) if isinstance(ckpt, dict) else False
    model = PhyTopoNet(residual=use_residual)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    dual_grid = not args.no_dual_grid
    triple_grid = not args.no_triple_grid

    print(f"✓ 模型 {model_path} residual={use_residual} device={device}")
    print(
        f"✓ predictor={args.predictor} stride={args.stride} dual_grid={dual_grid} triple_grid={triple_grid}"
    )
    print(f"✓ 待处理 {len(files)} 个 *_ERA5_sfc.nc → {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    manifest: list[dict] = []
    for fn in files:
        era5_path = os.path.join(era5_root, fn)
        out_nc = os.path.join(out_dir, fn.replace("_ERA5_sfc.nc", "_phytopo_recon.nc"))
        st = process_one_file(
            era5_path,
            geo_file,
            model,
            device,
            out_nc,
            args.patch_size,
            args.stride,
            min_bytes,
            args.overwrite,
            predictor=args.predictor,
            blend_window=args.blend_window,
            dual_grid=dual_grid,
            triple_grid=triple_grid,
            norm_uv_smooth_sigma=args.norm_uv_smooth_sigma,
        )
        print(f"  [{st}] {fn}")
        manifest.append({"file": fn, "status": st, "out": out_nc if st == "ok" else ""})

    man_path = os.path.join(out_dir, "batch_recon_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✓ 清单 → {man_path}")
    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"✓ 本 run 新写出 ok={n_ok} / {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
