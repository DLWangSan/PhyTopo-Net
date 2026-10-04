#!/usr/bin/env python3
"""
改进版全场推理（不改训练、不改 predict_full_field.py）。

思路组合：
1. **可分离 Hann 窗**：相对原径向余弦窗，边角权重更合理，重叠带过渡更平滑。
2. **双错位网格（dual_grid）**：第二套 patch 起点偏移 (stride/2, stride/2)，两套独立加权融合后再对有效格点取平均，
   削弱单一网格相位导致的周期条纹（类似 staggered sampling）。
3. **三错位网格（triple_grid，默认开）**：在双网格基础上增加起点 ``(stride/4, stride/4)``，三套独立融合后对有效预测取平均，进一步压低周期纹。
4. **归一化空间 UV 轻平滑**：融合后对通道 U/V 做极小 ``gaussian_filter``（默认 sigma≈0.35），专抑风速 mosaic，SLP/T2 不动。
5. **更小 stride**：推荐 16～24；PBS 默认已向 16 靠拢。

用法：与 ``predict_full_field.predict_full_field`` 相同签名外增加关键字参数；或在 ``infer_wrf_timeline_to_netcdf``
里把 ``pf.predict_full_field`` 换成 ``predict_full_field_blend_adv.predict_full_field_advanced``。

示例::

    from inference import predict_full_field_blend_adv as pfa
    pred_norm = pfa.predict_full_field_advanced(
        model, era5_vars, static_tensor, device,
        patch_size=64, stride=16, blend=\"hann\", triple_grid=True, norm_uv_smooth_sigma=0.35,
    )
"""
from __future__ import annotations

import os
import sys
from typing import Literal

_ROOT_BFA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT_BFA, _VIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import predict_full_field as _pf

normalize = _pf.normalize
denormalize = _pf.denormalize


BlendName = Literal["hann", "cosine"]


def _blend_window_cosine(patch_size: int) -> np.ndarray:
    """与原 predict_full_field._create_blend_window 一致的径向余弦窗。"""
    cy, cx = patch_size // 2, patch_size // 2
    y = np.arange(patch_size, dtype=np.float32)
    x = np.arange(patch_size, dtype=np.float32)
    r = np.sqrt((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2)
    max_r = np.sqrt(cx**2 + cy**2) + 1e-8
    r = np.clip(r / max_r, 0, 1)
    return (0.5 * (1 + np.cos(np.pi * r))).astype(np.float32)


def _blend_window_hann(patch_size: int) -> np.ndarray:
    """可分离 Hann，边缘为 0，中心平滑。"""
    wy = np.hanning(patch_size).astype(np.float32)
    wx = np.hanning(patch_size).astype(np.float32)
    w = wy[:, np.newaxis] * wx[np.newaxis, :]
    return np.maximum(w, 1e-8).astype(np.float32)


def _pick_blend(name: BlendName, patch_size: int) -> np.ndarray:
    if name == "hann":
        return _blend_window_hann(patch_size)
    if name == "cosine":
        return _blend_window_cosine(patch_size)
    raise ValueError(f"未知 blend={name!r}，使用 'hann' 或 'cosine'")


def _build_full_input(era5_vars: dict, static_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 predict_full_field.predict_full_field 前半段一致。"""
    H, W = static_tensor.shape[:2]
    era5_order = ["U10", "V10", "SLP", "T2", "SST", "BLH", "FSR"]
    era5_list = []
    for var in era5_order:
        if var in era5_vars:
            era5_list.append(normalize(era5_vars[var], var))
        else:
            era5_list.append(np.zeros((H, W)))
    era5_stack = np.stack(era5_list, axis=-1)
    full_input = np.concatenate([era5_stack, static_tensor], axis=-1)

    landmask = full_input[:, :, 11]
    full_input[:, :, 4][landmask > 0.5] = 0.0

    input_invalid_mask = np.zeros((H, W), dtype=bool)
    for c in range(7):
        input_invalid_mask |= np.isnan(full_input[:, :, c])
    full_input_filled = np.nan_to_num(full_input, nan=0.0)
    return full_input_filled, input_invalid_mask


def _accumulate_one_offset_grid(
    model: torch.nn.Module,
    full_input_filled: np.ndarray,
    input_invalid_mask: np.ndarray,
    device: torch.device | str,
    patch_size: int,
    stride: int,
    offset_y: int,
    offset_x: int,
    blend_window: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    单套网格滑窗推理并加权融合。
    返回 (prediction_full, weight_sum, final_invalid_mask)，均为归一化空间。
    """
    H, W = full_input_filled.shape[:2]
    patches: list[np.ndarray] = []
    patch_positions: list[tuple[int, int]] = []
    patch_invalid_masks: list[np.ndarray] = []

    for y in range(offset_y, H - patch_size + 1, stride):
        for x in range(offset_x, W - patch_size + 1, stride):
            patch = full_input_filled[y : y + patch_size, x : x + patch_size, :]
            patch_invalid = input_invalid_mask[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            patch_positions.append((y, x))
            patch_invalid_masks.append(patch_invalid)

    if not patches:
        n_out = 4
        empty = np.full((H, W, n_out), np.nan, dtype=np.float32)
        return empty, np.zeros((H, W), dtype=np.float32), input_invalid_mask.copy()

    patches_tensor = torch.from_numpy(np.stack(patches)).float().permute(0, 3, 1, 2)
    x_met = patches_tensor[:, :7, :, :]
    x_geo = patches_tensor[:, 7:, :, :]

    model.eval()
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc=f"pred patches oy={offset_y} ox={offset_x}"):
            batch_x_met = x_met[i : i + batch_size].to(device)
            batch_x_geo = x_geo[i : i + batch_size].to(device)
            batch_pred = model(batch_x_met, batch_x_geo)
            batch_pred = torch.clamp(batch_pred, 0.0, 1.0)
            all_predictions.append(batch_pred.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    n_out = predictions.shape[1]
    prediction_full = np.zeros((H, W, n_out), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    final_invalid_mask = np.zeros((H, W), dtype=bool)

    for i, (y, x) in enumerate(patch_positions):
        pred_patch = predictions[i].transpose(1, 2, 0)
        patch_invalid = patch_invalid_masks[i]
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        h_slice, w_slice = y_end - y, x_end - x

        patch_valid = ~patch_invalid[:h_slice, :w_slice]
        w = blend_window[:h_slice, :w_slice].copy()
        w[~patch_valid] = 0.0

        for c in range(n_out):
            pred_slice = pred_patch[:h_slice, :w_slice, c]
            pred_slice = np.where(patch_valid, pred_slice, 0.0)
            prediction_full[y:y_end, x:x_end, c] += w * pred_slice
        weight_sum[y:y_end, x:x_end] += w
        final_invalid_mask[y:y_end, x:x_end] |= patch_invalid[:h_slice, :w_slice]

    mask_cov = weight_sum > 1e-8
    safe_weight = np.where(mask_cov, weight_sum, 1.0)
    for c in range(n_out):
        prediction_full[:, :, c] = np.where(
            mask_cov, prediction_full[:, :, c] / safe_weight, np.nan
        )

    prediction_full[final_invalid_mask] = np.nan
    prediction_full[~mask_cov] = np.nan
    return prediction_full, weight_sum, final_invalid_mask


def _merge_multi_normalized(
    stacks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    多套错位网格：各格点在「该套预测有限且已覆盖」上对通道取算术平均；
    最后凡 ``inv`` 并集处置 NaN（与原先 dual 的 conservative OR 一致）。
    """
    if not stacks:
        raise ValueError("stacks 为空")
    if len(stacks) == 1:
        pred, _w, inv = stacks[0]
        out = pred.copy()
        out[inv] = np.nan
        return out.astype(np.float32)

    H, W, C = stacks[0][0].shape
    acc = np.zeros((H, W, C), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)
    inv_union = np.zeros((H, W), dtype=bool)

    for pred, wsum, inv in stacks:
        inv_union |= inv
        valid = (wsum > 1e-8) & np.all(np.isfinite(pred), axis=-1)
        vf = valid.astype(np.float64)
        for c in range(C):
            acc[:, :, c] += np.where(valid, pred[:, :, c].astype(np.float64), 0.0)
        cnt += vf

    out = np.full((H, W, C), np.nan, dtype=np.float32)
    ok = cnt > 1e-8
    cnt_safe = np.maximum(cnt, 1e-8)
    for c in range(C):
        out[:, :, c] = np.where(ok, (acc[:, :, c] / cnt_safe).astype(np.float32), np.nan)
    out[inv_union] = np.nan
    return out


def _smooth_norm_uv_channels(pred: np.ndarray, sigma_uv: float) -> np.ndarray:
    """仅对归一化后的 U/V（通道 0,1）做掩膜内高斯平滑，并 clamp 回 [0,1]。"""
    if sigma_uv <= 0:
        return pred
    out = pred.copy()
    for c in (0, 1):
        a = np.asarray(pred[:, :, c], dtype=np.float64)
        m = np.isfinite(a)
        if not np.any(m):
            continue
        fill = float(np.nanmean(a[m]))
        tmp = np.where(m, a, fill)
        sm = gaussian_filter(tmp, sigma=float(sigma_uv))
        sm = np.clip(sm, 0.0, 1.0)
        out[:, :, c] = np.where(m, sm, np.nan).astype(np.float32)
    return out


def predict_full_field_advanced(
    model,
    era5_vars: dict,
    static_tensor: np.ndarray,
    device: torch.device | str,
    patch_size: int = 64,
    stride: int = 24,
    blend: BlendName = "hann",
    dual_grid: bool = True,
    triple_grid: bool = True,
    norm_uv_smooth_sigma: float = 0.35,
    batch_size: int = 32,
) -> np.ndarray:
    """
    全场推理（改进融合）。返回与 ``predict_full_field`` 相同形状 ``(H,W,4)``，归一化空间。

    Parameters
    ----------
    stride
        更小重叠更大（更慢、通常更平滑）；推荐 16～24。
    blend
        ``\"hann\"`` 推荐；``\"cosine\"`` 接近原版径向窗。
    dual_grid
        False 时只做单套起点 ``(0,0)``。
    triple_grid
        True 且 ``dual_grid`` 时：额外第三套起点 ``(stride//4, stride//4)``（需 ``stride>=4``）。
    norm_uv_smooth_sigma
        融合后对归一化 U/V 做高斯平滑的 sigma（网格索引）；0 关闭。
    """
    full_input_filled, input_invalid_mask = _build_full_input(era5_vars, static_tensor)
    H, W = full_input_filled.shape[:2]
    blend_window = _pick_blend(blend, patch_size)

    offsets: list[tuple[int, int]] = [(0, 0)]
    if dual_grid and stride >= 2:
        offsets.append((stride // 2, stride // 2))
    if triple_grid and dual_grid and stride >= 4:
        q = stride // 4
        if (q, q) not in offsets:
            offsets.append((q, q))

    stacks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for oy, ox in offsets:
        if oy > H - patch_size or ox > W - patch_size:
            continue
        pred_i, w_i, inv_i = _accumulate_one_offset_grid(
            model,
            full_input_filled,
            input_invalid_mask,
            device,
            patch_size,
            stride,
            oy,
            ox,
            blend_window,
            batch_size,
        )
        if np.any(w_i > 1e-8):
            stacks.append((pred_i, w_i, inv_i))

    if not stacks:
        return np.full((H, W, 4), np.nan, dtype=np.float32)

    merged = _merge_multi_normalized(stacks)
    merged = _smooth_norm_uv_channels(merged, float(norm_uv_smooth_sigma))
    return merged


def main() -> None:
    """仅打印默认策略说明（快速自检）；完整推理仍用 predict_full_field.py 或业务脚本。"""
    print("predict_full_field_blend_adv: Hann + dual/triple offset + norm UV smooth")
    w = _blend_window_hann(64)
    print(f"Hann(64) min/max: {w.min():.4f} {w.max():.4f} sum={float(w.sum()):.2f}")


if __name__ == "__main__":
    main()
