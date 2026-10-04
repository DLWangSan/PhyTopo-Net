"""
直接预测某个时刻的完整场
1. 加载某个时刻的ERA5完整场（插值到WRF网格）
2. 加载静态数据
3. 用模型预测完整场（分成patches预测，然后拼接）
4. 加载对应时刻的WRF数据
5. 保存三个完整场用于对比
"""
import os
import sys
import torch
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import argparse
import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PhyTopoNet


def _create_blend_window(size):
    """创建2D余弦窗：中心权重1，边缘权重0，用于patch加权融合，减轻拼凑感"""
    cy, cx = size // 2, size // 2
    y = np.arange(size, dtype=np.float32)
    x = np.arange(size, dtype=np.float32)
    # 到中心的归一化距离 (0=中心, 1=最远角)
    r = np.sqrt((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2) + 1e-8
    r = np.clip(r / max_r, 0, 1)
    # 余弦窗: 中心1, 边缘0
    w = 0.5 * (1 + np.cos(np.pi * r))
    return w


def normalize(data, var_name):
    """归一化数据到[0, 1]"""
    NORM_STATS = {
        'U10': (-65.0, 65.0),
        'V10': (-65.0, 65.0),
        'SLP': (920.0, 1055.0),
        'T2': (250.0, 330.0),
        'RAIN': (0.0, 150.0),
        'SST': (270.0, 310.0),
        'BLH': (0.0, 5000.0),
        'FSR': (0.0, 1.0),
        'HGT': (0.0, 3000.0),
        'SLOPE': (0.0, 45.0),
        'LAT': (14.0, 28.0),
        'LON': (105.0, 125.0),
    }
    
    if var_name not in NORM_STATS:
        return data
    
    min_val, max_val = NORM_STATS[var_name]
    return (data - min_val) / (max_val - min_val)


def denormalize(data, var_name):
    """反归一化到物理单位"""
    NORM_STATS = {
        'U10': (-65.0, 65.0),
        'V10': (-65.0, 65.0),
        'SLP': (920.0, 1055.0),
        'T2': (250.0, 330.0),
        'RAIN': (0.0, 150.0),
        'SST': (270.0, 310.0),
        'BLH': (0.0, 5000.0),
        'FSR': (0.0, 1.0),
    }
    
    if var_name not in NORM_STATS:
        return data
    
    min_val, max_val = NORM_STATS[var_name]
    return data * (max_val - min_val) + min_val


def load_static_data(geo_file):
    """加载静态数据（与make_dataset.py保持一致）"""
    from netCDF4 import Dataset
    from scipy.ndimage import sobel, gaussian_filter
    
    nc = Dataset(geo_file)
    
    # 读取地形高度
    hgt = np.array(nc.variables['HGT_M'][0, :, :])  # (y, x)
    
    # 读取陆地掩膜
    landmask = np.array(nc.variables['LANDMASK'][0, :, :])  # (y, x)
    
    # 读取经纬度
    lat = np.array(nc.variables['XLAT_M'][0, :, :])
    lon = np.array(nc.variables['XLONG_M'][0, :, :])
    
    nc.close()
    
    # 计算坡度（使用Sobel算子近似）
    hgt_smooth = gaussian_filter(hgt, sigma=1.0)  # 轻微平滑
    grad_y = sobel(hgt_smooth, axis=0)
    grad_x = sobel(hgt_smooth, axis=1)
    # 假设网格间距约3km，转换为坡度（度）
    dx = 3000.0  # meters
    dy = 3000.0  # meters
    slope = np.arctan(np.sqrt((grad_x/dx)**2 + (grad_y/dy)**2)) * 180.0 / np.pi
    
    # 归一化
    hgt_norm = normalize(hgt, 'HGT')
    slope_norm = normalize(slope, 'SLOPE')
    lat_norm = normalize(lat, 'LAT')
    lon_norm = normalize(lon, 'LON')
    landmask_norm = landmask.astype(np.float32)  # 已经是0/1
    
    # 堆叠为 (H, W, C)
    static_tensor = np.stack([
        hgt_norm,
        slope_norm,
        lat_norm,
        lon_norm,
        landmask_norm
    ], axis=-1)
    
    return static_tensor, lat, lon


def find_era5_file_for_time(era5_dir, target_time):
    """查找包含指定时间的ERA5文件"""
    era5_files = sorted(glob.glob(os.path.join(era5_dir, 'ERA5_*.nc')))
    
    target_time_rounded = pd.to_datetime(target_time).round('h')
    
    for era5_file in era5_files:
        try:
            ds = xr.open_dataset(era5_file)
            
            # 处理expver和number维度
            if 'expver' in ds.dims:
                ds = ds.sel(expver=1)
            if 'number' in ds.dims:
                ds = ds.sel(number=0)
            
            # 重命名valid_time为time
            if 'valid_time' in ds.coords and 'time' not in ds.coords:
                ds = ds.rename({'valid_time': 'time'})
            
            if 'time' not in ds.coords:
                ds.close()
                continue
            
            # 统一时间格式
            ds['time'] = pd.to_datetime(ds.time.values).round('h')
            
            # 检查目标时间是否在范围内
            times = pd.to_datetime(ds.time.values)
            if times.min() <= target_time_rounded <= times.max():
                ds.close()
                return era5_file
            
            ds.close()
        except Exception:
            continue
    
    return None


def load_era5_for_time(
    era5_file, target_time, wrf_lat, wrf_lon, return_raw=True, era5_uv_interp="linear",
):
    """加载某个时刻的ERA5数据并插值到WRF网格。若 return_raw=True，同时返回原始0.25°数据。

    era5_uv_interp:
        linear（默认）— U10/V10 与变量一致，先线性再最近邻补洞；
        nearest — 仅 U10/V10 用最近邻插值到 WRF 网格（块状、少平滑），其余变量仍为线性。
    """
    ds = xr.open_dataset(era5_file)
    
    # 处理expver和number维度
    if 'expver' in ds.dims:
        ds = ds.sel(expver=1)
    if 'number' in ds.dims:
        ds = ds.sel(number=0)
    
    # 重命名valid_time为time
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    
    # 统一时间格式
    ds['time'] = pd.to_datetime(ds.time.values).round('h')
    target_time_rounded = pd.to_datetime(target_time).round('h')
    
    # 选择目标时间
    try:
        ds_sel = ds.sel(time=target_time_rounded, method='nearest')
    except (KeyError, ValueError, TypeError):
        ds.close()
        return None
    
    # 获取ERA5的经纬度
    if 'latitude' in ds_sel.coords:
        era5_lat = ds_sel.latitude.values
        era5_lon = ds_sel.longitude.values
    elif 'lat' in ds_sel.coords:
        era5_lat = ds_sel.lat.values
        era5_lon = ds_sel.lon.values
    else:
        ds.close()
        return None
    
    # 如果是1D，创建2D网格
    if era5_lat.ndim == 1:
        era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)
    else:
        era5_lat_2d = era5_lat
        era5_lon_2d = era5_lon
    
    # 准备插值点
    era5_points = np.column_stack([era5_lon_2d.flatten(), era5_lat_2d.flatten()])
    wrf_points = np.column_stack([wrf_lon.flatten(), wrf_lat.flatten()])
    
    # ERA5变量映射
    var_mapping = {
        'u10': 'U10',
        'v10': 'V10',
        'msl': 'SLP',
        't2m': 'T2',
        'sst': 'SST',
        'blh': 'BLH',
        'fsr': 'FSR'
    }
    
    # 提取并插值变量，同时保存原始数据（用于绘图）
    era5_vars = {}
    era5_raw = {}  # 原始 0.25° 数据
    for era5_var, wrf_var in var_mapping.items():
        if era5_var in ds_sel.data_vars:
            var_data = ds_sel[era5_var].values
            
            # 处理维度
            if var_data.ndim == 2:
                var_2d = var_data
            elif var_data.ndim == 3:
                var_2d = var_data[0, :, :]
            else:
                continue
            
            # 单位转换
            if era5_var == 'msl' and var_2d.max() > 2000:
                var_2d = var_2d / 100.0
            
            # 保存原始数据（U10, V10, SLP 用于绘图）
            if return_raw and wrf_var in ['U10', 'V10', 'SLP']:
                era5_raw[wrf_var] = var_2d.copy()
            
            # 插值（关键修复：先过滤掉ERA5原始数据中的NaN点）
            var_values_all = var_2d.flatten()
            
            # 过滤掉NaN点，只使用有效数据点进行插值
            valid_mask = ~np.isnan(var_values_all)
            era5_points_valid = era5_points[valid_mask]
            var_values_valid = var_values_all[valid_mask]
            
            # 如果有效数据点太少，跳过这个变量
            if len(var_values_valid) < 10:
                print(f"    警告: {era5_var} 有效数据点太少 ({len(var_values_valid)}), 跳过")
                continue
            
            use_nearest_uv = era5_uv_interp == "nearest" and wrf_var in ("U10", "V10")
            if use_nearest_uv:
                var_interp = griddata(
                    era5_points_valid, var_values_valid, wrf_points,
                    method="nearest", fill_value=np.nan
                )
                var_interp = var_interp.reshape(wrf_lat.shape)
            else:
                var_interp = griddata(
                    era5_points_valid, var_values_valid, wrf_points,
                    method="linear", fill_value=np.nan
                )
                var_interp = var_interp.reshape(wrf_lat.shape)
                if np.any(np.isnan(var_interp)):
                    var_interp_nn = griddata(
                        era5_points_valid, var_values_valid, wrf_points,
                        method="nearest"
                    )
                    var_interp_nn = var_interp_nn.reshape(wrf_lat.shape)
                    var_interp[np.isnan(var_interp)] = var_interp_nn[np.isnan(var_interp)]
            
            # 第三层：若仍有NaN（如SST在陆地上），用全局均值填充（与make_dataset一致）
            if np.any(np.isnan(var_interp)):
                global_mean = np.nanmean(var_values_valid)
                if np.isnan(global_mean):
                    global_mean = 0.0
                var_interp[np.isnan(var_interp)] = global_mean
            
            era5_vars[wrf_var] = var_interp
    
    ds.close()
    if return_raw and era5_raw:
        era5_raw['lat'] = era5_lat_2d
        era5_raw['lon'] = era5_lon_2d
        return era5_vars, era5_raw
    return era5_vars


def predict_full_field(model, era5_vars, static_tensor, device, patch_size=64, stride=32):
    """
    预测完整场（分成patches预测，然后拼接）
    
    Parameters:
    -----------
    model : PhyTopoNet
        训练好的模型
    era5_vars : dict
        ERA5变量字典 {var_name: (H, W)}
    static_tensor : np.array (H, W, 5)
        静态数据
    device : torch.device
        设备
    patch_size : int
        patch大小
    stride : int
        步长
    
    Returns:
    --------
    prediction : np.array (H, W, 5)
        预测的完整场
    """
    H, W = static_tensor.shape[:2]
    
    # 构建输入数据
    era5_order = ['U10', 'V10', 'SLP', 'T2', 'SST', 'BLH', 'FSR']
    era5_list = []
    for var in era5_order:
        if var in era5_vars:
            era5_list.append(normalize(era5_vars[var], var))
        else:
            era5_list.append(np.zeros((H, W)))
    
    era5_stack = np.stack(era5_list, axis=-1)  # (H, W, 7)
    full_input = np.concatenate([era5_stack, static_tensor], axis=-1)  # (H, W, 12)
    
    # 陆地上SST不参与计算：LANDMASK=1为陆地，将SST通道置0（与训练一致）
    landmask = full_input[:, :, 11]  # LANDMASK在通道11
    full_input[:, :, 4][landmask > 0.5] = 0.0  # SST在通道4，陆地置0
    
    # 记录原始输入中的NaN位置（用于后续mask输出）
    # 对于ERA5变量（前7个通道），如果原始数据是NaN，则标记为无效
    # 注意：不再主动mask边界区域，因为ERA5输入本身是正常的
    # 如果ERA5输入有NaN，会在插值时被处理；如果没有NaN，就不需要mask边界
    input_invalid_mask = np.zeros((H, W), dtype=bool)
    for c in range(7):  # ERA5变量通道
        input_invalid_mask |= np.isnan(full_input[:, :, c])
    
    # 不再主动mask边界区域，因为ERA5输入本身是正常的
    # 如果边界区域有异常值，会在后续的合理性检查中过滤
    
    # 生成patches
    patches = []
    patch_positions = []
    patch_invalid_masks = []  # 记录每个patch的无效区域
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = full_input[y:y+patch_size, x:x+patch_size, :]  # (64, 64, 12)
            patch_invalid = input_invalid_mask[y:y+patch_size, x:x+patch_size]
            
            # 用0填充NaN以便模型运行（但记录原始NaN位置）
            patch_filled = np.nan_to_num(patch, nan=0.0)
            
            patches.append(patch_filled)
            patch_positions.append((y, x))
            patch_invalid_masks.append(patch_invalid)
    
    if len(patches) == 0:
        raise ValueError("没有有效的patches")
    
    print(f"  生成了 {len(patches)} 个patches")
    
    # 转换为tensor
    patches_tensor = torch.from_numpy(np.stack(patches)).float()  # (N, 64, 64, 12)
    patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # (N, 12, 64, 64)
    
    # 分离ERA5和静态数据
    x_met = patches_tensor[:, :7, :, :]  # (N, 7, 64, 64)
    x_geo = patches_tensor[:, 7:, :, :]  # (N, 5, 64, 64)
    
    # 预测（分批处理以避免内存问题）
    model.eval()
    batch_size = 32
    all_predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc='预测patches'):
            batch_x_met = x_met[i:i+batch_size].to(device)
            batch_x_geo = x_geo[i:i+batch_size].to(device)
            batch_pred = model(batch_x_met, batch_x_geo)  # (B, 4, 64, 64) [U, V, SLP, T2]
            batch_pred = torch.clamp(batch_pred, 0.0, 1.0)
            all_predictions.append(batch_pred.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 4, 64, 64)
    
    # 拼接回完整场（4通道: U10, V10, SLP, T2）
    # 使用2D余弦窗加权融合，减轻patch边界拼凑感
    n_out = predictions.shape[1]
    prediction_full = np.zeros((H, W, n_out), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    blend_window = _create_blend_window(patch_size)  # (64, 64)
    final_invalid_mask = np.zeros((H, W), dtype=bool)
    
    for i, (y, x) in enumerate(patch_positions):
        pred_patch = predictions[i].transpose(1, 2, 0)  # (64, 64, n_out)
        patch_invalid = patch_invalid_masks[i]
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        h_slice, w_slice = y_end - y, x_end - x
        
        # 有效区域 mask（非 NaN）
        patch_valid = ~patch_invalid[:h_slice, :w_slice]
        # 该 patch 的权重：余弦窗 × 有效mask
        w = blend_window[:h_slice, :w_slice].copy()
        w[~patch_valid] = 0.0
        
        for c in range(n_out):
            pred_slice = pred_patch[:h_slice, :w_slice, c]
            pred_slice = np.where(patch_valid, pred_slice, 0.0)
            prediction_full[y:y_end, x:x_end, c] += w * pred_slice
        weight_sum[y:y_end, x:x_end] += w
        final_invalid_mask[y:y_end, x:x_end] |= patch_invalid[:h_slice, :w_slice]
    
    # 加权平均（避免除零）
    mask = weight_sum > 1e-8
    safe_weight = np.where(mask, weight_sum, 1.0)
    for c in range(n_out):
        prediction_full[:, :, c] = np.where(mask, prediction_full[:, :, c] / safe_weight, np.nan)
    
    # 将原始输入中无效的区域设为NaN（避免异常值）
    prediction_full[final_invalid_mask] = np.nan
    
    # 未覆盖区域也设为NaN
    prediction_full[~mask] = np.nan
    
    # 注意：合理性检查应该在反归一化之后进行（在main函数中）
    # 因为合理性检查的范围是物理单位，而这里的prediction_full还是归一化后的值
    return prediction_full


def main():
    parser = argparse.ArgumentParser(description='预测某个时刻的完整场')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--wrf_file', type=str, required=True)
    parser.add_argument('--era5_file', type=str, default=None,
                        help='ERA5文件路径（如果未指定，将从era5_dir自动查找）')
    parser.add_argument('--era5_dir', type=str, default=os.environ.get('ERA5_DIR', ''),
                        help='ERA5文件目录（用于自动查找）')
    parser.add_argument('--geo_file', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'geo_em.d03.nc'))
    parser.add_argument('--time_idx', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("预测完整场")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    use_residual = checkpoint.get('residual', False) if isinstance(checkpoint, dict) else False
    model = PhyTopoNet(residual=use_residual)
    if use_residual:
        print("  残差模式: output = input + 网络预测")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()
    print("✓ 模型加载完成")
    
    # 加载WRF数据
    print(f"\n加载WRF数据: {args.wrf_file}")
    ds_wrf = xr.open_dataset(args.wrf_file)
    wrf_lat = ds_wrf.lat.values
    wrf_lon = ds_wrf.lon.values
    times = pd.to_datetime(ds_wrf.time.values)
    target_time = times[args.time_idx]
    print(f"✓ 目标时间: {target_time}")
    print(f"✓ 网格大小: {wrf_lat.shape}")
    
    # 加载WRF标签数据
    wrf_t = ds_wrf.isel(time=args.time_idx)
    wrf_vars = {}
    for var in ['U10', 'V10', 'SLP', 'T2', 'RAIN']:
        if var in wrf_t.data_vars:
            wrf_vars[var] = wrf_t[var].values
    
    ds_wrf.close()
    
    # 加载静态数据
    print(f"\n加载静态数据: {args.geo_file}")
    static_tensor, static_lat, static_lon = load_static_data(args.geo_file)
    print("✓ 静态数据加载完成")
    
    # 查找或加载ERA5数据
    if args.era5_file is None:
        print(f"\n查找ERA5文件（目录: {args.era5_dir}）...")
        era5_file = find_era5_file_for_time(args.era5_dir, target_time)
        if era5_file is None:
            print(f"❌ 未找到时间 {target_time} 对应的ERA5文件")
            sys.exit(1)
        print(f"✓ 找到ERA5文件: {os.path.basename(era5_file)}")
    else:
        era5_file = args.era5_file
    
    print(f"\n加载ERA5数据: {os.path.basename(era5_file)}")
    result = load_era5_for_time(era5_file, target_time, wrf_lat, wrf_lon)
    if isinstance(result, tuple):
        era5_vars, era5_raw = result
    else:
        era5_vars = result
        era5_raw = None
    if era5_vars is None:
        print("❌ 无法加载ERA5数据")
        sys.exit(1)
    print("✓ ERA5数据加载完成")
    
    # 预测完整场
    print(f"\n预测完整场...")
    prediction = predict_full_field(
        model, era5_vars, static_tensor, args.device,
        patch_size=args.patch_size, stride=args.stride
    )
    print("✓ 预测完成")
    
    # 反归一化（4通道: U10, V10, SLP, T2）
    print("\n反归一化...")
    era5_order = ['U10', 'V10', 'SLP', 'T2']
    era5_full = np.zeros((wrf_lat.shape[0], wrf_lat.shape[1], 4))
    for i, var in enumerate(era5_order):
        if var in era5_vars:
            era5_full[:, :, i] = era5_vars[var]
        else:
            era5_full[:, :, i] = np.nan
    
    wrf_full = np.zeros((wrf_lat.shape[0], wrf_lat.shape[1], 4))
    for i, var in enumerate(era5_order):
        if var in wrf_vars:
            wrf_full[:, :, i] = wrf_vars[var]
        else:
            wrf_full[:, :, i] = np.nan
    
    prediction_phys = np.zeros_like(prediction)
    for i, var in enumerate(era5_order):
        prediction_phys[:, :, i] = denormalize(prediction[:, :, i], var)
    
    # 合理性检查：将明显超出物理范围的预测值设为NaN（在反归一化后）
    reasonable_ranges = {
        'U10': (-200, 200),
        'V10': (-200, 200),
        'SLP': (700, 1400),
        'T2': (200, 400),
    }
    
    for i, var in enumerate(era5_order):
        if var in reasonable_ranges:
            min_val, max_val = reasonable_ranges[var]
            invalid_mask = (prediction_phys[:, :, i] < min_val) | (prediction_phys[:, :, i] > max_val)
            n_invalid = np.sum(invalid_mask)
            if n_invalid > 0:
                prediction_phys[:, :, i][invalid_mask] = np.nan
                print(f"    过滤了 {n_invalid} 个超出范围的值（{var}，范围: [{min_val}, {max_val}]）")
    
    # 注意：由于clamp已经限制了U10/V10在[-65, 65]范围内，单个分量是合理的
    # U10和V10的组合可能导致风速>70 m/s（例如U10=60, V10=40 -> 72.1 m/s）
    # 但这是合理的，因为clamp已经限制了单个分量
    # 因此，我们移除风速阈值检查，只保留单个分量的范围检查（已在reasonable_ranges中）
    # 这样可以避免过度过滤clamp后的合理值
    
    # 保存
    print(f"\n保存到: {args.output_file}")
    save_dict = dict(
        prediction=prediction_phys,
        era5_input=era5_full,
        wrf_label=wrf_full,
        lat=wrf_lat,
        lon=wrf_lon,
        time_idx=args.time_idx,
        time=str(target_time)
    )
    if era5_raw is not None:
        save_dict['era5_raw_u'] = era5_raw['U10']
        save_dict['era5_raw_v'] = era5_raw['V10']
        save_dict['era5_raw_slp'] = era5_raw['SLP']
        save_dict['era5_raw_lat'] = era5_raw['lat']
        save_dict['era5_raw_lon'] = era5_raw['lon']
    # 地形（用于绘图叠加陆地/岛屿）
    from netCDF4 import Dataset
    nc = Dataset(args.geo_file)
    save_dict['terrain_hgt'] = np.array(nc.variables['HGT_M'][0, :, :])
    save_dict['terrain_landmask'] = np.array(nc.variables['LANDMASK'][0, :, :])
    save_dict['terrain_lat'] = np.array(nc.variables['XLAT_M'][0, :, :])
    save_dict['terrain_lon'] = np.array(nc.variables['XLONG_M'][0, :, :])
    nc.close()
    np.savez(args.output_file, **save_dict)
    print("✓ 完成！")


if __name__ == '__main__':
    main()
