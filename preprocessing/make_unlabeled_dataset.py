"""
无监督数据预处理脚本：为PhyTopo-Net生成Track B（无监督/自监督）训练数据
生成配对的.npy patches (Input: Coarse ERA5 + Static Topography; Label: ERA5伪标签)
数据结构必须与Track A完全一致，以便模型可以无缝加载
"""
import numpy as np
import xarray as xr
import os
import glob
from tqdm import tqdm
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import griddata
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
PATCH_SIZE = 64
STRIDE = 64  # Track B使用不重叠切片，步长等于patch大小

# 目标patch数量
TARGET_PATCHES = 50000

# 归一化范围（必须与Track A完全一致）
NORM_STATS = {
    'U10': (-65.0, 65.0),      # 实际范围: -56.924 到 56.301，留有余量
    'V10': (-65.0, 65.0),      # 实际范围: -55.615 到 57.273，留有余量
    'SLP': (920.0, 1055.0),    # 实际范围: 926.079 到 1043.491，留有余量
    'T2': (250.0, 330.0),      # 实际范围: 256.774 到 322.860，留有余量
    'RAIN': (0.0, 2200.0),     # 实际范围: 0.0 到 1931.472，留有余量
    'HGT': (0.0, 3000.0),      # m (静态数据，保持不变)
    'SLOPE': (0.0, 45.0),      # degrees (静态数据，保持不变)
    'LAT': (14.0, 28.0),      # degrees (静态数据，保持不变)
    'LON': (105.0, 125.0),    # degrees (静态数据，保持不变)
    'BLH': (0.0, 3000.0),      # m (ERA5变量，保持原配置)
    'FSR': (0.0, 1.0),         # roughness (ERA5变量，保持原配置)
    'SST': (270.0, 310.0),     # K (ERA5变量，保持原配置)
}

# 路径配置（支持环境变量覆盖，便于超算部署）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA5_DIR = os.environ.get('ERA5_DIR', os.path.join(_REPO_ROOT, 'data', 'era5'))
GEO_FILE = os.environ.get('GEO_FILE', os.path.join(_REPO_ROOT, 'assets', 'geo_em.d03.nc'))
WRF_DIR = os.environ.get('WRF_DIR', os.path.join(_REPO_ROOT, 'data', 'wrfout'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './dataset/unlabeled')

# 过滤条件：只排除NaN，保留所有有效patches（包括低风速）
# MIN_WIND_SPEED = 10.0  # 已移除，因为无监督数据集的目的就是补充低风速数据


def load_static_data(geo_file):
    """
    加载静态数据（地形、坡度、经纬度等）
    
    Returns:
    --------
    static_tensor : np.array (H, W, C)
        静态特征堆叠
    landmask : np.array (H, W)
        陆地掩膜
    wrf_lat : np.array (H, W)
        WRF纬度网格
    wrf_lon : np.array (H, W)
        WRF经度网格
    """
    print("=" * 60)
    print("加载静态数据...")
    print("=" * 60)
    
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
    
    print(f"  静态数据形状: {static_tensor.shape}")
    print(f"  特征: [HGT, SLOPE, LAT, LON, LANDMASK]")
    
    return static_tensor, landmask, lat, lon


def normalize(data, var_name):
    """Min-Max归一化到[0, 1]"""
    if var_name not in NORM_STATS:
        raise ValueError(f"未知变量: {var_name}")
    
    vmin, vmax = NORM_STATS[var_name]
    data_norm = (data - vmin) / (vmax - vmin)
    data_norm = np.clip(data_norm, 0.0, 1.0)
    return data_norm.astype(np.float32)


def get_wrf_time_ranges():
    """
    获取所有WRF数据的时间范围，用于排除这些时间段
    
    Returns:
    --------
    wrf_time_ranges : list of tuples
        [(start_time, end_time), ...] 每个WRF文件的时间范围
    """
    wrf_files = sorted(glob.glob(os.path.join(WRF_DIR, '*.nc')))
    
    if not wrf_files:
        print("  警告: 未找到WRF文件，将使用所有ERA5数据")
        return []
    
    wrf_time_ranges = []
    
    for wrf_file in wrf_files:
        try:
            ds = xr.open_dataset(wrf_file)
            if 'time' in ds.coords:
                times = pd.to_datetime(ds.time.values)
                start_time = times.min()
                end_time = times.max()
                wrf_time_ranges.append((start_time, end_time))
            ds.close()
        except Exception as e:
            print(f"  警告: 无法读取WRF文件 {os.path.basename(wrf_file)}: {e}")
            continue
    
    print(f"\n找到 {len(wrf_files)} 个WRF文件，时间范围:")
    for start, end in wrf_time_ranges[:5]:  # 只显示前5个
        print(f"  {start} 到 {end}")
    if len(wrf_time_ranges) > 5:
        print(f"  ... (共 {len(wrf_time_ranges)} 个文件)")
    
    return wrf_time_ranges


def get_available_era5_times(era5_dir, wrf_time_ranges):
    """
    获取所有可用的ERA5时间点（排除已有WRF的时间段）
    
    Parameters:
    -----------
    era5_dir : str
        ERA5文件目录
    wrf_time_ranges : list
        WRF时间范围列表
    
    Returns:
    --------
    available_times : list
        可用的时间点列表（pandas Timestamp）
    """
    print("\n" + "=" * 60)
    print("扫描ERA5文件，查找可用时间点...")
    print("=" * 60)
    
    era5_files = sorted(glob.glob(os.path.join(era5_dir, 'ERA5_*.nc')))
    
    if not era5_files:
        raise FileNotFoundError(f"未找到ERA5文件: {era5_dir}")
    
    print(f"找到 {len(era5_files)} 个ERA5文件")
    
    available_times = []
    
    for era5_file in tqdm(era5_files, desc="扫描ERA5文件"):
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
            times = pd.to_datetime(ds.time.values).round('H')
            
            # 检查每个时间点是否在WRF时间范围内
            for t in times:
                t_ts = pd.to_datetime(t)
                
                # 检查是否与任何WRF时间范围重叠
                is_in_wrf = False
                for wrf_start, wrf_end in wrf_time_ranges:
                    if wrf_start <= t_ts <= wrf_end:
                        is_in_wrf = True
                        break
                
                # 如果不在WRF范围内，添加到可用列表
                if not is_in_wrf:
                    available_times.append(t_ts)
            
            ds.close()
            
        except Exception as e:
            print(f"  警告: 处理文件 {os.path.basename(era5_file)} 时出错: {e}")
            continue
    
    # 去重并排序
    available_times = sorted(list(set(available_times)))
    
    print(f"\n找到 {len(available_times)} 个可用时间点（排除WRF时间段）")
    if available_times:
        print(f"  时间范围: {available_times[0]} 到 {available_times[-1]}")
    
    return available_times


def load_and_interp_era5(era5_file, target_time, wrf_lat, wrf_lon):
    """
    加载ERA5数据并插值到WRF网格
    
    Parameters:
    -----------
    era5_file : str
        ERA5文件路径
    target_time : pandas Timestamp
        目标时间
    wrf_lat : np.array (H, W)
        WRF纬度网格
    wrf_lon : np.array (H, W)
        WRF经度网格
    
    Returns:
    --------
    era5_interp : dict
        {var_name: np.array (H, W)} 插值后的变量
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
    ds['time'] = pd.to_datetime(ds.time.values).round('H')
    
    # 选择目标时间
    target_time_rounded = pd.to_datetime(target_time).round('H')
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
    
    # 准备插值后的数据容器
    era5_vars = {}
    
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
                var_2d = var_2d / 100.0  # Pa -> hPa
            
            # 插值
            var_values = var_2d.flatten()
            var_interp = griddata(
                era5_points, var_values, wrf_points,
                method='linear', fill_value=np.nan
            )
            var_interp = var_interp.reshape(wrf_lat.shape)
            
            # 处理NaN（使用最近邻填充）
            if np.any(np.isnan(var_interp)):
                var_interp_nn = griddata(
                    era5_points, var_values, wrf_points,
                    method='nearest'
                )
                var_interp_nn = var_interp_nn.reshape(wrf_lat.shape)
                var_interp[np.isnan(var_interp)] = var_interp_nn[np.isnan(var_interp)]
            
            era5_vars[wrf_var] = var_interp
    
    ds.close()
    
    return era5_vars


def denormalize(data_norm, var_name):
    """反归一化到物理单位"""
    if var_name not in NORM_STATS:
        raise ValueError(f"未知变量: {var_name}")
    
    vmin, vmax = NORM_STATS[var_name]
    return data_norm * (vmax - vmin) + vmin


def create_patches_unlabeled(input_data, label_data, landmask, patch_size, stride):
    """
    创建patches（保留所有有效patches，包括低风速）
    
    Parameters:
    -----------
    input_data : np.array (H, W, C_in)
        输入数据（ERA5 + Static）
    label_data : np.array (H, W, C_out)
        标签数据（ERA5伪标签）
    landmask : np.array (H, W)
        陆地掩膜
    patch_size : int
        Patch大小
    stride : int
        步长
    
    Returns:
    --------
    patches : list of dict
        [{'input': ..., 'label': ...}, ...]
    """
    H, W, C_in = input_data.shape
    _, _, C_out = label_data.shape
    
    patches = []
    
    # 滑动窗口提取patches
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # 提取patch
            input_patch = input_data[y:y+patch_size, x:x+patch_size, :]
            label_patch = label_data[y:y+patch_size, x:x+patch_size, :]
            landmask_patch = landmask[y:y+patch_size, x:x+patch_size]
            
            # 检查是否有NaN或无效值
            if np.any(np.isnan(input_patch)) or np.any(np.isnan(label_patch)):
                continue
            
            # 保留所有有效patches（包括低风速），因为无监督数据集的目的就是补充低风速数据
            # 只排除NaN，不进行风速过滤
            patches.append({
                'input': input_patch.astype(np.float32),
                'label': label_patch.astype(np.float32)
            })
    
    return patches


def find_era5_file_for_time(era5_dir, target_time):
    """
    查找包含指定时间的ERA5文件
    
    Parameters:
    -----------
    era5_dir : str
        ERA5文件目录
    target_time : pandas Timestamp
        目标时间
    
    Returns:
    --------
    era5_file : str or None
        对应的ERA5文件路径
    """
    era5_files = sorted(glob.glob(os.path.join(era5_dir, 'ERA5_*.nc')))
    
    target_time_rounded = pd.to_datetime(target_time).round('H')
    
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
            ds['time'] = pd.to_datetime(ds.time.values).round('H')
            
            # 检查目标时间是否在范围内
            times = pd.to_datetime(ds.time.values)
            if times.min() <= target_time_rounded <= times.max():
                ds.close()
                return era5_file
            
            ds.close()
        except Exception:
            continue
    
    return None


def process_unlabeled_data(static_tensor, landmask, wrf_lat, wrf_lon, available_times, target_patches):
    """
    处理无监督数据
    
    Parameters:
    -----------
    static_tensor : np.array (H, W, C)
        静态数据
    landmask : np.array (H, W)
        陆地掩膜
    wrf_lat : np.array (H, W)
        WRF纬度网格
    wrf_lon : np.array (H, W)
        WRF经度网格
    available_times : list
        可用时间点列表
    target_patches : int
        目标patch数量
    """
    print("\n" + "=" * 60)
    print("开始处理无监督数据...")
    print("=" * 60)
    
    # 随机采样时间点
    if len(available_times) > target_patches // 100:  # 假设每个时间点能生成约100个patches
        # 如果时间点太多，随机采样
        sampled_times = random.sample(available_times, min(len(available_times), target_patches // 100))
    else:
        sampled_times = available_times
    
    print(f"从 {len(available_times)} 个可用时间点中采样 {len(sampled_times)} 个")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_patches = []
    patch_idx = 0
    
    # 按时间处理
    for target_time in tqdm(sampled_times, desc="处理时间点"):
        # 查找对应的ERA5文件
        era5_file = find_era5_file_for_time(ERA5_DIR, target_time)
        
        if era5_file is None:
            continue
        
        # 加载并插值ERA5数据
        era5_vars = load_and_interp_era5(era5_file, target_time, wrf_lat, wrf_lon)
        
        if era5_vars is None or len(era5_vars) < 7:  # 至少需要7个ERA5变量
            continue
        
        # 构建输入数据（ERA5变量 + 静态变量）
        input_vars = []
        
        # ERA5变量（按顺序：U10, V10, SLP, T2, SST, BLH, FSR）
        era5_order = ['U10', 'V10', 'SLP', 'T2', 'SST', 'BLH', 'FSR']
        for var in era5_order:
            if var in era5_vars:
                var_data = era5_vars[var]
                var_norm = normalize(var_data, var)
                input_vars.append(var_norm)
        
        if len(input_vars) < 7:
            continue
        
        # 堆叠ERA5变量
        era5_stack = np.stack(input_vars, axis=-1)  # (H, W, 7)
        
        # 合并ERA5和静态数据
        full_input = np.concatenate([era5_stack, static_tensor], axis=-1)  # (H, W, 12)
        
        # 陆地上SST不参与计算：LANDMASK=1为陆地，将SST通道置0（与有标签数据一致）
        full_input[:, :, 4][landmask > 0.5] = 0.0  # SST在通道4，LANDMASK在通道11
        
        # 构建标签数据（ERA5伪标签：U10, V10, SLP, T2，去掉RAIN）
        label_vars = []
        label_order = ['U10', 'V10', 'SLP', 'T2']  # 去掉RAIN
        for var in label_order:
            if var in era5_vars:
                var_data = era5_vars[var]
                var_norm = normalize(var_data, var)
                label_vars.append(var_norm)
        
        if len(label_vars) < 4:
            continue
        
        label_stack = np.stack(label_vars, axis=-1)  # (H, W, 4)
        
        # 创建patches
        patches = create_patches_unlabeled(
            full_input, label_stack, landmask,
            PATCH_SIZE, STRIDE
        )
        
        # 保存patches
        for patch in patches:
            time_str = target_time.strftime('%Y%m%d%H')
            patch_file = os.path.join(OUTPUT_DIR, f'unlabeled_{time_str}_patch{patch_idx:06d}.npy')
            np.save(patch_file, patch)
            
            all_patches.append(patch)
            patch_idx += 1
            
            # 如果达到目标数量，停止
            if len(all_patches) >= target_patches:
                break
        
        if len(all_patches) >= target_patches:
            break
    
    print(f"\n✅ 生成了 {len(all_patches)} 个无监督patches")
    print(f"  保存路径: {OUTPUT_DIR}")


def main():
    """主函数"""
    print("=" * 60)
    print("PhyTopo-Net 无监督数据预处理 (Track B)")
    print("=" * 60)
    
    # 加载静态数据
    static_tensor, landmask, wrf_lat, wrf_lon = load_static_data(GEO_FILE)
    
    # 获取WRF时间范围（用于排除）
    wrf_time_ranges = get_wrf_time_ranges()
    
    # 获取可用的ERA5时间点
    available_times = get_available_era5_times(ERA5_DIR, wrf_time_ranges)
    
    if not available_times:
        print("\n⚠️  警告: 未找到可用的时间点！")
        print("  可能所有ERA5时间段都已有对应的WRF数据")
        return
    
    # 处理无监督数据
    process_unlabeled_data(
        static_tensor, landmask, wrf_lat, wrf_lon,
        available_times, TARGET_PATCHES
    )
    
    print("\n" + "=" * 60)
    print("无监督数据预处理完成！")
    print("=" * 60)
    
    # 统计
    unlabeled_files = glob.glob(os.path.join(OUTPUT_DIR, '*.npy'))
    print(f"\n无监督数据集: {len(unlabeled_files)} 个patches")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PhyTopo-Net 无监督数据生成')
    parser.add_argument('--era5_dir', default=None, help='ERA5 数据目录')
    parser.add_argument('--geo_file', default=None, help='静态地理文件路径')
    parser.add_argument('--wrf_dir', default=None, help='WRF 数据目录（用于排除已有时间段）')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    args = parser.parse_args()
    if args.era5_dir:
        ERA5_DIR = args.era5_dir
    if args.geo_file:
        GEO_FILE = args.geo_file
    if args.wrf_dir:
        WRF_DIR = args.wrf_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    main()
