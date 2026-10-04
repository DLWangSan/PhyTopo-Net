"""
数据预处理脚本：为PhyTopo-Net超分辨率模型生成训练数据
生成配对的.npy patches (Input: Coarse ERA5 + Static Topography; Label: High-Res WRF)
"""
import numpy as np
import xarray as xr
import os
import glob
from tqdm import tqdm
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
PATCH_SIZE = 64
STRIDE = 32

# 归一化范围（Min-Max）
# 根据所有台风数据统计结果调整，确保包含所有数据并留有余量
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

# 训练/测试/验证集划分
# 训练集：多个台风案例用于训练
TRAIN_CASES = ['Higos', 'Saola', 'Kompasu', 'Janu', 'July', 'April']  # 增加April以增加训练集数据量
# 验证集：用于调参和早停
VALID_CASES = ['Kuinu']  # 留一个台风做验证
# 测试集：用于最终评估
TEST_CASES = ['Yagi', 'Oct']  # 减少测试集案例，提高训练集占比

# 路径配置（支持环境变量覆盖，便于超算运行）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRF_DIR = os.environ.get('WRF_DIR', os.path.join(_REPO_ROOT, 'data', 'wrfout'))
ERA5_DIR = os.environ.get('ERA5_DIR', os.path.join(_REPO_ROOT, 'data', 'era5'))
GEO_FILE = os.environ.get('GEO_FILE', os.path.join(_REPO_ROOT, 'assets', 'geo_em.d03.nc'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './dataset')

# 过滤条件（已移除，保留所有patches）
# 注意：如果数据量太大，可以后续通过采样来减少
KEEP_ALL_PATCHES = True  # 保留所有patches，包括平静海洋区域


def load_static_data(geo_file):
    """
    加载静态数据（地形、坡度、经纬度等）
    
    Returns:
    --------
    static_tensor : np.array (H, W, C)
        静态特征堆叠
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
    from scipy.ndimage import sobel
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
    
    return static_tensor, landmask


def normalize(data, var_name):
    """Min-Max归一化到[0, 1]"""
    if var_name not in NORM_STATS:
        raise ValueError(f"未知变量: {var_name}")
    
    vmin, vmax = NORM_STATS[var_name]
    data_norm = (data - vmin) / (vmax - vmin)
    data_norm = np.clip(data_norm, 0.0, 1.0)
    return data_norm.astype(np.float32)


def denormalize(data_norm, var_name):
    """反归一化"""
    vmin, vmax = NORM_STATS[var_name]
    return data_norm * (vmax - vmin) + vmin


def load_wrf_data(wrf_file, static_tensor=None):
    """
    加载WRF数据
    
    Parameters:
    -----------
    wrf_file : str
        WRF文件路径
    static_tensor : np.array, optional
        静态数据张量，用于验证维度匹配
    
    Returns:
    --------
    ds_wrf : xarray.Dataset
        包含所有变量的WRF数据集
    """
    import pandas as pd
    
    print(f"\n加载WRF数据: {os.path.basename(wrf_file)}")
    
    ds = xr.open_dataset(wrf_file)
    
    # 检查变量（去掉RAIN，因为不再使用）
    required_vars = ['U10', 'V10', 'SLP', 'T2']
    missing_vars = [v for v in required_vars if v not in ds.data_vars]
    if missing_vars:
        print(f"  警告: 缺少变量 {missing_vars}")
    
    # 确保SLP单位是hPa
    if 'SLP' in ds.data_vars:
        if ds.SLP.max() > 2000:
            ds['SLP'] = ds.SLP / 100.0
    
    # 统一时间格式：转换为pandas时间并取整到小时
    ds['time'] = pd.to_datetime(ds.time.values).round('H')
    
    # 验证维度匹配（如果提供了static_tensor）
    # 支持不同的维度名称：x/west_east 和 y/south_north
    if static_tensor is not None:
        # 尝试不同的维度名称
        x_dim = None
        y_dim = None
        for dim_name in ['x', 'west_east', 'lon', 'longitude']:
            if dim_name in ds.dims:
                x_dim = dim_name
                break
        for dim_name in ['y', 'south_north', 'lat', 'latitude']:
            if dim_name in ds.dims:
                y_dim = dim_name
                break
        
        if x_dim is None or y_dim is None:
            print(f"  警告: 无法识别空间维度，可用维度: {list(ds.dims.keys())}")
        else:
            assert ds.dims[x_dim] == static_tensor.shape[1], \
                f"WRF {x_dim}维度 ({ds.dims[x_dim]}) 与静态数据宽度 ({static_tensor.shape[1]}) 不匹配"
            assert ds.dims[y_dim] == static_tensor.shape[0], \
                f"WRF {y_dim}维度 ({ds.dims[y_dim]}) 与静态数据高度 ({static_tensor.shape[0]}) 不匹配"
    
    print(f"  时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
    print(f"  时间步数: {len(ds.time)}")
    print(f"  空间维度: {ds.dims}")
    
    return ds


def find_era5_files(era5_dir, target_times):
    """
    根据目标时间查找对应的ERA5文件
    
    Parameters:
    -----------
    era5_dir : str
        ERA5文件目录
    target_times : list
        目标时间列表（pandas Timestamp或datetime）
    
    Returns:
    --------
    era5_files : dict
        {time: file_path} 映射
    """
    import pandas as pd
    
    # 获取所有ERA5文件
    era5_files = sorted(glob.glob(os.path.join(era5_dir, 'ERA5_*.nc')))
    
    if not era5_files:
        raise FileNotFoundError(f"未找到ERA5文件: {era5_dir}")
    
    print(f"\n找到 {len(era5_files)} 个ERA5文件")
    
    # 为每个目标时间查找对应的文件
    era5_dict = {}
    
    for target_time in target_times:
        if isinstance(target_time, (str, pd.Timestamp)):
            target_time = pd.to_datetime(target_time)
        
        # 查找包含该时间的文件
        found = False
        for era5_file in era5_files:
            try:
                ds_temp = xr.open_dataset(era5_file)
                
                # 检查是否有valid_time坐标
                if 'valid_time' in ds_temp.coords:
                    times = pd.to_datetime(ds_temp.valid_time.values)
                elif 'time' in ds_temp.coords:
                    times = pd.to_datetime(ds_temp.time.values)
                else:
                    ds_temp.close()
                    continue
                
                # 检查目标时间是否在范围内
                if times.min() <= target_time <= times.max():
                    era5_dict[target_time] = era5_file
                    found = True
                    ds_temp.close()
                    break
                
                ds_temp.close()
            except Exception as e:
                continue
        
        if not found:
            print(f"  警告: 未找到 {target_time} 对应的ERA5文件")
    
    return era5_dict


def load_and_interp_era5(era5_file, target_times, wrf_lat, wrf_lon):
    """
    加载ERA5数据并插值到WRF网格
    
    Parameters:
    -----------
    era5_file : str
        ERA5文件路径
    target_times : list
        目标时间列表
    wrf_lat : np.array (H, W)
        WRF纬度网格
    wrf_lon : np.array (H, W)
        WRF经度网格
    
    Returns:
    --------
    era5_interp : dict
        {var_name: np.array (H, W)} 插值后的变量
    """
    import pandas as pd
    from scipy.interpolate import griddata
    
    print(f"  加载ERA5文件: {os.path.basename(era5_file)}")
    
    ds = xr.open_dataset(era5_file)
    
    # 处理expver和number维度
    if 'expver' in ds.dims:
        ds = ds.sel(expver=1)  # 使用ERA5（不是ERA5T）
    if 'number' in ds.dims:
        ds = ds.sel(number=0)  # 使用第一个ensemble成员
    
    # 重命名valid_time为time（如果需要）
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    
    # 统一时间格式：转换为pandas时间并取整到小时
    ds['time'] = pd.to_datetime(ds.time.values).round('H')
    
    # 选择目标时间（统一取整到小时后直接索引）
    target_time = pd.to_datetime(target_times[0]).round('H')
    
    # 直接使用xarray的sel方法（时间已统一格式，应该能直接匹配）
    try:
        ds_sel = ds.sel(time=target_time, method='nearest')
        # 检查时间差（如果超过6小时则警告）
        actual_time = pd.to_datetime(ds_sel.time.values).round('H')
        time_diff_hours = abs((actual_time - target_time).total_seconds() / 3600.0)
        if time_diff_hours > 6:
            print(f"    警告: 时间差较大 ({time_diff_hours:.1f} 小时)")
    except (KeyError, ValueError, TypeError) as e:
        # 如果仍然失败，给出更详细的错误信息
        print(f"    错误: 无法匹配时间 {target_time}，可用时间范围: {ds.time.values[0]} 到 {ds.time.values[-1]}")
        raise
    
    # 获取ERA5的经纬度（可能是1D或2D）
    if 'latitude' in ds_sel.coords:
        era5_lat = ds_sel.latitude.values
        era5_lon = ds_sel.longitude.values
    elif 'lat' in ds_sel.coords:
        era5_lat = ds_sel.lat.values
        era5_lon = ds_sel.lon.values
    else:
        raise ValueError("无法找到ERA5经纬度坐标")
    
    # 如果是1D，创建2D网格
    if era5_lat.ndim == 1:
        era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)
    else:
        era5_lat_2d = era5_lat
        era5_lon_2d = era5_lon
    
    # 准备插值后的数据容器
    era5_vars = {}
    
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
    
    for era5_var, wrf_var in var_mapping.items():
        if era5_var in ds_sel.data_vars:
            var_data = ds_sel[era5_var].values
            
            # 处理维度
            if var_data.ndim == 2:
                var_2d = var_data
            elif var_data.ndim == 3:
                var_2d = var_data[0, :, :]  # 取第一个时间/ensemble
            else:
                continue
            
            # 单位转换
            if era5_var == 'msl' and var_2d.max() > 2000:
                var_2d = var_2d / 100.0  # Pa -> hPa
            
            # 使用scipy插值到WRF网格
            era5_points = np.column_stack([era5_lon_2d.flatten(), era5_lat_2d.flatten()])
            wrf_points = np.column_stack([wrf_lon.flatten(), wrf_lat.flatten()])
            var_values = var_2d.flatten()
            
            # 插值
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
    
    print(f"  插值完成，变量: {list(era5_vars.keys())}")
    
    return era5_vars


def create_patches(input_data, label_data, static_data, landmask, patch_size, stride):
    """
    创建patches（保留所有patches，不进行过滤）
    
    Parameters:
    -----------
    input_data : np.array (T, H, W, C_in)
        输入数据（ERA5 + Static）
    label_data : np.array (T, H, W, C_out)
        标签数据（WRF）
    static_data : np.array (H, W, C_static)
        静态数据
    landmask : np.array (H, W)
        陆地掩膜（用于记录，但不用于过滤）
    patch_size : int
        Patch大小
    stride : int
        步长
    
    Returns:
    --------
    patches : list of dict
        [{'input': ..., 'label': ...}, ...]
    stats : dict
        NaN过滤统计信息
    """
    T, H, W, C_in = input_data.shape
    _, _, _, C_out = label_data.shape
    
    patches = []
    stats = {
        'total_attempted': 0,
        'skipped_nan_input': 0,
        'skipped_nan_label': 0,
        'skipped_nan_both': 0,
        'kept': 0
    }
    
    # 滑动窗口提取patches（保留所有patches）
    for t in range(T):
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                stats['total_attempted'] += 1
                
                # 提取patch
                input_patch = input_data[t, y:y+patch_size, x:x+patch_size, :]
                label_patch = label_data[t, y:y+patch_size, x:x+patch_size, :]
                
                # 检查是否有NaN或无效值
                has_nan_input = np.any(np.isnan(input_patch))
                has_nan_label = np.any(np.isnan(label_patch))
                
                if has_nan_input and has_nan_label:
                    stats['skipped_nan_both'] += 1
                    continue
                elif has_nan_input:
                    stats['skipped_nan_input'] += 1
                    continue
                elif has_nan_label:
                    stats['skipped_nan_label'] += 1
                    continue
                
                # 保留所有有效patches（包括平静海洋区域）
                patches.append({
                    'input': input_patch.astype(np.float32),
                    'label': label_patch.astype(np.float32)
                })
                stats['kept'] += 1
    
    return patches, stats


def check_case_completed(case_name, split):
    """
    检查案例是否已经完成处理
    
    Parameters:
    -----------
    case_name : str
        案例名称（从WRF文件名提取）
    split : str
        数据集划分：'train', 'valid', 'test'
    
    Returns:
    --------
    bool
        True表示已完成，False表示未完成
    """
    output_dir = os.path.join(OUTPUT_DIR, split)
    pattern = os.path.join(output_dir, f"{case_name}_t*.npy")
    existing_files = glob.glob(pattern)
    return len(existing_files) > 0


def process_case(wrf_file, static_tensor, landmask, split='train'):
    """
    处理单个WRF案例
    
    Parameters:
    -----------
    wrf_file : str
        WRF文件路径
    static_tensor : np.array
        静态数据
    landmask : np.array
        陆地掩膜
    split : str
        数据集划分：'train', 'valid', 'test'
    """
    case_name = os.path.basename(wrf_file).replace('.nc', '').replace('_WRF_3km', '')
    split_names = {'train': '训练集', 'valid': '验证集', 'test': '测试集'}
    print("\n" + "=" * 60)
    print(f"处理案例: {case_name} ({split_names.get(split, split)})")
    print("=" * 60)
    
    # 检查是否已完成
    if check_case_completed(case_name, split):
        existing_count = len(glob.glob(os.path.join(OUTPUT_DIR, split, f"{case_name}_t*.npy")))
        print(f"  ⏭️  跳过已完成案例: {case_name} (已有 {existing_count} 个patches)")
        return
    
    # 加载WRF数据（传入static_tensor进行维度验证）
    ds_wrf = load_wrf_data(wrf_file, static_tensor=static_tensor)
    
    # 获取WRF网格
    wrf_lat = ds_wrf.lat.values
    wrf_lon = ds_wrf.lon.values
    
    # 获取时间（已统一格式并取整到小时）
    import pandas as pd
    wrf_times = pd.to_datetime(ds_wrf.time.values)
    
    # 查找对应的ERA5文件
    era5_dict = find_era5_files(ERA5_DIR, wrf_times)
    
    if not era5_dict:
        print(f"  跳过 {case_name}: 未找到对应的ERA5数据")
        ds_wrf.close()
        return
    
    # 按时间处理（优化：按ERA5文件分组，避免重复打开文件）
    # 内存优化：按ERA5文件分批保存，避免全量累积导致OOM
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)
    patch_offset = 0  # 全局patch索引，用于文件命名
    case_stats = {
        'total_attempted': 0,
        'skipped_nan_input': 0,
        'skipped_nan_label': 0,
        'skipped_nan_both': 0,
        'kept': 0
    }
    
    # 按ERA5文件分组时间
    file_time_groups = {}
    for t_idx, wrf_time in enumerate(wrf_times):
        if wrf_time not in era5_dict:
            continue
        era5_file = era5_dict[wrf_time]
        if era5_file not in file_time_groups:
            file_time_groups[era5_file] = []
        file_time_groups[era5_file].append((t_idx, wrf_time))
    
    # 按文件处理（每个文件只打开一次）
    for era5_file, time_list in tqdm(file_time_groups.items(), desc=f"处理 {case_name}"):
        nan_warned_vars = set()  # 本文件内已警告过的变量，避免重复打印
        try:
            # 一次性加载整个ERA5文件
            import pandas as pd
            from scipy.interpolate import griddata
            
            print(f"  加载ERA5文件: {os.path.basename(era5_file)}")
            ds_era5 = xr.open_dataset(era5_file)
            
            # 处理expver和number维度
            if 'expver' in ds_era5.dims:
                ds_era5 = ds_era5.sel(expver=1)
            if 'number' in ds_era5.dims:
                ds_era5 = ds_era5.sel(number=0)
            
            # 重命名valid_time为time
            if 'valid_time' in ds_era5.coords and 'time' not in ds_era5.coords:
                ds_era5 = ds_era5.rename({'valid_time': 'time'})
            
            # 统一时间格式：转换为pandas时间并取整到小时
            ds_era5['time'] = pd.to_datetime(ds_era5.time.values).round('H')
            
            # 获取ERA5的经纬度
            if 'latitude' in ds_era5.coords:
                era5_lat = ds_era5.latitude.values
                era5_lon = ds_era5.longitude.values
            elif 'lat' in ds_era5.coords:
                era5_lat = ds_era5.lat.values
                era5_lon = ds_era5.lon.values
            else:
                print(f"    警告: 无法找到ERA5经纬度坐标")
                ds_era5.close()
                continue
            
            # 如果是1D，创建2D网格
            if era5_lat.ndim == 1:
                era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)
            else:
                era5_lat_2d = era5_lat
                era5_lon_2d = era5_lon
            
            # 验证ERA5和WRF的覆盖范围
            era5_lat_min, era5_lat_max = era5_lat_2d.min(), era5_lat_2d.max()
            era5_lon_min, era5_lon_max = era5_lon_2d.min(), era5_lon_2d.max()
            wrf_lat_min, wrf_lat_max = wrf_lat.min(), wrf_lat.max()
            wrf_lon_min, wrf_lon_max = wrf_lon.min(), wrf_lon.max()
            
            # 检查ERA5是否完全覆盖WRF范围
            if (wrf_lat_min < era5_lat_min or wrf_lat_max > era5_lat_max or
                wrf_lon_min < era5_lon_min or wrf_lon_max > era5_lon_max):
                print(f"    ⚠️  警告: ERA5覆盖范围可能不足")
                print(f"      ERA5: lat [{era5_lat_min:.2f}, {era5_lat_max:.2f}], lon [{era5_lon_min:.2f}, {era5_lon_max:.2f}]")
                print(f"      WRF:  lat [{wrf_lat_min:.2f}, {wrf_lat_max:.2f}], lon [{wrf_lon_min:.2f}, {wrf_lon_max:.2f}]")
            else:
                print(f"    ✅ ERA5覆盖范围验证通过")
                print(f"      ERA5: lat [{era5_lat_min:.2f}, {era5_lat_max:.2f}], lon [{era5_lon_min:.2f}, {era5_lon_max:.2f}]")
                print(f"      WRF:  lat [{wrf_lat_min:.2f}, {wrf_lat_max:.2f}], lon [{wrf_lon_min:.2f}, {wrf_lon_max:.2f}]")
            
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
            
            # 准备插值点
            era5_points = np.column_stack([era5_lon_2d.flatten(), era5_lat_2d.flatten()])
            wrf_points = np.column_stack([wrf_lon.flatten(), wrf_lat.flatten()])
            
            # 处理每个时间步
            for t_idx, wrf_time in time_list:
                try:
                    # 选择最近的时间（时间已统一格式并取整到小时，直接索引）
                    target_time = pd.to_datetime(wrf_time).round('H')
                    try:
                        ds_sel = ds_era5.sel(time=target_time, method='nearest')
                        # 检查时间差（如果超过6小时则警告）
                        actual_time = pd.to_datetime(ds_sel.time.values).round('H')
                        time_diff_hours = abs((actual_time - target_time).total_seconds() / 3600.0)
                        if time_diff_hours > 6:
                            print(f"    警告: 时间差较大 ({time_diff_hours:.1f} 小时)")
                    except (KeyError, ValueError, TypeError) as e:
                        # 如果仍然失败，给出更详细的错误信息
                        print(f"    错误: 无法匹配时间 {target_time}，可用时间范围: {ds_era5.time.values[0]} 到 {ds_era5.time.values[-1]}")
                        continue
                    
                    # 提取并插值变量
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
                                var_2d = var_2d / 100.0
                            
                            # 插值（三层策略：linear -> nearest -> global mean）
                            var_values = var_2d.flatten()
                            
                            # 第一层：线性插值
                            var_interp = griddata(
                                era5_points, var_values, wrf_points,
                                method='linear', fill_value=np.nan
                            )
                            var_interp = var_interp.reshape(wrf_lat.shape)
                            
                            # 第二层：最近邻填充NaN
                            if np.any(np.isnan(var_interp)):
                                var_interp_nn = griddata(
                                    era5_points, var_values, wrf_points,
                                    method='nearest'
                                )
                                var_interp_nn = var_interp_nn.reshape(wrf_lat.shape)
                                var_interp[np.isnan(var_interp)] = var_interp_nn[np.isnan(var_interp)]
                            
                            # 第三层：如果仍有NaN，使用全局均值填充
                            if np.any(np.isnan(var_interp)):
                                global_mean = np.nanmean(var_2d)
                                if np.isnan(global_mean):
                                    global_mean = 0.0  # 如果全局均值也是NaN，使用0
                                var_interp[np.isnan(var_interp)] = global_mean
                                if wrf_var not in nan_warned_vars:
                                    nan_warned_vars.add(wrf_var)
                                    print(f"      警告: {wrf_var}有NaN，使用全局均值 {global_mean:.2f} 填充（本文件后续不再重复提示）")
                            
                            # 最终验证：确保没有NaN
                            if np.any(np.isnan(var_interp)):
                                raise ValueError(f"{wrf_var}插值后仍存在NaN，无法继续")
                            
                            era5_vars[wrf_var] = var_interp
                    
                    if not era5_vars:
                        continue
                    
                    # 提取当前时间的WRF数据
                    wrf_t = ds_wrf.isel(time=t_idx)
                    
                    # 构建输入数据（ERA5变量 + 静态变量）
                    input_vars = []
                    
                    # ERA5变量（按顺序：U10, V10, SLP, T2, SST, BLH, FSR）
                    era5_order = ['U10', 'V10', 'SLP', 'T2', 'SST', 'BLH', 'FSR']
                    for var in era5_order:
                        if var in era5_vars:
                            var_data = era5_vars[var]
                            var_norm = normalize(var_data, var)
                            input_vars.append(var_norm)
                    
                    # 静态变量（已经在static_tensor中归一化）
                    static_expanded = np.expand_dims(static_tensor, axis=0)  # (1, H, W, C)
                    
                    # 堆叠ERA5变量
                    if input_vars:
                        era5_stack = np.stack(input_vars, axis=-1)  # (H, W, C_era5)
                        era5_stack = np.expand_dims(era5_stack, axis=0)  # (1, H, W, C_era5)
                        
                        # 合并ERA5和静态数据
                        full_input = np.concatenate([era5_stack, static_expanded], axis=-1)  # (1, H, W, C_total)
                    else:
                        full_input = static_expanded
                    
                    # 构建标签数据（WRF变量，去掉RAIN）
                    label_vars = []
                    label_order = ['U10', 'V10', 'SLP', 'T2']  # 去掉RAIN
                    for var in label_order:
                        if var in wrf_t.data_vars:
                            var_data = wrf_t[var].values
                            var_norm = normalize(var_data, var)
                            label_vars.append(var_norm)
                    
                    if not label_vars:
                        continue
                    
                    label_stack = np.stack(label_vars, axis=-1)  # (H, W, C_label)
                    label_stack = np.expand_dims(label_stack, axis=0)  # (1, H, W, C_label)
                    
                    # 创建patches
                    patches, stats = create_patches(
                        full_input, label_stack, static_tensor, landmask,
                        PATCH_SIZE, STRIDE
                    )
                    
                    # 累计统计
                    for key in case_stats:
                        case_stats[key] += stats[key]
                    
                    # 按时间步立即保存，避免内存累积（单个ERA5文件可能含数百时间步，会导致OOM）
                    if patches:
                        for i, patch in enumerate(patches):
                            patch_file = os.path.join(output_dir, f"{case_name}_t{patch_offset + i:06d}.npy")
                            np.save(patch_file, patch)
                        patch_offset += len(patches)
                        # 每5000个或每批>500个时打印进度
                        if patch_offset % 5000 < len(patches) or len(patches) >= 500:
                            print(f"    已保存 {len(patches)} 个patches (累计 {patch_offset})", flush=True)
                    del patches  # 立即释放
                    
                except Exception as e:
                    print(f"    警告: 处理时间 {wrf_time} 时出错: {e}", flush=True)
                    continue
            
            ds_era5.close()
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  警告: 处理文件 {os.path.basename(era5_file)} 时出错: {e}")
            continue
    
    ds_wrf.close()
    
    # 打印NaN过滤统计
    print("\n" + "-" * 60)
    print(f"NaN过滤统计 ({case_name}):")
    print(f"  总尝试patches: {case_stats['total_attempted']}")
    print(f"  跳过(输入NaN): {case_stats['skipped_nan_input']} ({100*case_stats['skipped_nan_input']/max(case_stats['total_attempted'],1):.1f}%)")
    print(f"  跳过(标签NaN): {case_stats['skipped_nan_label']} ({100*case_stats['skipped_nan_label']/max(case_stats['total_attempted'],1):.1f}%)")
    print(f"  跳过(两者NaN): {case_stats['skipped_nan_both']} ({100*case_stats['skipped_nan_both']/max(case_stats['total_attempted'],1):.1f}%)")
    print(f"  保留patches: {case_stats['kept']} ({100*case_stats['kept']/max(case_stats['total_attempted'],1):.1f}%)")
    print("-" * 60)
    
    # 汇总保存结果（patches已分批保存，此处仅打印统计）
    if patch_offset > 0:
        print(f"  ✅ 共保存了 {patch_offset} 个patches")
    else:
        print(f"  ⚠️  未生成任何patches")


def main():
    """主函数"""
    print("=" * 60)
    print("PhyTopo-Net 数据预处理")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)
    
    # 加载静态数据
    static_tensor, landmask = load_static_data(GEO_FILE)
    
    # 获取所有WRF文件
    wrf_files = sorted(glob.glob(os.path.join(WRF_DIR, '*.nc')))
    
    if not wrf_files:
        raise FileNotFoundError(f"未找到WRF文件: {WRF_DIR}")
    
    print(f"\n找到 {len(wrf_files)} 个WRF文件")
    
    # 统计跳过和处理的案例
    skipped_count = 0
    processed_count = 0
    
    # 处理每个案例
    for wrf_file in wrf_files:
        case_name = os.path.basename(wrf_file)
        
        # 判断是训练集、验证集还是测试集
        is_train = any(case in case_name for case in TRAIN_CASES)
        is_valid = any(case in case_name for case in VALID_CASES)
        is_test = any(case in case_name for case in TEST_CASES)
        
        if is_train:
            split = 'train'
        elif is_valid:
            split = 'valid'
        elif is_test:
            split = 'test'
        else:
            print(f"\n跳过未知案例: {case_name}")
            continue
        
        # 检查是否已完成
        case_name_short = os.path.basename(wrf_file).replace('.nc', '').replace('_WRF_3km', '')
        if check_case_completed(case_name_short, split):
            existing_count = len(glob.glob(os.path.join(OUTPUT_DIR, split, f"{case_name_short}_t*.npy")))
            print(f"\n⏭️  跳过已完成案例: {case_name_short} ({split}, 已有 {existing_count} 个patches)")
            skipped_count += 1
            continue
        
        processed_count += 1
        process_case(wrf_file, static_tensor, landmask, split=split)
    
    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)
    
    # 统计
    print(f"\n处理统计:")
    print(f"  跳过已完成案例: {skipped_count} 个")
    print(f"  新处理案例: {processed_count} 个")
    
    train_files = glob.glob(os.path.join(OUTPUT_DIR, 'train', '*.npy'))
    valid_files = glob.glob(os.path.join(OUTPUT_DIR, 'valid', '*.npy'))
    test_files = glob.glob(os.path.join(OUTPUT_DIR, 'test', '*.npy'))
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_files)} 个patches")
    print(f"  验证集: {len(valid_files)} 个patches")
    print(f"  测试集: {len(test_files)} 个patches")


if __name__ == "__main__":
    import sys
    import argparse
    # 禁用输出缓冲，确保日志实时写入
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    parser = argparse.ArgumentParser(description='PhyTopo-Net 数据集生成')
    parser.add_argument('--wrf_dir', default=None, help='WRF 数据目录')
    parser.add_argument('--era5_dir', default=None, help='ERA5 数据目录')
    parser.add_argument('--geo_file', default=None, help='静态地理文件路径')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    args = parser.parse_args()
    
    if args.wrf_dir:
        WRF_DIR = args.wrf_dir
    if args.era5_dir:
        ERA5_DIR = args.era5_dir
    if args.geo_file:
        GEO_FILE = args.geo_file
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    
    main()

