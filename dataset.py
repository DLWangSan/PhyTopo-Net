"""
数据加载模块：处理.npy格式的数据集
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


# 旧的归一化范围（数据集生成时使用的）
OLD_NORM_STATS = {
    'U10': (-65.0, 65.0),
    'V10': (-65.0, 65.0),
    'SLP': (920.0, 1055.0),
    'T2': (250.0, 330.0),
    'RAIN': (0.0, 2200.0),  # 旧的范围太大
    'HGT': (0.0, 3000.0),
    'SLOPE': (0.0, 45.0),
    'LAT': (14.0, 28.0),
    'LON': (105.0, 125.0),
    'BLH': (0.0, 3000.0),
    'FSR': (0.0, 1.0),
    'SST': (270.0, 310.0),
}

# 新的归一化范围（修正后的）
NEW_NORM_STATS = {
    'U10': (-65.0, 65.0),
    'V10': (-65.0, 65.0),
    'SLP': (920.0, 1055.0),
    'T2': (250.0, 330.0),
    'RAIN': (0.0, 150.0),  # 修正：缩小范围到小时雨量
    'HGT': (0.0, 3000.0),
    'SLOPE': (0.0, 45.0),
    'LAT': (14.0, 28.0),
    'LON': (105.0, 125.0),
    'BLH': (0.0, 3000.0),
    'FSR': (0.0, 1.0),
    'SST': (270.0, 310.0),
}

# Label通道顺序（新数据集已去掉RAIN，仅4通道）
LABEL_VAR_NAMES = ['U10', 'V10', 'SLP', 'T2', 'RAIN']


def renormalize_label(label_tensor, old_stats, new_stats, var_names):
    """
    重新归一化label数据
    
    参数:
    - label_tensor: (C, H, W) 已经归一化到[0,1]的数据
    - old_stats: 旧的归一化范围
    - new_stats: 新的归一化范围
    - var_names: 变量名称列表
    """
    label_renorm = label_tensor.clone()
    
    for ch_idx, var_name in enumerate(var_names):
        if ch_idx >= label_tensor.shape[0]:
            break  # 新数据集只有4通道(U10,V10,SLP,T2)，无RAIN
        if var_name not in old_stats or var_name not in new_stats:
            continue
        
        # 1. 反归一化（用旧范围）
        old_min, old_max = old_stats[var_name]
        old_data = label_tensor[ch_idx] * (old_max - old_min) + old_min
        
        # 2. 重新归一化（用新范围）
        new_min, new_max = new_stats[var_name]
        new_data = (old_data - new_min) / (new_max - new_min)
        new_data = torch.clamp(new_data, 0.0, 1.0)
        
        label_renorm[ch_idx] = new_data
    
    return label_renorm


class WindDataset(Dataset):
    """
    风速场数据集类
    
    数据格式：
    - Input: (64, 64, 12) - ERA5 + Static Topography
        - Channels 0-6: ERA5 (U10, V10, SLP, T2, SST, BLH, FSR)
        - Channels 7-11: Static (HGT, SLOPE, LAT, LON, LANDMASK)
    - Label: (64, 64, 4) - WRF High-Resolution
        - Channels: U10, V10, SLP, T2 (removed RAIN)
    """
    
    def __init__(self, data_dir, split='train'):
        """
        初始化数据集
        
        Parameters:
        -----------
        data_dir : str
            数据集根目录路径
        split : str
            数据集划分 ('train', 'valid', 'test')
        """
        self.data_dir = data_dir
        self.split = split
        
        # 获取所有.npy文件路径
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"数据集目录不存在: {split_dir}")
        
        self.file_list = sorted(glob.glob(os.path.join(split_dir, '*.npy')))
        
        if len(self.file_list) == 0:
            raise ValueError(f"在 {split_dir} 中未找到.npy文件")
        
        print(f"加载 {split} 数据集: {len(self.file_list)} 个样本")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
        --------
        x_met : torch.Tensor
            气象数据 (7, 64, 64) - ERA5变量
        x_geo : torch.Tensor
            地形数据 (5, 64, 64) - 静态地形变量
        label : torch.Tensor
            标签数据 (4, 64, 64) - WRF高分辨率数据 [U10, V10, SLP, T2]
        """
        # 加载.npy文件
        # 兼容新旧版本numpy的pickle格式
        import pickle
        import sys
        
        # 方法1: 尝试标准numpy.load
        try:
            data = np.load(self.file_list[idx], allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
        except Exception:
            # 方法2: 手动读取numpy文件格式并处理pickle兼容性
            with open(self.file_list[idx], 'rb') as f:
                # 重置文件指针
                f.seek(0)
                # 读取numpy文件头
                magic = f.read(6)
                if magic != b'\x93NUMPY':
                    raise ValueError(f"Not a valid numpy file: {self.file_list[idx]}")
                
                # 重置文件指针并读取版本
                f.seek(0)
                version = np.lib.format.read_magic(f)
                
                # 读取数组头
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
                
                # 现在文件指针在pickle数据开始处
                # 创建一个兼容的Unpickler来处理旧版本numpy的类
                class NumpyCompatUnpickler(pickle.Unpickler):
                    def __init__(self, file, **kwargs):
                        # Python 3.8+需要encoding参数
                        if sys.version_info >= (3, 8):
                            kwargs.setdefault('encoding', 'latin1')
                            kwargs.setdefault('errors', 'ignore')
                        super().__init__(file, **kwargs)
                    
                    def find_class(self, module, name):
                        # 处理numpy旧版本的类映射
                        if module == 'numpy.core.multiarray':
                            if name == 'scalar':
                                return np.generic
                            elif name == '_reconstruct':
                                # 使用numpy的真正_reconstruct函数
                                try:
                                    from numpy.core.multiarray import _reconstruct
                                    return _reconstruct
                                except ImportError:
                                    # 如果导入失败，创建一个兼容版本
                                    def _reconstruct(subtype, shape, dtype):
                                        if not issubclass(subtype, np.ndarray):
                                            # 确保subtype是ndarray的子类
                                            subtype = np.ndarray
                                        return np.ndarray.__new__(subtype, shape, dtype)
                                    return _reconstruct
                        
                        # 尝试标准导入
                        try:
                            return super().find_class(module, name)
                        except (AttributeError, ModuleNotFoundError, ImportError, TypeError):
                            # 如果失败，尝试numpy的兼容映射
                            if 'numpy' in module:
                                # 尝试从numpy直接导入
                                try:
                                    mod = __import__('numpy', fromlist=[name])
                                    return getattr(mod, name)
                                except:
                                    pass
                            # 最后尝试：如果是_reconstruct，返回一个兼容函数
                            if name == '_reconstruct':
                                def _reconstruct(subtype, shape, dtype):
                                    if not isinstance(subtype, type) or not issubclass(subtype, np.ndarray):
                                        subtype = np.ndarray
                                    return np.ndarray.__new__(subtype, shape, dtype)
                                return _reconstruct
                            raise
                
                # 使用兼容的Unpickler加载数据
                unpickler = NumpyCompatUnpickler(f)
                data = unpickler.load()
                
                # 处理numpy object数组
                if isinstance(data, np.ndarray) and data.dtype == object:
                    data = data.item()
        
        # 提取input和label
        input_data = data['input']  # (64, 64, C) where C should be 12
        label_data = data['label']   # (64, 64, 5)
        
        # 检查并处理通道数不一致的情况
        input_channels = input_data.shape[2] if len(input_data.shape) == 3 else input_data.shape[-1]
        
        # 如果通道数不足12，进行填充（用零填充最后一个通道）
        if input_channels < 12:
            # 创建12通道的数组
            H, W = input_data.shape[0], input_data.shape[1]
            input_data_padded = np.zeros((H, W, 12), dtype=input_data.dtype)
            input_data_padded[:, :, :input_channels] = input_data
            input_data = input_data_padded
        
        # 转换为torch.FloatTensor（使用clone和contiguous确保不共享内存，避免多进程DataLoader错误）
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data.copy()).float()  # (64, 64, 12)
        else:
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        if isinstance(label_data, np.ndarray):
            label_tensor = torch.from_numpy(label_data.copy()).float()  # (64, 64, 5)
        else:
            label_tensor = torch.tensor(label_data, dtype=torch.float32)
        
        # Permute: (H, W, C) -> (C, H, W)
        input_tensor = input_tensor.permute(2, 0, 1).contiguous()  # (12, 64, 64)
        label_tensor = label_tensor.permute(2, 0, 1).contiguous()  # (5, 64, 64)
        
        # 陆地上SST不参与计算：LANDMASK=1为陆地，将SST通道置0
        landmask_ch = input_tensor[11, :, :]  # LANDMASK在通道11
        input_tensor[4, :, :][landmask_ch > 0.5] = 0.0  # SST在通道4，陆地置0
        
        # 通道分割 (Physics-Aware)
        # Stream A (Meteorology): Channels 0-6
        # ERA5变量: U10, V10, SLP, T2, SST, BLH, FSR
        x_met = input_tensor[0:7, :, :]  # (7, 64, 64)
        
        # Stream B (Topography): Channels 7-11
        # Static变量: HGT, SLOPE, LAT, LON, LANDMASK
        x_geo = input_tensor[7:12, :, :]  # (5, 64, 64)
        
        # 重新归一化label（修正RAIN的归一化范围）
        label_tensor = renormalize_label(
            label_tensor, 
            OLD_NORM_STATS, 
            NEW_NORM_STATS, 
            LABEL_VAR_NAMES
        )
        
        return x_met, x_geo, label_tensor

