"""
评估指标模块：计算R²、Bias、CC、SSIM等指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr


# 归一化参数（已修正，与dataset.py中的NEW_NORM_STATS保持一致）
NORM_STATS = {
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

# 变量名称（按通道顺序，新数据集已去掉RAIN，仅4通道）
VAR_NAMES = ['U10', 'V10', 'SLP', 'T2']


def denormalize(data_norm, var_name):
    """
    反归一化
    
    Parameters:
    -----------
    data_norm : torch.Tensor or np.ndarray
        归一化后的数据 [0, 1]
    var_name : str
        变量名称
    
    Returns:
    --------
    data : torch.Tensor or np.ndarray
        反归一化后的数据
    """
    if var_name not in NORM_STATS:
        raise ValueError(f"未知变量: {var_name}")
    
    vmin, vmax = NORM_STATS[var_name]
    
    if isinstance(data_norm, torch.Tensor):
        return data_norm * (vmax - vmin) + vmin
    else:
        return data_norm * (vmax - vmin) + vmin


def calculate_r2(pred, target):
    """
    计算R² (R-Squared)
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W) 或 (N,)
    target : torch.Tensor
        真实值 (B, C, H, W) 或 (N,)
    
    Returns:
    --------
    r2 : float
        R²值
    """
    # 展平并过滤 NaN/Inf
    pred_flat = pred.flatten().cpu().numpy().astype(np.float64)
    target_flat = target.flatten().cpu().numpy().astype(np.float64)
    valid = np.isfinite(pred_flat) & np.isfinite(target_flat)
    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]
    
    if len(pred_flat) < 2:
        return 0.0
    
    # 计算R²
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def calculate_bias(pred, target, var_name):
    """
    计算Bias (Mean Error)，需要反归一化
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W) 或 (N,)
    target : torch.Tensor
        真实值 (B, C, H, W) 或 (N,)
    var_name : str
        变量名称，用于反归一化
    
    Returns:
    --------
    bias : float
        Bias值（物理单位）
    """
    # 反归一化
    pred_denorm = denormalize(pred, var_name)
    target_denorm = denormalize(target, var_name)
    
    # 展平并过滤 NaN/Inf
    pred_flat = pred_denorm.flatten().cpu().numpy().astype(np.float64)
    target_flat = target_denorm.flatten().cpu().numpy().astype(np.float64)
    valid = np.isfinite(pred_flat) & np.isfinite(target_flat)
    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]
    
    if len(pred_flat) == 0:
        return 0.0
    
    bias = np.mean(pred_flat - target_flat)
    return float(bias)


def calculate_cc(pred, target):
    """
    计算皮尔逊相关系数 (Correlation Coefficient)
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W) 或 (N,)
    target : torch.Tensor
        真实值 (B, C, H, W) 或 (N,)
    
    Returns:
    --------
    cc : float
        相关系数
    """
    # 展平
    pred_flat = pred.flatten().cpu().numpy().astype(np.float64)
    target_flat = target.flatten().cpu().numpy().astype(np.float64)
    
    # 过滤 NaN/Inf，避免 pearsonr 报错
    valid = np.isfinite(pred_flat) & np.isfinite(target_flat)
    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]
    
    if len(pred_flat) < 2 or len(target_flat) < 2:
        return 0.0
    
    if np.std(pred_flat) == 0 or np.std(target_flat) == 0:
        return 0.0
    
    cc, _ = pearsonr(pred_flat, target_flat)
    return float(cc)


def calculate_ssim(pred, target, window_size=11, size_average=True):
    """
    计算SSIM (Structural Similarity Index)
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W) 或 (C, H, W) 或 (H, W)
    target : torch.Tensor
        真实值 (B, C, H, W) 或 (C, H, W) 或 (H, W)
    window_size : int
        滑动窗口大小
    size_average : bool
        是否对所有通道和批次求平均
    
    Returns:
    --------
    ssim : float
        SSIM值
    """
    # 确保输入是4D张量
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
    
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(0)
    
    # 确保pred和target在同一设备
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # SSIM参数（对于归一化到[0,1]的数据）
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 获取通道数
    B, C, H, W = pred.shape
    
    # 创建平均池化窗口（使用高斯窗口的简化版本）
    # 需要为每个通道创建独立的窗口，使用groups=C
    window = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype) / (window_size ** 2)
    
    # 对每个通道分别计算SSIM，然后求平均
    ssim_channels = []
    for c in range(C):
        pred_ch = pred[:, c:c+1, :, :]  # (B, 1, H, W)
        target_ch = target[:, c:c+1, :, :]  # (B, 1, H, W)
        
        # 计算均值
        mu1 = F.conv2d(pred_ch, window, padding=window_size // 2)
        mu2 = F.conv2d(target_ch, window, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(pred_ch * pred_ch, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target_ch * target_ch, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred_ch * target_ch, window, padding=window_size // 2) - mu1_mu2
        
        # 计算SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / (denominator + 1e-8)  # 避免除零
        
        ssim_channels.append(ssim_map)
    
    # 合并所有通道
    ssim_all = torch.cat(ssim_channels, dim=1)  # (B, C, H, W)
    
    if size_average:
        return ssim_all.mean().item()
    else:
        return ssim_all.mean(dim=(1, 2, 3)).cpu().numpy()


def calculate_all_metrics(pred, target, channel_idx, var_name):
    """
    计算所有指标（R², Bias, CC, SSIM）
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W)
    target : torch.Tensor
        真实值 (B, C, H, W)
    channel_idx : int
        通道索引
    var_name : str
        变量名称
    
    Returns:
    --------
    metrics : dict
        包含所有指标的字典
    """
    # 提取指定通道
    pred_ch = pred[:, channel_idx:channel_idx+1, :, :]
    target_ch = target[:, channel_idx:channel_idx+1, :, :]
    
    # 计算各项指标
    r2 = calculate_r2(pred_ch, target_ch)
    bias = calculate_bias(pred_ch, target_ch, var_name)
    cc = calculate_cc(pred_ch, target_ch)
    
    # SSIM计算（如果数据在GPU上，先移到CPU以节省内存）
    if pred_ch.is_cuda:
        pred_ch_cpu = pred_ch.cpu()
        target_ch_cpu = target_ch.cpu()
        ssim = calculate_ssim(pred_ch_cpu, target_ch_cpu)
    else:
        ssim = calculate_ssim(pred_ch, target_ch)
    
    return {
        'r2': r2,
        'bias': bias,
        'cc': cc,
        'ssim': ssim
    }


def calculate_metrics_all_channels(pred, target, target_channels=[0, 1, 2]):
    """
    计算所有通道的指标
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, 4, H, W) - [U, V, SLP, T2]
    target : torch.Tensor
        真实值 (B, 4, H, W) - [U, V, SLP, T2]
    target_channels : list
        用于计算ALL指标的通道索引，默认[0, 1, 2]对应U, V, SLP
    
    Returns:
    --------
    metrics_dict : dict
        包含所有通道指标的字典
    """
    metrics_dict = {}
    n_channels = pred.shape[1]
    
    # 对每个通道计算指标（只计算实际存在的通道）
    for ch_idx, var_name in enumerate(VAR_NAMES):
        if ch_idx >= n_channels:
            break
        metrics = calculate_all_metrics(pred, target, ch_idx, var_name)
        metrics_dict[var_name] = metrics
    
    # 计算整体指标（只使用目标通道：U, V, SLP）
    # 这样可以避免不关心的变量（如T2, RAIN）影响整体指标
    r2_list = []
    bias_list = []
    target_var_names = [VAR_NAMES[i] for i in target_channels]
    
    for var_name in target_var_names:
        if var_name in metrics_dict:
            r2_list.append(metrics_dict[var_name]['r2'])
            bias_list.append(metrics_dict[var_name]['bias'])
    
    # 提取目标通道用于计算CC和SSIM
    pred_target = pred[:, target_channels, :, :]
    target_target = target[:, target_channels, :, :]
    
    metrics_all = {
        'r2': np.mean(r2_list) if r2_list else 0.0,  # 只计算U, V, SLP的平均R²
        'cc': calculate_cc(pred_target, target_target)  # CC只计算U, V, SLP
    }
    
    # SSIM计算（只计算U, V, SLP通道）
    if pred_target.is_cuda:
        pred_cpu = pred_target.cpu()
        target_cpu = target_target.cpu()
        metrics_all['ssim'] = calculate_ssim(pred_cpu, target_cpu)
    else:
        metrics_all['ssim'] = calculate_ssim(pred_target, target_target)
    
    # 计算整体Bias（只计算U, V, SLP的平均值）
    metrics_all['bias'] = np.mean(bias_list) if bias_list else 0.0
    
    metrics_dict['ALL'] = metrics_all
    
    return metrics_dict

