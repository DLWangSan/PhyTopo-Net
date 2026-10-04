"""
损失函数模块：实现像素损失、梯度损失和频谱损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelLoss(nn.Module):
    """像素级L1损失（只计算U, V, SLP通道）"""
    
    def __init__(self, channels=[0, 1, 2]):
        """
        Parameters:
        -----------
        channels : list
            要计算损失的通道索引，默认[0, 1, 2]对应U, V, SLP
        """
        super(PixelLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.channels = channels
    
    def forward(self, pred, target):
        """
        Parameters:
        -----------
        pred : torch.Tensor
            预测值 (B, C, H, W)
        target : torch.Tensor
            真实值 (B, C, H, W)
        
        Returns:
        --------
        loss : torch.Tensor
            像素损失（只对指定通道计算）
        """
        # 只提取U, V, SLP通道
        pred_selected = pred[:, self.channels, :, :]
        target_selected = target[:, self.channels, :, :]
        
        return self.criterion(pred_selected, target_selected)


class GradientLoss(nn.Module):
    """
    梯度损失：使用Sobel算子计算梯度，主要应用于U, V, SLP通道
    """
    
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
        # 定义Sobel算子（x方向）
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        
        # 定义Sobel算子（y方向）
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # 注册为buffer（不参与梯度更新，会自动移动到正确的设备）
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_gradient(self, x):
        """
        计算输入张量的梯度
        
        Parameters:
        -----------
        x : torch.Tensor
            输入 (B, C, H, W)
        
        Returns:
        --------
        grad : torch.Tensor
            梯度幅值 (B, C, H, W)
        """
        # 确保Sobel算子在正确的设备上
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        
        # 对每个通道分别计算梯度
        B, C, H, W = x.shape
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        
        for c in range(C):
            # x方向梯度
            grad_x[:, c:c+1, :, :] = F.conv2d(
                x[:, c:c+1, :, :],
                sobel_x,
                padding=1
            )
            # y方向梯度
            grad_y[:, c:c+1, :, :] = F.conv2d(
                x[:, c:c+1, :, :],
                sobel_y,
                padding=1
            )
        
        # 梯度幅值
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return grad
    
    def forward(self, pred, target, channels=[0, 1, 2]):
        """
        计算梯度损失（主要应用于U, V, SLP通道）
        
        Parameters:
        -----------
        pred : torch.Tensor
            预测值 (B, C, H, W)
        target : torch.Tensor
            真实值 (B, C, H, W)
        channels : list
            要计算梯度损失的通道索引 (默认: [0, 1, 2] 对应 U, V, SLP)
        
        Returns:
        --------
        loss : torch.Tensor
            梯度损失
        """
        # 提取指定通道
        pred_grad = self.compute_gradient(pred[:, channels, :, :])
        target_grad = self.compute_gradient(target[:, channels, :, :])
        
        return self.criterion(pred_grad, target_grad)


class SpectralLoss(nn.Module):
    """
    频谱损失：在频域计算功率谱密度(PSD)的差异（只计算U, V, SLP通道）
    """
    
    def __init__(self, eps=1e-8, channels=[0, 1, 2]):
        super(SpectralLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.eps = eps
        self.channels = channels  # 只计算U, V, SLP通道
    
    def compute_psd(self, x):
        """
        计算功率谱密度
        
        Parameters:
        -----------
        x : torch.Tensor
            输入 (B, C, H, W)
        
        Returns:
        --------
        psd : torch.Tensor
            功率谱密度 (B, C, H, W//2+1)
        """
        # 2D FFT (只计算实数FFT)
        fft = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1)
        
        # 计算功率谱密度 (magnitude squared)
        psd = torch.abs(fft) ** 2
        
        return psd
    
    def forward(self, pred, target):
        """
        计算频谱损失（只对U, V, SLP通道）
        
        Parameters:
        -----------
        pred : torch.Tensor
            预测值 (B, C, H, W)
        target : torch.Tensor
            真实值 (B, C, H, W)
        
        Returns:
        --------
        loss : torch.Tensor
            频谱损失（只对指定通道计算）
        """
        # 只提取U, V, SLP通道
        pred_selected = pred[:, self.channels, :, :]
        target_selected = target[:, self.channels, :, :]
        
        # 计算PSD
        psd_pred = self.compute_psd(pred_selected)
        psd_target = self.compute_psd(target_selected)
        
        # 对数空间计算损失（避免数值不稳定）
        log_psd_pred = torch.log(psd_pred + self.eps)
        log_psd_target = torch.log(psd_target + self.eps)
        
        return self.criterion(log_psd_pred, log_psd_target)


class CycleConsistencyLoss(nn.Module):
    """
    循环一致性损失：确保高分辨率预测降采样后与输入的低分辨率ERA5一致
    只对风场(U, V)和气压(SLP)计算，对应通道索引[0, 1, 2]
    """
    
    def __init__(self, pool_size=8):
        """
        Parameters:
        -----------
        pool_size : int
            平均池化核大小，用于将3km高分辨率降采样到25km低分辨率
            64x64 -> 8x8，对应约8倍降采样
        """
        super(CycleConsistencyLoss, self).__init__()
        self.pool_size = pool_size
        self.criterion = nn.L1Loss()
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, pred_high_res, input_low_res):
        """
        计算循环一致性损失
        
        Parameters:
        -----------
        pred_high_res : torch.Tensor
            模型生成的高分辨率预测 (B, 5, H, W) - [U, V, SLP, T, RAIN]
        input_low_res : torch.Tensor
            输入的ERA5低分辨率数据 (B, 7, H, W) - [U, V, SLP, T, SST, BLH, FSR]
            注意：input_low_res已经是插值到高分辨率的，但物理上代表低分辨率信息
        
        Returns:
        --------
        loss : torch.Tensor
            循环一致性损失（只对U, V, SLP通道计算）
        """
        # 提取预测中的风场和气压 (通道0, 1, 2: U, V, SLP)
        pred_wind_press = pred_high_res[:, [0, 1, 2], :, :]  # (B, 3, H, W)
        
        # 提取输入中的ERA5风场和气压 (通道0, 1, 2: U, V, SLP)
        input_wind_press = input_low_res[:, [0, 1, 2], :, :]  # (B, 3, H, W)
        
        # 对高分辨率预测进行平均池化降采样
        pred_downsampled = self.avg_pool(pred_wind_press)  # (B, 3, H//pool_size, W//pool_size)
        
        # 对输入也进行相同的降采样（确保空间维度匹配）
        input_downsampled = self.avg_pool(input_wind_press)  # (B, 3, H//pool_size, W//pool_size)
        
        # 计算L1损失
        loss = self.criterion(pred_downsampled, input_downsampled)
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数：像素损失 + 梯度损失 + 频谱损失
    只对U, V, SLP通道计算损失
    """
    
    def __init__(self, w_pixel=1.0, w_grad=0.5, w_spec=0.1, target_channels=[0, 1, 2]):
        super(CombinedLoss, self).__init__()
        self.w_pixel = w_pixel
        self.w_grad = w_grad
        self.w_spec = w_spec
        self.target_channels = target_channels  # [0, 1, 2] 对应 U, V, SLP
        
        # 初始化损失函数，传入目标通道
        self.pixel_loss = PixelLoss(channels=target_channels)
        self.grad_loss = GradientLoss()
        self.spec_loss = SpectralLoss(channels=target_channels)
    
    def forward(self, pred, target):
        """
        计算组合损失（只对U, V, SLP通道）
        
        Parameters:
        -----------
        pred : torch.Tensor
            预测值 (B, 5, H, W) - [U, V, SLP, T, RAIN]
        target : torch.Tensor
            真实值 (B, 5, H, W) - [U, V, SLP, T, RAIN]
        
        Returns:
        --------
        total_loss : torch.Tensor
            总损失
        loss_dict : dict
            各项损失的字典
        """
        # 像素损失（只计算U, V, SLP通道）
        l_pixel = self.pixel_loss(pred, target)
        
        # 梯度损失（U, V, SLP通道：索引0, 1, 2）
        l_grad = self.grad_loss(pred, target, channels=self.target_channels)
        
        # 频谱损失（只计算U, V, SLP通道）
        l_spec = self.spec_loss(pred, target)
        
        # 组合损失
        total_loss = self.w_pixel * l_pixel + self.w_grad * l_grad + self.w_spec * l_spec
        
        loss_dict = {
            'total': total_loss.item(),
            'pixel': l_pixel.item(),
            'grad': l_grad.item(),
            'spec': l_spec.item()
        }
        
        return total_loss, loss_dict

