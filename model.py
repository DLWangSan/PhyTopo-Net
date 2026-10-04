"""
PhyTopo-Net模型架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """标准ResNet残差块"""
    
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    Query: Meteorology features
    Key & Value: Topography features
    """
    
    def __init__(self, channels=64):
        super(CrossAttentionFusion, self).__init__()
        self.channels = channels
        
        # Query projection (from Met features)
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Key & Value projection (from Geo features)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, feat_met, feat_geo):
        """
        Parameters:
        -----------
        feat_met : torch.Tensor
            气象特征 (B, C, H, W)
        feat_geo : torch.Tensor
            地形特征 (B, C, H, W)
        
        Returns:
        --------
        fused_feat : torch.Tensor
            融合后的特征 (B, C, H, W)
        """
        B, C, H, W = feat_met.shape
        
        # Project to Query, Key, Value
        Q = self.query_conv(feat_met)  # (B, C, H, W)
        K = self.key_conv(feat_geo)    # (B, C, H, W)
        V = self.value_conv(feat_geo)  # (B, C, H, W)
        
        # Reshape for attention computation
        Q = Q.view(B, C, H * W)  # (B, C, HW)
        K = K.view(B, C, H * W)  # (B, C, HW)
        V = V.view(B, C, H * W)  # (B, C, HW)
        
        # Compute attention scores
        # Attention = Q @ K^T / sqrt(C)
        attention = torch.bmm(Q.transpose(1, 2), K)  # (B, HW, HW)
        attention = attention / (C ** 0.5)
        attention = self.softmax(attention)
        
        # Apply attention to Value
        attended = torch.bmm(V, attention.transpose(1, 2))  # (B, C, HW)
        attended = attended.view(B, C, H, W)  # (B, C, H, W)
        
        # Project output
        attended = self.out_conv(attended)
        
        # Residual connection: add back to Met features
        fused_feat = feat_met + attended
        
        return fused_feat


class PhyTopoNet(nn.Module):
    """
    PhyTopo-Net: 使用精细地形引导优化粗分辨率ERA5风场
    
    residual: 若为 True，输出 = input + 网络预测的残差，即学习 WRF - ERA5 的修正量
              若为 False（默认），输出 = 网络预测的全场，与阶段一保持一致，便于复现
    """
    
    def __init__(self, residual=False):
        super(PhyTopoNet, self).__init__()
        self.residual = residual
        
        # ========== Dual Encoders ==========
        # Meteorology Encoder
        self.met_encoder = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Topography Encoder
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # ========== Cross-Attention Fusion ==========
        self.fusion = CrossAttentionFusion(channels=64)
        
        # ========== Backbone (ResNet Blocks) ==========
        self.backbone = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        
        # ========== Dual-Head Decoder ==========
        # Head Wind: Outputs U, V, T (3 channels, removed RAIN)
        self.head_wind = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)  # Output: U, V, T
        )
        
        # Head Pressure: Outputs SLP (1 channel)
        self.head_press = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # Output: SLP
        )
    
    def forward(self, x_met, x_geo):
        """
        Forward pass
        
        Parameters:
        -----------
        x_met : torch.Tensor
            气象输入 (B, 7, 64, 64) - ERA5变量
        x_geo : torch.Tensor
            地形输入 (B, 5, 64, 64) - 静态地形变量
        
        Returns:
        --------
        output : torch.Tensor
            输出 (B, 4, 64, 64) - [U, V, SLP, T]
        """
        # Dual Encoders
        feat_met = self.met_encoder(x_met)  # (B, 64, 64, 64)
        feat_geo = self.geo_encoder(x_geo)  # (B, 64, 64, 64)
        
        # Cross-Attention Fusion
        fused_feat = self.fusion(feat_met, feat_geo)  # (B, 64, 64, 64)
        
        # Backbone Processing
        features = self.backbone(fused_feat)  # (B, 64, 64, 64)
        
        # Dual-Head Decoder
        out_wind = self.head_wind(features)  # (B, 3, 64, 64) - [U, V, T]
        out_press = self.head_press(features)  # (B, 1, 64, 64) - [SLP]
        
        # Reassemble output to match label order: [U, V, SLP, T]
        u = out_wind[:, 0:1, :, :]      # (B, 1, 64, 64)
        v = out_wind[:, 1:2, :, :]      # (B, 1, 64, 64)
        t = out_wind[:, 2:3, :, :]      # (B, 1, 64, 64)
        
        # Concatenate: [U, V, SLP, T]
        raw_out = torch.cat([u, v, out_press, t], dim=1)  # (B, 4, 64, 64)
        
        if self.residual:
            # 残差学习：output = input + residual，学习 WRF - ERA5 的修正量
            final_out = x_met[:, :4] + raw_out
        else:
            final_out = raw_out
        
        return final_out

