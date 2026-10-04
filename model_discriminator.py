"""
条件判别器：判断「在此地形下，该风场是否合理」
输入：风场 (U, V, SLP) + 地形 (x_geo)
"""
import torch
import torch.nn as nn


class ConditionalDiscriminator(nn.Module):
    """
    PatchGAN 风格条件判别器
    输入: wind_field (B, 3, H, W) [U, V, SLP] + x_geo (B, 5, H, W)
    输出: (B, 1, H', W') 每个 patch 的真/假分数
    """
    
    def __init__(self, in_channels=8):
        """
        in_channels: 3 (U,V,SLP) + 5 (geo) = 8
        """
        super(ConditionalDiscriminator, self).__init__()
        self.in_channels = in_channels
        
        def d_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            d_block(in_channels, 64, norm=False),
            d_block(64, 128),
            d_block(128, 256),
            d_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, wind_field, x_geo):
        """
        wind_field: (B, 3, H, W) - U, V, SLP
        x_geo: (B, 5, H, W) - 地形
        """
        x = torch.cat([wind_field, x_geo], dim=1)
        return self.model(x)
