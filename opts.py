"""
配置参数模块：使用argparse管理训练参数
"""
import argparse
import os


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PhyTopo-Net Training')
    
    # 数据路径
    parser.add_argument('--data_dir', type=str, default='./dataprocess/dataset',
                        help='数据集根目录路径 (包含train/valid/test子目录)')
    
    # 实验设置
    parser.add_argument('--exp_name', type=str, default='phy_topo_net',
                        help='实验名称')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                        help='模型和日志保存目录')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小 (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数 (default: 4)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='是否使用pin_memory加速GPU传输')
    
    # 损失函数权重
    parser.add_argument('--w_pixel', type=float, default=1.0,
                        help='像素损失权重 (default: 1.0)')
    parser.add_argument('--w_grad', type=float, default=0.5,
                        help='梯度损失权重 (default: 0.5)')
    parser.add_argument('--w_spec', type=float, default=0.1,
                        help='频谱损失权重 (default: 0.1)')
    
    # 半监督训练参数（可选）
    parser.add_argument('--unlabeled_data_dir', type=str, default=None,
                        help='无监督数据目录路径 (例如: ./dataprocess/dataset/unlabeled)，None表示纯监督模式')
    parser.add_argument('--batch_size_unlabeled', type=int, default=None,
                        help='无监督数据的Batch Size (默认与batch_size相同)')
    parser.add_argument('--w_cycle', type=float, default=0.1,
                        help='循环一致性损失权重 (default: 0.1，仅在提供unlabeled_data_dir时生效)')
    parser.add_argument('--cycle_pool_size', type=int, default=8,
                        help='循环一致性的平均池化核边长 (仅半监督)。越大越偏重大尺度/低频，默认 8 (64→8 格)；可试 16 (64→4 格)')
    
    # 模型设置
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--finetune', action='store_true',
                        help='微调模式：只加载模型权重，从epoch 0开始，优化器重置')
    parser.add_argument(
        '--model_arch',
        type=str,
        default='phytopo',
        choices=['phytopo', 'phytopo_no_attn', 'unet_concat'],
        help='phytopo: 双编码器+交叉注意力; phytopo_no_attn: 双编码器+通道拼接1×1融合(无注意力); '
             'unet_concat: 经典 U-Net，输入为 ERA5||地形 拼接',
    )
    parser.add_argument('--residual', action='store_true',
                        help='残差学习：output = input + 网络预测，学习 WRF - ERA5 的修正量')
    # Teacher-Guided Physics Loss (可选，不影响原分支)
    parser.add_argument('--physics_guided', action='store_true',
                        help='启用 Teacher-Guided 物理损失：散度/涡度/流入角与WRF一致')
    parser.add_argument('--w_kinematic', type=float, default=0.2,
                        help='运动学损失权重 (散度+涡度)，仅 physics_guided 时生效')
    parser.add_argument('--w_inflow', type=float, default=0.1,
                        help='摩擦流入角损失权重，仅 physics_guided 时生效')
    parser.add_argument('--w_coast', type=float, default=0.1,
                        help='海岸线梯度惩罚权重，仅 physics_guided 时生效')
    parser.add_argument('--physics_scale', type=float, default=1e3,
                        help='散度/涡度量级缩放，避免被 pixel loss 淹没')
    # GAN 对抗损失 (可选，与 physics_guided 可同时使用)
    parser.add_argument('--gan', action='store_true',
                        help='启用 GAN 对抗损失，增强高频纹理')
    parser.add_argument('--w_adv', type=float, default=0.005,
                        help='对抗损失权重，仅 gan 时生效，建议 0.001~0.01')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument(
        '--zero_geo_input',
        action='store_true',
        help='消融：地形支路中 HGT/SLOPE 置零（LAT/LON/LANDMASK 保留；仍走 geo_encoder）',
    )
    
    # 验证和保存
    parser.add_argument('--val_freq', type=int, default=1,
                        help='每N个epoch验证一次 (default: 1)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='每N个epoch保存一次checkpoint (default: 10)')
    
    args = parser.parse_args()
    
    # 创建保存目录
    args.exp_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    return args


