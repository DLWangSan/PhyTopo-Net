"""
PhyTopo-Net训练脚本
"""
import os
import sys
import time
import logging
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv
from datetime import datetime
from itertools import cycle

from opts import parse_args
from dataset import WindDataset
from model import BaselineMetGeoUNet, PhyTopoNet
from loss import CombinedLoss, CycleConsistencyLoss
from loss_physics import TeacherGuidedPhysicsLoss
from model_discriminator import ConditionalDiscriminator
from metrics import calculate_metrics_all_channels, VAR_NAMES


def apply_zero_geo_ablation(x_geo: torch.Tensor) -> torch.Tensor:
    """消融：清零 HGT/SLOPE；保留 LAT/LON/LANDMASK（便于物理损失仍使用真实掩膜）。"""
    x_geo[:, 0:2, :, :] = 0
    return x_geo


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def calculate_rmse(pred, target, channels=None):
    """
    计算RMSE
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测值 (B, C, H, W)
    target : torch.Tensor
        真实值 (B, C, H, W)
    channels : list or None
        要计算的通道索引，None表示所有通道
    
    Returns:
    --------
    rmse : float
        RMSE值
    """
    if channels is not None:
        pred = pred[:, channels, :, :]
        target = target[:, channels, :, :]
    
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse).item()
    return rmse


def train_epoch(model, dataloader_sup, criterion, optimizer, device, logger, epoch,
                dataloader_unsup=None, cycle_criterion=None, w_cycle=0.0,
                physics_criterion=None, discriminator=None, optimizer_d=None, w_adv=0.005,
                zero_geo_input=False):
    """
    训练一个epoch（支持半监督模式）
    
    Parameters:
    -----------
    model : nn.Module
        模型
    dataloader_sup : DataLoader
        有监督数据加载器
    criterion : CombinedLoss
        有监督损失函数
    optimizer : optim.Optimizer
        优化器
    device : torch.device
        设备
    logger : logging.Logger
        日志记录器
    epoch : int
        当前epoch
    dataloader_unsup : DataLoader, optional
        无监督数据加载器，None表示纯监督模式
    cycle_criterion : CycleConsistencyLoss, optional
        循环一致性损失函数，None表示不使用
    w_cycle : float
        循环一致性损失权重
    
    Returns:
    --------
    avg_loss : float
        平均总损失
    avg_pixel : float
        平均像素损失
    avg_grad : float
        平均梯度损失
    avg_spec : float
        平均频谱损失
    avg_cycle : float
        平均循环一致性损失（如果使用半监督）
    """
    model.train()
    total_loss = 0.0
    loss_pixel = 0.0
    loss_grad = 0.0
    loss_spec = 0.0
    loss_cycle = 0.0
    loss_physics = 0.0
    loss_adv = 0.0
    num_batches = 0
    
    bce = nn.BCEWithLogitsLoss() if (discriminator is not None) else None
    
    # 如果使用半监督，创建无监督数据迭代器
    unsup_iter = None
    if dataloader_unsup is not None:
        unsup_iter = cycle(dataloader_unsup)
    
    pbar = tqdm(dataloader_sup, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (x_met, x_geo, label) in enumerate(pbar):
        # 移动到设备
        x_met = x_met.to(device)
        x_geo = x_geo.to(device)
        label = label.to(device)
        if zero_geo_input:
            apply_zero_geo_ablation(x_geo)

        # 前向传播（有监督）
        optimizer.zero_grad()
        pred_sup = model(x_met, x_geo)
        
        # 计算有监督损失
        loss_sup, loss_dict = criterion(pred_sup, label)
        
        # 总损失初始化为有监督损失
        total_loss_batch = loss_sup
        
        # Teacher-Guided 物理损失（可选）
        if physics_criterion is not None:
            landmask = x_geo[:, 4:5, :, :]  # LANDMASK 通道
            l_phys, phys_dict = physics_criterion(pred_sup, label, landmask)
            total_loss_batch = total_loss_batch + l_phys
            loss_physics += phys_dict['total_physics']
        
        # GAN 对抗损失（可选）
        if discriminator is not None and optimizer_d is not None:
            wind_pred = pred_sup[:, 0:3, :, :]   # U, V, SLP
            wind_real = label[:, 0:3, :, :]
            # 更新判别器
            optimizer_d.zero_grad()
            real_logits = discriminator(wind_real, x_geo)
            fake_logits = discriminator(wind_pred.detach(), x_geo)
            real_labels = torch.ones_like(real_logits, device=device)
            fake_labels = torch.zeros_like(fake_logits, device=device)
            loss_d_real = bce(real_logits, real_labels)
            loss_d_fake = bce(fake_logits, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            # 生成器对抗损失：让 D 认为 pred 为真
            fake_logits_g = discriminator(wind_pred, x_geo)
            l_adv = bce(fake_logits_g, real_labels)
            total_loss_batch = total_loss_batch + w_adv * l_adv
            loss_adv += l_adv.item()
        
        # 如果使用半监督，添加循环一致性损失
        if unsup_iter is not None and cycle_criterion is not None:
            try:
                # 获取无监督batch
                u_met, u_geo, _ = next(unsup_iter)
                u_met = u_met.to(device)
                u_geo = u_geo.to(device)
                if zero_geo_input:
                    apply_zero_geo_ablation(u_geo)

                # 前向传播（无监督）
                pred_unsup = model(u_met, u_geo)
                
                # 计算循环一致性损失
                # 注意：u_met是ERA5气象数据（7通道），pred_unsup是模型输出（5通道）
                # 我们需要将u_met和pred_unsup对齐到相同的通道（U, V, SLP）
                loss_cycle_batch = cycle_criterion(pred_unsup, u_met)
                
                # 添加到总损失
                total_loss_batch = total_loss_batch + w_cycle * loss_cycle_batch
                loss_cycle += loss_cycle_batch.item()
            except StopIteration:
                # 如果无监督数据耗尽，重新创建迭代器
                unsup_iter = cycle(dataloader_unsup)
        
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss_dict['total']
        loss_pixel += loss_dict['pixel']
        loss_grad += loss_dict['grad']
        loss_spec += loss_dict['spec']
        num_batches += 1
        
        # 更新进度条
        postfix = {
            'Loss': f"{loss_dict['total']:.4f}",
            'Pix': f"{loss_dict['pixel']:.4f}",
            'Grad': f"{loss_dict['grad']:.4f}",
            'Spec': f"{loss_dict['spec']:.4f}"
        }
        if unsup_iter is not None and cycle_criterion is not None:
            postfix['Cycle'] = f"{loss_cycle/num_batches:.4f}"
        if physics_criterion is not None:
            postfix['Phys'] = f"{loss_physics/num_batches:.4f}"
        if discriminator is not None:
            postfix['Adv'] = f"{loss_adv/num_batches:.4f}"
        pbar.set_postfix(postfix)
    
    # 平均损失
    avg_loss = total_loss / num_batches
    avg_pixel = loss_pixel / num_batches
    avg_grad = loss_grad / num_batches
    avg_spec = loss_spec / num_batches
    avg_cycle = loss_cycle / num_batches if num_batches > 0 else 0.0
    avg_physics = loss_physics / num_batches if num_batches > 0 else 0.0
    avg_adv = loss_adv / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_pixel, avg_grad, avg_spec, avg_cycle, avg_physics, avg_adv


def validate(model, dataloader, criterion, device, logger, physics_criterion=None, zero_geo_input=False):
    """验证"""
    model.eval()
    total_loss = 0.0
    loss_pixel = 0.0
    loss_grad = 0.0
    loss_spec = 0.0
    
    # RMSE统计
    rmse_wind = 0.0  # U, V通道 (0, 1)
    rmse_press = 0.0  # SLP通道 (2)
    rmse_all = 0.0
    
    # 收集所有预测和标签用于计算指标
    all_preds = []
    all_labels = []
    
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for x_met, x_geo, label in pbar:
            # 移动到设备
            x_met = x_met.to(device)
            x_geo = x_geo.to(device)
            label = label.to(device)
            if zero_geo_input:
                apply_zero_geo_ablation(x_geo)

            # 前向传播
            pred = model(x_met, x_geo)
            
            # 计算损失
            loss, loss_dict = criterion(pred, label)
            if physics_criterion is not None:
                landmask = x_geo[:, 4:5, :, :]
                l_phys, _ = physics_criterion(pred, label, landmask)
                loss_dict_total = loss_dict['total'] + l_phys.item()
            else:
                loss_dict_total = loss_dict['total']
            
            # 累计损失
            total_loss += loss_dict_total
            loss_pixel += loss_dict['pixel']
            loss_grad += loss_dict['grad']
            loss_spec += loss_dict['spec']
            
            # 计算RMSE
            rmse_wind += calculate_rmse(pred, label, channels=[0, 1])
            rmse_press += calculate_rmse(pred, label, channels=[2])
            rmse_all += calculate_rmse(pred, label, channels=[0, 1, 2])  # 只计算U, V, SLP
            
            # 收集预测和标签（用于计算R², Bias, CC, SSIM）
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())
            
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total']:.4f}",
                'RMSE_W': f"{rmse_wind/num_batches:.4f}",
                'RMSE_P': f"{rmse_press/num_batches:.4f}"
            })
    
    # 合并所有批次（移动到CPU以节省GPU内存）
    all_preds = torch.cat(all_preds, dim=0)  # (N, 5, 64, 64)
    all_labels = torch.cat(all_labels, dim=0)  # (N, 5, 64, 64)
    
    # 计算详细指标（R², Bias, CC, SSIM）
    # 注意：这些计算在CPU上进行，避免GPU内存溢出
    # ALL指标只计算U, V, SLP通道（与损失函数保持一致）
    metrics_dict = calculate_metrics_all_channels(all_preds, all_labels, target_channels=[0, 1, 2])
    
    # 释放内存
    del all_preds, all_labels
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 平均指标
    avg_loss = total_loss / num_batches
    avg_pixel = loss_pixel / num_batches
    avg_grad = loss_grad / num_batches
    avg_spec = loss_spec / num_batches
    avg_rmse_wind = rmse_wind / num_batches
    avg_rmse_press = rmse_press / num_batches
    avg_rmse_all = rmse_all / num_batches
    
    return {
        'loss': avg_loss,
        'pixel': avg_pixel,
        'grad': avg_grad,
        'spec': avg_spec,
        'rmse_wind': avg_rmse_wind,
        'rmse_press': avg_rmse_press,
        'rmse_all': avg_rmse_all,
        'metrics': metrics_dict  # 添加详细指标
    }


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(os.path.join(args.exp_dir, 'train.log'))
    
    logger.info("="*60)
    logger.info("PhyTopo-Net Training")
    logger.info("="*60)
    logger.info(f"实验名称: {args.exp_name}")
    logger.info(f"设备: {device}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"损失权重 - Pixel: {args.w_pixel}, Grad: {args.w_grad}, Spec: {args.w_spec}")
    if getattr(args, "zero_geo_input", False):
        logger.info("消融: zero_geo_input — 地形支路 HGT/SLOPE 置零（LAT/LON/LANDMASK 保留）")
    
    # 检查是否使用半监督模式
    use_semi_supervised = args.unlabeled_data_dir is not None and os.path.exists(args.unlabeled_data_dir)
    
    if use_semi_supervised:
        logger.info(f"\n半监督训练模式已启用")
        logger.info(f"无监督数据目录: {args.unlabeled_data_dir}")
        logger.info(f"循环一致性损失权重: {args.w_cycle}")
        batch_size_unsup = args.batch_size_unlabeled if args.batch_size_unlabeled is not None else args.batch_size
        logger.info(f"无监督批次大小: {batch_size_unsup}")
    else:
        logger.info(f"\n纯监督训练模式")
    
    # 创建数据集
    logger.info("\n加载数据集...")
    train_dataset = WindDataset(args.data_dir, split='train')
    valid_dataset = WindDataset(args.data_dir, split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # 如果使用半监督，创建无监督数据加载器
    train_loader_unsup = None
    if use_semi_supervised:
        try:
            # 无监督数据集：直接从unlabeled目录加载
            # 检查unlabeled_data_dir是否直接包含.npy文件，还是包含子目录
            unlabeled_files = sorted(glob.glob(os.path.join(args.unlabeled_data_dir, '*.npy')))
            
            if len(unlabeled_files) == 0:
                # 尝试在子目录中查找
                unlabeled_files = sorted(glob.glob(os.path.join(args.unlabeled_data_dir, '**', '*.npy'), recursive=True))
            
            if len(unlabeled_files) == 0:
                raise ValueError(f"在 {args.unlabeled_data_dir} 中未找到.npy文件")
            
            # 创建一个简化的数据集类来加载无监督数据
            # 复用WindDataset的逻辑，但直接指定文件列表
            class UnlabeledDataset(WindDataset):
                def __init__(self, file_list):
                    self.file_list = file_list
                    self.data_dir = os.path.dirname(file_list[0]) if file_list else ''
                    self.split = 'unlabeled'
                    print(f"加载无监督数据集: {len(self.file_list)} 个样本")
            
            unlabeled_dataset = UnlabeledDataset(unlabeled_files)
            
            batch_size_unsup = args.batch_size_unlabeled if args.batch_size_unlabeled is not None else args.batch_size
            train_loader_unsup = DataLoader(
                unlabeled_dataset,
                batch_size=batch_size_unsup,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory
            )
            logger.info(f"无监督样本数: {len(unlabeled_dataset)}")
        except Exception as e:
            logger.warning(f"无法加载无监督数据: {e}")
            logger.warning("将退化为纯监督模式")
            use_semi_supervised = False
            train_loader_unsup = None
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(valid_dataset)}")
    
    # 创建模型
    use_residual = getattr(args, 'residual', False)
    model_arch = getattr(args, 'model_arch', 'phytopo')
    logger.info("\n初始化模型...")
    if model_arch == 'unet_concat':
        model = BaselineMetGeoUNet(residual=use_residual).to(device)
    else:
        fusion_mode = 'cross_attn' if model_arch == 'phytopo' else 'concat'
        model = PhyTopoNet(residual=use_residual, fusion_mode=fusion_mode).to(device)
    logger.info(f"model_arch={model_arch}")
    if use_residual:
        logger.info("残差学习模式: output = input + 网络预测")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 创建损失函数
    criterion = CombinedLoss(
        w_pixel=args.w_pixel,
        w_grad=args.w_grad,
        w_spec=args.w_spec
    )
    
    # 如果使用半监督，创建循环一致性损失函数
    cycle_criterion = None
    if use_semi_supervised:
        pool_sz = getattr(args, 'cycle_pool_size', 8)
        cycle_criterion = CycleConsistencyLoss(pool_size=pool_sz)
        logger.info(f"循环一致性: AvgPool2d(kernel={pool_sz}), 高分辨率 {pool_sz}× 下采样后与 ERA5 对齐（偏大尺度）")
    
    # Teacher-Guided 物理损失（可选，不影响原分支）
    physics_criterion = None
    if getattr(args, 'physics_guided', False):
        physics_criterion = TeacherGuidedPhysicsLoss(
            w_kinematic=args.w_kinematic,
            w_inflow=args.w_inflow,
            w_coast=args.w_coast,
            physics_scale=args.physics_scale
        ).to(device)
        logger.info(f"Teacher-Guided 物理损失: w_kinematic={args.w_kinematic}, "
                   f"w_inflow={args.w_inflow}, w_coast={args.w_coast}, scale={args.physics_scale}")
    
    # GAN 判别器（可选）
    discriminator = None
    optimizer_d = None
    if getattr(args, 'gan', False):
        discriminator = ConditionalDiscriminator(in_channels=8).to(device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr * 2)  # D 略高学习率
        logger.info(f"GAN 对抗损失: w_adv={args.w_adv}")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 恢复训练/微调
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 判断是微调模式还是恢复训练模式
        # 微调模式：--finetune 或 提供了无监督数据（use_semi_supervised=True）
        if getattr(args, 'finetune', False) or use_semi_supervised:
            # 微调模式：只加载模型权重，从epoch 0开始
            logger.info(f"\n微调模式：加载预训练模型 {args.resume}")
            logger.info("只加载模型权重，从epoch 0开始训练")
            logger.info("优化器状态已重置（使用新的学习率: {})".format(args.lr))
            start_epoch = 0
            best_val_loss = float('inf')  # 重置最佳损失
        else:
            # 恢复训练模式：恢复所有状态
            logger.info(f"\n恢复训练: {args.resume}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"从epoch {start_epoch}继续训练")
    
    # 创建CSV文件记录指标
    metrics_csv_path = os.path.join(args.exp_dir, 'metrics.csv')
    csv_file_exists = os.path.exists(metrics_csv_path)
    
    # 训练循环
    logger.info("\n开始训练...")
    logger.info("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_result = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch,
            dataloader_unsup=train_loader_unsup,
            cycle_criterion=cycle_criterion,
            w_cycle=args.w_cycle if use_semi_supervised else 0.0,
            physics_criterion=physics_criterion,
            discriminator=discriminator,
            optimizer_d=optimizer_d,
            w_adv=args.w_adv if discriminator is not None else 0.0,
            zero_geo_input=args.zero_geo_input,
        )
        train_loss, train_pixel, train_grad, train_spec, train_cycle, train_physics, train_adv = train_result
        
        # 验证
        if (epoch + 1) % args.val_freq == 0:
            val_metrics = validate(
                model,
                valid_loader,
                criterion,
                device,
                logger,
                physics_criterion=physics_criterion,
                zero_geo_input=args.zero_geo_input,
            )
            
            # 更新学习率
            scheduler.step(val_metrics['loss'])
            
            # 记录日志
            logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss_str = f"训练损失: {train_loss:.6f} (Pixel: {train_pixel:.6f}, "
            train_loss_str += f"Grad: {train_grad:.6f}, Spec: {train_spec:.6f}"
            if use_semi_supervised:
                train_loss_str += f", Cycle: {train_cycle:.6f}"
            if physics_criterion is not None:
                train_loss_str += f", Phys: {train_physics:.6f}"
            if discriminator is not None:
                train_loss_str += f", Adv: {train_adv:.6f}"
            logger.info(train_loss_str)
            logger.info(f"验证损失: {val_metrics['loss']:.6f} (Pixel: {val_metrics['pixel']:.6f}, "
                       f"Grad: {val_metrics['grad']:.6f}, Spec: {val_metrics['spec']:.6f})")
            logger.info(f"验证RMSE - Wind: {val_metrics['rmse_wind']:.6f}, "
                       f"Pressure: {val_metrics['rmse_press']:.6f}, "
                       f"All: {val_metrics['rmse_all']:.6f}")
            
            # 记录详细指标（R², Bias, CC, SSIM）
            metrics_dict = val_metrics['metrics']
            logger.info("\n详细指标 (按变量):")
            for var_name in VAR_NAMES:
                if var_name in metrics_dict:
                    m = metrics_dict[var_name]
                    logger.info(f"  {var_name:6s} - R²: {m['r2']:7.4f}, "
                               f"Bias: {m['bias']:8.4f}, "
                               f"CC: {m['cc']:7.4f}, "
                               f"SSIM: {m['ssim']:7.4f}")
            
            # 整体指标
            if 'ALL' in metrics_dict:
                m_all = metrics_dict['ALL']
                logger.info(f"  {'ALL':6s} - R²: {m_all['r2']:7.4f}, "
                           f"Bias: {m_all['bias']:8.4f}, "
                           f"CC: {m_all['cc']:7.4f}, "
                           f"SSIM: {m_all['ssim']:7.4f}")
            
            # 保存指标到CSV
            with open(metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头（仅第一次）
                if not csv_file_exists:
                    header = ['Epoch', 'Train_Loss', 'Train_Pixel', 'Train_Grad', 'Train_Spec']
                    if use_semi_supervised:
                        header.append('Train_Cycle')
                    if physics_criterion is not None:
                        header.append('Train_Phys')
                    if discriminator is not None:
                        header.append('Train_Adv')
                    header.extend(['Val_Loss', 'Val_Pixel', 'Val_Grad', 'Val_Spec',
                                 'RMSE_Wind', 'RMSE_Press', 'RMSE_All'])
                    
                    # 为每个变量添加指标列
                    for var_name in VAR_NAMES:
                        header.extend([f'{var_name}_R2', f'{var_name}_Bias', f'{var_name}_CC', f'{var_name}_SSIM'])
                    
                    # 整体指标
                    header.extend(['ALL_R2', 'ALL_Bias', 'ALL_CC', 'ALL_SSIM'])
                    
                    writer.writerow(header)
                    csv_file_exists = True
                
                # 写入数据
                row = [epoch + 1, train_loss, train_pixel, train_grad, train_spec]
                if use_semi_supervised:
                    row.append(train_cycle)
                if physics_criterion is not None:
                    row.append(train_physics)
                if discriminator is not None:
                    row.append(train_adv)
                row.extend([val_metrics['loss'], val_metrics['pixel'], val_metrics['grad'], val_metrics['spec'],
                          val_metrics['rmse_wind'], val_metrics['rmse_press'], val_metrics['rmse_all']])
                
                # 每个变量的指标
                for var_name in VAR_NAMES:
                    if var_name in metrics_dict:
                        m = metrics_dict[var_name]
                        row.extend([m['r2'], m['bias'], m['cc'], m['ssim']])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0.0])
                
                # 整体指标
                if 'ALL' in metrics_dict:
                    m_all = metrics_dict['ALL']
                    row.extend([m_all['r2'], m_all['bias'], m_all['cc'], m_all['ssim']])
                else:
                    row.extend([0.0, 0.0, 0.0, 0.0])
                
                writer.writerow(row)
            
            logger.info(f"指标已保存到: {metrics_csv_path}")
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = os.path.join(args.exp_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                    'residual': use_residual,
                    'model_arch': model_arch,
                }, best_model_path)
                logger.info(f"保存最佳模型: {best_model_path} (Val Loss: {best_val_loss:.6f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'residual': use_residual,
                'model_arch': model_arch,
            }, checkpoint_path)
            logger.info(f"保存checkpoint: {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch时间: {epoch_time:.2f}秒")
        logger.info("-"*60)
    
    logger.info("\n训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()

