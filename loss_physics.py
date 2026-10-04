"""
Teacher-Guided Physics Loss: 运动学与动力学约束
不强迫物理量为0，而是约束预测场的物理导数与WRF真值一致。
- L_div: 散度一致性 (辐合辐散)
- L_vort: 涡度一致性 (气旋旋转)
- L_inflow: 摩擦流入角一致性 (跨等压线流入)
- L_coast: 海岸线梯度惩罚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_conv2d(x, kernel):
    """对单通道应用卷积"""
    return F.conv2d(x, kernel, padding=1)


def _compute_derivatives(u, v, p, sobel_x, sobel_y):
    """
    计算 u, v, p 的空间导数
    u, v: (B, 1, H, W)
    p: (B, 1, H, W)
    Returns: du/dx, du/dy, dv/dx, dv/dy, dp/dx, dp/dy
    """
    du_dx = _sobel_conv2d(u, sobel_x)
    du_dy = _sobel_conv2d(u, sobel_y)
    dv_dx = _sobel_conv2d(v, sobel_x)
    dv_dy = _sobel_conv2d(v, sobel_y)
    dp_dx = _sobel_conv2d(p, sobel_x)
    dp_dy = _sobel_conv2d(p, sobel_y)
    return du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy


class TeacherGuidedPhysicsLoss(nn.Module):
    """
    Teacher-Guided Physics Loss
    约束预测的散度、涡度、流入角与WRF真值一致
    """
    
    def __init__(self, w_kinematic=0.2, w_inflow=0.1, w_coast=0.1, physics_scale=1e3):
        super(TeacherGuidedPhysicsLoss, self).__init__()
        self.w_kinematic = w_kinematic
        self.w_inflow = w_inflow
        self.w_coast = w_coast
        self.physics_scale = physics_scale
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _get_uv_slp(self, pred, target):
        """提取 U, V, SLP 通道 (0, 1, 2)"""
        u_pred = pred[:, 0:1, :, :]
        v_pred = pred[:, 1:2, :, :]
        p_pred = pred[:, 2:3, :, :]
        u_wrf = target[:, 0:1, :, :]
        v_wrf = target[:, 1:2, :, :]
        p_wrf = target[:, 2:3, :, :]
        return u_pred, v_pred, p_pred, u_wrf, v_wrf, p_wrf
    
    def forward(self, pred, target, landmask=None):
        """
        Parameters:
        -----------
        pred : (B, C, H, W) 预测 [U, V, SLP, T]
        target : (B, C, H, W) WRF真值
        landmask : (B, 1, H, W) or (B, H, W), optional. 1=陆地, 0=海洋
        
        Returns:
        --------
        total : scalar
        loss_dict : dict with div, vort, inflow, coast
        """
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        u_pred, v_pred, p_pred, u_wrf, v_wrf, p_wrf = self._get_uv_slp(pred, target)
        
        du_dx_p, du_dy_p, dv_dx_p, dv_dy_p, dp_dx_p, dp_dy_p = _compute_derivatives(
            u_pred, v_pred, p_pred, sobel_x, sobel_y)
        du_dx_w, du_dy_w, dv_dx_w, dv_dy_w, dp_dx_w, dp_dy_w = _compute_derivatives(
            u_wrf, v_wrf, p_wrf, sobel_x, sobel_y)
        
        # L_div: 散度一致性
        div_pred = du_dx_p + dv_dy_p
        div_wrf = du_dx_w + dv_dy_w
        l_div = torch.mean(torch.abs(div_pred - div_wrf)) * self.physics_scale
        
        # L_vort: 涡度一致性
        vort_pred = dv_dx_p - du_dy_p
        vort_wrf = dv_dx_w - du_dy_w
        l_vort = torch.mean(torch.abs(vort_pred - vort_wrf)) * self.physics_scale
        
        # L_inflow: 摩擦流入角 (∇p·v)
        inflow_pred = dp_dx_p * u_pred + dp_dy_p * v_pred
        inflow_wrf = dp_dx_w * u_wrf + dp_dy_w * v_wrf
        l_inflow = torch.mean(torch.abs(inflow_pred - inflow_wrf))
        
        total = self.w_kinematic * (l_div + l_vort) + self.w_inflow * l_inflow
        
        loss_dict = {'div': l_div.item(), 'vort': l_vort.item(), 'inflow': l_inflow.item(), 'coast': 0.0}
        
        # L_coast: 海岸线梯度惩罚 (若有 landmask)
        if landmask is not None:
            if landmask.dim() == 3:
                landmask = landmask.unsqueeze(1)  # (B, 1, H, W)
            # Sobel(landmask) 得到海岸线
            coast_mask = torch.abs(_sobel_conv2d(landmask, sobel_x)) + \
                         torch.abs(_sobel_conv2d(landmask, sobel_y))
            coast_mask = (coast_mask > 0.1).float()
            if coast_mask.sum() > 0:
                diff_uv = torch.abs(u_pred - u_wrf) + torch.abs(v_pred - v_wrf)
                l_coast = (coast_mask * diff_uv).sum() / (coast_mask.sum() + 1e-8)
                total = total + self.w_coast * l_coast
                loss_dict['coast'] = l_coast.item()
        
        loss_dict['total_physics'] = total.item()
        return total, loss_dict
