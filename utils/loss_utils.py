#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def TV_loss(x, mask=None):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def color_aware_dice_loss(pred_rgb, gt_rgb, sigma=10.0, eps=1e-6):
    """
    :param pred_rgb: 模型输出 (B, 3, H, W) 范围 [0,1]
    :param gt_rgb: Ground Truth (B, 3, H, W) 范围 [0,1]
    :param sigma: 控制颜色相似度的温度参数
    """
    # 计算颜色相似度矩阵 (像素级) difference -> large -> sim -> small
    pred_rgb = torch.clamp(pred_rgb, 0, 1)
    color_sim = torch.exp(-torch.sum((pred_rgb - gt_rgb)**2, dim=1)) * sigma  # (B, H, W)
    
    # 区域重叠计算
    intersection = torch.sum(color_sim * (pred_rgb * gt_rgb).sum(dim=1), dim=(1,2))
    union = torch.sum(color_sim * (pred_rgb**2).sum(dim=1), dim=(1,2)) + \
            torch.sum(color_sim * (gt_rgb**2).sum(dim=1), dim=(1,2))
    
    dice = (2. * intersection + eps) / (union + eps)
    if dice.mean().item()>1:
        print(f"{dice.mean().item():.4f}", f"{intersection.mean().item():.4f}", f"{union.mean().item():.4f}")
        exit()
    return 1 - dice.mean()

def boundary_aware_contrastive_loss(pred_rgb, gt_mask, margin=0.3, kernel_size=5):
    """
    :param gt_mask: 二值化边界掩膜 (B, 1, H, W)
    """
    
    # Sobel 边缘检测获取预测边界
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred_rgb.dtype, device=pred_rgb.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred_rgb.dtype, device=pred_rgb.device)
    
    grad_x = F.conv2d(pred_rgb.mean(dim=1, keepdim=True), sobel_x.view(1,1,3,3), padding=1)
    grad_y = F.conv2d(pred_rgb.mean(dim=1, keepdim=True), sobel_y.view(1,1,3,3), padding=1)
    
    pred_edge = torch.sqrt(grad_x**2 + grad_y**2)  # (B, 1, H, W)
    
    # 边界区域对比损失
    pos_pairs = pred_edge[gt_mask == 1]
    neg_pairs = pred_edge[gt_mask == 0]
    
    # 边界应比非边界区域响应更强
    loss = torch.mean(torch.relu(margin - (pos_pairs.mean() - neg_pairs.mean())))
    return loss



def region_smooth_loss(gt, img):
    ''' c, h, w
    '''
    gt_gray = 0.2989*gt[0:1]+0.5870*gt[1:2]+0.1140*gt[2:3]
    loss = 0
    for i in gt_gray.unique():
        region_mask = ((gt_gray==i))
        if region_mask.sum()<1000:
            continue
        loss += torch.abs(img[region_mask.repeat(3,1,1)]-img[region_mask.repeat(3,1,1)].mean()).mean()
    return loss