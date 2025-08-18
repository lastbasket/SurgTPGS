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
import torchvision.transforms.functional as TF
import cv2

def tensor2array(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor
    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is None:
        mse_mask = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        if mask.shape[1] == 3:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10))
        else:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10)*3.0)

    return 20 * torch.log10(1.0 / torch.sqrt(mse_mask))

def rmse(a, b, mask):
    """Compute rmse.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)
    if len(mask.shape) == len(a.shape) - 1:
        mask = mask[..., None]
    mask_sum = np.sum(mask) + 1e-10
    rmse = (((a - b)**2 * mask).sum() / (mask_sum))**0.5
    return rmse


def sobel_boundary(gt, threshold=0.1):
    """
    使用 Sobel 算子提取边界
    :param gt_mask: Ground Truth 标签图 (H, W)
    :param threshold: 梯度幅值的阈值（归一化到 [0,1])
    :return: 边界二值化掩膜 (H, W)
    """
    # 将 gt_mask 转换为单通道灰度图
    gt_mask = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    
    # 计算 Sobel 梯度
    sobel_x = cv2.Sobel(gt_mask, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    sobel_y = cv2.Sobel(gt_mask, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 归一化梯度幅值到 [0,1]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # 根据阈值生成边界掩膜
    boundary_mask = (gradient_magnitude > threshold).astype(np.uint8)
    
    return boundary_mask

def smooth_sobel_edge_detection(image_tensor, threshold=0.1, sigma=1.0):
    """
    先平滑再使用 Sobel 算子提取边界
    :param sigma: 高斯核的标准差
    """
    # 高斯平滑
    kernel_size = int(2 * sigma + 1)
    
    image_tensor = 0.2989*image_tensor[0:1]+0.5870*image_tensor[1:2]+0.1140*image_tensor[2:3]    
    smoothed = TF.gaussian_blur(image_tensor.unsqueeze(0), kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    return sobel_edge_detection(smoothed, threshold)


def sobel_edge_detection(image_tensor, threshold=0.1):
    """
    使用 Sobel 算子提取边界
    :param image_tensor: 输入图像 Tensor (1, 1, H, W)，值范围 [0,1]
    :param threshold: 梯度幅值的阈值（归一化到 [0,1])
    :return: 边界二值化掩膜 (H, W)
    """
    # 定义 Sobel 卷积核
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=image_tensor.dtype).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=image_tensor.dtype).view(1, 1, 3, 3)
    
    # 计算水平和垂直方向的梯度
    grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1)  # (1, 1, H, W)
    grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1)  # (1, 1, H, W)
    
    # 计算梯度幅值
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(0)  # (1, 1, H, W)
    
    # 归一化梯度幅值到 [0,1]
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / \
                         (gradient_magnitude.max() - gradient_magnitude.min())
    
    # 根据阈值生成边界掩膜
    boundary_mask = (gradient_magnitude > threshold).squeeze(0)  # (H, W)
    
    return boundary_mask