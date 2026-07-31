from re import X
import os
import torch,timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional
Tensor = torch.Tensor
from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

from torchvision.ops import deform_conv2d 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.input_resolution = input_resolution  # 输入特征图的尺寸
        self.num_heads = num_heads  # 注意力头的数量
        self.window_size = window_size  # 注意力窗口大小
        self.shift_size = shift_size  # 窗口的循环移动大小
        self.mlp_ratio = mlp_ratio  # MLP隐藏层的比例

        # 如果输入分辨率小于或等于窗口大小，调整窗口大小
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # 确保 shift_size 在合理范围内
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)  # 第一个归一化层

        self.attn = WindowAttention(# 注意力模块
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()# 路径丢弃层

        self.norm2 = norm_layer(dim)# 第二个归一化层

        # MLP的隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)# MLP模块

        if self.shift_size > 0:
            # 计算 SW-MSA 的注意力掩码
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1  # 创建图像掩码
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt  # 填充掩码
                    cnt += 1

            # 获取窗口掩码
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1  # 按窗口分区
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)   # 变换形状
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # 创建注意力掩码
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # 设置掩码值
        else:
            attn_mask = None # 没有注意力掩码

        self.register_buffer("attn_mask", attn_mask) # 注册注意力掩码

    def forward(self, x):# 前向传播方法
        # H, W = self.input_resolution
        B, L, C = x.shape   # B: batch size, L: sequence length, C: channels
        H, W = int(L**0.5), int(L**0.5) # 计算输入特征图的高和宽
        assert L == H * W, "input feature has wrong size"   # 确保输入特征形状正确

        shortcut = x    # 残差连接
        x = x.view(B, H, W, C)  # 变换形状为 (B, H, W, C)

        # 循环移动
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))# 移动特征图
        else:
            shifted_x = x

        # 按窗口分区
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C  # 按窗口分区
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C  # 变换形状

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C   # 计算注意力

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C   # 逆窗口操作

        # 反向循环移动
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))   # 反向移动
        else:
            x = shifted_x
        x = x.view(B, H * W, C)     # 变换回 (B, L, C)
        x = shortcut + self.drop_path(self.norm1(x))        # 残差连接与归一化

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))     # MLP 处理和残差连接

        return x        # 返回输出特征

    def extra_repr(self) -> str:    # 返回类的额外描述
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):# 计算 FLOPS（每秒浮点运算次数）
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W       # norm1 的 FLOPS
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size        # 窗口数量
        flops += nW * self.attn.flops(self.window_size * self.window_size)      # W-MSA/SW-MSA 的 FLOPS
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio       # MLP 的 FLOPS
        # norm2
        flops += self.dim * H * W       # norm2 的 FLOPS
        return flops        # 返回总 FLOPS


class PatchMerging(nn.Module):
    

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution        # 存储输入特征图的分辨率
        self.dim = dim      # 输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)    # 定义线性层以减少特征维度
        self.norm = norm_layer(2 * dim)     # 归一化层，用于后续处理

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution        # 获取输入特征图的尺寸
        B, L, C = x.shape       # B: batch size, L: sequence length, C: channels
        assert L == H * W, "input feature has wrong size"  # 确保输入特征的形状正确
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."  # 确保H和W都是偶数

        x = x.view(B, H, W, C)      # 变换形状为 (B, H, W, C)

        # 提取四个子区域的特征
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C     # 选择偶数行和偶数列
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C     # 选择奇数行和偶数列
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C     # 选择偶数行和奇数列
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C     # 选择奇数行和奇数列

        # 将四个子区域的特征拼接在一起
        x = torch.cat([x0, x1, x2, x3], -1)  # 变为 (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # 变换为 (B, H/2 * W/2, 4*C

        x = self.reduction(x)  # 通过线性层减少特征维度到 (B, H/2 * W/2, 2*C)
        x = self.norm(x)  # 归一化处理

        return x  # 返回合并和标准化后的特征图

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"      # 返回输入分辨率和通道数的描述

    def flops(self):
        H, W = self.input_resolution        # 获取输入特征图的高度和宽度
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim       # 计算合并后的特征图的 FLOPs     # 每个合并操作涉及 4 个输入通道，输出通道数为 2 * dim
        flops += H * W * self.dim // 2      # 加上归一化层的 FLOPs     # 归一化操作，考虑到输出特征图的维度减半
        return flops        # 返回总的 FLOPs


class BasicLayer(nn.Module):
   

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim  # 输入通道数
        self.input_resolution = input_resolution  # 输入特征图的分辨率
        self.depth = depth  # transformer 块的数量
        self.use_checkpoint = use_checkpoint  # 是否使用梯度检查点

        # 当特征图小于 window_size 时, 缩小窗口以适配 (256x256 输入时 Stage4 为 8x8)
        effective_ws = min(window_size, min(input_resolution))

        # 构建 transformer 块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=effective_ws,
                                 shift_size=0 if (i % 2 == 0) else effective_ws // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])     # 创建指定深度的 transformer 块

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None      # 如果不需要下采样，则设置为 None

    def forward(self, x):
        for blk in self.blocks:     # 遍历每个 transformer 块
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)       # 使用梯度检查点
            else:
                x = blk(x)      # 正常前向传播
        if self.downsample is not None:
            x = self.downsample(x)# 如果有下采样层，则执行下采样
        return x  # 返回输出特征

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)  # 初始化第一个归一化层的偏置为 0
            nn.init.constant_(blk.norm1.weight, 0)  # 初始化第一个归一化层的权重为 0
            nn.init.constant_(blk.norm2.bias, 0)  # 初始化第二个归一化层的偏置为 0
            nn.init.constant_(blk.norm2.weight, 0)  # 初始化第二个归一化层的权重为 0


class LCSRB(nn.Module):#long connection swin-transformer residual block 
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=512, patch_size=8, resi_connection='3conv',pretrained_window_size=0):
        super(LCSRB, self).__init__()

        self.dim = dim*2 if downsample else dim     # 设置输入通道数，如果需要下采样则通道数加倍
        self.input_resolution = input_resolution

        # 构建残差组，包含多个 transformer 块
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, 
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         pretrained_window_size=pretrained_window_size)

        # 根据选择构建残差连接的卷积层
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1)      # 单卷积层
        elif resi_connection == '3conv':# 使用三个卷积层以节省参数和内存
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(self.dim, self.dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(self.dim // 4, self.dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(self.dim // 4, self.dim, 3, 1, 1))
            
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=1, in_chans=self.dim, embed_dim=self.dim,
        #     norm_layer=norm_layer)

        # self.patch_unembed = PatchUnEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=self.dim, embed_dim=self.dim,
        #     norm_layer=norm_layer)

    def forward(self, x):
        # return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x)))) + x

        # 通过残差组处理输入
        x = self.residual_group(x)
        B, HW, C = x.shape  # 获取批量大小、特征长度和通道数
        H, W = int(math.sqrt(HW)), int(math.sqrt(HW))  # 计算特征图的高度和宽度
        x_ = x.transpose(1, 2).view(B, C, H, W)  # 变换形状为 (B, C, H, W)
        x_ = self.conv(x_)  # 通过卷积层
        x_ = x_.flatten(2).transpose(1, 2)  # 变换形状回 (B, HW, C)
        return x_ + x  # 返回残差连接的结果

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()  # 计算残差组的 FLOPs
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9  # 计算卷积操作的 FLOPs

        return flops  # 返回总的 FLOPs

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def _init_respostnorm(self):
        for blk in self.residual_group.blocks:
            nn.init.constant_(blk.norm1.bias, 0)  # 初始化第一个归一化层的偏置为 0
            nn.init.constant_(blk.norm1.weight, 0)  # 初始化第一个归一化层的权重为 0
            nn.init.constant_(blk.norm2.bias, 0)  # 初始化第二个归一化层的偏置为 0
            nn.init.constant_(blk.norm2.weight, 0)  # 初始化第二个归一化层的权重为 0



class PatchUnEmbed(nn.Module):
    

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 将图像大小转换为元组
        patch_size = to_2tuple(patch_size)  # 将补丁大小转换为元组
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 计算补丁分辨率
        self.img_size = img_size  # 存储图像大小
        self.patch_size = patch_size  # 存储补丁大小
        self.patches_resolution = patches_resolution  # 存储补丁分辨率
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 计算补丁数量

        self.in_chans = in_chans  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入维度

    def forward(self, x):
        B, HW, C = x.shape  # 获取批量大小、特征长度和通道数
        x = x.transpose(1, 2).view(B, self.embed_dim, int(HW ** 0.5), int(HW ** 0.5))  # 将补丁嵌入转换为 (B, C, H, W) 形状
        return x  # 返回重构后的特征图

    def flops(self):
        flops = 0  # 目前没有计算操作
        return flops  # 返回 FLOPs

class PatchEmbed(nn.Module):
    

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 将输入图像大小转为元组形式
        patch_size = to_2tuple(patch_size)  # 将补丁大小转为元组形式
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 计算补丁的分辨率
        self.img_size = img_size  # 存储图像大小
        self.patch_size = patch_size  # 存储补丁大小
        self.patches_resolution = patches_resolution  # 存储补丁的分辨率
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 计算总补丁数

        self.in_chans = in_chans  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入的维度

        # 卷积层用于补丁嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果提供了归一化层，则初始化归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None  # 如果没有，则设置为 None

    def forward(self, x):
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 通过卷积层进行嵌入，将输入图像转换为补丁嵌入
        x = self.proj(x).flatten(2).transpose(1, 2)  # 变换形状为 (B, Ph*Pw, C)
        if self.norm is not None:
            x = self.norm(x)  # 如果指定了归一化，则应用归一化
        return x  # 返回补丁嵌入

    def flops(self):
        Ho, Wo = self.patches_resolution  # 补丁的高度和宽度
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])  # 计算卷积操作的 FLOPs
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim  # 如果有归一化层，则加上归一化操作的 FLOPs
        return flops  # 返回总的 FLOPs


class SwinTransformerV2(nn.Module):
    

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        # 初始化网络参数
        self.num_classes = num_classes  # 输出类别数
        self.num_layers = len(depths)  # 网络层数
        self.embed_dim = embed_dim  # 嵌入维度
        self.ape = ape  # 是否使用绝对位置嵌入
        self.patch_norm = patch_norm  # 是否在补丁嵌入后应用归一化
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 最终特征维度
        self.mlp_ratio = mlp_ratio  # MLP 比例

        # 将图像分割成不重叠的补丁
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  # 初始化补丁嵌入层
        num_patches = self.patch_embed.num_patches  # 获取补丁数量
        patches_resolution = self.patch_embed.patches_resolution  # 获取补丁分辨率
        self.patches_resolution = patches_resolution  # 存储补丁分辨率

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # 创建位置嵌入参数
            trunc_normal_(self.absolute_pos_embed, std=.02)  # 初始化位置嵌入

        self.pos_drop = nn.Dropout(p=drop_rate)  # 位置嵌入的丢弃层

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建各层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)# 将层添加到模块列表中

        self.norm = norm_layer(self.num_features)  # 最后的归一化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类头

        self.apply(self._init_weights)  # 初始化权重
        for bly in self.layers:  # 初始化每层的归一化
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 对线性层的权重进行截断正态初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 归一化层的偏置初始化为 0
            nn.init.constant_(m.weight, 1.0)  # 归一化层的权重初始化为 1.0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}       # 返回不使用权重衰减的参数

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}       # 返回不使用权重衰减的关键词

    def forward_features(self, x):
        x = self.patch_embed(x)  # 通过补丁嵌入层处理输入
        if self.ape:
            x = x + self.absolute_pos_embed  # 加上绝对位置嵌入
        x = self.pos_drop(x)  # 应用位置丢弃

        for layer in self.layers:  # 逐层处理
            x = layer(x)

        x = self.norm(x)  # 最后的归一化
        x = self.avgpool(x.transpose(1, 2))  # 自适应平均池化
        x = torch.flatten(x, 1)  # 展平特征
        return x  # 返回提取的特征

    def extra_features(self,x):
        x = self.patch_embed(x)  # 处理输入
        if self.ape:
            x = x + self.absolute_pos_embed  # 加上绝对位置嵌入
        x = self.pos_drop(x)  # 应用位置丢弃
        feature = []

        for layer in self.layers:  # 逐层处理
            x = layer(x)
            bs, n, f = x.shape  # 获取批量大小、补丁数量和特征维度
            h = int(n ** 0.5)  # 计算高度

            feature.append(x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous())  # 保存特征
        return feature  # 返回各层特征

    
    def get_unet_feature(self,x):
        x = self.patch_embed(x)  # 处理输入
        if self.ape:
            x = x + self.absolute_pos_embed  # 加上绝对位置嵌入
        x = self.pos_drop(x)  # 应用位置丢弃
        bs, n, f = x.shape  # 获取批量大小、补丁数量和特征维度
        h = int(n ** 0.5)  # 计算高度
        feature = [x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous()]  # 保存第一个层的特征

        for layer in self.layers:  # 逐层处理
            x = layer(x)
            bs, n, f = x.shape
            h = int(n ** 0.5)  # 计算高度

            feature.append(x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous())  # 保存特征
        return feature   #返回特征列表,长度为5的feature map
 
    def forward(self, x):
        x = self.forward_features(x)  # 提取特征
        x = self.head(x)  # 通过分类头
        return x  # 返回分类结果


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()  # 计算补丁嵌入的 FLOPs
        for i, layer in enumerate(self.layers):
            flops += layer.flops()  # 计算每层的 FLOPs
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (
                    2 ** self.num_layers)  # 计算池化后的 FLOPs
        flops += self.num_features * self.num_classes  # 计算分类头的 FLOPs
        return flops  # 返回总的 FLOPs


class SwinTransformerV2_LCSRB(SwinTransformerV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__(mg_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs)  # 调用父类构造函数
        # 初始化网络参数
        self.num_classes = num_classes  # 输出类别数
        self.num_layers = len(depths)  # 网络层数
        self.embed_dim = embed_dim  # 嵌入维度
        self.ape = ape  # 是否使用绝对位置嵌入
        self.patch_norm = patch_norm  # 是否在补丁嵌入后应用归一化
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 最终特征维度
        self.mlp_ratio = mlp_ratio  # MLP 比例

        # 将图像分割成不重叠的补丁
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  # 初始化补丁嵌入层
        num_patches = self.patch_embed.num_patches  # 获取补丁数量
        patches_resolution = self.patch_embed.patches_resolution  # 获取补丁分辨率
        self.patches_resolution = patches_resolution  # 存储补丁分辨率

        # 绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # 创建位置嵌入参数
            trunc_normal_(self.absolute_pos_embed, std=.02)  # 初始化位置嵌入

        self.pos_drop = nn.Dropout(p=drop_rate)  # 位置嵌入的丢弃层

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建 LCSRB 层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # layer = LCSRB(
            #                    dim=embed_dim,
            #                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
            #                                      patches_resolution[1] // (2 ** i_layer)),
            #                    depth=depths[i_layer],
            #                    num_heads=num_heads[i_layer],
            #                    window_size=window_size,
            #                    mlp_ratio=self.mlp_ratio,
            #                    qkv_bias=qkv_bias,
            #                    drop=drop_rate, attn_drop=attn_drop_rate,
            #                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            #                    norm_layer=norm_layer,
            #                    downsample =None,
            #                    use_checkpoint=use_checkpoint,
            #                    resi_connection='3conv',
            #                    pretrained_window_size=pretrained_window_sizes[i_layer])

            layer = LCSRB(     # 创建 LCSRB 层
                                dim=int(embed_dim * 2 ** i_layer),
                    
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               resi_connection='3conv', # 使用 3 个卷积的残差连接
                               pretrained_window_size=pretrained_window_sizes[i_layer]) # 使用预训练窗口大小
            self.layers.append(layer)   # 将层添加到模块列表中

        self.norm = norm_layer(self.num_features)  # 最后的归一化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类头

        self.apply(self._init_weights)  # 初始化权重
        for bly in self.layers:  # 初始化每层的残差组归一化
            bly.residual_group._init_respostnorm()  # 初始化残差组的归一化



def swin_v2(img_size=512, **kwargs):
    # 创建 Swin Transformer V2 模型实例
    model = SwinTransformerV2(
        img_size=img_size,
        window_size=16,  # 设置窗口大小为 16
        embed_dim=128,  # 嵌入维度设置为 128
        depths=[2, 2, 18, 2],  # 每个阶段的层数
        num_heads=[4, 8, 16, 32],  # 每层的注意力头数量
        **kwargs  # 其他参数
    )
    checkpoint=torch.load(r'C:\Users\LCY\pretrain\swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth')["model"]        # 加载预训练的权重
    # 根据图像大小调整加载的权重
    if img_size != 256:  # 检查图像大小是否为 256
        # 删除与注意力掩码相关的权重，因为它们在不同的图像大小下可能不适用
        del checkpoint["layers.0.blocks.1.attn_mask"]
        del checkpoint["layers.1.blocks.1.attn_mask"]
        del checkpoint["layers.3.blocks.0.attn.relative_coords_table"]
        del checkpoint["layers.3.blocks.0.attn.relative_position_index"]
        del checkpoint["layers.3.blocks.1.attn.relative_coords_table"]
        del checkpoint["layers.3.blocks.1.attn.relative_position_index"]

    # 将加载的权重加载到模型中，允许部分不匹配
    model.load_state_dict(checkpoint, strict=False)

    return model        # 返回模型实例

_SWIN_V2_CONFIGS = {
    'tiny':  dict(embed_dim=96,  depths=[2, 2, 6, 2],  num_heads=[3, 6, 12, 24], window_size=16),
    'small': dict(embed_dim=96,  depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=16),
    'base':  dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=16),
}

# 预训练权重文件名 (不含路径)
_SWIN_V2_PRETRAINED = {
    'tiny':  'swinv2_tiny_patch4_window16_256.pth',
    'small': 'swinv2_small_patch4_window16_256.pth',
    'base':  'swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
}


def _find_pretrained(variant):
    """自动查找预训练权重, 兼容本地 / Kaggle / 任意环境.
    优先: 环境变量 SWIN_PRETRAIN_DIR > 脚本目录 > pretrain/ 子目录 > 递归搜索
    """
    fname = _SWIN_V2_PRETRAINED[variant]
    search_dirs = [os.environ.get('SWIN_PRETRAIN_DIR', '')]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs += [
        script_dir,
        os.path.join(script_dir, 'pretrain'),
        os.getcwd(),
        os.path.join(os.getcwd(), 'pretrain'),
        r'E:\月球_dataset\Mss-Net\pretrain',  # 本地默认
        '/kaggle/input',  # Kaggle 典型挂载点
    ]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        # 直接查找
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            return path
        # Kaggle: pretrain 通常在 dataset 的子目录里, 递归一层
        try:
            for sub in os.listdir(d):
                path = os.path.join(d, sub, fname)
                if os.path.isfile(path):
                    return path
        except PermissionError:
            pass
    raise FileNotFoundError(
        f'找不到预训练权重: {fname}\n'
        f'请设置环境变量 SWIN_PRETRAIN_DIR 指向包含 {fname} 的目录, '
        f'或将文件放到脚本所在目录或 pretrain/ 子目录'
    )

def swin_v2_LCSRB(img_size=512, in_chans=3, size='base', **kwargs):
    # 解析模型规模
    variant = size.lower()
    for key in _SWIN_V2_CONFIGS:
        if key in variant:
            variant = key
            break
    cfg = _SWIN_V2_CONFIGS[variant]
    print(f'[swin_v2_LCSRB] using variant={variant}, embed_dim={cfg["embed_dim"]}, depths={cfg["depths"]}')

    # 提前取出 pretrained 参数, 不传给 SwinTransformerV2_LCSRB
    pretrained = kwargs.pop('pretrained', True)

    # 创建 Swin Transformer V2 LCSRB 模型实例
    model = SwinTransformerV2_LCSRB(
        img_size=img_size,
        in_chans=in_chans,
        window_size=cfg['window_size'],
        embed_dim=cfg['embed_dim'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        **kwargs
    )
    # 加载预训练的权重
    if pretrained:
        pretrained_path = _find_pretrained(variant)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
    else:
        checkpoint = {}

    # ===== 将 3 通道的 patch_embed.proj.weight 扩展到 in_chans 通道 =====
    pe_key = 'patch_embed.proj.weight'
    if pe_key in checkpoint and in_chans != 3:
        w = checkpoint[pe_key]  # (embed_dim, 3, patch, patch)
        if w.shape[1] == 3:
            if in_chans < 3:
                new_w = w[:, :in_chans]
            else:
                extra = in_chans - 3
                # 用 3 通道均值去初始化多出来的通道 (论文常用做法)
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = torch.cat([w, mean_w.repeat(1, extra, 1, 1)], dim=1)
            # 保持总体尺度不变
            new_w = new_w * (3.0 / in_chans)
            checkpoint[pe_key] = new_w
            print(f'[swin_v2_LCSRB] expand patch_embed 3 -> {in_chans} channels')

    # 根据图像大小调整加载的权重 (用 pop 避免 KeyError)
    if img_size != 256:
        for k_del in [
            "layers.0.blocks.1.attn_mask",
            "layers.1.blocks.1.attn_mask",
            "layers.3.blocks.0.attn.relative_coords_table",
            "layers.3.blocks.0.attn.relative_position_index",
            "layers.3.blocks.1.attn.relative_coords_table",
            "layers.3.blocks.1.attn.relative_position_index",
        ]:
            checkpoint.pop(k_del, None)
    #创建一个新的字典，用于存储修改后的键值对
    new_dict = {}
    model_dict = model.state_dict()  # 获取模型的状态字典
    # 遍历预训练权重的键值对
    for k, v in checkpoint.items():
        if "blocks." in k:
            new_key = k.replace("blocks.", "residual_group.blocks.")
        elif "downsample." in k:
            new_key = k.replace("downsample.", "residual_group.downsample.")
        else:
            new_key = k
        # 跳过模型中不存在的 key
        if new_key not in model_dict:
            continue
        # 跳过形状不匹配的 key
        if v.shape != model_dict[new_key].shape:
            print(f'skip: {k} pretrain={v.shape} model={model_dict[new_key].shape}')
            continue
        new_dict[new_key] = v
    # 将新字典中的权重加载到模型中，允许部分不匹配
    model.load_state_dict(new_dict, strict=False)

    return model        # 返回模型实例

class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    # 在原始实现中使用精确的 RoI 池化，而不是自适应平均池化
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)        # 计算每个阶段的输出通道数
        # 创建不同池化尺寸的阶段
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        # 创建瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),# 3x3 卷积层
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout2d(0.1)  # 2D 丢弃层
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)  # 自适应平均池化层
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1 卷积层
        bn = nn.BatchNorm2d(out_channels)  # 批归一化
        relu = nn.ReLU(inplace=True)  # 激活函数
        return nn.Sequential(prior, conv, bn, relu)  # 返回顺序块

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]# 获取输入特征图的高和宽
        pyramids = [features]# 初始化金字塔特征列表
        # 对每个阶段进行池化并调整大小
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                        align_corners=True) for stage in self.stages])
        # 拼接所有金字塔特征并通过瓶颈层
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output# 返回输出特征图




class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()       # 调用父类的构造方法

    def forward(self, x):
        return x * torch.sigmoid(x)     # 返回 Swish 激活函数的计算结果



class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0     # 确保输入通道数可以被组数整除
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        # 初始化权重参数
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            # self.bias = nn.Parameter(torch.empty(out_dim))
            # 初始化偏置参数
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask      # 是否使用掩码
        if with_mask:
            # 创建参数生成器，包括偏移和掩码
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:# 创建参数生成器，仅包括偏移
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            # 从参数生成器中获取偏移和掩码
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)  # 拼接偏移
            mask = mask.sigmoid()  # 应用 Sigmoid 函数
        else:
            offset = self.param_generator(x)  # 仅获取偏移
            mask = None
        # 调用 deform_conv2d 函数执行可变形卷积
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x  # 返回输出张量

class DeformablePSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6],with_mask=True):
        super(DeformablePSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)        # 计算每个阶段的输出通道数
        # 创建不同池化尺寸的阶段
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s,with_mask) 
                                                        for b_s in bin_sizes])
        # 创建瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),# 3x3 卷积层
            nn.BatchNorm2d(in_channels),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout2d(0.1)  # 2D 丢弃层
        )
        

    def _make_stages(self, in_channels, out_channels, bin_sz,with_mask):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)        # 自适应平均池化层
        conv = DeformableConv2d(in_dim=in_channels, out_dim=out_channels, kernel_size=1, groups=1, offset_groups=1, with_mask=with_mask)        # 可变形卷积层
        # 使用 GroupNorm 替代 BatchNorm2d, 避免 bin_sz=1 且 batch=1 时空间维度不足的问题
        num_groups = min(32, out_channels)
        bn = nn.GroupNorm(num_groups, out_channels)
        relu = nn.ReLU(inplace=True)  # 激活函数
        return nn.Sequential(prior, conv, bn, relu)  # 返回顺序块


    def forward(self, features):
        h, w = features.size()[2], features.size()[3]  # 获取输入特征图的高和宽
        pyramids = [features]  # 初始化金字塔特征列表
        # 对每个阶段进行池化并调整大小
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bicubic', 
                                        align_corners=True) for stage in self.stages])
        # 拼接所有金字塔特征并通过瓶颈层
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output  # 返回输出特征图





def up_and_add(x, y):       #用于将张量 x 上采样到张量 y 的大小，然后将它们相加
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN, self).__init__()
        assert feature_channels[0] == fpn_out       # 确保 FPN 输出通道数与第一个特征通道数一致
        # 1x1 卷积层用于减少通道数
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        # 平滑卷积层，增强特征图
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        

    def forward(self, features):
        # 使用 1x1 卷积减少通道数
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]##
        # 从高到低层进行上采样并相加
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        # 平滑处理特征图
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]     # 反转特征图列表
        P = list(reversed(P))       # 添加最后一层特征图
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        # 获取输出特征图的高和宽
        H, W = P[0].size(2), P[0].size(3)
        # 将 P[1:] 的特征图上采样到相同的高和宽
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        return P        # 返回输出特征图
        # x = self.conv_fusion(torch.cat((P), dim=1))
        # return x

class Downsampler_Conv(nn.Module):

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 **kwargs):

        super().__init__()
        # 计算填充
        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)
        # 检查填充和步幅的有效性
        if padding < 0:
            raise ValueError('negative padding is not supported for Conv2d')
        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')
        # 创建卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        return self.conv(x)     # 返回卷积层的输出

    


class FPNPAN(nn.Module):

    def __init__(self,
                 feature_channels = [256, 512, 1024, 2048], 
                 fpn_out = 256,

                 up_mode: str = 'nearest'):

        super().__init__()
        # FPN: 1x1 卷积层用于通道数调整
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        # 平滑卷积层
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))

        # PAN: 侧向连接
        self.laterals = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, 1) for i in range(len(feature_channels))])

        # 下采样层
        self.downsamples = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, 1, 2, padding=0, bias=True)
                                              for _ in range(len(feature_channels) - 2)])
        self.downsamples.append(nn.Conv2d(fpn_out, fpn_out, 1, bias=True))

        # 融合层
        self.fuses = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, 3, 1, 1, bias=True) for _ in range(len(feature_channels))])



    def forward(self, features: List[Tensor]) -> List[Tensor]:
        # FPN: 通道数调整和特征融合
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]##
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]        # 添加最后一层特征图

        # PAN: 侧向连接和下采样
        p_features = []
        for i in range(len(P)):
            p = self.laterals[i](P[i])      # 侧向连接

            if p_features:
                d = self.downsamples[i - 1](p_features[-1])  # 下采样
                p += d  # 累加特征

            p = self.fuses[i](p)        # 融合特征
            p_features.append(p)
        # 上采样到相同的高和宽
        H, W = p_features[0].size(2), p_features[0].size(3)
        p_features[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in p_features[1:]]

        return p_features       # 返回处理后的特征图




class SwinUperNet_LCSRB(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(SwinUperNet_LCSRB, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size,img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small","tiny"]:
            feature_channels = [192,384,768,768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256,512,1024,1024]
        # 初始化 Pyramidal Pooling Module 和 Feature Pyramid Network
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN(feature_channels, fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])     # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])  # 使用 PPM 处理最后一层特征
        features = self.FPN(features)  # 使用 FPN 处理特征
        features = self.conv_fusion(torch.cat((features), dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头
        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features# 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()       #返回一个生成器，用于获取主干网络的参数。

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())      #返回一个生成器，用于获取 PPN、FPN 和解码头的参数。

    def freeze_bn(self):#通过将批处理规范化图层设置为评估模式来冻结它们。该方法将网络中的所有BatchNorm2d层设置为评估模式防止他们在训练期间更新他们的跑步统计数据。
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class Swin_LCSRB_PSP_FPNPAN(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_LCSRB_PSP_FPNPAN, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size,img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small","tiny"]:
            feature_channels = [192,384,768,768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256,512,1024,1024]
        # 初始化 Pyramidal Pooling Module 和 FPNPAN
        self.PPN = PSPModule(feature_channels[-1])
        self.FPNPAN = FPNPAN(feature_channels,fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])# 获取输入的高和宽

        # 提取特征
        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])  # 使用 PPM 处理最后一层特征
        features = self.FPNPAN(features)  # 使用 FPNPAN 处理特征
        features = self.conv_fusion(torch.cat((features), dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')# 上采样到原始输入尺寸
        return features# 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class TerrainEncoder(nn.Module):
    """轻量地形编码器 (Terrain Encoder)

    从地形通道 (DEM/Slope/TPI/Curvature) 单独提取多尺度地形语义特征,
    用于引导主干网络的光学特征。输出尺度与 Swin V2 backbone 对齐:
    Stage 0: 1/4, Stage 1: 1/8, Stage 2: 1/16, Stage 3: 1/16
    """
    def __init__(self, in_channels=4, channels=(192, 384, 768, 768)):
        super().__init__()
        # Stage 0: 1/4 (两次 stride=2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )
        # Stage 1: 1/8
        self.s1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        )
        # Stage 2: 1/16
        self.s2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
        )
        # Stage 3: 1/16 (与 Swin Stage 3 对齐, 不再下采样)
        self.s3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
        )

    def forward(self, x):
        f0 = self.stem(x)   # 1/4
        f1 = self.s1(f0)    # 1/8
        f2 = self.s2(f1)    # 1/16
        f3 = self.s3(f2)    # 1/16
        return [f0, f1, f2, f3]


class TerrainGuidedFusion(nn.Module):
    """地形引导融合 (Terrain-Guided Fusion)

    跨模态条件门控: 同时利用主干特征 F_i 和地形特征 T_i 生成门控信号:
        G_i = sigmoid( Conv1x1( [F_i, T_i] ) )
        F'_i = F_i * G_i + F_i

    相比旧版 (门控只看 T_i), 新版让网络知道"当前位置主干特征是什么",
    再结合地形决定该不该放大, 避免 DEM 噪声导致系统性假阳性。
    """
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
            ) for c in channels
        ])

    def forward(self, main_feats, terrain_feats):
        out = []
        for f, t, gate in zip(main_feats, terrain_feats, self.gates):
            if t.shape[-2:] != f.shape[-2:]:
                t = F.interpolate(t, size=f.shape[-2:], mode='bilinear', align_corners=False)
            g = torch.sigmoid(gate(torch.cat([f, t], dim=1)))
            out.append(f * g + f)
        return out


class CoordinateAttention(nn.Module):
    """坐标注意力模块 (Coordinate Attention)

    通过方向感知的通道注意力选择性增强线性构造特征。
    与 Strip Pooling 不同, CA 学习哪些通道/位置是重要的,
    不会盲目引入全局噪声, 更适合构造稀疏的月球表面。

    Reference: Hou et al., "Coordinate Attention for Efficient
               Mobile Network Design", CVPR 2021.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mid_c = max(8, in_channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        # 共享变换
        self.conv1 = nn.Conv2d(in_channels, mid_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.act = nn.Hardswish(inplace=True)

        # 分支: H 方向和 W 方向独立生成注意力
        self.conv_h = nn.Conv2d(mid_c, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_c, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码位置信息
        x_h = self.pool_h(x)                          # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # (B, C, W, 1)

        # 拼接后共享变换
        y = torch.cat([x_h, x_w], dim=2)             # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))         # (B, mid_c, H+W, 1)

        # 分割回 H 和 W
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)               # (B, mid_c, 1, W)

        # 生成方向注意力图
        a_h = torch.sigmoid(self.conv_h(x_h))        # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))        # (B, C, 1, W)

        return x * a_h * a_w


class StripPooling(nn.Module):
    """条带池化模块 (Strip Pooling Module)

    通过 H×1 和 1×W 条带池化捕获全局水平/垂直上下文,
    增强对长线性构造 (皱脊/月溪/断层/地堑) 的连通性特征提取。

    Reference: Hou et al., "Strip Pooling: Rethinking Spatial Pooling
               for Scene Parsing", CVPR 2020.
    """
    def __init__(self, in_channels):
        super().__init__()
        mid_c = in_channels // 4

        # 水平条带: 沿高度池化为 1×W, 捕获水平方向全局上下文
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, mid_c, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )

        # 垂直条带: 沿宽度池化为 H×1, 捕获垂直方向全局上下文
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, mid_c, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
        )

        # 融合: 投射回原通道数, 生成注意力权重
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_c, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        _, _, H, W = x.shape

        # 水平条带分支
        h = self.pool_h(x)                     # (B, C, 1, W)
        h = self.conv_h(h)                     # (B, mid_c, 1, W)
        h = F.interpolate(h, size=(H, W), mode='bilinear', align_corners=True)

        # 垂直条带分支
        v = self.pool_v(x)                     # (B, C, H, 1)
        v = self.conv_v(v)                     # (B, mid_c, H, 1)
        v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=True)

        # 注意力融合 + 残差连接
        att = torch.sigmoid(self.fuse(h + v))  # (B, C, H, W)
        return x * att + x


class LocalCNNBranch(nn.Module):
    """Local CNN 双分支 — 保留极细线局部空间细节

    Stage 级并行: Transformer (全局语义) + 3x3 Conv×2 (局部细节) → Concat+1x1 融合

    设计原则:
      - 用普通 3x3 Conv 而非 DWConv: 极细线需要跨通道空间协同
      - Concat 融合而非 Add: 保留两条路径的独立信号, 让网络自行学习融合权重
      - 只在 Stage2/3 使用: 分辨率尚可 (H/8, H/16), 细线结构未完全丢失
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # 融合层: Concat(Transformer, LocalCNN) → 1x1 Conv 回原通道数
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: Swin Transformer 特征图 (B, C, H, W)
        local = self.conv2(self.conv1(x))       # 3x3 Conv 提取局部细线细节
        fused = self.fuse(torch.cat([x, local], dim=1))  # Concat 融合
        return fused


class Swin_LCSRB_DeformablePSP_FPNPAN(nn.Module):
    # Implementing only the object path
    def __init__(self, size="base", img_size=512, num_classes=1, in_channels=3, pretrained=True,
                 use_strip_pooling=False, use_coord_attention=False,
                 use_local_cnn=False,
                 use_dem_guided=False, terrain_channels=4,
                 use_deep_supervision=False):
        super(Swin_LCSRB_DeformablePSP_FPNPAN, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size, img_size=img_size, in_chans=in_channels, pretrained=pretrained)
        # 根据模型大小设置特征通道数
        variant = size.lower()
        if 'base' in variant:
            feature_channels = [256, 512, 1024, 1024]
        else:  # tiny / small
            feature_channels = [192, 384, 768, 768]

        # ===== 注意力模块选择 =====
        self.use_strip_pooling = use_strip_pooling
        self.use_coord_attention = use_coord_attention

        if use_coord_attention:
            self.ca2 = CoordinateAttention(feature_channels[2])  # Stage 2
            self.ca3 = CoordinateAttention(feature_channels[3])  # Stage 3
            print(f'[CoordAttention] enabled on Stage 2 ({feature_channels[2]}ch) & Stage 3 ({feature_channels[3]}ch)')
        elif use_strip_pooling:
            self.sp2 = StripPooling(feature_channels[2])  # Stage 2
            self.sp3 = StripPooling(feature_channels[3])  # Stage 3
            print(f'[StripPooling] enabled on Stage 2 ({feature_channels[2]}ch) & Stage 3 ({feature_channels[3]}ch)')

        # ===== Local CNN 双分支 (Stage 级并行的局部细节分支) =====
        self.use_local_cnn = use_local_cnn
        if use_local_cnn:
            self.lc2 = LocalCNNBranch(feature_channels[2])  # Stage 2
            self.lc3 = LocalCNNBranch(feature_channels[3])  # Stage 3
            print(f'[LocalCNN] enabled on Stage 2 ({feature_channels[2]}ch) & Stage 3 ({feature_channels[3]}ch)')

        # ===== DEM-guided Fusion =====
        self.use_dem_guided = use_dem_guided
        self.terrain_channels = terrain_channels
        if use_dem_guided:
            self.terrain_encoder = TerrainEncoder(
                in_channels=terrain_channels,
                channels=tuple(feature_channels),
            )
            self.terrain_fusion = TerrainGuidedFusion(feature_channels)
            print(f'[DEM-guided] enabled, terrain_channels={terrain_channels}, '
                  f'channels={feature_channels}')

        # 初始化可变形金字塔池化模块和 FPNPAN
        self.PPN = DeformablePSPModule(feature_channels[-1])
        fpn_out = feature_channels[0]
        self.FPNPAN = FPNPAN(feature_channels, fpn_out=fpn_out)
        # 融合卷积层 (fpn_out * 4 个拼接通道)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)

        # Deep Supervision: 在 FPNPAN 前3个特征图上加辅助头
        self.use_deep_supervision = use_deep_supervision
        if use_deep_supervision:
            aux_out = fpn_out // 2
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fpn_out, aux_out, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(aux_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(aux_out, num_classes, 1)
                ) for _ in range(3)
            ])
            print(f'[DeepSup] 3 aux heads on FPNPAN features (fpn_out={fpn_out})')

    def freeze_backbone_stages(self, num_stages=2):
        """冻结 backbone 前 num_stages 个阶段 (含 patch_embed)."""
        # 冻结 patch_embed
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        # 冻结前 num_stages 个 layer
        for i in range(min(num_stages, len(self.backbone.layers))):
            for p in self.backbone.layers[i].parameters():
                p.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f'[freeze] {frozen}/{total} params frozen (前 {num_stages} 阶段 + patch_embed)')



    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽

        # 提取主干特征 (使用全部输入通道)
        features = self.backbone.extra_features(x)

        # DEM-guided fusion: 用地形通道生成门控信号, 调制主干特征
        if self.use_dem_guided:
            # 取后 terrain_channels 个通道作为地形输入 (DEM/Slope/TPI/Curv)
            terrain_input = x[:, -self.terrain_channels:, :, :]
            terrain_feats = self.terrain_encoder(terrain_input)
            features = self.terrain_fusion(features, terrain_feats)

        # 注意力增强深层特征
        if self.use_coord_attention:
            features[2] = self.ca2(features[2])  # Stage 2
            features[3] = self.ca3(features[3])  # Stage 3
        elif self.use_strip_pooling:
            features[2] = self.sp2(features[2])  # Stage 2
            features[3] = self.sp3(features[3])  # Stage 3

        # Local CNN 双分支: 与 Transformer 并行, 保留极细线局部细节
        if self.use_local_cnn:
            features[2] = self.lc2(features[2])  # Stage 2 (H/8, 细线特征丰富)
            features[3] = self.lc3(features[3])  # Stage 3 (H/16, 中等尺度)

        features[-1] = self.PPN(features[-1])  # 使用可变形 PPM 处理最后一层特征
        features = self.FPNPAN(features)  # 使用 FPNPAN 处理特征

        # Deep Supervision: 在主 head 之前, 用 aux heads 产生辅助输出
        aux_logits = None
        if self.use_deep_supervision:
            aux_logits = [aux_head(f) for aux_head, f in zip(self.aux_heads, features[:3])]
            # 上采样到输入尺寸
            aux_logits = [F.interpolate(a, size=input_size, mode='bilinear', align_corners=True)
                          for a in aux_logits]

        features = self.conv_fusion(torch.cat((features), dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        if aux_logits is not None:
            return features, aux_logits
        return features  # 返回最终的输出特征图

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class ASPP(nn.Module):
   
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        # 创建 ASPP 块列表
        self.aspp_blocks = nn.ModuleList()
        for rate in rates:
            # 为每个扩张率构建一个卷积块
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        # 全局池化层
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化到 1x1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 卷积
            nn.BatchNorm2d(out_channels),  # 批归一化
            nn.ReLU()  # 激活函数
        )
       
    def forward(self, x):
        size = x.shape[-2:]  # 获取输入张量的高和宽
        # 对全局池化的输出进行上采样
        res = [F.interpolate(self.global_pooling(x), size=size, mode='bilinear', align_corners=False)]
        # 处理每个 ASPP 块并将结果添加到输出列表
        for block in self.aspp_blocks:
            res.append(block(x))
        # 将所有结果在通道维度上连接
        return torch.cat(res, dim=1)  # 返回连接后的张量


class Swin_LCSRB_FPNPAN(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_LCSRB_FPNPAN, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size,img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small","tiny"]:
            feature_channels = [192,384,768,768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256,512,1024,1024]
        # 初始化 FPNPAN
        self.FPNPAN = FPNPAN(feature_channels,fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽

        # 提取特征
        features = self.backbone.extra_features(x)
        features = self.FPNPAN(features)  # 处理特征
        features = self.conv_fusion(torch.cat((features), dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class Swin_FPNPAN_PSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_FPNPAN_PSP, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2(size=size,img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small","tiny"]:
            feature_channels = [192,384,768,768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256,512,1024,1024]
        # 初始化 Pyramidal Pooling Module 和 FPNPAN
        self.PPN = PSPModule(feature_channels[-1])
        self.FPNPAN = FPNPAN(feature_channels,fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)




    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 提取特征
        features = self.backbone.extra_features(x)
        features = self.PPN(features[-1])  # 使用 PPM 处理最后一层特征
        features = self.FPNPAN(features)  # 使用 FPNPAN 处理特征
        features = self.conv_fusion(torch.cat(features, dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头

        return features  # 返回最终的输出特征图


    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class Swin_FPNPAN_DeformablePSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_FPNPAN_DeformablePSP, self).__init__()
        # 初始化主干网络
        self.backbone = swin_v2(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化可变形金字塔池化模块和 FPNPAN
        self.PPN = DeformablePSPModule(feature_channels[-1])
        self.FPNPAN = FPNPAN(feature_channels, fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)




    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽

        # 提取特征
        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])  # 使用可变形 PPM 处理最后一层特征
        features = self.FPNPAN(features)  # 使用 FPNPAN 处理特征
        features = self.conv_fusion(torch.cat(features, dim=1))  # 融合特征
        features = self.head(features)  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features  # 返回最终的输出特征图


    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class Swin_LCSRB_PSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_LCSRB_PSP, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化金字塔池化模块
        self.PPN = PSPModule(feature_channels[-1])
        # 解码头
        self.head = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽

        # 提取特征
        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])  # 使用金字塔池化模块处理最后一层特征
        features = self.head(features[-1])  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class Swin_LCSRB_DeformablePSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_LCSRB_DeformablePSP, self).__init__()
        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化可变形金字塔池化模块
        self.PPN = DeformablePSPModule(feature_channels[-1], with_mask=False)
        # 解码头
        self.head = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])  # 使用可变形 PPM 处理最后一层特征
        features = self.head(features[-1])  # 经过解码头

        features = F.interpolate(features, size=input_size, mode='bilinear')  # 上采样到原始输入尺寸
        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



class Swin_LCSRB(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_LCSRB, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2_LCSRB(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 解码头，用于输出特征图
        self.head = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        # 经过解码头处理最后一层特征
        features = self.head(features[-1])
        # 上采样到原始输入尺寸
        features = F.interpolate(features, size=input_size, mode='bilinear')
        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class Swin_FPNPAN(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_FPNPAN, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化 FPNPAN 模块
        self.FPNPAN = FPNPAN(feature_channels, fpn_out=feature_channels[0])
        # 融合卷积层
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 解码头
        self.head = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        # 使用 FPNPAN 处理特征
        features = self.FPNPAN(features)
        # 融合特征
        features = self.conv_fusion(torch.cat(features, dim=1))
        # 经过解码头处理
        features = self.head(features)
        # 上采样到原始输入尺寸
        features = F.interpolate(features, size=input_size, mode='bilinear')

        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class Swin_PSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_PSP, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化金字塔池化模块
        self.PSP = PSPModule(feature_channels[-1])
        # 解码头
        self.head = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        # 使用金字塔池化模块处理最后一层特征
        features[-1] = self.PSP(features[-1])
        # 经过解码头处理
        features = self.head(features[-1])
        # 上采样到原始输入尺寸
        features = F.interpolate(features, size=input_size, mode='bilinear')

        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
    
class Swin_DeformablePSP(nn.Module):
    # Implementing only the object path
    def __init__(self,size="swinv2_base_window16_256",img_size=512,num_classes=1, in_channels=3, pretrained=True):
        super(Swin_DeformablePSP, self).__init__()

        # 初始化主干网络
        self.backbone = swin_v2(size=size, img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        # 初始化可变形金字塔池化模块
        self.PSP = DeformablePSPModule(feature_channels[-1], with_mask=False)
        # 解码头
        self.head = nn.Conv2d(1024, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # 将单通道输入复制为三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = (x.size()[2], x.size()[3])  # 获取输入的高和宽
        # 提取特征
        features = self.backbone.extra_features(x)
        # 使用可变形池化模块处理最后一层特征
        features[-1] = self.PSP(features[-1])
        # 经过解码头处理
        features = self.head(features[-1])
        # 上采样到原始输入尺寸
        features = F.interpolate(features, size=input_size, mode='bilinear')

        return features  # 返回最终的输出特征图

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

from segmentation_models_pytorch.base import modules as md

try:
    from timm.layers.cbam import CbamModule
except ImportError:
    from timm.models.layers.cbam import CbamModule

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        # 卷积层1，结合输入和跳跃连接的通道
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # 注意力机制，基于指定的类型
        if attention_type=="cbam":
            self.attention1 = CbamModule(channels=in_channels + skip_channels)
        else:
            self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        # 卷积层2
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # 第二个注意力机制
        if attention_type=="cbam":
            self.attention2 = CbamModule(channels=out_channels)
        else:
            self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        # 存储输入、输出和跳跃连接的通道数
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
    def forward(self, x, skip=None):
        # 如果没有跳跃连接，进行上采样
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            # 如果输入和跳跃连接的空间维度不同，进行上采样
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 如果有跳跃连接，进行拼接并应用注意力机制
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # 在通道维度拼接
            x = self.attention1(x)  # 应用注意力机制
        # 通过卷积层和注意力机制处理输入
        x = self.conv1(x)  # 第一个卷积层
        x = self.conv2(x)  # 第二个卷积层
        x = self.attention2(x)  # 应用第二个注意力机制
        return x  # 返回处理后的输出


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        # 创建第一个卷积层，包含 ReLU 激活和可选的批量归一化
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # 创建第二个卷积层，同样包含 ReLU 激活和可选的批量归一化
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # 将两个卷积层作为顺序模块的组件
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()
        # 检查解码器块的数量是否与提供的通道数匹配
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # 移除第一个跳跃连接通道（具有相同的空间分辨率）
        encoder_channels = encoder_channels[1:]
        # 反转通道顺序，从编码器的头部开始
        encoder_channels = encoder_channels[::-1]

        # 计算每个块的输入和输出通道数
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        out_channels = decoder_channels
        # 如果需要中心块，则初始化 CenterBlock；否则使用 nn.Identity
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # 组合解码器的关键字参数
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # 移除第一个跳跃连接（具有相同的空间分辨率）
        features = features[::-1]  # 反转通道顺序，从编码器的头部开始

        head = features[0]  # 获取头部特征
        skips = features[1:]  # 获取跳跃连接特征

        x = self.center(head)  # 处理头部特征
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None  # 获取相应的跳跃连接
            x = decoder_block(x, skip)  # 通过解码器块处理

        return x  # 返回最终输出

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)# 创建卷积层
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()# 根据需要创建上采样层
        super().__init__(conv2d, upsampling)# 将卷积层和上采样层作为顺序模块的组件

class SwinUnet(nn.Module):

    def __init__(
        self,size="swinv2_base_window16_256",img_size=512,num_classes=1 #"base" "large"
    ):
        super().__init__()
        # 初始化编码器
        self.encoder = swin_v2(size=size,img_size=img_size)
        # 根据模型大小设置特征通道数
        if size.split("_")[1] in ["small","tiny"]:
            feature_channels = (3,192,384,768,768)
        elif size.split("_")[1] in ["base"]:
            feature_channels = (3,256,512,1024,1024)
        # 初始化解码器
        self.decoder = UnetDecoder(encoder_channels=feature_channels,n_blocks=4,decoder_channels=(512,256,128,64),attention_type=None)
        # 初始化分割头
        self.segmentation_head = SegmentationHead(in_channels=64,out_channels=num_classes,kernel_size=3,upsampling=4
        )

    def forward(self, input):
        # 将单通道输入复制为三通道
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        # 获取编码器特征
        encoder_feature = self.encoder.get_unet_feature(input)
        # 获取解码器输出
        decoder_output = self.decoder(*encoder_feature)
        # 通过分割头获得最终掩码
        masks = self.segmentation_head(decoder_output)

        return masks  # 返回分割掩码
    

if __name__=="__main__":
    S = 512
    inp = torch.randn((4,3,S,S))# 创建一个随机输入张量，形状为 (4, 3, S, S)，# 其中 4 是批量大小，3 是通道数（RGB），S 是图像的高度和宽度。

    # 初始化 Swin LCSRB Deformable PSP 模型，指定输入图像大小、模型大小和类别数量。
    model = Swin_LCSRB_DeformablePSP(img_size=S,size="swinv2_base_window16_256",num_classes=1).cuda()

    # 也可以使用另一个模型：Swin_LCSRB_DeformablePSP_FPNPAN。
    # model = Swin_LCSRB_DeformablePSP_FPNPAN(img_size=S,size="swinv2_base_window16_256",num_classes=1).cuda()

    # 将输入张量移动到 GPU，并通过模型进行前向传播，获取输出。
    out = model(inp.cuda())
    # 打印输出的形状，以便检查模型的输出是否符合预期。
    print(out.shape)



    