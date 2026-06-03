# Copied from taco-group/MWFormer (model/EncDec.py)
# https://github.com/taco-group/MWFormer
# Network_top: StyleFilter 의 style vector 를 조건으로 받아 이미지를 복원하는 메인 backbone.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mwformer.base_networks import (ConvLayer, UpsampleConvLayer,
                                     ResidualBlock, strip_prefix_if_present)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM Block
# ─────────────────────────────────────────────────────────────────────────────

class FilmBlock(nn.Module):
    def __init__(self, cin_x, cin_y, x_out_channels):
        super().__init__()
        self.Conv_0 = nn.Conv2d(cin_x, x_out_channels, 3, 1, 1)
        self.Conv_1 = nn.Conv2d(cin_y, x_out_channels, 1, 1)
        self.LayerNorm_x = nn.LayerNorm([x_out_channels])
        self.in_project_x = nn.Linear(x_out_channels, x_out_channels)
        self.gelu1 = nn.GELU()
        self.LayerNorm_y = nn.LayerNorm([x_out_channels])
        self.in_project_y = nn.Linear(x_out_channels, x_out_channels)
        self.w_project_y = nn.Linear(x_out_channels, x_out_channels)
        self.b_project_y = nn.Linear(x_out_channels, x_out_channels)
        self.gelu2 = nn.GELU()
        self.out_project_x = nn.Linear(x_out_channels, x_out_channels)

    def forward(self, x, y):
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        shortcut_x = x
        x = self.LayerNorm_x(x.permute(0, 2, 3, 1)).contiguous()
        x = self.gelu1(self.in_project_x(x))
        y = self.LayerNorm_y(y.permute(0, 2, 3, 1)).contiguous()
        y = self.gelu2(self.in_project_y(y))
        x = x * self.w_project_y(y) + self.b_project_y(y)
        x = self.out_project_x(x).permute(0, 3, 1, 2).contiguous()
        return x + shortcut_x


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Attention / MLP / Block (Encoder 용 — hyper-network 포함)
# ─────────────────────────────────────────────────────────────────────────────

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.dwconv(x.transpose(1, 2).view(B, C, H, W))
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hyper=False, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.hyper = hyper
        self.fc1 = nn.Linear(in_features, hidden_features)
        if hyper:
            self.hypernet = nn.Sequential(
                nn.Linear(64, hidden_features * 3), nn.ReLU(),
                nn.Linear(hidden_features * 3, hidden_features * 9))
        else:
            self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W, vec):
        x = self.fc1(x)
        if self.hyper:
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W).contiguous()
            weight = self.hypernet(vec).reshape(-1, 1, 3, 3)
            x = F.conv2d(x.view(1, -1, H, W), weight, groups=B * C, padding=1).view(B, -1, H, W)
            x = x.flatten(2).transpose(1, 2)
        else:
            x = self.dwconv(x, H, W)
        return self.drop(self.fc2(self.drop(self.act(x))))


class Attention(nn.Module):
    def __init__(self, dim, hyper=False, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.hyper = hyper
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if hyper:
            self.hypernet1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, dim * dim))
            self.hypernet2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                            nn.Linear(64, 2 * dim * dim))
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, vec):
        B, N, C = x.shape
        nh = self.num_heads
        hd = C // nh
        if self.hyper:
            Wq = self.hypernet1(vec).reshape(B, C, C)
            Wkv = self.hypernet2(vec).reshape(B, 2 * C, C)
            q = torch.stack([F.linear(x[b], Wq[b]) for b in range(B)])
            q = q.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
            src = x
            if self.sr_ratio > 1:
                src = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W))
                                .reshape(B, C, -1).permute(0, 2, 1))
            kv = torch.stack([F.linear(src[b], Wkv[b]) for b in range(B)])
            kv = kv.reshape(B, -1, 2, nh, hd).permute(2, 0, 3, 1, 4)
        else:
            q = self.q(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)
            src = x
            if self.sr_ratio > 1:
                src = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W))
                                .reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(src).reshape(B, -1, 2, nh, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return x


class Block(nn.Module):
    def __init__(self, hyper_attn, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, hyper=hyper_attn, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hyper=True,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, vec):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, vec))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, vec))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Block (task-query attention, hyper-network 없음)
# ─────────────────────────────────────────────────────────────────────────────

class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.task_query = nn.Parameter(torch.randn(1, 48, dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        nh = self.num_heads
        hd = C // nh
        task_q = self.task_query
        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1).squeeze(1)
        q = self.q(task_q).reshape(B, task_q.shape[1], nh, hd).permute(0, 2, 1, 3)
        src = x
        if self.sr_ratio > 1:
            src = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W))
                            .reshape(B, C, -1).permute(0, 2, 1))
        kv = self.kv(src).reshape(B, -1, 2, nh, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, task_q.shape[1], C)))
        return x


class MlpDec(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W, vec=None):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        return self.drop(self.fc2(self.drop(self.act(x))))


class Block_dec(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpDec(in_features=dim, hidden_features=int(dim * mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / Decoder Transformer
# ─────────────────────────────────────────────────────────────────────────────

class EncoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, feature_chans=64,
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.layernorm = nn.LayerNorm([feature_chans])
        self.input_film = FilmBlock(in_chans, feature_chans, embed_dims[0])
        self.patch_embed1 = OverlapPatchEmbed(img_size, 7, 4, embed_dims[0], embed_dims[0])
        self.film1 = FilmBlock(embed_dims[0], feature_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.film2 = FilmBlock(embed_dims[1], feature_chans, embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.film3 = FilmBlock(embed_dims[2], feature_chans, embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])
        self.mini_patch_embed1 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.mini_patch_embed2 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.mini_patch_embed3 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])
        self.mini_patch_embed4 = OverlapPatchEmbed(img_size // 32, 3, 2, embed_dims[0], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(True, embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.patch_block1 = nn.ModuleList([Block(True, embed_dims[1], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for _ in range(1)])
        self.pnorm1 = norm_layer(embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(False, embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.patch_block2 = nn.ModuleList([Block(False, embed_dims[2], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for _ in range(1)])
        self.pnorm2 = norm_layer(embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(False, embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.patch_block3 = nn.ModuleList([Block(False, embed_dims[3], num_heads[1], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for _ in range(1)])
        self.pnorm3 = norm_layer(embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(False, embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x, feature_vec):
        B, _, H, W = x.shape
        vec = feature_vec
        fv = feature_vec.unsqueeze(2).unsqueeze(3)
        _, c, _, _ = fv.shape
        x = self.input_film(x, fv.expand(B, c, H, W))
        outs = []
        ED = [64, 128, 320, 512]

        x1, H1, W1 = self.patch_embed1(x)
        x2, H2, W2 = self.mini_patch_embed1(x1.permute(0, 2, 1).reshape(B, ED[0], H1, W1))
        for blk in self.block1:
            x1 = blk(x1, H1, W1, vec)
        x1 = self.norm1(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        for blk in self.patch_block1:
            x2 = blk(x2, H2, W2, vec)
        x2 = self.pnorm1(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        x1 = self.film1(x1, fv.expand(B, c, H1, W1))
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, ED[1], H1, W1) + x2
        x2, H2, W2 = self.mini_patch_embed2(x1)
        x1 = x1.view(B, ED[1], -1).permute(0, 2, 1)
        for blk in self.block2:
            x1 = blk(x1, H1, W1, vec)
        x1 = self.norm2(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        for blk in self.patch_block2:
            x2 = blk(x2, H2, W2, vec)
        x2 = self.pnorm2(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        x1 = self.film2(x1, fv.expand(B, c, H1, W1))
        x1, H1, W1 = self.patch_embed3(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, ED[2], H1, W1) + x2
        x2, H2, W2 = self.mini_patch_embed3(x1)
        x1 = x1.view(B, ED[2], -1).permute(0, 2, 1)
        for blk in self.block3:
            x1 = blk(x1, H1, W1, vec)
        x1 = self.norm3(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        for blk in self.patch_block3:
            x2 = blk(x2, H2, W2, vec)
        x2 = self.pnorm3(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        x1 = self.film3(x1, fv.expand(B, c, H1, W1))
        x1, H1, W1 = self.patch_embed4(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, ED[3], H1, W1) + x2
        x1 = x1.view(B, ED[3], -1).permute(0, 2, 1)
        for blk in self.block4:
            x1 = blk(x1, H1, W1, vec)
        x1 = self.norm4(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        return outs


class DecoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[3], embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Block_dec(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x[3]
        B = x.shape[0]
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return [x]


# ─────────────────────────────────────────────────────────────────────────────
# Conv Projection (Decoder → 이미지)
# ─────────────────────────────────────────────────────────────────────────────

class convprojection(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.convd32x = UpsampleConvLayer(512, 512, 4, 2)
        self.convd16x = UpsampleConvLayer(512, 320, 4, 2)
        self.dense_4  = nn.Sequential(ResidualBlock(320))
        self.convd8x  = UpsampleConvLayer(320, 128, 4, 2)
        self.dense_3  = nn.Sequential(ResidualBlock(128))
        self.convd4x  = UpsampleConvLayer(128, 64,  4, 2)
        self.dense_2  = nn.Sequential(ResidualBlock(64))
        self.convd2x  = UpsampleConvLayer(64,  16,  4, 2)
        self.dense_1  = nn.Sequential(ResidualBlock(16))
        self.convd1x  = UpsampleConvLayer(16,  8,   4, 2)

    def _pad_if_needed(self, ref, x):
        ph = ref.shape[2] - x.shape[2]
        pw = ref.shape[3] - x.shape[3]
        if ph != 0 or pw != 0:
            x = F.pad(x, (0, -pw if pw < 0 else 0, 0, -ph if ph < 0 else 0))
        return x

    def forward(self, x1, x2):
        r = self._pad_if_needed(x1[3], self.convd32x(x2[0]))
        r = self.convd16x(r + x1[3])
        r = self._pad_if_needed(x1[2], r)
        r = self.convd8x(self.dense_4(r) + x1[2])
        r = self.convd4x(self.dense_3(r) + x1[1])
        r = self.convd2x(self.dense_2(r) + x1[0])
        r = self.convd1x(self.dense_1(r))
        return r


# ─────────────────────────────────────────────────────────────────────────────
# Top-level 모델 (공개 API)
# ─────────────────────────────────────────────────────────────────────────────

class Tenc(EncoderTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 4, 4], mlp_ratios=[2, 2, 2, 2],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[4, 2, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class Tdec(DecoderTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class Network_top(nn.Module):
    """MWFormer 복원 backbone. StyleFilter 의 feature_vec 을 입력으로 받음."""

    def __init__(self, **kwargs):
        super().__init__()
        self.Tenc     = Tenc()
        self.Tdec     = Tdec()
        self.convtail = convprojection()
        self.clean    = ConvLayer(8, 3, 3, 1, 1)

    def forward(self, x, feature_vec):
        x1 = self.Tenc(x, feature_vec)
        x2 = self.Tdec(x1)
        x  = self.convtail(x1, x2)
        return self.clean(x)
