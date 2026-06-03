# Copied from taco-group/MWFormer (model/style_filter64.py)
# https://github.com/taco-group/MWFormer
# StyleFilter_Top: 입력 이미지에서 날씨 유형을 나타내는 64-dim style vector 추출.

import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (style_filter 전용 — backbone 의 것과 독립적)
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
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act   = act_layer()
        self.fc2   = nn.Linear(hidden_features, out_features)
        self.drop  = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        return self.drop(self.fc2(self.drop(self.act(self.dwconv(self.fc1(x), H, W)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio  = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
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
        nh, hd = self.num_heads, C // self.num_heads
        q = self.q(x).reshape(B, N, nh, hd).permute(0, 2, 1, 3)
        src = x
        if self.sr_ratio > 1:
            src = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W))
                            .reshape(B, C, -1).permute(0, 2, 1))
        kv = self.kv(src).reshape(B, -1, 2, nh, hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
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


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj  = nn.Conv2d(in_chans, embed_dim, patch_size, stride,
                               (patch_size[0] // 2, patch_size[1] // 2))
        self.norm  = nn.LayerNorm(embed_dim)
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
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


# ─────────────────────────────────────────────────────────────────────────────
# Gram matrix & MLP heads
# ─────────────────────────────────────────────────────────────────────────────

def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    t = tensor.view(b, d, h * w)
    return torch.bmm(t, t.permute(0, 2, 1))


class StyleFilter_conv1(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.hidden  = nn.Linear(inputsize, inputsize // 2)
        self.hidden2 = nn.Linear(inputsize // 2, inputsize // 4)
        self.output  = nn.Linear(inputsize // 4, 64)
        self.act     = nn.LeakyReLU()

    def forward(self, x):
        return self.output(self.act(self.hidden2(self.act(self.hidden(x)))))


class StyleFilter_res1(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.hidden = nn.Linear(inputsize, inputsize // 8)
        self.output = nn.Linear(inputsize // 8, 64)
        self.act    = nn.LeakyReLU()

    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 2-scale Encoder for StyleFilter
# ─────────────────────────────────────────────────────────────────────────────

class StyleEncoder(nn.Module):
    def __init__(self, img_size=224, embed_dims=[64, 128], num_heads=[1, 2],
                 mlp_ratios=[2, 2], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[2, 2], sr_ratios=[4, 2]):
        super().__init__()
        self.depths = depths
        self.patch_embed1      = OverlapPatchEmbed(img_size, 7, 4, 3, embed_dims[0])
        self.patch_embed2      = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])
        self.mini_patch_embed1 = OverlapPatchEmbed(img_size // 4, 3, 2, embed_dims[0], embed_dims[1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1      = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1       = norm_layer(embed_dims[0])
        self.patch_block1 = nn.ModuleList([Block(embed_dims[1], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for _ in range(1)])
        self.pnorm1      = norm_layer(embed_dims[1])
        cur += depths[0]
        self.block2      = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate, dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2       = norm_layer(embed_dims[1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        ED = [64, 128]
        x1, H1, W1 = self.patch_embed1(x)
        x2, H2, W2 = self.mini_patch_embed1(x1.permute(0, 2, 1).reshape(B, ED[0], H1, W1))
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        for blk in self.patch_block1:
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm1(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = (x1.permute(0, 2, 1).reshape(B, ED[1], H1, W1) + x2).view(B, ED[1], -1).permute(0, 2, 1)
        for blk in self.block2:
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        return [x2, x1]  # [stage1_feat, stage2_feat]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level 모델 (공개 API)
# ─────────────────────────────────────────────────────────────────────────────

class StyleFilter_Top(nn.Module):
    """입력 이미지 → 64-dim 날씨 style vector."""

    def __init__(self):
        super().__init__()
        self.encoder       = StyleEncoder(
            patch_size=4, embed_dims=[64, 128], num_heads=[1, 2],
            mlp_ratios=[2, 2], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2], sr_ratios=[4, 2],
            drop_rate=0.0, drop_path_rate=0.1)
        self.style_filter1 = StyleFilter_conv1(2080)
        self.style_filter2 = StyleFilter_res1(8256)
        self.out1_fc       = nn.Linear(128, 64)
        self.layernorm     = nn.LayerNorm([64])

    def forward(self, x):
        enc_out = self.encoder(x)
        g1 = gram_matrix(enc_out[0])
        g2 = gram_matrix(enc_out[1])
        b1, d1, _ = g1.shape
        b2, d2, _ = g2.shape
        idx1 = torch.triu(torch.ones(d1, d1)) == 1
        idx2 = torch.triu(torch.ones(d2, d2)) == 1
        v1 = self.style_filter1(g1[:, idx1].view(b1, -1))
        v2 = self.style_filter2(g2[:, idx2].view(b2, -1))
        return self.layernorm(self.out1_fc(torch.cat([v1, v2], dim=1)))
