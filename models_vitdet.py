# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
import os
import shutil
from functools import partial

import cv2
import numpy as np
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block, DropPath, HybridEmbed, Mlp

# if os.path.exists('images'):
#     shutil.rmtree('images')
# os.makedirs('images', exist_ok=True)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowedAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=14,
                 pad_mode="constant"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.pad_mode = pad_mode

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode=self.pad_mode)

        qkv = F.unfold(qkv,
                       kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        # if self.mask:
        #     attn = attn * mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(x,
                   output_size=(H_, W_),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size))  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 windowed=False,
                 window_size=14,
                 pad_mode="constant",
                 layer_scale=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if windowed:
            self.attn = WindowedAttention(dim,
                                          num_heads=num_heads,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          attn_drop=attn_drop,
                                          proj_drop=drop,
                                          window_size=window_size,
                                          pad_mode=pad_mode)
        else:
            self.attn = Attention(dim,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  attn_drop=attn_drop,
                                  proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x.clone()), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer, nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,
                 global_pool=False,
                 mask_ratio=None,
                 mask_type='random',
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 window_attn=False,
                 window_size=14):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        self.window_attn = window_attn

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.blocks = nn.ModuleList([
        #     Block(dim=embed_dim,
        #           num_heads=num_heads,
        #           mlp_ratio=mlp_ratio,
        #           qkv_bias=qkv_bias,
        #           qk_scale=qk_scale,
        #           drop=drop_rate,
        #           attn_drop=attn_drop_rate,
        #           drop_path=dpr[i],
        #           norm_layer=norm_layer) for i in range(depth)
        # ])
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  windowed=window_attn[i],
                  window_size=window_size[i]) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.global_pool = global_pool
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        if mask_ratio is not None:
            assert self.mask_type == 'uniform'
            print(f'mask_ratio: {mask_ratio}, mask_type: {mask_type}')
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def masking(self, x, H, W):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if True:  # self.mask_type == 'uniform'
            M = int(L**0.5)
            noise = rearrange(noise, 'n (h p1 w p2) -> (n h w) (p1 p2)', n=N, p1=2, p2=2, h=M // 2, w=M // 2)
            if self.mask_ratio == 0.75:
                index = noise.min(-1)[1]
                noise[range(len(index)), index] = -1
                H, W = H // 2, W // 2
            elif self.mask_ratio == 0.5:
                index = noise.topk(k=2, dim=-1, largest=False)[1]
                noise[range(len(index)), index[:, 0]] = -len(index) + torch.arange(
                    (len(index)), device=x.device).float()
                noise[range(len(index)), index[:, 1]] = -len(index) + torch.arange(
                    (len(index)), device=x.device).float()
                H, W = H // 2, W
            else:
                raise NotImplementedError
            noise = rearrange(noise, '(n h w) (p1 p2)-> n (h p1 w p2) ', n=N, p1=2, p2=2, h=M // 2, w=M // 2)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # to save memory, do not calculate the mask
        mask = None
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, H, W

    def forward_features(self, x):
        B, _, HH, WW = x.shape
        # img = x.clone()
        x, H, W = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.mask_ratio is not None and self.training:
            x, mask, H, W = self.masking(x, H, W)

            # save image
            # mean = np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, -1)
            # std = np.array(IMAGENET_DEFAULT_STD).reshape(1, 1, -1)
            # N, L = mask.shape
            # M = int(L**0.5)
            # mask = mask.reshape(N, M, M)
            # mask = mask.repeat_interleave(HH // M, 1).repeat_interleave(WW // M, 2).unsqueeze(1).contiguous().permute(
            #     0, 2, 3, 1).cpu().numpy()  # (N, H, W, 1)
            # img = img.permute(0, 2, 3, 1).cpu().numpy()
            # for i in range(N):
            #     real_img = cv2.cvtColor(np.uint8(255 * ((img[i] * std) + mean)), cv2.COLOR_RGB2BGR)
            #     mask_img = cv2.cvtColor(np.uint8(255 * ((img[i] * (1 - mask[i]) * std) + mean)), cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(f'images/{img[i][:2,0,0]}.png', np.concatenate([real_img, mask_img], 1))

        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if self.window_attn[i]:
                x[:, 1:, :] = blk(x[:, 1:, :], H, W)
            else:
                x = blk(x, H, W)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vitdet_base_patch16(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              window_attn=[True, True, False] * 4,
                              window_size=[14, 14, None] * 4,
                              **kwargs)
    return model


def vitdet_large_patch16(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              window_attn=[True, True, True, True, True, False] * 4,
                              window_size=[14, 14, 14, 14, 14, None] * 4,
                              **kwargs)
    return model


def vitdet_huge_patch14(**kwargs):
    model = VisionTransformer(patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              window_attn=[True, True, True, True, True, True, True, False] * 4,
                              window_size=[14, 14, 14, 14, 14, 14, 14, None] * 4,
                              **kwargs)
    return model
