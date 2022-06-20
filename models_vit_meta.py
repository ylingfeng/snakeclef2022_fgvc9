# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MetaFormer: https://github.com/dqshuai/MetaFormer
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class ResNormLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, meta_dims=[4, 3], **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.meta_dims = meta_dims
        for ind, meta_dim in enumerate(meta_dims):
            meta_head = nn.Sequential(
                nn.Linear(meta_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dim),
                ResNormLayer(self.embed_dim),
            ) if meta_dim > 0 else nn.Identity()
            setattr(self, f"meta_{ind+1}_head", meta_head)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1 + len(meta_dims), self.embed_dim),
                                      requires_grad=True)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def forward_features(self, x, meta):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        # forward metadata
        extra_tokens = [cls_tokens]
        if len(self.meta_dims) > 1:
            metas = torch.split(meta, self.meta_dims, dim=1)
        else:
            metas = (meta, )
        for ind, cur_meta in enumerate(metas):
            meta_head = getattr(self, f"meta_{ind+1}_head")
            meta = meta_head(cur_meta)
            meta = meta.reshape(B, -1, self.embed_dim)
            extra_tokens.append(meta)

        extra_tokens.append(x)
        x = torch.cat(extra_tokens, dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, meta):
        x = self.forward_features(x, meta)
        x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              **kwargs)
    return model
