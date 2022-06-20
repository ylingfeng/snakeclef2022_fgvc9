# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
from einops.einops import rearrange

# if os.path.exists('images'):
#     shutil.rmtree('images')
# os.makedirs('images', exist_ok=True)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_ratio=None, mask_type='random', **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        print(f'mask_type: {mask_type}, mask_ratio: {mask_ratio}')
        self.num_patches = self.patch_embed.num_patches
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.mask_type == 'uniform':
            M = int(L**0.5)
            noise = rearrange(noise, 'n (h p1 w p2) -> (n h w) (p1 p2)', n=N, p1=2, p2=2, h=M // 2, w=M // 2)
            if self.mask_ratio == 0.75:
                index = noise.min(-1)[1]
                noise[range(len(index)), index] = -1
            elif self.mask_ratio == 0.5:
                index = noise.topk(k=2, dim=-1, largest=False)[1]
                noise[range(len(index)), index[:, 0]] = -1
                noise[range(len(index)), index[:, 1]] = -1
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
        return x_masked, mask

    def forward_features(self, x, mask):
        B, _, H, W = x.shape
        # img = x.clone()
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.mask_ratio is not None and self.mask_ratio > 0 and self.training:
            # masking: length -> length * mask_ratio
            if False:
                ids_shuffle = torch.argsort(mask, dim=1)  # ascend: small is keep, large is remove
                ids_keep = ids_shuffle[:, :14 * 14]
                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1024))
            else:
                assert mask is None
                x, mask = self.masking(x)

            # save image
            # mean = np.array(IMAGENET_DEFAULT_MEAN).reshape(1, 1, -1)
            # std = np.array(IMAGENET_DEFAULT_STD).reshape(1, 1, -1)
            # N, L = mask.shape
            # M = int(L**0.5)
            # mask = mask.reshape(N, M, M)
            # mask = mask.repeat_interleave(H // M, 1).repeat_interleave(W // M, 2).unsqueeze(1).contiguous().permute(
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

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
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


def vit_large_patch8(**kwargs):
    model = VisionTransformer(patch_size=8,
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
