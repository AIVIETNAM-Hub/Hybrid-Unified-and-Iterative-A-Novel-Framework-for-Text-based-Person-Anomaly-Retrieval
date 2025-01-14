# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer


class SwinTransformerForSimMIM(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        

        self.encoder = encoder
        # print(self.encoder.num_classes)
        # assert self.encoder.num_classes == 0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.encoder.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.encoder.ape:
            x = x + self.absolute_pos_embed
        x = self.encoder.pos_drop(x)

        for layer in self.encoder.layers:
            x = layer(x)
        x = self.encoder.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.encoder.in_chans
        self.patch_size = self.encoder.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config, encoder=None):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        # if encoder==None:
        #     encoder = SwinTransformerForSimMIM(
        #     img_size=config.DATA.IMG_SIZE,
        #     patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #     in_chans=config.MODEL.SWIN.IN_CHANS,
        #     num_classes=0,
        #     embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #     depths=config.MODEL.SWIN.DEPTHS,
        #     num_heads=config.MODEL.SWIN.NUM_HEADS,
        #     window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #     qk_scale=config.MODEL.SWIN.QK_SCALE,
        #     drop_rate=config.MODEL.DROP_RATE,
        #     drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #     ape=config.MODEL.SWIN.APE,
        #     patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        # else:
        encoder = SwinTransformerForSimMIM(encoder)
        encoder_stride = 32

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride).to('cuda')

    return model
