import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import init

from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.simmim import SimMIM, build_simmim
from models.bert import BertConfig, BertForMaskedLM

from utils import read_json
from models.config import get_config
import numpy as np
from torchvision.transforms import ToTensor
from argparse import Namespace


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply


def build_vision_encoder(config):

    vision_config = read_json(config['vision_config'])
    vision_width = vision_config['vision_width']

    vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                     h=vision_config['h'],
                                     w=vision_config['w'],
                                     embed_dim=vision_config['embed_dim'],
                                     depths=vision_config['depths'],
                                     num_heads=vision_config['num_heads'],
                                     window_size=vision_config['window_size'],
                                     drop_rate=vision_config['drop_rate'],
                                     drop_path_rate=vision_config['drop_path_rate'],
                                     num_classes=0)

    if config['load_params_vision']:
        # download from https://github.com/microsoft/Swin-Transformer
        state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']
        print(f"build_vision_encoder: load swin {vision_config['h']} x {vision_config['w']} ====>")
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys, flush=True)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width):

    config_text = BertConfig.from_json_file(config['text_config'])
    config_text.encoder_width = vision_width
    text_encoder, msg = BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                        output_loading_info=True)
    if config['load_params_text']:
        print("build_text_encoder: load bert ====>")
        for k, v in msg.items():
            print(f"{k}: {sorted(v)}")
    # print(text_encoder)
    # for name, param in text_encoder.state_dict().items():
    #     print(f"Parameter Name: {name}, Shape: {param.shape}")
    return text_encoder


def feature_mapping(input_dim, output_dim, dropout_p=0):
    mlp = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(input_dim, output_dim),
    )
    init.normal_(mlp[2].weight.data, std=0.00001)
    init.constant_(mlp[2].bias.data, 0.0)
    return mlp


class CMP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # image encoder
        self.vision_encoder, vision_width = build_vision_encoder(config)
        self.vision_width = vision_width

        # text & cross encoder
        self.text_encoder = build_text_encoder(config, vision_width=self.vision_width)
        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width

        self.embed_dim = config['embed_dim']
        self.avgpool = nn.AdaptiveAvgPool1d(3)
        self.vision_proj = feature_mapping(3072, self.embed_dim, config['itc_dp'])
        self.text_proj = feature_mapping(3072, self.embed_dim, config['itc_dp'])

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        
        args = Namespace(cfg=config['config_mim'])
        self.config_mim = get_config(args)
        self.simmim = build_simmim(self.config_mim, self.vision_encoder)
        self.maskgen = MaskGenerator()

        self.epsilon = config['label_smooth']
        print('Label Smooth:', self.epsilon)

        input_dim = self.text_width
        output_dim = 2
        self.itm_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim))

        self.init_params = []  # train with 2 * lr
        # image encoder
        for i in range(2, 4):
            for name, param in self.vision_encoder.layers[i].named_parameters():
                self.init_params.extend(['vision_encoder.layers.' + str(i) + '.' + name])

        # text & cross encoder
        temp_encoder = self.text_encoder.bert
        temp_name = 'text_encoder.bert.encoder.layer.'
        temp_list = [4, 5, 10, 11]
        for i in temp_list:
            for name, param in temp_encoder.encoder.layer[i].named_parameters():
                self.init_params.extend([temp_name + str(i) + '.' + name])
        self.init_params.extend(
            ['text_encoder.cls.' + n for n, _ in self.text_encoder.cls.named_parameters()])

        self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
        self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])
        self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])


    def load_pretrained(self, ckpt_rpath):
        checkpoint = torch.load(ckpt_rpath, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys])
        print("unexpected_keys: ", msg.unexpected_keys)


    def get_vision_embeds(self, image):
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts


    def get_text_embeds(self, text_ids, text_atts):
        encoder = self.text_encoder.bert
        return encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text').last_hidden_state


    def get_cross_embeds(self, image_embeds, image_atts, text_embeds, text_atts,):
        encoder = self.text_encoder.bert
        return encoder(encoder_embeds=text_embeds,
                       attention_mask=text_atts,
                       encoder_hidden_states=image_embeds,
                       encoder_attention_mask=image_atts,
                       return_dict=True,
                       mode='fusion',
                       ).last_hidden_state


    def get_image_feat(self, image_embeds):
        x = self.avgpool(image_embeds[:, 1:, ].transpose(1, 2))
        x = x.transpose(1, 2)
        x = torch.cat([x[:, 0, :], x[:, 1, :], x[:, 2, :]], dim=1)
        image_feat = self.vision_proj(x)
        return image_feat


    def get_text_feat(self, text_embeds):
        x = self.avgpool(text_embeds[:, 1:, ].transpose(1, 2))  # B C 3
        x = x.transpose(1, 2)  # B 3 C
        x = torch.cat([text_embeds[:, 0, :], x[:, 0, :], x[:, 1, :], x[:, 2, :]], dim=1)
        text_feat = self.text_proj(x)
        return text_feat


    def get_contrastive_loss(self, image_feat, text_feat, idx):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        idx = idx.view(-1, 1)
        assert idx.size(0) == image_feat.size(0)
        idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)

        if 0 < self.epsilon < 1:
            _, num_classes = logits.shape
            labels = (1 - self.epsilon) * labels + self.epsilon / num_classes

        loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2


    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None, ):
        """
        Matching Loss with in-batch hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            idx = idx.view(-1, 1)
            assert idx.size(0) == bs
            mask = torch.eq(idx, idx.t())
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts,
                                          text_embeds=text_embeds, text_atts=text_atts,)[:, 0, :]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all,
                                          text_embeds=text_embeds_all, text_atts=text_atts_all,)[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        return itm_loss


    def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids, ):
        return self.text_encoder(text_ids_masked,
                                 attention_mask=text_atts,
                                 encoder_hidden_states=image_embeds,
                                 encoder_attention_mask=image_atts,
                                 return_dict=True,
                                 labels=masked_ids,
                                 masked_pos=masked_pos).loss
    
    def get_mim_loss(self, x):
        mask = self.maskgen()
        totensor = ToTensor()
        mask = totensor(mask).to('cuda')
        return self.simmim(x, mask)

