from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from lib.models.mambalct.utils import combine_tokens, recover_tokens
from lib.models.mambalct.mamba import vim_small_patch16_224


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = True
        self.add_sep_seg = False



    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.cls_token_len =cfg.MODEL.BACKBONE.CLS_TOKEN_LEN

        patch_pos_embed = self.absolute_pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and self.cls_token_len > 0:
            cls_pos_embed = self.pos_embed[:, 0:self.cls_token_len, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        self.mamba_moudle = vim_small_patch16_224(embed_dim=self.embed_dim)
        self.mamba_layers = cfg.MODEL.BACKBONE.MAMBA_LAYER



    def forward_features(self, z, x, mask=None,temporal_query=None):
        B = x.shape[0]

        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        z = self.patch_embed(z)


        x = self.patch_embed(x)


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, self.cls_token_len, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        for blk in self.blocks[:-self.num_main_blocks]:
            x = blk(x)
            z = blk(z)

        x = x[..., 0, 0, :]
        z = z[..., 0, 0, :]

        z += self.pos_embed_z
        x += self.pos_embed_x

        if T_z > 1:  # multiple memory frames
            z = z.view(B, T_z, -1, z.size()[-1]).contiguous()
            z = z.flatten(1, 2)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        len_search = x.shape[1]
        len_template = z.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            if temporal_query is None:
                x = torch.cat([cls_tokens, x], dim=1)
            else:
                x = torch.cat([temporal_query, x], dim=1)

        x = self.pos_drop(x)


        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            if i in self.mamba_layers:
                x,att = blk(x)
                x = self.mamba_moudle(x,len_search,len_template,self.cls_token_len)
            else:
                x,att = blk(x)

        # for blk in self.blocks[-self.num_main_blocks:]:
        #     x = blk(x)
        #     x = self.mamba_moudle(x,len_search,len_template)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        if self.add_cls_token:
            cls = x[:,:self.cls_token_len]
            x = x[:,self.cls_token_len:]

        aux_dict = {"attn": att,
                    "temproal_token": cls}
        x = self.norm_(x)

        return x, aux_dict

    def forward(self, z, x,temporal_query=None, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic HiViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x,temporal_query=temporal_query )

        return x, aux_dict
