import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mambalct.hivit import hivit_small,hivit_base



class MambaLCT(nn.Module):
    """ This is the base class for ODTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            #256(bs,3,256,256),(bs,3,128,128)-->patch(bs,384,512)
            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (aux_dict["temproal_token"].clone()).detach()
                
            att = torch.matmul(enc_opt,self.track_query.transpose(1, 2))  # (B, HW, N)
            att = torch.mean(att,dim=2,keepdim=True)

            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_mambalct(cfg, training=True):
    # current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    current_dir = "/home/local_data/lxh/history_work/MambaLCT"
    pretrained_path = os.path.join(current_dir,'pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''


    if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
        backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)


    elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
        backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)


    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = MambaLCT(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )


    return model
