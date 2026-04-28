"""
TBSI_Track model. Developed on OSTrack.
"""
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv, MLP
from lib.models.layers.score_decoder import ScoreDecoder
from lib.models.tbsi_track.vit_tbsi_care import vit_base_patch16_224_tbsi
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_xywh_to_xyxy


class TBSITrack(nn.Module):
    """ This is the base class for TBSITrack developed on OSTrack (Ye et al. ECCV 2022) """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", cls_head=None, cls_head_tir=None, cls_flag=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cls_head = cls_head
        self.cls_head_tir = cls_head_tir

        self.cls_flag = cls_flag

        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.tbsi_fuse_search = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head

        self.cls_fuse_search = conv(hidden_dim * 2, hidden_dim)
        self.cls_fuse_template = conv(hidden_dim * 2, hidden_dim)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gt_bboxes=None):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, gt_score_map=None, cls_flag=self.cls_flag, gt_bboxes=gt_bboxes)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None, cls_flag=False, gt_bboxes=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, 64:320, :]
        enc_opt2 = cat_feature[:, 384:640, :]  # [bs,256,768]
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)  # [bs,256,768x2]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # [bs,1,768x2,256]
        bs, Nq, C, HW = opt.size()
        HW = int(HW / 2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # [bs,768x2,16,16]
        opt_feat = self.tbsi_fuse_search(opt_feat)  # [bs,768,16,16]

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            # return out
        elif self.head_type == "CENTER":
            # run the center head # bbox 为 (cx,cy,w,h) 格式
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            # return out
        else:
            raise NotImplementedError

        if cls_flag:  # update zzf
            template_rgb = cat_feature[:, :64, :]
            search_rgb = cat_feature[:, 64:320, :]
            template_tir = cat_feature[:, 320:384, :]
            search_tir = cat_feature[:, 384:640, :]

            template_rgb = template_rgb.permute((0, 2, 1)).contiguous().view(-1, 768, 8, 8)
            template_tir = template_tir.permute((0, 2, 1)).contiguous().view(-1, 768, 8, 8)
            search_rgb = search_rgb.permute((0, 2, 1)).contiguous().view(-1, 768, 16, 16)
            search_tir = search_tir.permute((0, 2, 1)).contiguous().view(-1, 768, 16, 16)

            # search_fuse = self.cls_fuse_search(torch.cat([search_rgb, search_tir], dim=1))
            # template_fuse = self.cls_fuse_template(torch.cat([template_rgb, template_tir], dim=1))

            if gt_bboxes is None:
                new_gt_box = box_cxcywh_to_xyxy(outputs_coord.clone().view(-1, 4))  # outputs_coord=(cx,cy,w,h)
            else:
                new_gt_box = box_xywh_to_xyxy(gt_bboxes.clone().view(-1, 4))  # gt_bboxes=(x,y,w,h)
            out.update({'cls_score': self.cls_head(search_rgb, template_rgb, new_gt_box).view(-1)})  # (bs,1,1) 先search再template
            out.update({'cls_score_tir': self.cls_head_tir(search_tir, template_tir, new_gt_box.clone()).view(-1)})

        return out


def build_tbsi_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TBSITrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_tbsi':
        backbone = vit_base_patch16_224_tbsi(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                             tbsi_loc=cfg.MODEL.BACKBONE.TBSI_LOC,
                                             tbsi_drop_path=cfg.TRAIN.TBSI_DROP_PATH,
                                             num_template=cfg.DATA.TEMPLATE.NUMBER
                                             )
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    # the proposed score prediction module (SPM)
    score_branch = ScoreDecoder(pool_size=4, hidden_dim=768, num_heads=768 // 64)
    score_branch_tir = ScoreDecoder(pool_size=4, hidden_dim=768, num_heads=768 // 64)

    model = TBSITrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cls_head=score_branch,
        cls_head_tir=score_branch_tir,
        cls_flag=cfg.TRAIN.TRAIN_CLS
    )

    if 'TBSITrack' in cfg.MODEL.PRETRAIN_FILE and training:
        if cfg.TRAIN.TRAIN_CLS:
            pretrained_file = cfg.MODEL.PRETRAIN_FILE
            checkpoint = torch.load(pretrained_file, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        else:
            pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
            checkpoint = torch.load(pretrained_file, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
