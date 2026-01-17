"""
Basic LightUAV model.
"""
import math
import os
# import sys
# sys.path.insert(0,'./lib/models/lightUAV')
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones


from lib.models.layers.head import build_box_head
from .utils import combine_tokens, token2feature, feature2token
from lib.models.lightUAV.vit import vit_base_patch16_224
from lib.models.lightUAV.vit import CAE_Base_patch16_224_Async
from timm.models.layers import Mlp
from timm.models.layers import to_2tuple
from lib.utils.box_ops import box_xyxy_to_cxcywh



class LightUAV(nn.Module):
    """ This is the base class for LightUAV """
    #0708
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CENTER", add_target_token=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        #0707
        # if box_head_extreme is not None:
        #     self.box_head_extreme = box_head_extreme

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = self.feat_size_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.add_target_token = add_target_token
        if self.add_target_token:
            self.target_token_embed = Mlp(4, out_features=self.backbone.embed_dim)
            
    def forward_z(self, template: torch.Tensor, template_bbox=None):
        target_token = None
        if self.add_target_token:
            target_token = self.target_token_embed(template_bbox).unsqueeze(-2)
        return self.backbone(z=template, target_token=target_token, mode='z')

    def forward(self, template: torch.Tensor, search: torch.Tensor, loader_type='train_mix', template_bb=None, mode='train'):
        # Forward backbone
        if self.add_target_token and mode == 'z' and template_bb is not None:
            target_token = self.target_token_embed(template_bb).unsqueeze(-2)
        else:
            target_token = None
        x = self.backbone(z=template, x=search, mode=mode, target_token=target_token, loader_type=loader_type)
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        
        #0707
        # if 'extreme' in loader_type:
        #     out = self.forward_head_extreme(feat_last, None)
        # else:
        #0708/0718
        out = self.forward_head(feat_last, None, loader_type)
        # out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None, loader_type='train_mix'):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
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
            #0718
            if loader_type == 'train_extreme':
                score_map_ctr, bbox, size_map, offset_map, topk_score = self.box_head(x=opt_feat, feat_size=self.feat_sz_s, gt_score_map=gt_score_map,
                                                                        loader_type=loader_type)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(bs, topk_score.shape[1], 4)
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_score': topk_score}
            else:
                score_map_ctr, bbox, size_map, offset_map = self.box_head(x=opt_feat, feat_size=self.feat_sz_s, gt_score_map=gt_score_map,
                                                                        loader_type=loader_type)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map}
            # outputs_coord = box_xyxy_to_cxcywh(bbox)

            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # out = {'pred_boxes': outputs_coord_new,
            #        'score_map': score_map_ctr,
            #        'size_map': size_map,
            #        'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    
    #0707
    # def forward_head_extreme(self, cat_feature, gt_score_map=None):
    #     """
    #     cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
    #     """
    #     enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
    #     opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
    #     bs, Nq, C, HW = opt.size()
    #     opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

    #     if self.head_type == "CORNER":
    #         # run the corner head
    #         pred_box, score_map = self.box_head(opt_feat, True)
    #         outputs_coord = box_xyxy_to_cxcywh(pred_box)
    #         outputs_coord_new = outputs_coord.view(bs, Nq, 4)
    #         out = {'pred_boxes': outputs_coord_new,
    #                'score_map': score_map,
    #                }
    #         return out

    #     elif self.head_type == "CENTER":
    #         # run the center head
    #         score_map_ctr, bbox, size_map, offset_map = self.box_head(x=opt_feat, feat_size=self.feat_sz_s, gt_score_map=gt_score_map)
    #         # outputs_coord = box_xyxy_to_cxcywh(bbox)
    #         outputs_coord = bbox
    #         outputs_coord_new = outputs_coord.view(bs, Nq, 4)
    #         out = {'pred_boxes': outputs_coord_new,
    #                'score_map': score_map_ctr,
    #                'size_map': size_map,
    #                'offset_map': offset_map}
    #         return out
    #     else:
    #         raise NotImplementedError


def build_lightUAV(cfg, training=True, extreme=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('lightUAV' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_CAE':
        backbone = CAE_Base_patch16_224_Async(pretrained,
                                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              self_blocks_num=cfg.MODEL.BACKBONE.SELF_BLOCKS_NUM,
                                              cross_blocks_num=cfg.MODEL.BACKBONE.CROSS_BLOCKS_NUM,
                                              depth=cfg.MODEL.BACKBONE.DEPTH,
                                              add_target_token=cfg.MODEL.ADD_TARGET_TOKEN,
                                              attention=cfg.MODEL.BACKBONE.ATTENTION_TYPE,
                                              search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                              template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                              new_patch_size=cfg.MODEL.BACKBONE.STRIDE
                                              )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    #0707
    # box_head_extreme = build_box_head(cfg, hidden_dim)

    model = LightUAV(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        add_target_token=cfg.MODEL.ADD_TARGET_TOKEN
    )

    if 'lightUAV' in cfg.MODEL.PRETRAIN_FILE and training and not extreme:
        checkpoint_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print("missing_keys:{}".format(missing_keys))
        print("unexpected_keys:{}".format(unexpected_keys))
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
