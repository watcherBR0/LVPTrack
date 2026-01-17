import os
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, map_boxes_back, map_boxes_back_batch, clip_box, clip_box_batch, batch_bbox_voting
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from lib.utils.pseudo_label_save import write_to_txt
class LightUAVActor(BaseActor):
    """ Actor for training LightUAV models """
    #0708
    def __init__(self, net, net_extreme, objective, loss_weight, settings, cfg=None):
        super().__init__(net, net_extreme, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    #0703
    def __call__(self, data, loader_type):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, loader_type)

        # compute losses
        loss, status = self.compute_losses(out_dict, data, loader_type=loader_type)

        #0712
        if loader_type != 'train_extreme':
            return loss, status
        else:
            return loss, status, out_dict

    def forward_pass(self, data, loader_type):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        template_data = data['template_anno'][0]
        template_bb = box_xywh_to_xyxy(template_data).clamp(min=0.0,max=1.0)

        if len(template_list) == 1:
            template_list = template_list[0]

        #0708
        if "extreme" in loader_type:
            out_dict = self.net_extreme(template=template_list,
                            search=search_img,
                            template_bb=template_bb,
                            mode='train',
                            loader_type=loader_type)
            #0709
            # if loader_type == "train_extreme":
            #     #save pseudo-label
            #     out_dict['img_paths'] = data['search_images_path'][0]
            #     #0710
            #     #mapback pred boxes
            #     #permute
            #     data['search_box_extract'] = data['search_box_extract'].permute(1,0)
            #     data['search_resize_factors'] = data['search_resize_factors'].permute(1,0)
            #     data['search_original_shape'] = data['search_original_shape'].permute(1,0)

            #     #0719
            #     bbox_optimize = batch_bbox_voting(out_dict['pred_boxes'], out_dict['topk_score'])

            #     # out_dict['mapback_pred_boxes'] = torch.tensor(out_dict['pred_boxes'].mean(dim=1) * 256 / data['search_resize_factors'])
            #     # out_dict['mapback_pred_boxes'] = torch.tensor(out_dict['pred_boxes'][:, 0, :] * 256 / data['search_resize_factors'])
            #     out_dict['mapback_pred_boxes'] = torch.tensor(bbox_optimize.squeeze() * 256 / data['search_resize_factors'])

            #     # test_one = clip_box(map_boxes_back(data['search_box_extract'][1], out_dict['mapback_pred_boxes'][1],
            #     #                                                          data['search_resize_factors'][1]), 
            #     #                                                          data['search_original_shape'][1][0], 
            #     #                                                          data['search_original_shape'][1][1], 
            #     #                                                          margin=10)

            #     out_dict['mapback_pred_boxes'] = clip_box_batch(map_boxes_back_batch(data['search_box_extract'], out_dict['mapback_pred_boxes'],
            #                                                              data['search_resize_factors']),
            #                                                              data['search_original_shape'], 
            #                                                              margin=10)

            #     for i in range(len(out_dict['img_paths'])):
            #         #split img_path to list
            #         path_list = out_dict['img_paths'][i].split('/')
            #         #pseudo-label save path
            #         pl_save_dir = os.path.join(data['settings'].save_dir, 'pseudo_label', path_list[-3], path_list[-2])
            #         #img index
            #         img_id = int(path_list[-1].split('.')[0])
            #         txt_path = os.path.join(pl_save_dir, 'pl.txt')
            #         if not os.path.exists(pl_save_dir):
            #             os.makedirs(pl_save_dir)
            #             #create pl.txt
            #             open(txt_path, 'a').close()
            #         write_to_txt(txt_path, img_id, out_dict['mapback_pred_boxes'][i])

        else:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                template_bb=template_bb,
                                mode='train',
                                loader_type=loader_type)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, loader_type=''):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        #0719
        if loader_type == 'train_extreme':
            pred_boxes = pred_dict['pred_boxes'][:, 0:1, :]
        else:
            pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
