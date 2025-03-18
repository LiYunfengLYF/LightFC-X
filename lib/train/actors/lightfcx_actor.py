import numpy as np

from . import BaseActor, ACTOR_Registry
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap


@ACTOR_Registry.register()
class RGBTer_empty_Actor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.last_3epoch_l1 = True

    def __call__(self, data):
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
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['visible']['template_images']) == 1
        # assert len(data['visible']['search_images']) == 1

        template_img_v = data['visible']['template_images'][:, 0, :].view(-1, *data['visible']['template_images'].shape[
                                                                               2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][:, 0, :].view(-1,
                                                                           *data['infrared']['template_images'].shape[
                                                                            2:])  # (batch, 3, 128, 128)

        search_img_v = data['visible']['search_images'][:, 0, :].view(-1, *data['visible']['search_images'].shape[
                                                                           2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][:, 0, :].view(-1, *data['infrared']['search_images'].shape[
                                                                            2:])  # (batch, 3, 320, 320)
        return None

    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        # compute giou and iou
        giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss = torch.tensor(0.0, device='cuda:0')  # (BN,4) (BN,4)

        # compute location loss
        location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = torch.tensor(0.0, device=l1_loss.device)
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


@ACTOR_Registry.register()
class RGBTer_lightfc_Actor(BaseActor):

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
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
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'], epoch=data['epoch'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['visible']['template_images']) == 1
        # assert len(data['visible']['search_images']) == 1

        template_img_v = data['visible']['template_images'][:, 0, :].view(-1, *data['visible']['template_images'].shape[
                                                                               2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][:, 0, :].view(-1,
                                                                           *data['infrared']['template_images'].shape[
                                                                            2:])  # (batch, 3, 128, 128)

        search_img_v = data['visible']['search_images'][:, 0, :].view(-1, *data['visible']['search_images'].shape[
                                                                           2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][:, 0, :].view(-1, *data['infrared']['search_images'].shape[
                                                                            2:])  # (batch, 3, 320, 320)

        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i], )
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, epoch=0):

        # gt gaussian map
        bs, n, _ = gt_dict['search_anno'].shape
        gt_bbox = gt_dict['search_anno'].view(bs, 4)

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'].view(n, bs, 4), self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

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

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss
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


@ACTOR_Registry.register()
class RGBTer_lightfc_update_Actor(BaseActor):

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
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
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'], epoch=data['epoch'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['visible']['template_images']) == 1
        # assert len(data['visible']['search_images']) == 1

        template_img_v_1 = data['visible']['template_images'][:, 0, :].view(-1,
                                                                            *data['visible']['template_images'].shape[
                                                                             2:])  # (batch, 3, 128, 128)
        template_img_i_1 = data['infrared']['template_images'][:, 0, :].view(-1,
                                                                             *data['infrared']['template_images'].shape[
                                                                              2:])  # (batch, 3, 128, 128)

        template_img_v_2 = data['visible']['template_images'][:, 1, :].view(-1,
                                                                            *data['visible']['template_images'].shape[
                                                                             2:])  # (batch, 3, 128, 128)
        template_img_i_2 = data['infrared']['template_images'][:, 1, :].view(-1,
                                                                             *data['infrared']['template_images'].shape[
                                                                              2:])  # (batch, 3, 128, 128)

        search_img_v = data['visible']['search_images'][:, 0, :].view(-1, *data['visible']['search_images'].shape[
                                                                           2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][:, 0, :].view(-1, *data['infrared']['search_images'].shape[
                                                                            2:])  # (batch, 3, 320, 320)

        out_dict = self.net(template=[[template_img_v_1, template_img_i_1], [template_img_v_2, template_img_i_2]],
                            search=[search_img_v, search_img_i], )
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, epoch=0):

        # gt gaussian map
        bs, n, _ = gt_dict['search_anno'].shape
        gt_bbox = gt_dict['search_anno'].view(bs, 4)

        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'].view(n, bs, 4), self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

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

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'focal'] * location_loss
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


@ACTOR_Registry.register()
class RGBSer_lightfc_Actor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
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
        out_rgb, out_s = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_rgb, out_s, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        template_img_1, template_img_2, search_img_1, search_img_2 = self.data2temp_search(data)

        out_dict = self.net(template=[template_img_1, template_img_2],
                            search=[search_img_1, search_img_2], )
        return out_dict

    def compute_losses(self, predrgb_dict, preds_dict, gt_dict, return_status=True):
        rgb_seach_anno = gt_dict['search_anno'][:, 0, :].unsqueeze(dim=1)
        sonar_seach_anno = gt_dict['search_anno'][:, 1, :].unsqueeze(dim=1)

        # rgb gt gaussian map
        bs, n, _ = rgb_seach_anno.shape
        rgb_gt_bbox = rgb_seach_anno.view(bs, 4)

        rgb_gt_gaussian_maps = generate_heatmap(rgb_seach_anno.view(n, bs, 4), self.cfg.DATA.SEARCH.SIZE,
                                                self.cfg.MODEL.BACKBONE.STRIDE)
        rgb_gt_gaussian_maps = rgb_gt_gaussian_maps[-1].unsqueeze(1)

        # sonar gt gaussian map
        bs, n, _ = sonar_seach_anno.shape
        s_gt_bbox = sonar_seach_anno.view(bs, 4)

        s_gt_gaussian_maps = generate_heatmap(sonar_seach_anno.view(n, bs, 4), self.cfg.DATA.SEARCH.SIZE,
                                              self.cfg.MODEL.BACKBONE.STRIDE)
        s_gt_gaussian_maps = s_gt_gaussian_maps[-1].unsqueeze(1)

        # rgb Get boxes
        rgb_pred_boxes = predrgb_dict['pred_boxes']

        # s Get boxes
        s_pred_boxes = preds_dict['pred_boxes']

        if torch.isnan(rgb_pred_boxes).any() or torch.isnan(s_pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")

        # rgb loss
        rgb_pred_boxes_vec = box_cxcywh_to_xyxy(rgb_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        rgb_gt_boxes_vec = box_xywh_to_xyxy(rgb_gt_bbox).view(-1, 4).clamp(min=0.0,
                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        try:
            rgb_giou_loss, rgb_iou = self.objective['giou'](rgb_pred_boxes_vec, rgb_gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            rgb_giou_loss, rgb_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        rgb_l1_loss = self.objective['l1'](rgb_pred_boxes_vec, rgb_gt_boxes_vec)  # (BN,4) (BN,4)
        if 'score_map' in predrgb_dict:
            rgb_location_loss = self.objective['focal'](predrgb_dict['score_map'], rgb_gt_gaussian_maps)
        else:
            rgb_location_loss = torch.tensor(0.0, device=rgb_l1_loss.device)

        # weighted sum
        rgb_loss = self.loss_weight['giou'] * rgb_giou_loss + self.loss_weight['l1'] * rgb_l1_loss + self.loss_weight[
            'focal'] * rgb_location_loss

        # sonar loss
        pred_boxes_vec = box_cxcywh_to_xyxy(s_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(s_gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        try:
            s_giou_loss, s_iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            s_giou_loss, s_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        s_l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        if 'score_map' in preds_dict:
            s_location_loss = self.objective['focal'](preds_dict['score_map'], s_gt_gaussian_maps)
        else:
            s_location_loss = torch.tensor(0.0, device=s_l1_loss.device)

        s_loss = self.loss_weight['giou'] * s_giou_loss + self.loss_weight['l1'] * s_l1_loss + self.loss_weight[
            'focal'] * s_location_loss

        loss = rgb_loss + s_loss
        if return_status:
            # status for log
            # mean_iou = ((rgb_iou + s_iou) / 2).detach().mean()
            rgb_iou = rgb_iou.detach().mean()
            s_iou = s_iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": (rgb_giou_loss + s_giou_loss).item(),
                      "Loss/l1": (rgb_l1_loss + s_l1_loss).item(),
                      "Loss/location": (rgb_location_loss + s_location_loss).item(),
                      "RGB-IoU": rgb_iou.item(),
                      "S-IoU": s_iou.item(),
                      }
            return loss, status
        else:
            return loss

    def data2temp_search(self, data):
        # template rgb (batch, 3, 128, 128)
        template_img_1 = data['template_images'][:, 0, :].view(-1, *data['template_images'].shape[2:])
        template_img_2 = data['template_images'][:, 1, :].view(-1, *data['template_images'].shape[2:])

        # search rgb (batch, 3, 320, 320)
        search_img_1 = data['search_images'][:, 0, :].view(-1, *data['search_images'].shape[2:])
        search_img_2 = data['search_images'][:, 1, :].view(-1, *data['search_images'].shape[2:])

        return template_img_1, template_img_2, search_img_1, search_img_2

