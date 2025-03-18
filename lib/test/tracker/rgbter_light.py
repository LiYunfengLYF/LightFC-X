from copy import deepcopy

from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.models.rgbter_light.trackerModel_lightfc import TRACKER_REGISTRY
from lib.test.utils.hann import hann2d
from lib.train.data.bounding_box_utils import batch_xywh2center2
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box


class RGBTer_light(BaseTracker):
    def __init__(self, params, dataset_name=None):
        super(RGBTer_light, self).__init__(params)
        self.cfg = params.cfg

        network = TRACKER_REGISTRY.get(params.cfg.MODEL.NETWORK)(cfg=params.cfg, env_num=None, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'],
                                strict=True)

        # for template update
        self.use_template_update = getattr(self.cfg.TEST, 'USE_UPDATE', False)
        if self.use_template_update:
            self.update_interval = self.cfg.TEST.UPDATE_INTERVAL
            self.update_score = self.cfg.TEST.UPDATE_SCORE

        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.template_feat = []

    def initialize(self, image, info: dict):

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'],
                                                                self.params.template_factor,
                                                                output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            if not self.use_template_update:
                self.template_feat = self.network.forward_backbone(
                    [template.tensors[:, :3, :, :], template.tensors[:, 3:, :, :]])
            else:
                template_feat = self.network.forward_backbone(
                    [template.tensors[:, :3, :, :], template.tensors[:, 3:, :, :]])
                self.init_feat = template_feat
                # self.template_feat = self.network.updatenet(template_feat, template_feat)
                self.template_feat = template_feat

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.use_template_update:
            self.update_image = image
            self.update_box = deepcopy(info['init_bbox'])

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            if not self.use_template_update:
                out_dict = self.network.forward_track(
                    template=self.template_feat,
                    search=[search.tensors[:, :3, :, :], search.tensors[:, 3:, :, :]])
            else:
                out_dict = self.network.forward_track(
                    template=self.template_feat,
                    search=[search.tensors[:, :3, :, :], search.tensors[:, 3:, :, :]])

        # max score and memory
        pred_score_map = out_dict['score_map']
        max_score = out_dict['score_map'].max().cpu().numpy()

        # add hann windows
        response = self.output_window * pred_score_map

        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # template update v1
        if self.use_template_update:
            if max_score > self.update_score:
                self.update_image = image
                self.update_box = deepcopy(self.state)

            if self.frame_id % self.update_interval == 0 and self.frame_id > 100:
                zd_patch_arr, _, zd_amask_arr = sample_target(self.update_image, self.update_box,
                                                              self.params.template_factor,
                                                              output_sz=self.params.template_size)
                template = self.preprocessor.process(zd_patch_arr, zd_amask_arr)
                template_feat = self.network.forward_backbone(
                    [template.tensors[:, :3, :, :], template.tensors[:, 3:, :, :]])
                self.template_feat = self.network.updatenet(self.init_feat, template_feat)


        # if self.use_template_update: # v2 res
        #     if max_score > self.update_score and self.frame_id > self.update_interval:
        #         self.update_image = image
        #         self.update_box = deepcopy(self.state)
        #
        #     if self.frame_id % self.update_interval == 0 and self.frame_id > self.update_interval:
        #         zd_patch_arr, _, zd_amask_arr = sample_target(self.update_image, self.update_box,
        #                                                       self.params.template_factor,
        #                                                       output_sz=self.params.template_size)
        #         template = self.preprocessor.process(zd_patch_arr, zd_amask_arr)
        #         template_feat = self.network.forward_backbone(
        #             [template.tensors[:, :3, :, :], template.tensors[:, 3:, :, :]])
        #         self.template_feat = self.network.updatenet(self.init_feat, template_feat) + self.init_feat
        # if self.use_template_update: v3
        #
        #     if self.frame_id % self.update_interval == 0 and self.frame_id > self.update_interval:
        #         zd_patch_arr, _, zd_amask_arr = sample_target(image, self.state,
        #                                                       self.params.template_factor,
        #                                                       output_sz=self.params.template_size)
        #         template = self.preprocessor.process(zd_patch_arr, zd_amask_arr)
        #         template_feat = self.network.forward_backbone(
        #             [template.tensors[:, :3, :, :], template.tensors[:, 3:, :, :]])
        #         self.template_feat = self.network.updatenet(self.init_feat, template_feat)
        return {"target_bbox": self.state, 'best_score': max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


class RGBTer_light_seq(BaseTracker):
    def __init__(self, params, dataset_name):
        super(RGBTer_light_seq, self).__init__(params)
        # network = build_tbsi_track(params.cfg, training=False)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        network = TRACKER_REGISTRY.get(params.cfg.MODEL.NETWORK)(cfg=params.cfg, env_num=None, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'],
                                strict=True)

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):

        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        # if self.cfg.MODEL.BACKBONE.CE_LOC:
        #     template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
        #                                                 template.tensors.device).squeeze(1)
        #     self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=[self.z_dict1.tensors[:, :3, :, :], self.z_dict1.tensors[:, 3:, :, :]],
                search=[x_dict.tensors[:, :3, :, :], x_dict.tensors[:, 3:, :, :]], ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state, 'best_score': pred_score_map.max()}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def batch_init(self, images, template_bbox, initial_bbox) -> dict:

        initial_bbox = batch_xywh2center2(initial_bbox)  # ndarray:(2*num_seq,4)

        self.center_pos = initial_bbox[:, :2]  # ndarray:(2*num_seq,2)
        self.size = initial_bbox[:, 2:]  # ndarray:(2*num_seq,2)

        # get crop
        z_crop_list = []
        for i in range(len(images)):
            z_patch_arr, resize_factor, z_amask_arr = sample_target(images[i], template_bbox[i, :2],
                                                                    self.params.template_factor,
                                                                    output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)
            z_crop_list.append(template)
        z_crop = torch.cat(z_crop_list, dim=0)  # Tensor(2*num_seq,3,128,128)

        self.network.forward_batch_backbone(z_crop)
        out = {'template_images': z_crop}  # Tensor(2*num_seq,3,128,128)
        self.state_seq = template_bbox
        self.box_seq = initial_bbox
        return out

    def batch_track(self, img, gt_boxes, action_mode='max') -> dict:
        x_crop_list = []
        x_resize_factor_list = []
        for i in range(len(img)):
            x_patch_arr, resize_factor, x_amask_arr = sample_target(img[i], self.box_seq[i, :2],
                                                                    self.params.template_factor,
                                                                    output_sz=self.params.template_size)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            x_crop_list.append(search)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return RGBTer_light
