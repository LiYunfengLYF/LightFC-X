import torch
from torch import nn

from lib.models.lightfcx.template_update.crosattn_update import updatenetv1
from lib.utils.registry import MODEL_REGISTRY, TRACKER_REGISTRY
from lib.utils.load import load_pretrain
from lib.utils.token_utils import token2patch


@TRACKER_REGISTRY.register()
class RGBTer_lightfc(nn.Module):
    def __init__(self, cfg, env_num, training=False):
        super().__init__()

        self.backbone = MODEL_REGISTRY.get(cfg.MODEL.BACKBONE.TYPE)(**cfg.MODEL.BACKBONE.PARAMS)

        if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        if cfg.MODEL.NECK.USE_NECK:
            self.neck = MODEL_REGISTRY.get(cfg.MODEL.NECK.TYPE)(**cfg.MODEL.NECK.PARAMS)
        else:
            self.neck = None

        if cfg.MODEL.FUSION.USE_FUSION:
            self.fusion = MODEL_REGISTRY.get(cfg.MODEL.FUSION.TYPE)(**cfg.MODEL.FUSION.PARAMS)
        else:
            self.fusion = None

        self.head_type = cfg.MODEL.HEAD.TYPE
        self.head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)

    def forward(self,
                template: list,
                search: list,
                ):
        z = self.backbone(template)
        x = self.backbone(search)

        opt = self.fusion(z, x)
        out = self.head(opt)
        return out

    def forward_backbone(self, template: list):
        return self.backbone(template)

    def forward_track(self, template, search):
        xf = self.backbone(search)
        opt = self.fusion(template, xf)
        opt = self.head(opt)
        return opt


@TRACKER_REGISTRY.register()
class RGBTer_lightfc_update_eval(nn.Module):
    def __init__(self, cfg, env_num, training=False):
        super().__init__()

        self.backbone = MODEL_REGISTRY.get(cfg.MODEL.BACKBONE.TYPE)(**cfg.MODEL.BACKBONE.PARAMS)

        if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        if cfg.MODEL.NECK.USE_NECK:
            self.neck = MODEL_REGISTRY.get(cfg.MODEL.NECK.TYPE)(**cfg.MODEL.NECK.PARAMS)
        else:
            self.neck = None

        if cfg.MODEL.FUSION.USE_FUSION:
            self.fusion = MODEL_REGISTRY.get(cfg.MODEL.FUSION.TYPE)(**cfg.MODEL.FUSION.PARAMS)
        else:
            self.fusion = None

        self.head_type = cfg.MODEL.HEAD.TYPE
        self.head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)

        self.updatenet = MODEL_REGISTRY.get(cfg.MODEL.UPDATENET.TYPE)(**cfg.MODEL.UPDATENET.PARAMS)

    def forward(self, template: list, search: list, **kwargs):
        self.backbone.eval()
        self.fusion.eval()
        self.head.eval()

        z1 = self.backbone(template[0])
        z2 = self.backbone(template[1])

        z = self.updatenet(z1, z2)
        x = self.backbone(search)

        opt = self.fusion(z, x)
        out = self.head(opt)
        return out

    def forward_backbone(self, template: list):
        return self.backbone(template)

    def forward_track(self, template, search):
        xf = self.backbone(search)
        opt = self.fusion(template, xf)
        opt = self.head(opt)
        return opt


@TRACKER_REGISTRY.register()
class RGBSer_lightfc(nn.Module):
    def __init__(self, cfg, env_num, training=False):
        super().__init__()

        self.backbone = MODEL_REGISTRY.get(cfg.MODEL.BACKBONE.TYPE)(**cfg.MODEL.BACKBONE.PARAMS)

        if training and cfg.MODEL.BACKBONE.USE_PRETRAINED:
            load_pretrain(self.backbone, env_num=env_num, training=training, cfg=cfg, mode=cfg.MODEL.BACKBONE.LOAD_MODE)

        if cfg.MODEL.NECK.USE_NECK:
            self.neck = MODEL_REGISTRY.get(cfg.MODEL.NECK.TYPE)(**cfg.MODEL.NECK.PARAMS)
        else:
            self.neck = None

        if cfg.MODEL.FUSION.USE_FUSION:
            self.fusion = MODEL_REGISTRY.get(cfg.MODEL.FUSION.TYPE)(**cfg.MODEL.FUSION.PARAMS)
        else:
            self.fusion = None

        self.head_type = cfg.MODEL.HEAD.TYPE
        self.head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)
        self.sonar_head = MODEL_REGISTRY.get(cfg.MODEL.HEAD.TYPE)(**cfg.MODEL.HEAD.PARAMS)

    def forward(self,
                template: list,
                search: list,
                **kwargs):

        z = self.backbone(template)
        x = self.backbone(search)

        opt_v, opt_s = self.fusion(z, x)
        opt_v = self.head(opt_v)
        opt_s = self.sonar_head(opt_s)
        return opt_v, opt_s

    def forward_backbone(self, template: list):
        return self.backbone(template)

    def forward_track(self, zf, x):
        xf = self.backbone(x)
        xv, xi = torch.split(xf, (160, 160), dim=1)
        opt_v, opt_s = self.fusion(zf, [xv, xi])
        opt_v = self.head(opt_v)
        opt_s = self.sonar_head(opt_s)
        return opt_v, opt_s
