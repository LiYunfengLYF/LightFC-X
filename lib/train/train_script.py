import os
import random
import numpy as np
import torch

from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *
from lib.models.rgbter_light.trackerModel_lightfc import TRACKER_REGISTRY
from lib.train.actors import ACTOR_Registry
from ..utils.focal_loss import FocalLoss
from ..utils.load import load_yaml, load_sot_pretrain
from .box_loss import wiou_loss, eiou_loss, siou_loss, ciou_loss


def run(settings):
    settings.description = 'Training script for RGB-T Tracker'

    cfg = load_yaml(settings.cfg_file)
    update_settings(settings, cfg)
    # init seed
    random.seed(cfg.TRAIN.LEARN.SEED)
    np.random.seed(cfg.TRAIN.LEARN.SEED)
    torch.manual_seed(cfg.TRAIN.LEARN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.LEARN.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds are initialized: {cfg.TRAIN.LEARN.SEED}.")

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == 'lightfcx':
        net = TRACKER_REGISTRY.get(cfg.MODEL.NETWORK)(cfg, env_num=settings.env_num, training=True)
        if cfg.TRAIN.PRETRAIN.SOT_PRETRAIN:
            try:
                load_sot_pretrain(net, env_num=settings.env_num, cfg=cfg)
            except:
                print('fail to load sot pretrain model')
            if hasattr(net,'sonar_head'):
                try:
                    net.sonar_head.load_state_dict(net.head.state_dict())
                    print('SUCCESS LOAD SONAR HEAD WEIGHT from RGB HEAD')
                except:
                    pass

    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)

    # Loss functions and Actors
    if settings.script_name == 'lightfcx':
        focal_loss = FocalLoss()
        # wiou_loss, eiou_loss, siou_loss, ciou_loss
        if cfg.TRAIN.LEARN.IOU_TYPE == 'giou':
            objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        elif cfg.TRAIN.LEARN.IOU_TYPE == 'wiou':
            objective = {'giou': wiou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        elif cfg.TRAIN.LEARN.IOU_TYPE == 'eiou':
            objective = {'giou': eiou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        elif cfg.TRAIN.LEARN.IOU_TYPE == 'siou':
            objective = {'giou': siou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        elif cfg.TRAIN.LEARN.IOU_TYPE == 'ciou':
            objective = {'giou': ciou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        else:
            raise
        print(f'Using loss function: {cfg.TRAIN.LEARN.IOU_TYPE}')
        loss_weight = {'giou': cfg.TRAIN.LEARN.GIOU_WEIGHT, 'l1': cfg.TRAIN.LEARN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = ACTOR_Registry.get(cfg.TRAIN.ACTOR.TYPE)(net=net, objective=objective, loss_weight=loss_weight,
                                                         settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    rgb_mode = getattr(cfg.DATA, 'RGB_SAMPLE_PROCESS', False)
    print(f'LTR RGB_SAMPLE_PROCESS MODE: {rgb_mode}')
    print(f'Use AMP: {cfg.TRAIN.TRAINER.AMP.USED}')

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler,use_amp=cfg.TRAIN.TRAINER.AMP.USED,
                         rgb_mode=rgb_mode)

    # train process
    trainer.train(cfg.TRAIN.LEARN.EPOCH, load_latest=True, fail_safe=True)
