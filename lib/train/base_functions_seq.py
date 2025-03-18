import torch
from torch.utils.data.distributed import DistributedSampler

from lib.train.data.seq_loader import SLTLoader
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, LasHeR
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, sequence_sampler
import lib.train.data.transforms as tfm
from lib.train.dataset.rgbt234 import RGBT234
from lib.train.dataset.got10k_rgbt import Got10k_rgbt
from lib.utils.misc import is_main_process


def update_settings_seq(settings, cfg):
    # settings.print_interval = cfg.TRAIN.TRAINER.PRINT_INTERVAL
    # settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
    #                                'search': cfg.DATA.SEARCH.FACTOR}
    # settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
    #                       'search': cfg.DATA.SEARCH.SIZE}
    # settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
    #                                  'search': cfg.DATA.SEARCH.CENTER_JITTER}
    # settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
    #                                 'search': cfg.DATA.SEARCH.SCALE_JITTER}
    # settings.grad_clip_norm = cfg.TRAIN.TRAINER.AMP.GRAD_CLIP_NORM
    # settings.print_stats = None
    # settings.batchsize = cfg.TRAIN.LEARN.BATCH_SIZE
    # settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    # settings.save_interval = cfg.TRAIN.TRAINER.SAVE_INTERVAL
    settings.num_epoch = 10
    settings.num_per_epoch = 1000
    settings.num_seq = 8
    settings.num_seq_backward = 2
    settings.num_frames = 24
    settings.slt_loss_weight = 15.0
    settings.clip_grad = True
    settings.grad_max_norm = 100.0

    settings.max_gap = 300
    settings.max_interval = 10
    settings.interval_prob = 0.3


def names2datasets_seq(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "LasHeR_train", "LasHeR_test", 'RGBT234', 'GOT10K_RGBT']
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(
                    Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        # Dataset classes supports for LasHeR variants, no LMDB by default
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_train_dir, split='train', image_loader=image_loader))
        if name == "LasHeR_test":
            datasets.append(LasHeR(settings.env.lasher_test_dir, split='test', image_loader=image_loader))
        if name == 'RGBT234':
            datasets.append(RGBT234(settings.env.rgbt234_dir, image_loader=image_loader))
        if name == 'GOT10K_RGBT':
            datasets.append(Got10k_rgbt(settings.env.got10k_dir, image_loader=image_loader, env_num=settings.env_num))
    return datasets


def build_dataloaders_seq(cfg, settings):
    dataset_train = sequence_sampler.SequenceSampler(
        datasets=names2datasets_seq(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=settings.num_per_epoch,
        max_gap=settings.max_gap,
        max_interval=settings.max_interval,
        num_search_frames=settings.num_frames, num_template_frames=1,
        frame_sample_mode='random_interval', prob=settings.interval_prob)

    loader_train = SLTLoader('train', dataset_train, training=True, batch_size=settings.num_seq,
                             num_workers=settings.num_workers,
                             shuffle=False, drop_last=True)

    return loader_train
