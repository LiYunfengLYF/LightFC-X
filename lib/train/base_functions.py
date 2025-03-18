import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, LasHeR, VTUAV
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader,processing_nomask
import lib.train.data.transforms as tfm
from lib.train.dataset.depthtrack import DepthTrack
from lib.train.dataset.got10k_rgbt import Got10k_rgbt
from lib.train.dataset.sarset_seq import SARDETSeq
from lib.train.dataset.visevent import VisEvent
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.TRAINER.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.TRAINER.AMP.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.LEARN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.save_interval = cfg.TRAIN.TRAINER.SAVE_INTERVAL

    settings.rgbs_srst = getattr(cfg.DATA, 'RGBS_SRST', False)
    if settings.rgbs_srst:
        print(f'train RGBS SRST {settings.rgbs_srst}')


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "SARDET", "LasHeR_train", "LasHeR_test",
                        'GOT10K_RGBT', 'VTUAV_ST', 'VisEvent', "DepthTrack_train", "DepthTrack_val"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader,
                                      rgbs_mode=settings.rgbs_srst))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader,
                                       rgbs_mode=settings.rgbs_srst ))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(
                    Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader,
                                       rgbs_mode=settings.rgbs_srst, ))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader,
                                       rgbs_mode=settings.rgbs_srst, ))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader,
                                       rgbs_mode=settings.rgbs_srst, ))
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

        # SAR-DET
        if name == "SARDET":
            datasets.append(SARDETSeq(settings.env.sardet_train_dir, split='train', image_loader=image_loader))


        # Dataset classes supports for LasHeR variants, no LMDB by default
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_train_dir, split='train', image_loader=image_loader))
        if name == "LasHeR_test":
            datasets.append(LasHeR(settings.env.lasher_test_dir, split='test', image_loader=image_loader))
        if name == 'VTUAV_ST':
            datasets.append(VTUAV(settings.env.vtuav_train_st, image_loader=image_loader, split='train_st',
                                  env_num=settings.env_num))

        # RGBE
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='rgbrgb', split='train_subset'))
        # RGBD
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='train'))
        if name == "DepthTrack_test":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='test'))

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0),  # No grayscale for thermal infrared data
                                    tfm.RandomHorizontalFlip(probability=0.5))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # data_processing_train = processing_nomask.STARKProcessing_nomask(search_area_factor=search_area_factor,
    #                                                                  output_sz=output_sz,
    #                                                                  center_jitter_factor=settings.center_jitter_factor,
    #                                                                  scale_jitter_factor=settings.scale_jitter_factor,
    #                                                                  mode='sequence',
    #                                                                  transform=transform_train,
    #                                                                  joint_transform=transform_joint,
    #                                                                  settings=settings)
    #
    # data_processing_val = processing_nomask.STARKProcessing_nomask(search_area_factor=search_area_factor,
    #                                                                output_sz=output_sz,
    #                                                                center_jitter_factor=settings.center_jitter_factor,
    #                                                                scale_jitter_factor=settings.scale_jitter_factor,
    #                                                                mode='sequence',
    #                                                                transform=transform_val,
    #                                                                joint_transform=transform_joint,
    #                                                                settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    rgb_mode = getattr(cfg.DATA, 'RGB_SAMPLE_PROCESS', False)
    if rgb_mode:
        print('USE RGB_SAMPLE_PROCESS')
    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
        num_template_frames=settings.num_template, processing=data_processing_train,
        frame_sample_mode=sampler_mode, train_cls=train_cls,
        rgb_mode=rgb_mode)


    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train,
                             training=True,
                             batch_size=cfg.TRAIN.LEARN.BATCH_SIZE,
                             shuffle=shuffle,
                             num_workers=cfg.TRAIN.TRAINER.NUM_WORKER,
                             drop_last=True,
                             pin_memory=False,
                             stack_dim=1,
                             sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
                                          num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template,
                                          processing=data_processing_val,
                                          frame_sample_mode=sampler_mode,
                                          train_cls=train_cls,
                                          rgb_mode=rgb_mode)

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val',
                           dataset_val,
                           training=False,
                           batch_size=cfg.TRAIN.LEARN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.TRAINER.NUM_WORKER,
                           drop_last=True,
                           pin_memory=False,
                           stack_dim=1,
                           sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.TRAINER.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    SOT_PRETRAIN = getattr(cfg.TRAIN.PRETRAIN, 'SOT_PRETRAIN', False)
    update = getattr(cfg.TRAIN.PRETRAIN, 'UPDATE_FINETUNE', False)
    if update:
        print('Use update finetune is True')
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "update" in n and p.requires_grad]},
        ]

        for n, p in net.named_parameters():
            if "update" not in n:
                p.requires_grad = False

        if is_main_process():
            print("Learnable parameters are shown below for sot pretraining setting.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    else:
        if train_cls:
            print("Only training classification head. Learnable parameters are shown below.")
            param_dicts = [
                {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
            ]

            for n, p in net.named_parameters():
                if "cls" not in n:
                    p.requires_grad = False
                else:
                    print(n)
        elif SOT_PRETRAIN:
            print('Use rgbt is True')
            param_dicts = [
                {"params": [p for n, p in net.named_parameters() if "rgbt" in n and p.requires_grad]},
                {
                    "params": [p for n, p in net.named_parameters() if "rgbt" not in n and p.requires_grad],
                    "lr": cfg.TRAIN.LEARN.LR * cfg.TRAIN.OPTIMIZER.BACKBONE_MULTIPLIER,
                },
            ]
            if is_main_process():
                print("Learnable parameters are shown below for sot pretraining setting.")
                for n, p in net.named_parameters():
                    if p.requires_grad:
                        print(n)
        else:
            param_dicts = [
                {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": cfg.TRAIN.LEARN.LR * cfg.TRAIN.OPTIMIZER.BACKBONE_MULTIPLIER,
                },
            ]
            if is_main_process():
                print("Learnable parameters are shown below.")
                for n, p in net.named_parameters():
                    if p.requires_grad:
                        print(n)

    if cfg.TRAIN.OPTIMIZER.TYPE == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=cfg.TRAIN.LEARN.LR,
                                      weight_decay=cfg.TRAIN.LEARN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.SCHEDULER.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
