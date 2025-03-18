import os
import os.path
from collections import OrderedDict

import numpy as np
import torch
import csv
import pandas
from .base_video_dataset import BaseVideoDataset
from lib.train.data import opencv_loader
from lib.train.admin import env_settings


class VTUAV(BaseVideoDataset):
    def __init__(self, root=None, image_loader=opencv_loader, split='train_st', env_num=5):

        root = env_settings().vtuav_dir if root is None else root
        super().__init__('VTUAV', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if split == 'train_st':
                with open(os.path.join(self.root, 'ST_train_split.txt')) as f:
                    sequence_list = list(csv.reader(f))
            elif split == 'val':
                with open(os.path.join(self.root, 'ST_val_split.txt')) as f:
                    sequence_list = list(csv.reader(f))
            else:
                raise ValueError('Unknown split name.')
        # self.seq_ids = seq_ids
        self.init_idx = np.load(os.path.join(self.root, 'init_frame.npy'),
                                allow_pickle=True).item()
        self.sequence_list = sequence_list

        self.object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})

    def get_name(self):
        return 'UAV_RGBT'

    def has_class_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):

        return os.listdir(self.root)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "rgb.txt")
        gt = np.loadtxt(bb_anno_file).astype(np.float32)

        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id][0])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, modality, frame_id):
        seq_name = seq_path.split('/')[-1]
        if seq_name in self.init_idx:
            init_idx = self.init_idx[seq_name]
        else:
            init_idx = 0
        nz = 6
        return os.path.join(seq_path, modality, str(frame_id * 10 + init_idx).zfill(nz) + '.jpg')  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return np.concatenate((self.image_loader(self._get_frame_path(seq_path, 'rgb', frame_id)),
                               self.image_loader(self._get_frame_path(seq_path, 'ir', frame_id))), 2)

    def get_frames(self, seq_id, frame_ids, anno=None):

        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, self.object_meta
