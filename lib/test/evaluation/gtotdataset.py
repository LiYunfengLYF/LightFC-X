import numpy as np
import torch

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

from lib.utils.box_ops import box_xyxy_to_xywh


class GtotDataset(BaseDataset):
    def __init__(self, env_num):
        super().__init__(env_num)
        self.base_path = self.env_settings.gtot_path
        self.sequence_list = self.get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self._get_sequence_list()])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundTruth_i.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = box_xyxy_to_xywh(torch.tensor(load_text(str(anno_path), delimiter=' ', dtype=np.float64))).numpy()


        frames_path_i = '{}/{}/i'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/v'.format(self.base_path, sequence_name)

        frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".png")]
        if frame_list_i ==[]:
            frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".bmp")]
        frame_list_i.sort()

        frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".png")]
        if frame_list_v ==[]:
            frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".bmp")]
        frame_list_v.sort()

        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'gtot', ground_truth_rect.reshape(-1, 4))

    def _get_sequence_list(self):
        sequence_list = ['BlackCar', 'BlackSwan1', 'BlueCar', 'BusScale', 'BusScale1', 'carNig', 'Crossing', 'crowdNig',
                         'Cycling', 'DarkNig', 'Exposure2', 'Exposure4', 'fastCar2', 'FastCarNig', 'FastMotor',
                         'FastMotorNig', 'Football', 'GarageHover', 'Gathering', 'GoTogether', 'Jogging', 'LightOcc',
                         'Minibus', 'Minibus1', 'MinibusNig', 'MinibusNigOcc', 'Motorbike', 'Motorbike1', 'MotorNig',
                         'occBike', 'OccCar-1', 'OccCar-2', 'Otcbvs', 'Otcbvs1', 'Pool', 'Quarreling', 'RainyCar1',
                         'RainyCar2', 'RainyMotor1', 'RainyMotor2', 'RainyPeople', 'Running', 'Torabi', 'Torabi1',
                         'Tricycle', 'tunnel', 'Walking', 'WalkingNig', 'WalkingNig1', 'WalkingOcc']
        return sequence_list

    def __len__(self):
        return len(self.sequence_list)


if __name__ == '__main__':
    print(len(['BlackCar', 'BlackSwan1', 'BlueCar', 'BusScale', 'BusScale1', 'carNig', 'Crossing', 'crowdNig',
               'Cycling', 'DarkNig', 'Exposure2', 'Exposure4', 'fastCar2', 'FastCarNig', 'FastMotor',
               'FastMotorNig', 'Football', 'GarageHover', 'Gathering', 'GoTogether', 'Jogging', 'LightOcc',
               'Minibus', 'Minibus1', 'MinibusNig', 'MinibusNigOcc', 'Motorbike', 'Motorbike1', 'MotorNig',
               'occBike', 'OccCar-1', 'OccCar-2', 'Otcbvs', 'Otcbvs1', 'Pool', 'Quarreling', 'RainyCar1',
               'RainyCar2', 'RainyMotor1', 'RainyMotor2', 'RainyPeople', 'Running', 'Torabi', 'Torabi1',
               'Tricycle', 'tunnel', 'Walking', 'WalkingNig', 'WalkingNig1', 'WalkingOcc']))
