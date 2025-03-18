import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class Rgbt210Dataset(BaseDataset):
    def __init__(self, env_num):
        super().__init__(env_num)
        self.base_path = self.env_settings.rgbt210_path
        self.sequence_list = self.get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self._get_sequence_list()])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path_i = '{}/{}/infrared'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/visible'.format(self.base_path, sequence_name)
        frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".jpg")]
        frame_list_i.sort()
        frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".jpg")]
        frame_list_v.sort()
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'rgbt210', ground_truth_rect.reshape(-1, 4))

    def _get_sequence_list(self):
        sequence_list = ['afterrain', 'aftertree', 'baby', 'baginhand', 'baketballwaliking', 'balancebike',
                         'basketball2', 'bicyclecity', 'bike', 'bikeman', 'bikemove1', 'biketwo', 'bluebike', 'blueCar',
                         'boundaryandfast', 'bus6', 'call', 'car', 'car10', 'car20', 'car37', 'car41', 'car66',
                         'caraftertree', 'carLight', 'carnotfar', 'carnotmove', 'carred', 'child', 'child3', 'child4',
                         'children2', 'children3', 'children4', 'crossroad', 'crouch', 'diamond', 'dog', 'dog10',
                         'dog11', 'elecbike', 'elecbike10', 'elecbike2', 'elecbike3', 'elecbikechange2',
                         'elecbikeinfrontcar', 'elecbikewithhat', 'elecbikewithlight', 'elecbikewithlight1', 'face1',
                         'floor-1', 'flower1', 'flower2', 'fog', 'fog6', 'glass', 'glass2', 'graycar2', 'green',
                         'greentruck', 'greyman', 'greywoman', 'guidepost', 'hotglass', 'hotkettle', 'inglassandmobile',
                         'jump', 'kettle', 'kite2', 'kite4', 'luggage', 'man22', 'man23', 'man24', 'man26', 'man28',
                         'man29', 'man3', 'man4', 'man45', 'man5', 'man55', 'man68', 'man69', 'man7', 'man8', 'man88',
                         'man9', 'manafterrain', 'mancross', 'mancross1', 'mancrossandup', 'mandrivecar', 'manfaraway',
                         'maninblack', 'maninglass', 'maningreen2', 'maninred', 'manlight', 'manoccpart',
                         'manonboundary', 'manonelecbike', 'manontricycle', 'manout2', 'manup', 'manwithbag',
                         'manwithbag4', 'manwithbasketball', 'manwithluggage', 'manwithumbrella', 'manypeople',
                         'mobile', 'night2', 'nightcar', 'nightrun', 'nightthreepeople', 'notmove', 'oldman', 'oldman2',
                         'oldwoman', 'people', 'people1', 'people3', 'playsoccer', 'push', 'rainingwaliking',
                         'raningcar', 'redbag', 'redcar', 'redcar2', 'redmanchange', 'rmo', 'run', 'run1', 'run2',
                         'scooter', 'shake', 'shoeslight', 'soccer', 'soccer2', 'soccerinhand', 'straw', 'stroller',
                         'supbus', 'supbus2', 'takeout', 'tallman', 'threeman', 'threeman2', 'threewoman2', 'together',
                         'toy1', 'toy3', 'toy4', 'tree2', 'tree3', 'tree5', 'trees', 'tricycle', 'tricycle2',
                         'tricycle6', 'tricycle9', 'tricyclefaraway', 'tricycletwo', 'twoelecbike', 'twoelecbike1',
                         'twoman', 'twoman1', 'twoperson', 'twowoman', 'walking40', 'walking41', 'walkingman',
                         'walkingman1', 'walkingman12', 'walkingman20', 'walkingman41', 'walkingmantiny',
                         'walkingnight', 'walkingtogether', 'walkingtogether1', 'walkingtogetherright',
                         'walkingwithbag1', 'walkingwithbag2', 'walkingwoman', 'whitebag', 'whitecar', 'whitecar3',
                         'whitecar4', 'whitecarafterrain', 'whitesuv', 'woamn46', 'woamnwithbike', 'woman', 'woman100',
                         'woman4', 'woman48', 'woman6', 'woman89', 'woman96', 'woman99', 'womancross', 'womanfaraway',
                         'womaninblackwithbike', 'womanleft', 'womanpink', 'womanred', 'womanrun', 'womanwithbag6',
                         'yellowcar']
        return sequence_list

    def __len__(self):
        return len(self.sequence_list)


if __name__ == '__main__':
    data = Rgbt210Dataset(env_num=102)
    print(os.listdir(r'G:\data\rgbt\rgbt210'))
