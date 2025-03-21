import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class viseventDataset(BaseDataset):
    def __init__(self, env_num):
        super().__init__(env_num)
        self.base_path = self.env_settings.visevent_path
        self.sequence_list = self.get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self._get_sequence_list()])

    def _construct_sequence(self, sequence_name):
        anno_path = os.path.join(self.base_path, sequence_name, 'groundtruth.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path_event = os.path.join(self.base_path, sequence_name, 'event_imgs')
        frames_path_event = [os.path.join(frames_path_event, frame) for frame in os.listdir(frames_path_event) if
                             frame.endswith(".bmp")]
        frames_path_event.sort()

        frames_path_vis = os.path.join(self.base_path, sequence_name, 'vis_imgs')
        frames_path_vis = [os.path.join(frames_path_vis, frame) for frame in os.listdir(frames_path_vis) if
                           frame.endswith(".bmp")]
        frames_path_vis.sort()

        frames_list = [frames_path_event, frames_path_vis, ]
        return Sequence(sequence_name, frames_list, 'visevent', ground_truth_rect.reshape(-1, 4))

    def _get_sequence_list(self):
        sequence_list = ['00141_tank_outdoor2', '00147_tank_outdoor2', '00197_driving_outdoor3',
                         '00236_tennis_outdoor4', '00241_tennis_outdoor4', '00282_tennis_outdoor4',
                         '00292_tennis_outdoor4', '00297_tennis_outdoor4', '00314_UAV_outdoor5', '00325_UAV_outdoor5',
                         '00331_UAV_outdoor5', '00335_UAV_outdoor5',
                         '00340_UAV_outdoor6', '00345_UAV_outdoor6', '00351_UAV_outdoor6', '00355_UAV_outdoor6',
                         '00370_UAV_outdoor6', '00374_UAV_outdoor6', '00385_UAV_outdoor6', '00398_UAV_outdoor6',
                         '00404_UAV_outdoor6', '00406_UAV_outdoor6', '00408_UAV_outdoor6', '00410_UAV_outdoor6',
                         '00413_UAV_outdoor6', '00416_UAV_outdoor6', '00419_UAV_outdoor6', '00421_UAV_outdoor6',
                         '00423_UAV_outdoor6', '00425_UAV_outdoor6', '00428_UAV_outdoor6', '00430_UAV_outdoor6',
                         '00432_UAV_outdoor6', '00433_UAV_outdoor6', '00435_UAV_outdoor6', '00437_UAV_outdoor6',
                         '00439_UAV_outdoor6', '00442_UAV_outdoor6',
                         '00445_UAV_outdoor6', '00447_UAV_outdoor6', '00449_UAV_outdoor6', '00451_UAV_outdoor6',
                         '00453_UAV_outdoor6', '00455_UAV_outdoor6', '00458_UAV_outdoor6', '00462_UAV_outdoor6',
                         '00464_UAV_outdoor6', '00466_UAV_outdoor6', '00471_UAV_outdoor6', '00473_UAV_outdoor6',
                         '00478_UAV_outdoor6', '00483_UAV_outdoor6', '00490_UAV_outdoor6', '00503_UAV_outdoor6',
                         '00506_person_outdoor6', '00508_person_outdoor6', '00510_person_outdoor6',
                         '00511_person_outdoor6', '00514_person_outdoor6', 'basketball_0076', 'basketball_0078',
                         'dightNUM_001', 'dvSave-2021_02_04_20_41_53', 'dvSave-2021_02_04_20_49_43',
                         'dvSave-2021_02_04_20_56_55', 'dvSave-2021_02_04_21_04_05', 'dvSave-2021_02_04_21_18_52',
                         'dvSave-2021_02_04_21_20_22', 'dvSave-2021_02_04_21_21_24', 'dvSave-2021_02_06_08_33_09_cat',
                         'dvSave-2021_02_06_08_52_19_rotateball', 'dvSave-2021_02_06_08_56_18_windowPattern',
                         'dvSave-2021_02_06_08_56_40_windowPattern2', 'dvSave-2021_02_06_08_57_35_machineBrad',
                         'dvSave-2021_02_06_08_58_43_cat', 'dvSave-2021_02_06_09_09_44_blackcar',
                         'dvSave-2021_02_06_09_09_44_person3', 'dvSave-2021_02_06_09_09_44_person6',
                         'dvSave-2021_02_06_09_09_44_person7', 'dvSave-2021_02_06_09_10_52_car1',
                         'dvSave-2021_02_06_09_09_44_person6', 'dvSave-2021_02_06_09_09_44_person7',
                         'dvSave-2021_02_06_09_10_52_car1', 'dvSave-2021_02_06_09_10_52_car2',
                         'dvSave-2021_02_06_09_10_52_car3',
                         'dvSave-2021_02_06_09_11_41_car1', 'dvSave-2021_02_06_09_11_41_person1',
                         'dvSave-2021_02_06_09_11_41_person4',
                         'dvSave-2021_02_06_09_13_36_person0', 'dvSave-2021_02_06_09_13_36_person2',
                         'dvSave-2021_02_06_09_14_18_girl1',
                         'dvSave-2021_02_06_09_14_18_person5', 'dvSave-2021_02_06_09_14_18_whitecar1',
                         'dvSave-2021_02_06_09_16_06_person', 'dvSave-2021_02_06_09_16_35_car',
                         'dvSave-2021_02_06_09_17_11_person',
                         'dvSave-2021_02_06_09_21_53_car', 'dvSave-2021_02_06_09_22_41_person1',
                         'dvSave-2021_02_06_09_23_50_person1',
                         'dvSave-2021_02_06_09_24_26_Pedestrian1', 'dvSave-2021_02_06_09_24_39_oldman1',
                         'dvSave-2021_02_06_09_33_23_person1',
                         'dvSave-2021_02_06_09_33_23_person5', 'dvSave-2021_02_06_09_35_08_Pedestrian',
                         'dvSave-2021_02_06_09_36_15_Pedestrian',
                         'dvSave-2021_02_06_09_36_44_Pedestrian', 'dvSave-2021_02_06_09_58_27_DigitAI',
                         'dvSave-2021_02_06_10_03_17_GreenPlant',
                         'dvSave-2021_02_06_10_05_38_phone', 'dvSave-2021_02_06_10_09_04_bottle',
                         'dvSave-2021_02_06_10_11_59_paperClips',
                         'dvSave-2021_02_06_10_14_17_paperClip', 'dvSave-2021_02_06_10_17_16_paperClips',
                         'dvSave-2021_02_06_15_08_41_flag',
                         'dvSave-2021_02_06_15_12_44_car', 'dvSave-2021_02_06_15_14_26_blackcar',
                         'dvSave-2021_02_06_15_15_36_redcar',
                         'dvSave-2021_02_06_15_16_07_car', 'dvSave-2021_02_06_15_17_48_whitecar',
                         'dvSave-2021_02_06_15_18_36_redcar', 'dvSave-2021_02_06_17_15_20_whitecar',
                         'dvSave-2021_02_06_17_16_26_whitecar',
                         'dvSave-2021_02_06_17_20_28_personFootball', 'dvSave-2021_02_06_17_21_41_personFootball',
                         'dvSave-2021_02_06_17_23_26_personFootball',
                         'dvSave-2021_02_06_17_27_53_personFootball', 'dvSave-2021_02_06_17_31_03_personBasketball',
                         'dvSave-2021_02_06_17_33_01_personBasketball',
                         'dvSave-2021_02_06_17_34_58_personBasketball', 'dvSave-2021_02_06_17_36_49_personBasketball',
                         'dvSave-2021_02_06_17_41_45_personBasketball',
                         'dvSave-2021_02_06_17_45_17_personBasketball', 'dvSave-2021_02_06_17_47_49_personBasketball',
                         'dvSave-2021_02_06_17_49_51_personBasketball',
                         'dvSave-2021_02_06_17_51_05_personBasketball', 'dvSave-2021_02_06_17_53_39_personFootball',
                         'dvSave-2021_02_06_17_57_54_personFootball',
                         'dvSave-2021_02_06_18_04_18_person1', 'dvSave-2021_02_06_18_04_18_person3',
                         'dvSave-2021_02_08_21_02_13_car3', 'dvSave-2021_02_08_21_02_13_motor2',
                         'dvSave-2021_02_08_21_04_56_car3', 'dvSave-2021_02_08_21_04_56_car6',
                         'dvSave-2021_02_08_21_05_56_motor', 'dvSave-2021_02_08_21_06_03_car3',
                         'dvSave-2021_02_08_21_06_03_car5', 'dvSave-2021_02_08_21_06_03_car7',
                         'dvSave-2021_02_08_21_06_03_motor2', 'dvSave-2021_02_08_21_07_02_car2',
                         'dvSave-2021_02_08_21_07_52', 'dvSave-2021_02_08_21_15_49_car1',
                         'dvSave-2021_02_08_21_15_49_car3', 'dvSave-2021_02_08_21_15_49_car8',
                         'dvSave-2021_02_08_21_17_43_car3', 'dvSave-2021_02_08_21_17_43_car5',
                         'dvSave-2021_02_12_13_38_26', 'dvSave-2021_02_12_13_39_56',
                         'dvSave-2021_02_12_13_43_54', 'dvSave-2021_02_12_13_46_18',
                         'dvSave-2021_02_12_13_51_43', 'dvSave-2021_02_12_13_56_29',
                         'dvSave-2021_02_14_16_21_40', 'dvSave-2021_02_14_16_22_06',
                         'dvSave-2021_02_14_16_26_44_car3', 'dvSave-2021_02_14_16_26_44_girl',
                         'dvSave-2021_02_14_16_26_44_person1', 'dvSave-2021_02_14_16_28_37_car2',
                         'dvSave-2021_02_14_16_28_37_car3', 'dvSave-2021_02_14_16_28_37_person1',
                         'dvSave-2021_02_14_16_29_49_car2', 'dvSave-2021_02_14_16_30_05_car1',
                         'dvSave-2021_02_14_16_30_20_car2', 'dvSave-2021_02_14_16_31_07_blackcar2',
                         'dvSave-2021_02_14_16_31_07_person1', 'dvSave-2021_02_14_16_31_07_redtaxi01',
                         'dvSave-2021_02_14_16_31_07_whitecar1', 'dvSave-2021_02_14_16_31_07_whitecar4',
                         'dvSave-2021_02_14_16_34_11_car3', 'dvSave-2021_02_14_16_34_11_person1',
                         'dvSave-2021_02_14_16_34_48_car2', 'dvSave-2021_02_14_16_34_48_person1',
                         'dvSave-2021_02_14_16_35_40_car2', 'dvSave-2021_02_14_16_37_15_car2',
                         'dvSave-2021_02_14_16_37_15_car5', 'dvSave-2021_02_14_16_37_15_motor2',
                         'dvSave-2021_02_14_16_37_15_person', 'dvSave-2021_02_14_16_40_59_blackcar1',
                         'dvSave-2021_02_14_16_40_59_car1', 'dvSave-2021_02_14_16_40_59_car4',
                         'dvSave-2021_02_14_16_40_59_car7', 'dvSave-2021_02_14_16_40_59_motor1',
                         'dvSave-2021_02_14_16_42_14_car1', 'dvSave-2021_02_14_16_42_44_car3',
                         'dvSave-2021_02_14_16_42_44_car6', 'dvSave-2021_02_14_16_43_23_car3',
                         'dvSave-2021_02_14_16_43_23_car4', 'dvSave-2021_02_14_16_43_54_car2',
                         'dvSave-2021_02_14_16_43_54_car4', 'dvSave-2021_02_14_16_45_13_car1',
                         'dvSave-2021_02_14_16_45_13_car5', 'dvSave-2021_02_14_16_45_13_car7',
                         'dvSave-2021_02_14_16_46_34_car10', 'dvSave-2021_02_14_16_46_34_car12',
                         'dvSave-2021_02_14_16_46_34_car16', 'dvSave-2021_02_14_16_46_34_car3',
                         'dvSave-2021_02_14_16_46_34_car5', 'dvSave-2021_02_14_16_46_34_car8',
                         'dvSave-2021_02_14_16_48_45_car3', 'dvSave-2021_02_14_16_48_45_car5',
                         'dvSave-2021_02_14_16_48_45_car8', 'dvSave-2021_02_14_16_51_21_car1',
                         'dvSave-2021_02_14_16_51_21_motor1', 'dvSave-2021_02_14_16_53_15_flag',
                         'dvSave-2021_02_14_16_55_35_person1', 'dvSave-2021_02_14_16_56_01_house',
                         'dvSave-2021_02_14_16_56_18_car2', 'dvSave-2021_02_14_16_56_18_car4',
                         'dvSave-2021_02_14_16_56_18_car6', 'dvSave-2021_02_14_16_56_59_car2',
                         'dvSave-2021_02_14_16_56_59_car4', 'dvSave-2021_02_14_16_56_59_car6',
                         'dvSave-2021_02_14_17_00_48', 'dvSave-2021_02_14_17_02_37_roadflag',
                         'dvSave-2021_02_15_10_12_19_basketball', 'dvSave-2021_02_15_10_14_18_chicken',
                         'dvSave-2021_02_15_10_22_23_basketball', 'dvSave-2021_02_15_10_22_23_boyhead',
                         'dvSave-2021_02_15_10_23_05_basketall', 'dvSave-2021_02_15_10_23_05_boyhead',
                         'dvSave-2021_02_15_10_24_03_basketball', 'dvSave-2021_02_15_10_24_03_boyhead',
                         'dvSave-2021_02_15_10_26_11_chicken', 'dvSave-2021_02_15_10_26_52_basketball1',
                         'dvSave-2021_02_15_10_26_52_basketball2', 'dvSave-2021_02_15_10_26_52_personHead1',
                         'dvSave-2021_02_15_12_44_27_chicken', 'dvSave-2021_02_15_12_45_02_Duck',
                         'dvSave-2021_02_15_12_53_54_personHead', 'dvSave-2021_02_15_12_56_56_Fish',
                         'dvSave-2021_02_15_12_56_56_personHead', 'dvSave-2021_02_15_12_58_05_personHead2',
                         'dvSave-2021_02_15_12_58_56_person', 'dvSave-2021_02_15_13_01_16_Duck',
                         'dvSave-2021_02_15_13_02_21_Chicken', 'dvSave-2021_02_15_13_04_57_Duck',
                         'dvSave-2021_02_15_13_05_43_Chicken', 'dvSave-2021_02_15_13_08_12_blackcar',
                         'dvSave-2021_02_15_13_09_09_person', 'dvSave-2021_02_15_13_10_54_person',
                         'dvSave-2021_02_15_13_12_45_redcar', 'dvSave-2021_02_15_13_13_44_whitecar',
                         'dvSave-2021_02_15_13_14_18_blackcar', 'dvSave-2021_02_15_13_24_03_girlhead',
                         'dvSave-2021_02_15_13_24_49_girlhead', 'dvSave-2021_02_15_13_25_36_girlhead',
                         'dvSave-2021_02_15_13_27_20_bottle', 'dvSave-2021_02_15_13_28_20_cash',
                         'dvSave-2021_02_15_23_51_36', 'dvSave-2021_02_15_23_54_17',
                         'dvSave-2021_02_15_23_56_17', 'dvSave-2021_02_16_17_07_38',
                         'dvSave-2021_02_16_17_12_18', 'dvSave-2021_02_16_17_15_53',
                         'dvSave-2021_02_16_17_20_20', 'dvSave-2021_02_16_17_23_10',
                         'dvSave-2021_02_16_17_29_37', 'dvSave-2021_02_16_17_34_11',
                         'dvSave-2021_02_16_17_38_25', 'dvSave-2021_02_16_17_42_50',
                         'dydrant_001', 'roadLight_001',
                         'tennis_long_001', 'tennis_long_002', 'tennis_long_003', 'tennis_long_004',
                         'tennis_long_005', 'tennis_long_006', 'tennis_long_007', 'traffic_0006',
                         'traffic_0013', 'traffic_0019', 'traffic_0023', 'traffic_0028',
                         'traffic_0034', 'traffic_0037', 'traffic_0040', 'traffic_0043',
                         'traffic_0046', 'traffic_0049', 'traffic_0052', 'traffic_0055',
                         'traffic_0058', 'traffic_0061', 'traffic_0064', 'traffic_0067',
                         'traffic_0070', 'traffic_0073',
                         'UAV_long_001', 'video_0004',
                         'video_0005', 'video_0008', 'video_0009', 'video_0015',
                         'video_0018', 'video_0021', 'video_0026', 'video_0029',
                         'video_0032', 'video_0039', 'video_0041', 'video_0045',
                         'video_0049', 'video_0050', 'video_0054', 'video_0056',
                         'video_0058', 'video_0060', 'video_0064', 'video_0067',
                         'video_0070', 'video_0073', 'video_0076', 'video_0079']
        return sequence_list

    def __len__(self):
        return len(self.sequence_list)


if __name__ == '__main__':
    data = viseventDataset(env_num=102)
    print(os.listdir(r'/home/datasets/visevent/test_subset'))
    '''
    dataset= viseventDataset(root =XXX,env_num=102)

    a = dataset.sequence_list[0]
    print(a[0])
    '''
