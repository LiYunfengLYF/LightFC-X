import os

import etrack

from .data import Sequence


class RGBPS(object):
    _sequence_name = ['connected_polyhedron9', 'fake_person2', 'uuv7', 'iron_ball2', 'connected_polyhedron4',
                      'octahedron4', 'ball_and_polyhedron2', 'ball_and_polyhedron3', 'connected_polyhedron5',
                      'octahedron3', 'connected_polyhedron3', 'fake_person6', 'uuv9', 'uuv12', 'fake_person5', 'uuv5',
                      'uuv4', 'frustum5', 'octahedron6', 'connected_polyhedron7', 'octahedron1', 'iron_ball1',
                      'frustum6', 'frustum4', 'octahedron2', 'fake_person1', 'frustum1', 'uuv6', 'iron_ball3',
                      'frustum7', 'uuv1', 'connected_polyhedron1', 'connected_polyhedron6', 'uuv2', 'uuv3', 'uuv8',
                      'octahedron7', 'ball_and_polyhedron1', 'fake_person4', 'octahedron8', 'fake_person3', 'frustum2',
                      'octahedron5', 'fake_person7', 'uuv10', 'frustum3', 'connected_polyhedron8', 'frustum8', 'uuv11',
                      'connected_polyhedron2']

    def __init__(self, root_dir):
        super().__init__()

        self.name = 'RGBPS'
        self.root_dir = root_dir
        self.sequence_list = self.construct_sequence_list()

    def __getitem__(self, item):
        return self.sequence_list[item]

    def __len__(self):
        return len(self.sequence_list)

    def construct_sequence_list(self):
        sequence_list = []
        for name in self._sequence_name:
            sensor_rgb = os.path.join(self.root_dir, name, 'light', )
            sensor_s = os.path.join(self.root_dir, name, 'sonar')

            rgb_imgs = etrack.seqread(os.path.join(sensor_rgb, 'img'))
            s_imgs = etrack.seqread(os.path.join(sensor_s, 'img'))

            rgb_gt = etrack.txtread(os.path.join(sensor_rgb, 'groundtruth.txt'))
            s_gt = etrack.txtread(os.path.join(sensor_s, 'groundtruth.txt'))
            sequence_list.append(Sequence(name, rgb_imgs, s_imgs, rgb_gt, s_gt))
        return sequence_list


class RGBPS_OC(RGBPS):
    _sequence_name = ['ball_and_polyhedron3', 'connected_polyhedron6', 'connected_polyhedron7', 'frustum1', 'frustum2',
                      'frustum3', 'frustum5', 'octahedron2', 'uuv3', 'uuv4', 'uuv5', 'uuv6', 'uuv7']


class RGBPS_FOV(RGBPS):
    _sequence_name = ['connected_polyhedron1', 'connected_polyhedron2', 'connected_polyhedron3',
                      'connected_polyhedron4', 'connected_polyhedron5', 'connected_polyhedron6',
                      'connected_polyhedron7', 'fake_person3', 'fake_person4', 'frustum2', 'frustum4', 'frustum6',
                      'iron_ball1', 'iron_ball2', 'octahedron3', 'octahedron4', 'octahedron6', 'uuv1', 'uuv6', ]


class RGBPS_SA(RGBPS):
    _sequence_name = ['ball_and_polyhedron1', 'ball_and_polyhedron2', 'ball_and_polyhedron3', 'connected_polyhedron1',
                      'connected_polyhedron2', 'connected_polyhedron3', 'connected_polyhedron4',
                      'connected_polyhedron5', 'connected_polyhedron6', 'connected_polyhedron7',
                      'connected_polyhedron8', 'connected_polyhedron9', ]


class RGBPS_SV(RGBPS):
    _sequence_name = ['ball_and_polyhedron2', 'connected_polyhedron1', 'connected_polyhedron3', 'connected_polyhedron4',
                      'connected_polyhedron6', 'connected_polyhedron7', 'connected_polyhedron8', 'fake_person1',
                      'fake_person2', 'fake_person3', 'fake_person5', 'fake_person6', 'frustum1', 'octahedron1',
                      'octahedron5', 'uuv2', ]


class RGBPS_SC(RGBPS):
    _sequence_name = ['connected_polyhedron9', 'frustum2', 'iron_ball1', 'iron_ball2',
                      'iron_ball3', 'octahedron2', 'octahedron3', 'octahedron6', 'octahedron7', 'uuv3', 'uuv5', 'uuv7',
                      'uuv9', 'uuv11', 'uuv12', ]


class RGBPS_DEF(RGBPS):
    _sequence_name = ['fake_person2', 'frustum6', 'iron_ball2', 'iron_ball3', 'octahedron2', 'octahedron3',
                      'octahedron4', 'octahedron6', 'octahedron8', 'uuv1', 'uuv5', 'uuv6', 'uuv9', 'uuv11', ]


class RGBPS_VLR(RGBPS):
    _sequence_name = ['connected_polyhedron5', 'connected_polyhedron6', 'connected_polyhedron7',
                      'connected_polyhedron8', 'fake_person2', 'fake_person3', 'fake_person5', 'fake_person6',
                      'fake_person7', 'frustum1', 'frustum4', 'frustum5', 'frustum8', 'octahedron7', 'octahedron8',
                      'uuv10', 'uuv11', ]


class RGBPS_LSR(RGBPS):
    _sequence_name = ['fake_person1', 'fake_person2', 'fake_person3', 'fake_person5', 'frustum2', 'frustum3',
                      'frustum4', 'frustum5', 'frustum7', 'frustum8', 'uuv2', 'uuv3', 'uuv4', 'uuv5', 'uuv6', 'uuv7',
                      'uuv8', 'uuv9', 'uuv10', 'uuv11', 'uuv12', ]


class RGBPS_LI(RGBPS):
    _sequence_name = ['connected_polyhedron7', 'connected_polyhedron8', 'fake_person5', 'fake_person6', 'fake_person7',
                      'frustum8', 'octahedron7', 'octahedron8', 'uuv9', 'uuv10', ]
