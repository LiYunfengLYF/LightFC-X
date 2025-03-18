import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.rgbs_test.tracker.rgbs_tracker import RGBSerTracker



import lib.models
import multiprocessing
from lib.rgbs_test import ExperimentRGBPS


env_num = 0
if __name__ == '__main__':

    data_root = r'/home/code/dev/RGBS50'
    project_root = r'/home/code/rgbt-light'
    multiprocessing.set_start_method('spawn')
    e = ExperimentRGBPS(data_root, project_root)

    tracker = RGBSerTracker(r'RGBS_baseline_ep30')

    e.run(tracker)
    # e.eval(tracker.name)
