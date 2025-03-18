import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import matplotlib.pyplot as plt
from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
import argparse

plt.rcParams['figure.figsize'] = [8, 8]


parser = argparse.ArgumentParser(description='Parse args for training')
parser.add_argument('--tracker_name', type=str, default=None, help='test script name')
parser.add_argument('--tracker_param', type=str, default='baseline', help='yaml configure file name')

parser.add_argument('--dataset', type=str, default='rgbt234', help='Sequence number or name.')
parser.add_argument('--env_num', type=int, default=0, help='Use for multi environment developing, support: 0,1,2')
parser.add_argument('--send_email', type=bool, default=False,
                    help='Use for multi environment developing, support: 0,1,2')
args = parser.parse_args()

trackers = []
trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset,
                            run_ids=None, display_name=args.tracker_param, env_num=args.env_num))

dataset = get_dataset(args.dataset, env_num=args.env_num)
score = print_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
                      env_num=args.env_num)
