#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup
from tensorboard_visualizer_helper import gen_from_txt, plt_default_vs_econmt_full_training_perplexity, \
    plt_default_vs_econmt_full_training_validation_bleu, \
    plt_default_vs_econmt_full_training_metrics

plt_rc_setup()

plt_default_vs_econmt_full_training_perplexity('N', True)
plt_default_vs_econmt_full_training_perplexity('T', True)
plt_default_vs_econmt_full_training_validation_bleu('T', True , first_k_ckpts=(8, 8, 6))

plt_default_vs_econmt_full_training_metrics(metric='throughput', metric_unit='samples/s', 
                                            measurer=np.average,
                                            ylabel='Avg Throughput (samples/s)')
plt_default_vs_econmt_full_training_metrics(metric='memory_usage', metric_unit='GB', 
                                            measurer=np.max, 
                                            ylabel='Max Memory Consumption (GB)',)
