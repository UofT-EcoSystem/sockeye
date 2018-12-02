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

plt_default_vs_econmt_full_training_perplexity('N', True, prefix='iwslt15-vi_en-tbd-')
plt_default_vs_econmt_full_training_perplexity('T', True, prefix='iwslt15-vi_en-tbd-')
plt_default_vs_econmt_full_training_validation_bleu('T', True, first_k_ckpts=(7, 7, 6), bar=22.6, prefix='iwslt15-vi_en-tbd-')

plt_default_vs_econmt_full_training_metrics(metric='throughput', metric_unit='samples/s', 
                                            measurer=np.average,
                                            prefix='iwslt15-vi_en-tbd-',
                                            ylabel='Throughput (samples/s)')
plt_default_vs_econmt_full_training_metrics(metric='memory_usage', metric_unit='GB', 
                                            measurer=np.max,
                                            prefix='iwslt15-vi_en-tbd-',
                                            ylabel='Memory Consumption (GB)',)
