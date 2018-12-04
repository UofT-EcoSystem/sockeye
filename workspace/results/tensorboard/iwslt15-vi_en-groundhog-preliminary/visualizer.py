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
from tensorboard_visualizer_helper import \
    plt_default_vs_econmt_preliminary, \
    plt_default_vs_econmt_preliminary_metric


plt_rc_setup()

# plt_default_vs_econmt_preliminary(metric='perplexity')
# plt_default_vs_econmt_preliminary(metric='memory_usage', metric_unit='GB')
# plt_default_vs_econmt_preliminary(metric='throughput', metric_unit='samples/s')

plt_default_vs_econmt_preliminary_metric(prefix='iwslt15-vi_en-groundhog-', batch_size=80, 
                                         max_memory_usage=8,
                                         max_throughput=600)