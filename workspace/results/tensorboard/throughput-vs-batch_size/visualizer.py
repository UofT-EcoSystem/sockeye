#!/usr/bin/python

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")

from visualizer_helper import plt_rc_setup, plt_throughput_vs_batch_size

plt_rc_setup()
plt_throughput_vs_batch_size()
