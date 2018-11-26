#!/usr/bin/python

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")

from visualizer_helper import plt_rc_setup, gen_from_txt, plt_default_vs_econmt_preliminary

plt_rc_setup()

plt_default_vs_econmt_preliminary(metric='perplexity')
plt_default_vs_econmt_preliminary(metric='perplexity')
plt_default_vs_econmt_preliminary(metric='memory_usage', metric_unit='GB')
plt_default_vs_econmt_preliminary(metric='memory_usage', metric_unit='GB')
plt_default_vs_econmt_preliminary(metric='throughput', metric_unit='Samples/s')
plt_default_vs_econmt_preliminary(metric='throughput', metric_unit='Samples/s')
