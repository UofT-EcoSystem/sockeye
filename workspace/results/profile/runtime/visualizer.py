#!/usr/bin/python

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_breakdown


sorted_stats_list = [
    ("SeqRev", "")
]
