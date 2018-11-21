#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', type=str, default=None)
parser.add_argument('--metric', type=str, default=None)


def plt_rc_setup(dpi=400, fontsize=24):
    """
    Setup the RC parameters of Pyplot.

    :param dpi     : Figure Resolution (Default to 400)
    :param fontsize: Font Size (Default to 24)
    """
    plt.rc('figure', dpi=dpi)
    plt.rc('axes', axisbelow=True)
    plt.rc('mathtext', fontset='cm')
    plt.rc('mathtext', rm='Times New Roman')
    plt.rc('font', family='Times New Roman', size=fontsize)


def plt_legacy_vs_partial_fw_prop(prefix, metric):
    """
    Plot the comparison between legacy backpropagation and partial forward propagation.

    :param prefix: FileName Prefix
    :param metric: Metric recorded on Tensorboard
    """


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    args = parser.parse_args()

