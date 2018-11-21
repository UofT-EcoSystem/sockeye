#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--csv-prefix', help='Prefix of CSV Files'
                    type=str, default=None)
parser.add_argument('--metric', help='Metric to be Plotted', type=str, default=None)


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


def plt_legacy_vs_partial_fw_prop(csv_prefix, csv_suffix, metric,
                                  xlabel='Global Step', 
                                  ylabel=None, title=None):
    """
    Plot the comparison between legacy backpropagation and partial forward propagation.

    :param prefix: FileName Prefix
    :param metric: Metric recorded on Tensorboard
    """
    if ylabel is None:
        ylabel = metric.title().replace('_', ' ')
    if title is None:
        title ='%s-%s-legacy_vs_partial_fw_prop' % (csv_prefix, csv_suffix)

    legacy  = np.genfromtxt(fname='%s-legacy-%s/csv/%s.csv' % (csv_prefix, csv_suffix, metric),
                                    delimiter=',').astype(np.float64)[1:,:]
    partial = np.genfromtxt(fname='%s-partial_fw_prop-%s/csv/%s.csv' % \
                                          (csv_prefix, csv_suffix, metric),
                                    delimiter=',').astype(np.float64)[1:,:]
    
    plt.Figure()

    ax = plt.axes()
    ax.plot(legacy [0], legacy [1], linewidth=1, linestyle='--', 
            color='black', label='Legacy')
    ax.plot(partial[0], partial[1], linewidth=1, linestype='-',
            color='black', label='Partial FW Prop')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.savefig(title + ".png")


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    args = parser.parse_args()

