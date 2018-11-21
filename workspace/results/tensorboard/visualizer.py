#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--csv-prefix', help='Prefix of CSV Files',
                    type=str, default=None)
parser.add_argument('--metric', help='Metric to be Plotted', 
                    type=str, default=None)
parser.add_argument('--metric-unit', help='Metric Unit', 
                    type=str, default=None)

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


def plt_legacy_vs_partial_fw_prop(csv_prefix, metric, metric_unit, ymin=None,
                                  xlabel='Global Step', 
                                  xlabel_unit='Number of Training Batches', title=None):
    """
    Plot the comparison between legacy backpropagation and partial forward propagation.
    """
    ylabel = metric.title().replace('_', ' ')
    title ='%s-legacy_vs_partial_fw_prop-%s' % (csv_prefix, metric)

    legacy  = np.genfromtxt(fname='%s-legacy/csv/%s.csv' % (csv_prefix, metric),
                            delimiter=',').astype(np.float64)[1:,:]
    partial = np.genfromtxt(fname='%s-partial_fw_prop/csv/%s.csv' % \
                                  (csv_prefix, metric),
                            delimiter=',').astype(np.float64)[1:,:]
    
    plt.Figure()

    ax = plt.axes()
    ax.plot(legacy [:,1], legacy [:,2], linewidth=2, linestyle='--', 
            color='black', label='Legacy')
    ax.plot(partial[:,1], partial[:,2], linewidth=2, linestyle='-',
            color='black', label='Partial FW Prop')

    plt.xlabel("%s (%s)" % (xlabel, xlabel_unit))
    plt.ylabel("%s (%s)" % (ylabel, metric_unit) if metric_unit is not None else ylabel)

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    args = parser.parse_args()

    plt_legacy_vs_partial_fw_prop(csv_prefix=args.csv_prefix,
                                  metric=args.metric, 
                                  metric_unit=args.metric_unit)