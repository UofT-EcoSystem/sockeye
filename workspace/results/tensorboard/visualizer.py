#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import numpy as np
import matplotlib.pyplot as plt

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


def gen_from_txt(fname):
    data = np.genfromtxt(fname=fname, delimiter=',').astype(np.float64)[1:,:]
    data[:, 0] = (data[:, 0] - data[0, 0]) / 60.0

    return data


def plt_legacy_vs_partial_fw_prop(csv_prefix, metric, metric_unit=None, ymin=None, ymax=None,
                                  xlabel='Global Step', 
                                  xlabel_unit='Number of Training Batches', title=None):
    """
    Plot the comparison between legacy backpropagation and partial forward propagation.
    """
    ylabel = metric.title().replace('_', ' ')
    title ='%s-legacy_vs_partial_fw_prop-%s' % (csv_prefix, metric)

    legacy  = gen_from_txt(fname='%s-legacy/csv/%s.csv' % (csv_prefix, metric))
    partial = gen_from_txt(fname='%s-partial_fw_prop/csv/%s.csv' % (csv_prefix, metric))

    if metric == 'memory_usage' and metric_unit == 'GB':
        legacy [:,2] = legacy [:,2] / 1024
        partial[:,2] = partial[:,2] / 1024
    
    plt.figure()

    plt.plot(legacy [:,1], legacy [:,2], linewidth=2, linestyle='--', 
             color='black', label='Legacy')
    plt.plot(partial[:,1], partial[:,2], linewidth=2, linestyle='-',
             color='black', label='Partial FW Prop')

    plt.xlabel("%s (%s)" % (xlabel, xlabel_unit))
    plt.ylabel("%s (%s)" % (ylabel, metric_unit) if metric_unit is not None else ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if metric == 'memory_usage':
        plt.yticks(np.arange(0, 13, 4))
    if metric == 'perplexity':
        plt.yticks(np.arange(0, 1300, 400))

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    plt_legacy_vs_partial_fw_prop(csv_prefix='iwslt15-vi_en-groundhog-500', metric='speed', 
                                  metric_unit='Samples/s')
    plt_legacy_vs_partial_fw_prop(csv_prefix='iwslt15-vi_en-groundhog-500', metric='perplexity')
    plt_legacy_vs_partial_fw_prop(csv_prefix='iwslt15-vi_en-groundhog-500', metric='memory_usage',
                                  metric_unit='GB')