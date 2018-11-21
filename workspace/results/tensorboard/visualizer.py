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


def plt_default_vs_econmt(csv_prefix, metric, metric_unit=None, ymin=None, ymax=None,
                                  xlabel='Global Step', 
                                  xlabel_unit='Number of Training Batches', title=None):
    """
    Plot the comparison between legacy backpropagation and partial forward propagation.
    """
    ylabel = metric.title().replace('_', ' ')
    title ='%s-default_vs_econmt-%s' % (csv_prefix, metric)

    default = gen_from_txt(fname='%s-default/csv/%s.csv' % (csv_prefix, metric))
    econmt  = gen_from_txt(fname= '%s-econmt/csv/%s.csv' % (csv_prefix, metric))

    if metric == 'memory_usage' and metric_unit == 'GB':
        default[:,2] = default[:,2] / 1000
        econmt [:,2] = econmt [:,2] / 1000
    
    plt.figure()

    plt.plot(default[:,1], default[:,2], linewidth=2, linestyle='--', 
             color='black', label='Default')
    plt.plot(econmt [:,1], econmt [:,2], linewidth=2, linestyle='-',
             color='black', label='EcoNMT')

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

    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='speed', metric_unit='Samples/s')
    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='speed', metric_unit='Samples/s')
    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='perplexity')
    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='perplexity')
    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='memory_usage', metric_unit='GB')
    plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='memory_usage', metric_unit='GB')
