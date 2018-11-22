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
    # normalize the time axis to minutes
    data[:, 0] = (data[:, 0] - data[0, 0]) / 60.0

    return data


def plt_default_vs_econmt(csv_prefix, metric, metric_unit=None, ymin=None, ymax=None,
                          xlabel='Global Step', 
                          xlabel_unit='Number of Training Batches'):
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


def plt_throughput_vs_batch():
    B = [4, 8, 16, 32, 64, 128]

    resnet50_throughput, sockeye_throughput = [99.36, 137.38, 172.26, 197.28, 200.02, 206.91], []

    for batch_size in B:
        sockeye_throughput.append(gen_from_txt("iwslt15-vi_en-tbd-500-default-B_%d/csv/throughput.csv" % batch_size)[0, 2])
    print(sockeye_throughput)

    def _plt_throughput_vs_batch(batch, throughput, title):
        plt.figure()


        plt.plot(B, throughput, linewidth=2, linestyle='--', 
                 color='black', marker='X')

        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (Samples/s)")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # plt.legend(fontsize=20)
        plt.grid(linestyle='-.', linewidth=1)

        plt.tight_layout()
        plt.savefig("throughput_vs_batch-" + title + ".png")
    
    _plt_throughput_vs_batch(B, resnet50_throughput, "resnet_50")
    _plt_throughput_vs_batch(B,  sockeye_throughput, "sockeye")


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='perplexity')
    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='perplexity')
    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='memory_usage', metric_unit='GB')
    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='memory_usage', metric_unit='GB')
    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-groundhog-500', metric='throughput', metric_unit='Samples/s')
    # plt_default_vs_econmt(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='throughput', metric_unit='Samples/s')

    plt_throughput_vs_batch()