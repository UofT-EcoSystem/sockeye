#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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


def gen_from_txt(fname, metric, metric_unit=None):
    data = np.genfromtxt(fname=fname, delimiter=',').astype(np.float64)[1:,:]

    if metric == 'memory_usage' and metric_unit == 'GB':
        data[:,2] = data[:,2] / 1000
    if metric == 'validation_bleu':
        data[:,2] = data[:,2] * 100
        starting_walltime = np.genfromtxt(fname=fname.replace(metric, 'perplexity'), 
                                          delimiter=',').astype(np.float64)[1,0]
        data = np.insert(data, 0, [starting_walltime, 0, 0], axis=0)

    # normalize the time axis to minutes
    data[:,0] = (data[:,0] - data[0, 0]) / 60.0

    if metric == 'validation_bleu':
        data_new = np.zeros(shape=(100, 3)).astype(np.float64)

        data_new[:,0] = np.linspace(0, np.max(data[:,0]), num=100, endpoint=True)
        data_new[:,1] = np.linspace(1, np.max(data[:,1]), num=100, endpoint=True)
        f = interp1d(data[:,0], data[:,2], kind='quadratic')
        data_new[:,2] = f(data_new[:,0])

        data = data_new

    return data


def plt_default_vs_econmt_preliminary(csv_prefix, metric, metric_unit=None):
    """
    Plot the comparison between legacy backpropagation and 
    partial forward propagation (First 500 Updates, Preliminary Ver.).
    """
    ylabel = metric.title().replace('_', ' ')
    title ='%s-default_vs_econmt-preliminary-%s' % (csv_prefix, metric)

    default = gen_from_txt(fname='%s-default/csv/%s.csv' % (csv_prefix, metric),
                           metric=metric, metric_unit=metric_unit)
    econmt  = gen_from_txt(fname= '%s-econmt/csv/%s.csv' % (csv_prefix, metric),
                           metric=metric, metric_unit=metric_unit)
    
    # ==============================================================================================

    plt.figure()

    plt.plot(default[:,1], default[:,2], linewidth=2, linestyle='--', 
             color='black', label='Default')
    plt.plot(econmt [:,1], econmt [:,2], linewidth=2, linestyle='-',
             color='black', label='EcoNMT')

    plt.xlabel('Global Step (Number of Training Batches)')
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


def plt_throughput_vs_batch_size():
    B = [4, 8, 16, 32, 64, 128]

    resnet50_throughput = [99.36, 137.38, 172.26, 197.28, 200.02, 206.91]

    sockeye_throughput   = []
    sockeye_memory_usage = []

    for batch_size in B:
        sockeye_throughput  .append(gen_from_txt("iwslt15-vi_en-tbd-500-default-B_%d/csv/throughput.csv"   % batch_size,
                                    metric="throughput")[0, 2])
        sockeye_memory_usage.append(gen_from_txt("iwslt15-vi_en-tbd-500-default-B_%d/csv/memory_usage.csv" % batch_size,
                                    metric="memory_usage", metric_unit="GB")[-1, 2])

    # ==============================================================================================

    plt.figure()

    plt.plot(B, resnet50_throughput, linewidth=2, linestyle='-', 
             color='black', marker='o', markersize=5)

    plt.xlabel("Batch Size")
    plt.xlim(xmin=0, xmax=140)
    plt.xticks(B, ['%d' % batch_size if batch_size != 8 else '' \
        for batch_size in B], fontsize=20)
    plt.ylabel("Throughput (Samples/s)")
    plt.yticks(np.arange(0, 251, 50), fontsize=20)

    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig("throughput_vs_batch_size-resnet_50.png")

    # ==============================================================================================

    fig, axes = plt.subplots()

    throughput_plot = axes.plot(B, sockeye_throughput, linewidth=2, linestyle='-',
                                color='black', marker='o', markersize=5, label="Throughput")
    axes.set_xlabel("Batch Size")
    axes.set_xlim(xmin=0, xmax=140)
    axes.set_xticks(B)
    axes.set_xticklabels(['%d' % batch_size if batch_size != 8 else '' \
        for batch_size in B])
    axes.set_ylabel("Throughput (Samples/s)")
    axes.set_yticks(np.arange(0, 501, 100))
    
    for ticklabel in axes.get_xticklabels() + axes.get_yticklabels():
        ticklabel.set_fontsize(20)

    axes.grid(linestyle='-.', linewidth=1)

    # ==============================================================================================

    axes = axes.twinx()

    memory_usage_plot = axes.plot(B, sockeye_memory_usage, linewidth=2, linestyle='--',
                                  color='black', marker='X', markersize=5, label="Memory Usage")
    
    axes.set_ylabel("Memory Usage (GB)")
    axes.set_yticks(np.arange(0, 11, 2))

    legends = throughput_plot + memory_usage_plot
    axes.legend(legends, [legend.get_label() for legend in legends], fontsize=20)

    plt.tight_layout()
    plt.savefig("throughput_and_memory_usage_vs_batch_size-sockeye.png")


def plt_default_vs_econmt_full_training(csv_prefix, xscale, metric, metric_unit=None):
    """
    Plot the comparison between legacy backpropagation and 
    partial forward propagation (Full Training Ver.).
    """
    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    ylabel = metric.title().replace('_', ' ')
    title ='%s-default_vs_econmt-full_training-%s-%s' % (csv_prefix, xscale, metric)

    default_128_metric = gen_from_txt("%s-default-B_128/csv/%s.csv" % (csv_prefix, metric), metric, metric_unit)
    econmt_128_metric  = gen_from_txt( "%s-econmt-B_128/csv/%s.csv" % (csv_prefix, metric), metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "%s-econmt-B_256/csv/%s.csv" % (csv_prefix, metric), metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    plt.plot(default_128_metric[:,1] if xscale == 'N' else default_128_metric[:,0], 
             default_128_metric[:,2], linewidth=2, linestyle='--', 
             color='black', label='Default-B=128')
    plt.plot(econmt_128_metric [:,1] if xscale == 'N' else econmt_128_metric [:,0], 
             econmt_128_metric [:,2], linewidth=2, linestyle='-',
             color='black', label='EcoNMT-B=128')
    plt.plot(econmt_256_metric [:,1] if xscale == 'N' else econmt_256_metric [:,0], 
             econmt_256_metric [:,2], linewidth=2, linestyle='-',
             color='green', label='EcoNMT-B=256')

    plt.xlabel(r'Time (min)' if xscale == 'T' else \
               'Training Checkpoint Number' if metric == 'validation_bleu' else \
               'Global Step (Number of Training Batches)')
    plt.ylabel("Validation BLEU" if metric == 'validation_bleu' else \
               "%s (%s)" % (ylabel, metric_unit) if metric_unit is not None else ylabel)
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

    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-groundhog-500', metric='perplexity')
    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='perplexity')
    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-groundhog-500', metric='memory_usage', metric_unit='GB')
    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='memory_usage', metric_unit='GB')
    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-groundhog-500', metric='throughput', metric_unit='Samples/s')
    # plt_default_vs_econmt_preliminary(csv_prefix='iwslt15-vi_en-tbd-500'      , metric='throughput', metric_unit='Samples/s')

    # plt_throughput_vs_batch_size()

    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='N', metric='perplexity')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='N', metric='memory_usage', metric_unit='GB')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='N', metric='throughput', metric_unit='Samples/s')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='N', metric='validation_bleu')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='T', metric='perplexity')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='T', metric='memory_usage', metric_unit='GB')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='T', metric='throughput', metric_unit='Samples/s')
    plt_default_vs_econmt_full_training(csv_prefix='iwslt15-vi_en-tbd-10k', xscale='T', metric='validation_bleu')