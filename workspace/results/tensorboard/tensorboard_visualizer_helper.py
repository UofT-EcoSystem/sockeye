import numpy as np
import matplotlib.pyplot as plt

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")

from visualizer_helper import plt_legend


def gen_from_txt(fname, metric, metric_unit=None, skip=None):
    data = np.genfromtxt(fname=fname, delimiter=',').astype(np.float64)[1:,:]

    if metric == 'throughput' and skip is not None:
        data_filter = np.all([np.mod(np.arange(data.shape[0])+1,skip)!=0, \
                              np.mod(np.arange(data.shape[0])+1,skip)!=1, \
                              np.mod(np.arange(data.shape[0])+1,skip)!=2, \
                              np.mod(np.arange(data.shape[0])+1,skip)!=3], axis=0)

        data = data[data_filter,:]
    if metric == 'memory_usage' and metric_unit == 'GB':
        data[:,2] = data[:,2] / 1000
    if metric == 'validation_bleu':
        data[:,2] = data[:,2] * 100
        starting_walltime = np.genfromtxt(fname=fname.replace(metric, 'perplexity'), 
                                          delimiter=',').astype(np.float64)[1,0]
        data = np.insert(data, 0, [starting_walltime, 0, 0], axis=0)
        if skip is not None:
            data = data[np.arange(data.shape[0])!=skip,:]

    # normalize the time axis to minutes
    data[:,0] = (data[:,0] - data[0, 0]) / 60.0

    return data


def plt_default_vs_econmt_preliminary(metric, metric_unit=None):
    """
    Plot the comparison between legacy backpropagation and 
    partial forward propagation (First 500 Updates, Preliminary Ver.).
    """
    ylabel = metric.title().replace('_', ' ')
    title ='default_vs_econmt-%s' % metric

    default = gen_from_txt(fname='default/csv/%s.csv' % metric,
                           metric=metric, metric_unit=metric_unit)
    econmt  = gen_from_txt(fname= 'econmt/csv/%s.csv' % metric,
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
    plt.ylabel("Throughput (samples/s)")
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
    axes.set_ylabel("Throughput (samples/s)")
    axes.set_yticks(np.arange(0, 501, 100))
    
    for ticklabel in axes.get_xticklabels() + axes.get_yticklabels():
        ticklabel.set_fontsize(20)

    axes.grid(linestyle='-.', linewidth=1)

    # ==============================================================================================

    axes = axes.twinx()

    memory_usage_plot = axes.plot(B, sockeye_memory_usage, linewidth=2, linestyle='--',
                                  color='black', marker='X', markersize=5, label="Memory Usage")
    
    axes.set_ylabel("Memory Consumption (GB)")
    axes.set_yticks(np.arange(0, 11, 2))

    legends = throughput_plot + memory_usage_plot
    axes.legend(legends, [legend.get_label() for legend in legends], fontsize=20)

    plt.tight_layout()
    plt.savefig("throughput_vs_batch_size-sockeye.png")


def plt_default_vs_econmt_full_training_validation_bleu(xscale, par_rev, first_k_ckpts, 
                                                        prefix='', suffix='', discard=None):

    metric, metric_unit = 'validation_bleu', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='%sdefault_vs_econmt%s%s-%s-%s' % (prefix, suffix, '-par_rev' if par_rev else '', xscale, metric)

    default_128_metric = gen_from_txt("default-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit, 
                                      discard[0] if discard is not None else None)
    econmt_128_metric  = gen_from_txt( "econmt-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit,
                                      discard[0] if discard is not None else None)
    econmt_256_metric  = gen_from_txt( "econmt-B_256%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit,
                                      discard[1] if discard is not None else None)

    # ==============================================================================================

    plt.figure()

    if par_rev:
        default_128_raw_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, metric, metric_unit)
        plt.plot(default_128_raw_metric[:first_k_ckpts[0],1] if xscale == 'N' else default_128_raw_metric[:first_k_ckpts[0],0], 
                 default_128_raw_metric[:first_k_ckpts[0],2], linewidth=2, linestyle='-', 
                 marker='v', markersize=10,
                 color='black', label=r'Default$_{B=128}$')
    plt.plot(default_128_metric[:first_k_ckpts[1],1] if xscale == 'N' else default_128_metric[:first_k_ckpts[1],0], 
             default_128_metric[:first_k_ckpts[1],2], linewidth=2, linestyle='-', 
             marker='X', markersize=10,
             color='black', label=r'Default$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_128_metric [:first_k_ckpts[1],1] if xscale == 'N' else econmt_128_metric [:first_k_ckpts[1],0], 
             econmt_128_metric [:first_k_ckpts[1],2], linewidth=2, linestyle='-', 
             marker='.', markersize=10,
             color='black', label= r'EcoRNN$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_256_metric [:first_k_ckpts[2],1] if xscale == 'N' else econmt_256_metric [:first_k_ckpts[2],0], 
             econmt_256_metric [:first_k_ckpts[2],2], linewidth=2, linestyle='-', 
             marker='^', markersize=10,
             color='black', label= r'EcoRNN$_{B=256}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))

    plt.axhline(y=22.6, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Training Checkpoint Number' if xscale == 'N' else 'Time (min)')
    plt.ylabel("Validation BLEU Score")
    if xscale == 'N':
        plt.xticks(np.arange(0, 9, 2), fontsize=20)
    else:
        plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=22)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_default_vs_econmt_full_training_perplexity(xscale, par_rev, prefix='', suffix=''):

    metric, metric_unit = 'perplexity', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='%sdefault_vs_econmt%s%s-%s-%s' % (prefix, suffix, '-par_rev' if par_rev else '', xscale, metric)

    default_128_metric = gen_from_txt("default-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_128_metric  = gen_from_txt( "econmt-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "econmt-B_256%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    first_k_ckpts = 100

    if par_rev:
        default_128_raw_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, 
                                              metric, metric_unit)
        plt.plot(default_128_raw_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_raw_metric[:first_k_ckpts,0], 
                 default_128_raw_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
                 marker='v', markersize=10, markevery=(1, 10),
                 color='black', label=r'Default$_{B=128}$')
    plt.plot(default_128_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_metric[:first_k_ckpts,0], 
             default_128_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10, markevery=(3, 10),
             color='black', label=r'Default$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_128_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_128_metric [:first_k_ckpts,0], 
             econmt_128_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='.', markersize=10, markevery=(5, 10),
             color='black', label= r'EcoRNN$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_256_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_256_metric [:first_k_ckpts,0], 
             econmt_256_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10, markevery=(7, 10),
             color='black', label= r'EcoRNN$_{B=256}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))

    plt.xlabel('Global Step' if xscale == 'N' else 'Time (min)')
    plt.ylabel("Perplexity")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=22)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_default_vs_econmt_full_training_metrics(metric, metric_unit, measurer, ylabel,
                                                prefix='', suffix='', bar_width=0.3):

    title ='%sdefault_vs_econmt%s-%s' % (prefix, suffix, metric)

    default_128_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric,
                                      metric=metric, metric_unit=metric_unit, skip=40)
    default_128_par_rev_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    # cudnn_128_par_rev_metric   = gen_from_txt(  "cudnn-B_128-par_rev/csv/%s.csv" % metric,
    #                                           metric=metric, metric_unit=metric_unit)
    econmt_128_par_rev_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    econmt_256_par_rev_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=20)

    # ==============================================================================================

    default_128_metric = measurer(default_128_metric[:,2])
    default_128_par_rev_metric = measurer(default_128_par_rev_metric[:,2])
    # cudnn_128_par_rev_metric   = measurer(  cudnn_128_par_rev_metric[:,2])
    econmt_128_par_rev_metric  = measurer( econmt_128_par_rev_metric[:,2])
    econmt_256_par_rev_metric  = measurer( econmt_256_par_rev_metric[:,2])

    # plt.figure(figsize=(6, 8))
    plt.figure()

    def _annotate(x, metric):
        plt.annotate((r'$%.2f\times$') % (metric / default_128_par_rev_metric),
                 xy    =(x, metric + 0.04*plt.ylim()[1]), 
                 xytext=(x, metric + 0.04*plt.ylim()[1]), 
                 fontsize=18, ha='center', va='center', 
                 bbox=dict(boxstyle='square', facecolor='white', linewidth=3))

    handles = []

    handles.append(plt.bar(x=-2*bar_width, height=default_128_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='grey',
            label=r"Default$_{B=128}$"))
    handles.append(plt.bar(x=-1*bar_width, height=default_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='white', 
            label=r"Default$_{B=128}^\mathrm{par\_rev}$"))
    # handles.append(plt.bar(x=0, height=cudnn_128_par_rev_metric,
    #         width=bar_width, edgecolor='black', linewidth=3,
    #         color='white', 
    #         label=  r"CuDNN$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 0*bar_width, height= econmt_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color=np.array([0, 0.2, 0]),
            label= r"EcoRNN$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 1*bar_width, height= econmt_256_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color=np.array([0, 0.8, 0]),
            label= r"EcoRNN$_{B=256}^\mathrm{par\_rev}$"))

    _annotate(x=-2*bar_width, metric=default_128_metric)
    _annotate(x=-1*bar_width, metric=default_128_par_rev_metric)
    # _annotate(x= 0          , metric=  cudnn_128_par_rev_metric)
    _annotate(x= 0*bar_width, metric= econmt_128_par_rev_metric)
    _annotate(x= 1*bar_width, metric= econmt_256_par_rev_metric)

    plt.xlim([-3*bar_width, 2*bar_width])
    plt.xticks([])
    plt.yticks(fontsize=20)
    plt.ylabel(ylabel)

    # plt.legend(fontsize=20, ncol=1)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, "default_vs_econmt-legend")
    plt_legend(handles, "default_vs_econmt-legend-ncol_2", ncol=2)
    plt_legend(handles, "default_vs_econmt-legend-horizontal", ncol=len(handles))