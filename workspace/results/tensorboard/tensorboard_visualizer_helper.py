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


def plt_default_vs_econmt_full_training_validation_bleu(first_k_ckpts, prefix='', suffix='', bar=None, discard=None):

    metric, metric_unit = 'validation_bleu', None
    
    title ='%sdefault_vs_econmt%s-par_rev-T-%s' % (prefix, suffix, metric)

    default_128_raw_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, metric, metric_unit)
    default_128_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit, 
                                      discard[0] if discard is not None else None)
    econmt_128_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit,
                                      discard[0] if discard is not None else None)
    econmt_256_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit,
                                      discard[1] if discard is not None else None)

    # ==============================================================================================

    plt.figure()

    handles = []

    handles.append(plt.plot(default_128_raw_metric[:first_k_ckpts[0],0], 
                            default_128_raw_metric[:first_k_ckpts[0],2], linewidth=2, linestyle='-', 
                            marker='v', markersize=10,
                            color='black', label=r'Default$_{B=128}$')[0])
    handles.append(plt.plot(default_128_metric[:first_k_ckpts[1],0], 
                            default_128_metric[:first_k_ckpts[1],2], linewidth=2, linestyle='-', 
                            marker='X', markersize=10,
                            color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')[0])
    handles.append(plt.plot(econmt_128_metric [:first_k_ckpts[2],0], 
                            econmt_128_metric [:first_k_ckpts[2],2], linewidth=2, linestyle='-', 
                            marker='.', markersize=10,
                            color='black', label= r'EcoRNN$_{B=128}^{\mathrm{par\_rev}}$')[0])
    handles.append(plt.plot(econmt_256_metric [:first_k_ckpts[3],0], 
                            econmt_256_metric [:first_k_ckpts[3],2], linewidth=2, linestyle='-', 
                            marker='^', markersize=10, 
                            color='black', label= r'EcoRNN$_{B=256}^{\mathrm{par\_rev}}$')[0])

    def _annotate(x, y):
        plt.annotate(r"$%.2f\times$" % (x / default_128_metric[first_k_ckpts[1]-2,0]), 
                    xy    =(0, y),
                    xytext=(x, y),
                    fontsize=20,
                    va="center", ha="left",
                    bbox=dict(boxstyle="square", fc="white", ec='blue', linewidth=3),
                    arrowprops=dict(arrowstyle="<|-|>", color='blue', linewidth=3))
    
    _annotate(default_128_raw_metric[first_k_ckpts[0]-2,0], 4*bar/5)
    _annotate(default_128_metric[first_k_ckpts[1]-2,0], 3*bar/5)
    _annotate( econmt_128_metric[first_k_ckpts[2]-2,0], 2*bar/5)
    _annotate( econmt_256_metric[first_k_ckpts[3]-2,0], 1*bar/5)

    if bar is not None:
        plt.text(2, bar + 1.5, 'Target BLEU Score %.1f' % bar, fontsize=18,
                 bbox=dict(boxstyle="square", fc="white", ec='red', linewidth=3))
        plt.axhline(y=bar, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Time (min)')
    plt.ylabel("Validation BLEU Score")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    # plt.legend(fontsize=24)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, 'default_vs_econmt-plot-legend-horizontal', ncol=len(handles))


def plt_cudnn_vs_econmt_full_training_validation_bleu(first_k_ckpts, cross_bar, 
                                                      prefix='', suffix='', bar=None, discard=None):

    metric, metric_unit = 'validation_bleu', None
    
    title ='%scudnn_vs_econmt%s-par_rev-T-%s' % (prefix, suffix, metric)

    default_128_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit, 
                                      discard[0] if discard is not None else None)
    cudnn_128_metric   = gen_from_txt(  "cudnn-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit, 
                                      discard[0] if discard is not None else None)
    econmt_256_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit,
                                      discard[1] if discard is not None else None)

    # ==============================================================================================

    plt.figure()

    handles = []

    handles.append(plt.plot(default_128_metric[:first_k_ckpts[0],0], 
                            default_128_metric[:first_k_ckpts[0],2], linewidth=2, linestyle='-', 
                            marker='X', markersize=10,
                            color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')[0])
    handles.append(plt.plot(cudnn_128_metric  [:first_k_ckpts[1],0], 
                            cudnn_128_metric  [:first_k_ckpts[1],2], linewidth=2, linestyle='-', 
                            marker='P', markersize=10,
                            color='black', label=  r'CuDNN$_{B=128}^{\mathrm{par\_rev}}$')[0])
    handles.append(plt.plot(econmt_256_metric [:first_k_ckpts[2],0], 
                            econmt_256_metric [:first_k_ckpts[2],2], linewidth=2, linestyle='-', 
                            marker='^', markersize=10, 
                            color='black', label= r'EcoRNN$_{B=256}^{\mathrm{par\_rev}}$')[0])

    def _find_intersection(start, end, bar):
        x = np.arange(start[0], end[0], (end[0] - start[0]) / 100)
        y = np.arange(start[1], end[1], (end[1] - start[1]) / 100)
        idx = np.argwhere(np.diff(np.sign(y - bar))).flatten()
        return x[idx[0]]

    default_intersection = _find_intersection((default_128_metric[cross_bar[0]  ,0],
                                               default_128_metric[cross_bar[0]  ,2]),
                                              (default_128_metric[cross_bar[0]+1,0],
                                               default_128_metric[cross_bar[0]+1,2]), 20.0)
    cudnn_intersection   = _find_intersection((cudnn_128_metric  [cross_bar[1]  ,0],
                                               cudnn_128_metric  [cross_bar[1]  ,2]),
                                              (cudnn_128_metric  [cross_bar[1]+1,0],
                                               cudnn_128_metric  [cross_bar[1]+1,2]), 20.0)
    econmt_intersection  = _find_intersection((econmt_256_metric [cross_bar[2]  ,0],
                                               econmt_256_metric [cross_bar[2]  ,2]),
                                              (econmt_256_metric [cross_bar[2]+1,0],
                                               econmt_256_metric [cross_bar[2]+1,2]), 20.0)

    def _annotate(x, y):
        plt.annotate(r"$%.2f\times$" % (x / default_intersection), 
                    xy    =(0, y),
                    xytext=(x, y),
                    fontsize=20,
                    va="center", ha="left",
                    bbox=dict(boxstyle="square", fc="white", ec='blue', linewidth=3),
                    arrowprops=dict(arrowstyle="<|-|>", color='blue', linewidth=3))
    
    _annotate(default_intersection, 3*bar/4)
    _annotate(  cudnn_intersection, 2*bar/4)
    _annotate( econmt_intersection, 1*bar/4)

    # print(default_intersection, cudnn_intersection, econmt_intersection)

    if bar is not None:
        plt.text(2, bar + 1.5, 'Target BLEU Score %.1f' % bar, fontsize=18,
                 bbox=dict(boxstyle="square", fc="white", ec='red', linewidth=3))
        plt.axhline(y=bar, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Time (min)')
    plt.ylabel("Validation BLEU Score")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    # plt.legend(fontsize=24)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, 'cudnn_vs_econmt-plot-legend-horizontal', ncol=len(handles))


def plt_default_vs_econmt_full_training_perplexity(xscale, prefix='', suffix=''):

    metric, metric_unit = 'perplexity', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='%sdefault_vs_econmt%s-par_rev-%s-%s' % (prefix, suffix, xscale, metric)

    default_128_raw_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, metric, metric_unit)
    default_128_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit)
    econmt_128_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    first_k_ckpts = 100
        
    plt.plot(default_128_raw_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_raw_metric[:first_k_ckpts,0], 
             default_128_raw_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='v', markersize=10, markevery=(1, 10),
             color='black', label=r'Default$_{B=128}$')
    plt.plot(default_128_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_metric[:first_k_ckpts,0], 
             default_128_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10, markevery=(3, 10),
             color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')
    plt.plot(econmt_128_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_128_metric [:first_k_ckpts,0], 
             econmt_128_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='.', markersize=10, markevery=(5, 10),
             color='black', label= r'EcoRNN$_{B=128}^{\mathrm{par\_rev}}$')
    plt.plot(econmt_256_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_256_metric [:first_k_ckpts,0], 
             econmt_256_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10, markevery=(7, 10),
             color='black', label= r'EcoRNN$_{B=256}^{\mathrm{par\_rev}}$')

    plt.xlabel('Global Step' if xscale == 'N' else 'Time (min)')
    plt.ylabel("Perplexity")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    # plt.legend(fontsize=22)
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
    econmt_128_par_rev_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    econmt_256_par_rev_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=20)

    # ==============================================================================================

    default_128_metric = measurer(default_128_metric[:,2])
    default_128_par_rev_metric = measurer(default_128_par_rev_metric[:,2])
    econmt_128_par_rev_metric  = measurer(econmt_128_par_rev_metric [:,2])
    econmt_256_par_rev_metric  = measurer(econmt_256_par_rev_metric [:,2])

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
            color='black',
            label=r"Default$_{B=128}$"))
    handles.append(plt.bar(x=-1*bar_width, height=default_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='white', 
            label=r"Default$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 0*bar_width, height= econmt_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color=np.array([0, 0.5, 0]),
            label= r"EcoRNN$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 1*bar_width, height= econmt_256_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color=np.array([0, 0.9, 0]),
            label= r"EcoRNN$_{B=256}^\mathrm{par\_rev}$"))

    _annotate(x=-2*bar_width, metric=default_128_metric)
    _annotate(x=-1*bar_width, metric=default_128_par_rev_metric)
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
    plt_legend(handles, "default_vs_econmt-bar-legend-horizontal", ncol=len(handles))


def plt_cudnn_vs_econmt_full_training_metrics(metric, metric_unit, measurer, ylabel,
                                              prefix='', suffix='', bar_width=0.3):

    title ='%scudnn_vs_econmt%s-%s' % (prefix, suffix, metric)

    default_128_par_rev_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    cudnn_128_par_rev_metric   = gen_from_txt("cudnn-B_128-par_rev/csv/%s.csv"   % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    econmt_256_par_rev_metric  = gen_from_txt("econmt-B_256-par_rev/csv/%s.csv"  % metric,
                                              metric=metric, metric_unit=metric_unit, skip=20)

    # ==============================================================================================

    default_128_par_rev_metric = measurer(default_128_par_rev_metric[:,2])
    cudnn_128_par_rev_metric   = measurer(cudnn_128_par_rev_metric  [:,2])
    econmt_256_par_rev_metric  = measurer(econmt_256_par_rev_metric [:,2])

    # plt.figure(figsize=(6, 8))
    plt.figure()

    def _annotate(x, metric):
        plt.annotate((r'$%.2f\times$') % (metric / default_128_par_rev_metric),
                 xy    =(x, metric + 0.04*plt.ylim()[1]), 
                 xytext=(x, metric + 0.04*plt.ylim()[1]), 
                 fontsize=18, ha='center', va='center', 
                 bbox=dict(boxstyle='square', facecolor='white', linewidth=3))

    handles = []

    handles.append(plt.bar(x=-1*bar_width, height=default_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='white', 
            label=r"Default$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 0*bar_width, height=  cudnn_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='grey',
            label=  r"CuDNN$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 1*bar_width, height= econmt_256_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color=np.array([0, 0.8, 0]),
            label= r"EcoRNN$_{B=256}^\mathrm{par\_rev}$"))

    _annotate(x=-1*bar_width, metric=default_128_par_rev_metric)
    _annotate(x= 0*bar_width, metric=cudnn_128_par_rev_metric)
    _annotate(x= 1*bar_width, metric=econmt_256_par_rev_metric)

    plt.xlim([-2.5*bar_width, 2.5*bar_width])
    plt.xticks([])
    plt.yticks(fontsize=20)
    plt.ylabel(ylabel)

    # plt.legend(fontsize=20, ncol=1)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, "cudnn_vs_econmt-bar-legend-horizontal", ncol=len(handles))