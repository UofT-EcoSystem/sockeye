#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_legend
from tensorboard_visualizer_helper import gen_from_txt


def plt_default_vs_econmt_full_training_validation_bleu(xscale, par_rev):

    metric, metric_unit = 'validation_bleu', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='default_vs_econmt%s-%s-%s' % ('-par_rev' if par_rev else '', xscale, metric)

    default_128_metric = gen_from_txt("default-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_128_metric  = gen_from_txt( "econmt-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "econmt-B_256%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    first_k_ckpts = 7 if par_rev else 8

    if par_rev:
        default_128_raw_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, 
                                              metric, metric_unit)
        plt.plot(default_128_raw_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_raw_metric[:first_k_ckpts,0], 
                 default_128_raw_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
                 marker='v', markersize=10,
                 color='black', label=r'Default$_{B=128}$')
    plt.plot(default_128_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_metric[:first_k_ckpts,0], 
             default_128_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10,
             color='black', label=r'Default$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_128_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_128_metric [:first_k_ckpts,0], 
             econmt_128_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='.', markersize=10,
             color='black', label= r'EcoNMT$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_256_metric [:6,1] if xscale == 'N' else econmt_256_metric [:6,0], 
             econmt_256_metric [:6,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10,
             color='black', label= r'EcoNMT$_{B=256}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))

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

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_default_vs_econmt_full_training_perplexity(xscale, par_rev):

    metric, metric_unit = 'perplexity', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='default_vs_econmt%s-%s-%s' % ('-par_rev' if par_rev else '', xscale, metric)

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
             color='black', label= r'EcoNMT$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_256_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_256_metric [:first_k_ckpts,0], 
             econmt_256_metric [:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10, markevery=(7, 10),
             color='black', label= r'EcoNMT$_{B=256}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))

    plt.xlabel('Global Step' if xscale == 'N' else 'Time (min)')
    plt.ylabel("Perplexity")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_default_vs_econmt_full_training_end2end():
    metric, metric_unit = 'validation_bleu', None
    
    title ='default_vs_econmt-end2end-%s' % metric

    default_128_metric = gen_from_txt(       "default-B_128/csv/%s.csv" % metric, metric, metric_unit)
    econmt_256_metric  = gen_from_txt("econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    plt.plot(default_128_metric[:7,0],
             default_128_metric[:7,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10,
             color='black', label=r'Default$_{B=128}$')
    plt.plot(econmt_256_metric [:6,0], 
             econmt_256_metric [:6,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10,
             color='black', label= r'EcoNMT$_{B=256}^{\mathrm{par\_rev}}$')

    plt.axhline(y=22.6, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Time (min)')
    plt.ylabel("Validation BLEU Score")
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_default_vs_econmt_full_training_metrics(metric, metric_unit, measurer, ylabel,
                                                bar_width=0.3):

    title ='default_vs_econmt-%s' % metric

    default_128_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric,
                                      metric=metric, metric_unit=metric_unit, skip=40)
    # econmt_128_metric  = gen_from_txt( "econmt-B_128/csv/%s.csv" % metric, 
    #                                   metric=metric, metric_unit=metric_unit, skip=40)
    # econmt_256_metric  = gen_from_txt( "econmt-B_256/csv/%s.csv" % metric, 
    #                                   metric=metric, metric_unit=metric_unit, skip=20)
    default_128_par_rev_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    econmt_128_par_rev_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=40)
    econmt_256_par_rev_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric,
                                              metric=metric, metric_unit=metric_unit, skip=20)

    # ==============================================================================================

    default_128_metric = measurer(default_128_metric[:,2])
    # econmt_128_metric  = measurer( econmt_128_metric[:,2])
    # econmt_256_metric  = measurer( econmt_256_metric[:,2])
    default_128_par_rev_metric = measurer(default_128_par_rev_metric[:,2])
    econmt_128_par_rev_metric  = measurer( econmt_128_par_rev_metric[:,2])
    econmt_256_par_rev_metric  = measurer( econmt_256_par_rev_metric[:,2])

    plt.figure(figsize=(6, 8))
    # plt.figure()

    def _annotate(x, metric):
        plt.annotate((r'$%.1f\times$') % (metric / default_128_par_rev_metric),
                 xy    =(x, metric + 0.04*plt.ylim()[1]), 
                 xytext=(x, metric + 0.04*plt.ylim()[1]), 
                 fontsize=20, ha='center', va='center', 
                 bbox=dict(boxstyle='square', facecolor='white', linewidth=3))

    handles = []

    handles.append(plt.bar(x=-1.5*bar_width, height=default_128_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color= 'grey', hatch='/',
            label=r"Default$_{B=128}$"))
    # handles.append(plt.bar(x=-1.5*bar_width, height= econmt_128_metric,
    #         width=bar_width, edgecolor='black', linewidth=3,
    #         color='white',
    #         label= r"EcoRNN$_{B=128}$"))
    # handles.append(plt.bar(x=-0.5*bar_width, height= econmt_256_metric,
    #         width=bar_width, edgecolor='black', linewidth=3,
    #         color='green',
    #         label= r"EcoRNN$_{B=128}$"))
    handles.append(plt.bar(x=-0.5*bar_width, height=default_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color= 'grey', 
            label=r"Default$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 0.5*bar_width, height= econmt_128_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='white', 
            label= r"EcoRNN$_{B=128}^\mathrm{par\_rev}$"))
    handles.append(plt.bar(x= 1.5*bar_width, height= econmt_256_par_rev_metric,
            width=bar_width, edgecolor='black', linewidth=3,
            color='green', 
            label= r"EcoRNN$_{B=256}^\mathrm{par\_rev}$"))

    _annotate(x=-1.5*bar_width, metric=default_128_metric)
    # _annotate(x=-1.5*bar_width, metric= econmt_128_metric)
    # _annotate(x=-0.5*bar_width, metric= econmt_256_metric)
    _annotate(x=-0.5*bar_width, metric=default_128_par_rev_metric)
    _annotate(x= 0.5*bar_width, metric= econmt_128_par_rev_metric)
    _annotate(x= 1.5*bar_width, metric= econmt_256_par_rev_metric)

    plt.xticks([])
    plt.yticks(fontsize=20)
    plt.ylabel(ylabel, fontsize=32)

    # plt.legend(fontsize=20, ncol=1)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, "default_vs_econmt-legend")


plt_rc_setup()

plt_default_vs_econmt_full_training_perplexity('N', False)
plt_default_vs_econmt_full_training_perplexity('T', False)
plt_default_vs_econmt_full_training_perplexity('N', True)
plt_default_vs_econmt_full_training_perplexity('T', True)
plt_default_vs_econmt_full_training_validation_bleu('N', False)
plt_default_vs_econmt_full_training_validation_bleu('T', False)
plt_default_vs_econmt_full_training_validation_bleu('N', True)
plt_default_vs_econmt_full_training_validation_bleu('T', True)

# plt_default_vs_econmt_full_training_end2end()

plt_default_vs_econmt_full_training_metrics(metric='throughput', metric_unit='samples/s', 
                                            measurer=np.average,
                                            ylabel='Avg Throughput (samples/s)')
plt_default_vs_econmt_full_training_metrics(metric='memory_usage', metric_unit='GB', 
                                             measurer=np.max, 
                                            ylabel='Max Memory Consumption (GB)',)
