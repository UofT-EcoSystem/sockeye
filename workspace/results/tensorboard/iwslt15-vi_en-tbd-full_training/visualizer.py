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
from tensorboard_visualizer_helper import gen_from_txt, plt_default_vs_econmt_full_training_perplexity, \
    plt_default_vs_econmt_full_training_validation_bleu, \
    plt_default_vs_econmt_full_training_metrics, \
    plt_cudnn_vs_econmt_full_training_validation_bleu, \
    plt_cudnn_vs_econmt_full_training_metrics

plt_rc_setup()

plt_default_vs_econmt_full_training_perplexity('N', prefix='iwslt15-vi_en-tbd-')
plt_default_vs_econmt_full_training_validation_bleu(first_k_ckpts=(7, 7, 7, 6),
                                                    bar=22.6, prefix='iwslt15-vi_en-tbd-')
plt_cudnn_vs_econmt_full_training_validation_bleu  (first_k_ckpts=(7, 10, 6),
                                                    cross_bar=(2, 2, 2),
                                                    bar=22.6, prefix='iwslt15-vi_en-tbd-')

plt_default_vs_econmt_full_training_metrics(metric='throughput', metric_unit='samples/s',
                                            measurer=np.average,
                                            prefix='iwslt15-vi_en-tbd-',
                                            ylabel='Throughput (samples/s)')
plt_default_vs_econmt_full_training_metrics(metric='memory_usage', metric_unit='GB',
                                            measurer=np.max,
                                            prefix='iwslt15-vi_en-tbd-',
                                            ylabel='Memory Consumption (GB)',)
plt_cudnn_vs_econmt_full_training_metrics(metric='memory_usage', metric_unit='GB',
                                          measurer=np.max,
                                          prefix='iwslt15-vi_en-tbd-',
                                          ylabel='Memory Consumption (GB)',)

def plt_hparam_sweep_rnn_layers():

    # Number of LSTM RNN Layers
    title ='iwslt15-vi_en-tbd-default_vs_econmt-par_rev-hparam-rnn_layers'

    plt.figure()

    plt.plot([1, 2, 3], [7.931, 9.297, 10.715],
             linewidth=2, linestyle='-', 
             marker='X', markersize=10, 
             color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')
    plt.plot([3, 4], [10.715, 12.378],
             linewidth=2, linestyle='--', 
             marker='X', markersize=10, 
             color='black')
    plt.plot([1, 2, 3, 4], [3.077, 4.343, 5.779, 7.213], 
             linewidth=2, linestyle='-', 
             marker='^', markersize=10, 
             color='black', label= r'EcoRNN$_{B=128}^{\mathrm{par\_rev}}$')

    plt.xlabel('Number of RNN Layers')
    plt.ylabel("Memory Consumption (GB)")
    plt.xticks([1, 2, 3, 4], fontsize=20)
    plt.yticks(np.arange(0, 13, 4), fontsize=20)
    plt.xlim(xmin=0, xmax=5)
    plt.ylim(ymin=0, ymax=13)

    # plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_hparam_sweep_hidden_dimension():

    # Number of LSTM RNN Layers
    title ='iwslt15-vi_en-tbd-default_vs_econmt-par_rev-hparam-hidden_dimension'

    plt.figure()

    handles = []

    handles.append(plt.plot([256, 512], [5.357, 9.297],
             linewidth=2, linestyle='-', 
             marker='X', markersize=10, 
             color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')[0])
    plt.plot([512, 1024], [9.297, 18.326],
             linewidth=2, linestyle='--', 
             marker='X', markersize=10, 
             color='black')
    handles.append(plt.plot([256, 512, 1024], [2.889, 4.343, 7.657],
             linewidth=2, linestyle='-', 
             marker='^', markersize=10, 
             color='black', label= r'EcoRNN$_{B=128}^{\mathrm{par\_rev}}$')[0])

    plt.xlabel('Hidden Dimension')
    plt.ylabel("Memory Consumption (GB)")
    plt.xticks([256, 512, 1024], fontsize=20)
    plt.yticks(np.arange(0, 19, 4), fontsize=20)
    plt.xlim(xmin=0, xmax=1280)
    plt.ylim(ymin=0, ymax=19)

    # plt.legend(fontsize=20, loc=2)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt_legend(handles, 'legend-default_vs_econmt-hparam_plot-horizontal', len(handles))


plt_hparam_sweep_rnn_layers()
plt_hparam_sweep_hidden_dimension()