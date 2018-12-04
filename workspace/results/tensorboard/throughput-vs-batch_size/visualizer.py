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

from visualizer_helper import plt_rc_setup
from tensorboard_visualizer_helper import gen_from_txt


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
                                  color='black', marker='X', markersize=5, label="Memory\nConsumption")
    
    axes.set_ylabel("Memory Consumption (GB)")
    axes.set_yticks(np.arange(0, 11, 2))

    legends = throughput_plot + memory_usage_plot
    axes.legend(legends, [legend.get_label() for legend in legends], fontsize=20, loc=4)

    for ticklabel in axes.get_xticklabels() + axes.get_yticklabels():
        ticklabel.set_fontsize(20)

    plt.tight_layout()
    plt.savefig("throughput_vs_batch_size-sockeye.png")


plt_rc_setup()

plt_throughput_vs_batch_size()
