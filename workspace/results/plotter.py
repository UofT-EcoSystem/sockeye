#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from memory_profile_analysis import parse_memory_profile

parser = argparse.ArgumentParser()

parser.add_argument('--memory_profile', help='Path to Memory Profile', type=str, default=None)


def plt_legend(handles, title, ncol=1):
    """
    Plot the legend in a separate figure.
    
    :param handles: Legend Handles
    :param title  : Figure Title
    :param ncol   : Number of Columns (Default to 1)
    
    :return: None
    """
    lgd_fig = plt.figure()
    plt.axis('off')
    lgd = plt.legend(handles=handles,
                     loc='center', ncol=ncol)
    lgd_fig.canvas.draw()
    plt.savefig(title + ".png",
                bbox_inches=lgd.get_window_extent().transformed(lgd_fig.dpi_scale_trans.inverted()))


def plt_rc_setup(dpi=400, fontsize=24):
    """
    Setup the RC parameters of Pyplot.

    :param dpi     : Figure Resolution (Default to 400)
    :param fontsize: Font Size (Default to 24)
    
    :return: None
    """
    plt.rc('figure', dpi=dpi)
    plt.rc('axes', axisbelow=True)
    plt.rc('mathtext', fontset='cm')
    plt.rc('mathtext', rm='Times New Roman')
    plt.rc('font', family='Times New Roman', size=fontsize)


def plt_breakdown(memory_profile, bar_width=0.3,
                  annotation_fontsize=24,
                  annotation_length_ratio=0.8):
    """
    Plot the breakdown of memory consumption.
    
    :param memory_profile: A Dictionary that maps Labels to Memory Consumptions
    :param bar_width     : Bar Width (Default to 0.3)
    :param annotation_fontsize    : Fontsize of Annotation (Default to 24)
    :param annotation_length_ratio: Ratio between the Annotation Line and Bar Length (Default to 0.445)
    
    :return None
    """

    for i, (label, memory_consumption) in enumerate(memory_profile.items()):
        plt.bar(x=0, height=memory_consumption, bottom=np.sum(memory_profile.values()[i+1:]),
                width=bar_width, edgecolor='black', linewidth=3,
                color=np.array([1, 1, 1]) * i * 1.0 / (len(memory_profile.values()) - 1),
                label=label)
        middle_pos = memory_consumption / 2 + np.sum(memory_profile.values()[i+1:])
        bar_length = memory_consumption
        plt.annotate(label,
                     xy    =(0.7*bar_width, middle_pos), 
                     xytext=(1.1*bar_width, middle_pos), 
                     fontsize=24,
                     ha='left', 
                     va='center', 
                     bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                     arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % \
                        (annotation_length_ratio * bar_length), linewidth=2))

    plt.xlim([-2*bar_width, 2*bar_width])
    plt.xticks([]) # remove the ticklabels on the x-axis
    plt.ylabel("Memory Consumption (MB)")
    plt.grid(linestyle='-.', linewidth=1, axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


if __name__ == "__main__":
    # setup the RC parameters
    # plt_rc_setup()

    # plt_breakdown({'1' : 2, '3' : 4})
    # plt.tight_layout()
    # plt.savefig("sockeye-memory_profile-groundhog_iwslt15.png")
    args = parser.parse_args()

    parse_memory_profile(args.memory_profile)
