#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from memory_profile_analysis import parse_memory_profile, operator_regex_dict

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


def plt_memory_breakdown(sorted_stats_list,
                         regex_dict,
                         expected_sum,
                         fig_name,
                         top_k=7,
                         bar_width=0.3,
                         annotation_top_k=3,
                         annotation_fontsize=24,
                         annotation_length_ratio=0.0025):
    """
    Plot the breakdown of memory consumption.
    
    :param sorted_stats_list: A List that has Tuples of the form (Regex, [Memory Consumption, Labels])
    :param regex_dict       : Regular Expression Dictionary
    :param expected_sum     : Expected Total Consumption (reported by `nvidia-smi`)
    :param fig_name         : Name of the Saved Figure
    :param bar_width        : Bar Width (Default to 0.3)
    :param top_k            : The `top_k` Items to be Labeled
    :param annotation_top_k       : The `top_k` annotations that are on the Right Hand Side
    :param annotation_fontsize    : Fontsize of Annotation (Default to 24)
    :param annotation_length_ratio: Ratio between the Annotation Line and Bar Length (Default to 0.445)
    
    :return None
    """
    plt.figure(figsize=(6, 8))

    sorted_stats_klist = [regex_dict[kv[0]]        for kv in sorted_stats_list[:top_k]]
    sorted_stats_vlist = [kv[1][0] / (1024 * 1024) for kv in sorted_stats_list[:top_k]]
    sorted_stats_klist.append('Others')
    sorted_stats_vlist.append(np.sum([kv[1][0] / (1024 * 1024) for kv in sorted_stats_list[top_k:]]))
    sorted_stats_klist.append('Unrecognized')
    sorted_stats_vlist.append(expected_sum - np.sum([kv[1][0] / (1024 * 1024) for kv in sorted_stats_list[:]]))

    annotations = []

    for i in range(top_k + 2):
        plt.bar(x=0, height=sorted_stats_vlist[i], bottom=np.sum(sorted_stats_vlist[i+1:]),
                width=bar_width, edgecolor='black', linewidth=3,
                color=np.array([i * 1.0 / (top_k + 1), i * 1.0 / (top_k + 1), i * 1.0 / (top_k + 1)]),
                hatch='//' if sorted_stats_klist[i] is 'Unrecognized' else '',
                label=sorted_stats_klist[i])
        middle_pos = sorted_stats_vlist[i] / 2 + np.sum(sorted_stats_vlist[i+1:])
        bar_length = sorted_stats_vlist[i]

        switch_side_flag = True if i < annotation_top_k or i % 2 == 0 else False

        annotations.append(
            plt.annotate((sorted_stats_klist[i] + ' (%.2f%%)') % (sorted_stats_vlist[i] * 100.0 / expected_sum),
                         xy    =(0.6*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         xytext=(0.9*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         fontsize=18,
                         ha='left' if switch_side_flag is True else 'right', 
                         va='center', 
                         bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                         arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % 
                            (annotation_length_ratio * bar_length), linewidth=2)))
    # x- & y- axis
    plt.xlim([-3*bar_width, 3*bar_width])
    plt.xticks([])
    plt.ylabel("Memory Consumption (MB)")

    # Grid & Legend
    plt.grid(linestyle='-.', linewidth=1, axis='y')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Tighten Layout and Savefig
    # plt.tight_layout()
    plt.savefig(fig_name, bbox_extra_artists=tuple(annotations), bbox_inches='tight')


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    args = parser.parse_args()

    sorted_stats_list = parse_memory_profile(memory_profile=args.memory_profile, 
                                                 regex_dict=operator_regex_dict)

    plt_memory_breakdown(sorted_stats_list=sorted_stats_list, regex_dict=operator_regex_dict,
                         expected_sum=4477, fig_name='sockeye-memory_profile-groundhog_iwslt15.png')