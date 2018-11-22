#!/usr/bin/python
"""
    Author: Bojian Zheng (ArmageddonKnight@github)
    Description: This file plots the memory profile of the Sockeye NMT Toolkit.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from memory_profile_analysis import parse_memory_profile, \
                                    SOCKEYE_LAYER_REGEX_DICT, \
                                    SOCKEYE_FUNCTION_REGEX_DICT

parser = argparse.ArgumentParser()

parser.add_argument('--memory-profile', help='Path to Memory Profile', type=str, default=None)


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
    """
    plt.rc('figure', dpi=dpi)
    plt.rc('axes', axisbelow=True)
    plt.rc('mathtext', fontset='cm')
    plt.rc('mathtext', rm='Times New Roman')
    plt.rc('font', family='Times New Roman', size=fontsize)


def plt_memory_breakdown(sorted_stats_list,
                         expected_sum,
                         xlabel,
                         fig_name,
                         bar_width=0.3,
                         annotation_top_k=None,
                         annotation_fontsize=18,
                         annotation_length_ratio=0.11):
    """
    Plot the breakdown of memory consumption.
    
    :param sorted_stats_list: A List that has Tuples of the form (Regex, [Memory Consumption, Labels])
    :param expected_sum     : Expected Total Consumption (reported by `nvidia-smi`)
    :param fig_name         : Name of the Saved Figure
    :param bar_width        : Bar Width (Default to 0.3)
    :param annotation_top_k       : The `top_k` Annotations that are to be plotted (Default to 3)
    :param annotation_fontsize    : Fontsize of Annotations (Default to 24)
    :param annotation_length_ratio: Ratio between the Annotation Line and Bar Length (Default to 0.445)
    
    :return None
    """
    plt.figure(figsize=(8, 6))

    sorted_stats_klist = [kv[0]                           for kv in sorted_stats_list[:]]
    sorted_stats_vlist = [kv[1][0] / (1024 * 1024 * 1024) for kv in sorted_stats_list[:]]
    
    if expected_sum is not None:
        sorted_stats_klist.append('Untrackable')
        sorted_stats_vlist.append(expected_sum - np.sum([kv[1][0] / (1024 * 1024 * 1024) for kv in sorted_stats_list[:]]))

    assert len(sorted_stats_klist) == len(sorted_stats_vlist)

    # annotations = []

    sorted_stats_list_len = len(sorted_stats_klist)

    for i in range(sorted_stats_list_len):
        plt.bar(x=0, height=sorted_stats_vlist[i], bottom=np.sum(sorted_stats_vlist[i+1:]),
                width=bar_width * 0.8, edgecolor='black', linewidth=3,
                # color=np.array([1, i * 1.0 / 3, i * 1.0 / 3]) if i < 3 else 'white',
                color=np.array([i * 1.0 / (sorted_stats_list_len - 2), 
                                i * 1.0 / (sorted_stats_list_len - 2), 
                                i * 1.0 / (sorted_stats_list_len - 2)]) \
                                    if 'Other'       not in sorted_stats_klist[i] and \
                                       'Untrackable' not in sorted_stats_klist[i] else 'white',
                hatch='//' if sorted_stats_klist[i] is 'Untrackable' else '',
                label=sorted_stats_klist[i])
                # label=(sorted_stats_klist[i] + ' (%.2f%%)') % (sorted_stats_vlist[i] * 100.0 / expected_sum))
        if annotation_top_k is None or i < annotation_top_k:
            middle_pos = sorted_stats_vlist[i] / 2 + np.sum(sorted_stats_vlist[i+1:])
            bar_length = sorted_stats_vlist[i]
            switch_side_flag = False if i >= 2 and i % 2 == 0 else True
            
            plt.annotate(('%2.0f%%') % (sorted_stats_vlist[i] * 100.0 / expected_sum),
                         xy    =(0.5*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         xytext=(0.7*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         fontsize=annotation_fontsize,
                         ha='left' if switch_side_flag is True else 'right', 
                         va='center', 
                         bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                         arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % 
                            (annotation_length_ratio * annotation_fontsize * bar_length), linewidth=2))
        """
        middle_pos = sorted_stats_vlist[i] / 2 + np.sum(sorted_stats_vlist[i+1:])
        bar_length = sorted_stats_vlist[i]
        switch_side_flag = True if i < annotation_top_k or i % 2 == 0 else False
        
        annotations.append(
            plt.annotate((sorted_stats_klist[i] + ' (%.2f%%)') % (sorted_stats_vlist[i] * 100.0 / expected_sum),
                         xy    =(0.6*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         xytext=(0.8*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         fontsize=20,
                         ha='left' if switch_side_flag is True else 'right', 
                         va='center', 
                         bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                         arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % 
                            (annotation_length_ratio * bar_length), linewidth=2)))
        """
    # x- & y- axis
    plt.xlim([-2*bar_width, 2*bar_width])
    plt.xticks([])
    plt.xlabel(xlabel)
    plt.ylabel("Memory Consumption (GB)")

    # Grid & Legend
    plt.grid(linestyle='-.', linewidth=1, axis='y')
    plt.legend(loc=2, fontsize=18)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=annotation_fontsize)
    
    # Tighten Layout and Savefig
    plt.tight_layout()
    plt.savefig(fig_name)
    # plt.savefig(fig_name, bbox_extra_artists=tuple(annotations), bbox_inches='tight')


if __name__ == "__main__":
    # setup the RC parameters
    plt_rc_setup()

    args = parser.parse_args()

    sorted_stats_list = parse_memory_profile(memory_profile=args.memory_profile, 
                                             regex_dict=SOCKEYE_LAYER_REGEX_DICT)
    plt_memory_breakdown(sorted_stats_list=sorted_stats_list, 
                         expected_sum=4477.0 / 1024, 
                         xlabel="Layer Type",
                         fig_name='iwslt15-vi_en-groundhog-memory_profile-layer.png')
    sorted_stats_list = parse_memory_profile(memory_profile=args.memory_profile, 
                                             regex_dict=SOCKEYE_FUNCTION_REGEX_DICT)
    plt_memory_breakdown(sorted_stats_list=sorted_stats_list, 
                         expected_sum=4477.0 / 1024, 
                         xlabel="Data Structure",
                         fig_name='iwslt15-vi_en-groundhog-memory_profile-function.png')