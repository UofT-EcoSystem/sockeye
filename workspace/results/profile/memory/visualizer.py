#!/usr/bin/python

from memory_profile_analysis import parse_memory_profile, \
                                    SOCKEYE_LAYER_REGEX_DICT, \
                                    SOCKEYE_FUNCTION_REGEX_DICT

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_breakdown


def plt_memory_breakdown(memory_profile, expected_sum, yticks=None):
    sorted_stats_list = parse_memory_profile(memory_profile=memory_profile, 
                                             regex_dict=SOCKEYE_LAYER_REGEX_DICT)
    plt_breakdown(sorted_stats_list=sorted_stats_list, 
                  expected_sum=expected_sum / 1e3, 
                  xlabel="Layer Type",
                  ylabel="Memory Consumption (GB)", yticks=yticks, 
                  colors=[np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5]), np.array([1, 0, 0]),
                          np.array([1, 1, 1]), np.array([1, 1, 1])] if 'partial_fw_prop' in memory_profile else None,
                  fig_name=memory_profile.replace('.log', '-layer.png'),
                  annotation_top_k=4)
    sorted_stats_list = parse_memory_profile(memory_profile=memory_profile, 
                                             regex_dict=SOCKEYE_FUNCTION_REGEX_DICT)
    plt_breakdown(sorted_stats_list=sorted_stats_list, 
                  expected_sum=expected_sum / 1e3, 
                  xlabel="Data Structure",
                  ylabel="Memory Consumption (GB)", yticks=yticks, 
                  fig_name=memory_profile.replace('.log', '-function.png'),
                  annotation_top_k=3)


plt_rc_setup()

plt_memory_breakdown(memory_profile='iwslt15-vi_en-groundhog-memory_profile.log', 
                     expected_sum=4477)
plt_memory_breakdown(memory_profile='iwslt15-vi_en-tbd-memory_profile.log', 
                     expected_sum=9077, yticks=np.arange(0, 11, 2))


def _plt_default_vs_econmt(sorted_stats_list_default, sorted_stats_list_econmt, title, xlabel, 
                           annotation_top_k=None, colors=None):

    plt.figure(figsize=(8, 6))

    def _plt_breakdown(x, sorted_stats_list,
                       expected_sum,
                       xlabel, ylabel,
                       ymax=None,
                       yticks=np.arange(0, 11, 2),
                       colors=None,
                       bar_width=0.3,
                       annotation_top_k=None,
                       annotation_fontsize=18):

        sorted_stats_klist = [kv[0] for kv in sorted_stats_list[:]]

        sorted_stats_vlist = [kv[1][0] / 1e9 for kv in sorted_stats_list[:]]
        sorted_stats_klist.append('Untrackable')
        sorted_stats_vlist.append(expected_sum - np.sum([kv[1][0] / 1e9 for kv in sorted_stats_list[:]]))

        assert len(sorted_stats_klist) == len(sorted_stats_vlist)

        sorted_stats_list_len = len(sorted_stats_klist)

        if ymax is not None:
            plt.ylim(ymax=ymax)

        for i in range(sorted_stats_list_len):
            plt.bar(x=x, height=sorted_stats_vlist[i], bottom=np.sum(sorted_stats_vlist[i+1:]),
                    width=0.8*bar_width, edgecolor='black', linewidth=3,
                    # color=np.array([1, i * 1.0 / 3, i * 1.0 / 3]) if i < 3 else 'white',
                    color=colors[i] if colors is not None else np.array([1, 0, 0]) if i == 0 else \
                            'white' if 'Other'       in sorted_stats_klist[i] or \
                                        'Untrackable' in sorted_stats_klist[i] else \
                            np.array([0, 0, 0]) if sorted_stats_list_len <= 3 else \
                            np.array([(i-1) * 1.0 / (sorted_stats_list_len - 3), 
                                    (i-1) * 1.0 / (sorted_stats_list_len - 3), 
                                    (i-1) * 1.0 / (sorted_stats_list_len - 3)]),
                    hatch='//' if sorted_stats_klist[i] is 'Untrackable' else '',
                    label=sorted_stats_klist[i] if x == 0 else None)
                    # label=(sorted_stats_klist[i] + ' (%.2f%%)') % (sorted_stats_vlist[i] * 100.0 / expected_sum))
            if annotation_top_k is None or i < annotation_top_k:
                middle_pos = sorted_stats_vlist[i] / 2 + np.sum(sorted_stats_vlist[i+1:])
                bar_length = sorted_stats_vlist[i]
                switch_side_flag = False if i >= 2 and i % 2 == 0 else True
                
                plt.annotate(('%.0f%%') % (sorted_stats_vlist[i] * 100.0 / (expected_sum)),
                                xy    =(x+0.45*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                                xytext=(x+0.55*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                                fontsize=annotation_fontsize,
                                ha='left' if switch_side_flag is True else 'right', 
                                va='center', 
                                bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                                arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % 
                                (0.52 / (plt.ylim()[1] if yticks is None else yticks[-1]) * annotation_fontsize * bar_length), linewidth=2))
        # x- & y- axis
        plt.xlim([-1.3*bar_width, x + 1.3*bar_width])
        plt.xticks([])
        if yticks is not None:
            plt.yticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Grid & Legend
        plt.grid(linestyle='-.', linewidth=1, axis='y')
        plt.legend(loc=0, fontsize=24)


    _plt_breakdown(x=0, sorted_stats_list=sorted_stats_list_default, expected_sum=9.077,
                   xlabel=xlabel,
                   ylabel="Memory Consumption (GB)",
                   annotation_top_k=annotation_top_k)
    _plt_breakdown(x=0.6, sorted_stats_list=sorted_stats_list_econmt , expected_sum=4.197,
                   xlabel=xlabel,
                   ylabel="Memory Consumption (GB)", 
                   annotation_top_k=annotation_top_k,
                   colors=colors)

    plt.tight_layout()
    plt.savefig(title + ".png")


sorted_stats_list_default = parse_memory_profile(memory_profile='iwslt15-vi_en-tbd-memory_profile.log', 
                                                 regex_dict=SOCKEYE_LAYER_REGEX_DICT)
sorted_stats_list_econmt  = parse_memory_profile(memory_profile='iwslt15-vi_en-tbd-partial_fw_prop-memory_profile.log', 
                                                 regex_dict=SOCKEYE_LAYER_REGEX_DICT)
_plt_default_vs_econmt(sorted_stats_list_default, sorted_stats_list_econmt, 
                       "iwslt15-vi_en-tbd-memory_profile-default_vs_econmt-layer", 'Layer Type',
                       colors=[np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5]), np.array([1, 0, 0]),
                               np.array([1, 1, 1]), np.array([1, 1, 1])], annotation_top_k=4)

sorted_stats_list_default = parse_memory_profile(memory_profile='iwslt15-vi_en-tbd-memory_profile.log', 
                                                 regex_dict=SOCKEYE_FUNCTION_REGEX_DICT)
sorted_stats_list_econmt  = parse_memory_profile(memory_profile='iwslt15-vi_en-tbd-partial_fw_prop-memory_profile.log', 
                                                 regex_dict=SOCKEYE_FUNCTION_REGEX_DICT)
_plt_default_vs_econmt(sorted_stats_list_default, sorted_stats_list_econmt, 
                       "iwslt15-vi_en-tbd-memory_profile-default_vs_econmt-function", 'Data Structure', 
                       annotation_top_k=3)