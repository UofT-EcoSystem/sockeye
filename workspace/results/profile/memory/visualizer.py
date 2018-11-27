#!/usr/bin/python

from memory_profile_analysis import parse_memory_profile, \
                                    SOCKEYE_LAYER_REGEX_DICT, \
                                    SOCKEYE_FUNCTION_REGEX_DICT

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_breakdown


def plt_memory_breakdown(memory_profile, expected_sum, annotation_length_ratio):
    sorted_stats_list = parse_memory_profile(memory_profile=memory_profile, 
                                             regex_dict=SOCKEYE_LAYER_REGEX_DICT)
    plt_breakdown(sorted_stats_list=sorted_stats_list, 
                  expected_sum=expected_sum / 1e3, 
                  xlabel="Layer Type",
                  ylabel="Memory Consumption (GB)"
                  fig_name=memory_profile.replace('.log', '-layer.png'),
                  annotation_top_k=4,
                  annotation_length_ratio=annotation_length_ratio)
    sorted_stats_list = parse_memory_profile(memory_profile=memory_profile, 
                                             regex_dict=SOCKEYE_FUNCTION_REGEX_DICT)
    plt_breakdown(sorted_stats_list=sorted_stats_list, 
                  expected_sum=expected_sum / 1e3, 
                  xlabel="Data Structure",
                  ylabel="Memory Consumption (GB)",
                  fig_name=memory_profile.replace('.log', '-function.png'),
                  annotation_top_k=3,
                  annotation_length_ratio=annotation_length_ratio)


plt_rc_setup()

plt_memory_breakdown('iwslt15-vi_en-groundhog-memory_profile.log', 4477, 0.11)
plt_memory_breakdown('iwslt15-vi_en-tbd-memory_profile.log', 9077, 0.056)
