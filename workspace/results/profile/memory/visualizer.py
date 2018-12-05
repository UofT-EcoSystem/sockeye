#!/usr/bin/python

from memory_profile_analysis import parse_memory_profile, \
                                    SOCKEYE_LAYER_REGEX_DICT, \
                                    SOCKEYE_FUNCTION_REGEX_DICT

import os, sys
import numpy as np

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
plt_memory_breakdown(memory_profile='iwslt15-vi_en-tbd-partial_fw_prop-memory_profile.log', 
                     expected_sum=4271, yticks=np.arange(0, 11, 2))
