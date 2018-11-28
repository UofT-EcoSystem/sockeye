#!/usr/bin/python

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_breakdown


plt_rc_setup()

sorted_stats_list = [
    ("Sequence\nReverse", [62.587]),
    ("Fully-\nConnected"    , [17.417 + 12.912 + 8.2582 + \
               5.3654 + 4.2895 + 2.4036 + 1.4046])
]

sorted_stats_list.append(("Others", [62.587 / 0.4119 - sorted_stats_list[0][1][0] - \
                                                       sorted_stats_list[1][1][0]]))

plt_breakdown(sorted_stats_list=sorted_stats_list,
              expected_sum=62.587 / 0.4119,
              xlabel='GPU Kernel', ylabel='Runtime (ms)',
              fig_name='iwslt15-vi_en-groundhog-runtime_profile-gpu_kernel', ymax=200)

sorted_stats_list = [
    (r"$\mathtt{cudaSync}$"  , [119.15]),
    (r"$\mathtt{cudaLaunch}$", [1558.90 - 1505.42])
]

sorted_stats_list.append(("Others", [1558.90 / 0.4867 - 1505.42 - 1504.37 - 
                                                        sorted_stats_list[0][1][0] - \
                                                        sorted_stats_list[1][1][0]]))

plt_breakdown(sorted_stats_list=sorted_stats_list,
              expected_sum=62.587 / 0.4119,
              xlabel='CUDA API', ylabel='Runtime (ms)',
              fig_name='iwslt15-vi_en-groundhog-runtime_profile-cuda_api', ymax=200)