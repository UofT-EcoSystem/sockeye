#!/usr/bin/python

import os, sys

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup, plt_breakdown


plt_rc_setup()

gpu_kernel_sum, cuda_api_sum = 119.647957 / 0.69479557, 1624.542834 / 0.49061146 - 1534.320709 - 1532.120223

print("GPU Kernel: %f, CUDA API: %f, Total: %f" % (gpu_kernel_sum, 
                                                   cuda_api_sum, 
                                                   gpu_kernel_sum + cuda_api_sum))

sorted_stats_list = [
    ("Sequence\nReverse", [119.647957]),
    ("Fully-\nConnected", [8.194789 + 6.616071 + 5.485084 + \
                           2.129842 + 2.090799 + 1.734601 + 
                           1.697543 + 1.692487 + 0.487400 + 
                           0.185024 + 0.098914])
]

sorted_stats_list.append(("Others", [gpu_kernel_sum - sorted_stats_list[0][1][0] - \
                                                      sorted_stats_list[1][1][0]]))

plt_breakdown(sorted_stats_list=sorted_stats_list,
              expected_sum=gpu_kernel_sum, extra_sum=cuda_api_sum,
              xlabel='GPU Kernel', ylabel='Runtime (ms)', 
              fig_name='iwslt15-vi_en-groundhog-runtime_profile-gpu_kernel', ymax=250)

sorted_stats_list = [
    (r"$\mathtt{cudaSync}$"  , [143.812516]),
    (r"$\mathtt{cudaLaunch}$", [1624.542834 - 1534.320709])
]

sorted_stats_list.append(("Others", [cuda_api_sum - sorted_stats_list[0][1][0] - \
                                                    sorted_stats_list[1][1][0]]))

plt_breakdown(sorted_stats_list=sorted_stats_list,
              expected_sum=cuda_api_sum, extra_sum=gpu_kernel_sum,
              xlabel='CUDA API', ylabel='Runtime (ms)', 
              fig_name='iwslt15-vi_en-groundhog-runtime_profile-cuda_api', ymax=250)