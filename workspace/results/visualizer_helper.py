import numpy as np
import matplotlib.pyplot as plt

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

def plt_breakdown(sorted_stats_list,
                  expected_sum,
                  xlabel, ylabel,
                  fig_name,
                  ymax=None,
                  bar_width=0.3,
                  annotation_top_k=None,
                  annotation_fontsize=18):
    plt.figure(figsize=(8, 6))

    sorted_stats_klist = [kv[0]          for kv in sorted_stats_list[:]]

    if 'memory_profile' in fig_name:
        sorted_stats_vlist = [kv[1][0] / 1e9 for kv in sorted_stats_list[:]]
        sorted_stats_klist.append('Untrackable')
        sorted_stats_vlist.append(expected_sum - np.sum([kv[1][0] / 1e9 for kv in sorted_stats_list[:]]))
    else:
        sorted_stats_vlist = [kv[1][0] for kv in sorted_stats_list[:]]

    assert len(sorted_stats_klist) == len(sorted_stats_vlist)

    # annotations = []

    sorted_stats_list_len = len(sorted_stats_klist)

    if ymax is not None:
        plt.ylim(ymax=ymax)

    for i in range(sorted_stats_list_len):
        plt.bar(x=0, height=sorted_stats_vlist[i], bottom=np.sum(sorted_stats_vlist[i+1:]),
                width=bar_width * 0.8, edgecolor='black', linewidth=3,
                # color=np.array([1, i * 1.0 / 3, i * 1.0 / 3]) if i < 3 else 'white',
                color=np.array([1, 0, 0]) if i == 0 else \
                      'white' if 'Other'       in sorted_stats_klist[i] or \
                                 'Untrackable' in sorted_stats_klist[i] else \
                      np.array([0, 0, 0]) if sorted_stats_list_len <= 3 else \
                      np.array([(i-1) * 1.0 / (sorted_stats_list_len - 3), 
                                (i-1) * 1.0 / (sorted_stats_list_len - 3), 
                                (i-1) * 1.0 / (sorted_stats_list_len - 3)]),
                hatch='//' if sorted_stats_klist[i] is 'Untrackable' else '',
                label=sorted_stats_klist[i])
                # label=(sorted_stats_klist[i] + ' (%.2f%%)') % (sorted_stats_vlist[i] * 100.0 / expected_sum))
        if annotation_top_k is None or i < annotation_top_k:
            middle_pos = sorted_stats_vlist[i] / 2 + np.sum(sorted_stats_vlist[i+1:])
            bar_length = sorted_stats_vlist[i]
            switch_side_flag = False if i >= 2 and i % 2 == 0 else True
            
            plt.annotate(('%.0f%%') % (sorted_stats_vlist[i] * 100.0 / expected_sum),
                         xy    =(0.45*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         xytext=(0.55*bar_width * (1 if switch_side_flag is True else -1), middle_pos), 
                         fontsize=annotation_fontsize,
                         ha='left' if switch_side_flag is True else 'right', 
                         va='center', 
                         bbox=dict(boxstyle='square', facecolor='white', linewidth=3),
                         arrowprops=dict(arrowstyle="-[, widthB=%f, lengthB=0.3" % 
                            (0.52 / plt.ylim()[1] * annotation_fontsize * bar_length), linewidth=2))
    # x- & y- axis
    plt.xlim([-2*bar_width, 2*bar_width])
    plt.xticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Grid & Legend
    plt.grid(linestyle='-.', linewidth=1, axis='y')
    plt.legend(loc=2, fontsize=18)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=annotation_fontsize)
    
    # Tighten Layout and Savefig
    plt.tight_layout()
    plt.savefig(fig_name)
    # plt.savefig(fig_name, bbox_extra_artists=tuple(annotations), bbox_inches='tight')