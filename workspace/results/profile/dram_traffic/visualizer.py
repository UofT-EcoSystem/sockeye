#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


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


def parse_dram_traffic_profile(fname):
    import csv
    
    dram_read_trans, dram_write_trans = 0, 0

    with open(fname, 'r') as csv_file:
        csv_fin = csv.reader(csv_file)
        
        for row in csv_fin:
            if row[0][0] == "=" or row[0] == "Device":
                continue
            # The parsed `row` is a list of metrics that consist of the following information.
            # "Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
            assert len(row) == 8, "Length of `row` must be 8."
            
            if row[3] == "dram_read_transactions":
                dram_read_trans += int(row[2]) * int(row[7])
            if row[3] == "dram_write_transactions":
                dram_write_trans += int(row[2]) * int(row[7])
    
    return int(dram_read_trans / 1e6), int(dram_write_trans / 1e6)
        

if __name__ == "__main__":
    dram_read_legacy , dram_write_legacy  = parse_dram_traffic_profile("iwslt15-vi_en-groundhog-legacy.csv")
    dram_read_partial, dram_write_partial = parse_dram_traffic_profile("iwslt15-vi_en-groundhog-partial_fw_prop.csv")
    
    plt_rc_setup()

    bar_width = 0.3

    plt.figure()

    plt.bar(x=0, height=dram_read_legacy, bottom=dram_write_legacy,
            width=bar_width, edgecolor='black', linewidth=3, 
            color='white', label='DRAM Read')
    plt.bar(x=0, height=dram_write_legacy, bottom=0,
            width=bar_width, edgecolor='black', linewidth=3, 
            color='black', label='DRAM Write')
    plt.bar(x=1, height=dram_write_partial, bottom=0,
            width=bar_width, edgecolor='black', linewidth=3, 
            color='black')
    plt.bar(x=1, height=dram_read_partial, bottom=dram_write_partial,
            width=bar_width, edgecolor='black', linewidth=3, 
            color='white')
    
    xticklabels = ['Default', 'EcoNMT']

    plt.xlim  ([-2*bar_width, 1+2*bar_width])
    plt.xticks(range(len(xticklabels)), xticklabels)
    plt.yticks(np.arange(0, 900, 225), fontsize=20)

    plt.ylabel(r"DRAM Transactions ($10^6$)")

    plt.legend()
    plt.grid(linestyle='-.', linewidth=1, axis='y')

    plt.tight_layout()
    plt.savefig("DRAM_Traffic_Comparison-Default_vs_EcoNMT.png")

