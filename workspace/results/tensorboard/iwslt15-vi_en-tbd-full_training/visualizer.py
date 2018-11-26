#!/usr/bin/python

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")

from visualizer_helper import plt_rc_setup, gen_from_txt


plt_rc_setup()

# ==================================================================================================


def plt_default_vs_econmt_full_training_validation_bleu(xscale):

    metric, metric_unit = 'validation_bleu', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    ylabel = metric.title().replace('_', ' ')
    title ='default_vs_econmt-%s-%s' % (xscale, metric)

    default_128_metric = gen_from_txt("default-B_128/csv/%s.csv" % metric, metric, metric_unit)
    # econmt_128_metric  = gen_from_txt( "econmt-B_128/csv/%s.csv" % metric, metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "econmt-B_256/csv/%s.csv" % metric, metric, metric_unit)

    default_128_par_rev_metric = gen_from_txt("default-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit)
    econmt_128_par_rev_metric  = gen_from_txt( "econmt-B_128-par_rev/csv/%s.csv" % metric, metric, metric_unit)
    econmt_256_par_rev_metric  = gen_from_txt( "econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    plt.plot(default_128_par_rev_metric[:9,1] if xscale == 'N' else default_128_par_rev_metric[:9,0], 
             default_128_par_rev_metric[:9,2], linewidth=2, linestyle='-', 
             marker='x', markersize=10,
             color='black', label=r'Default$_{B=128}^{\mathrm{par\_rev}}$')
    plt.plot(econmt_128_par_rev_metric [:7,1] if xscale == 'N' else econmt_128_par_rev_metric [:7,0], 
             econmt_128_par_rev_metric [:7,2], linewidth=2, linestyle='-', 
             marker='.', markersize=10,
             color='black', label= r'EcoNMT$_{B=128}^{\mathrm{par\_rev}}$')
    plt.plot(econmt_256_par_rev_metric [:6,1] if xscale == 'N' else econmt_256_par_rev_metric [:6,0], 
             econmt_256_par_rev_metric [:6,2], linewidth=2, linestyle='-', 
             marker='+', markersize=10,
             color='black', label= r'EcoNMT$_{B=256}^{\mathrm{par\_rev}}$')

    plt.axhline(y=22.6, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Training Checkpoint Number' if xscale == 'N' else r'Time (min)')
    plt.ylabel("Validation BLEU Score")
    if xscale == 'N':
        plt.xticks(np.arange(0, 9, 2), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")

plt_default_vs_econmt_full_training_validation_bleu('N')
plt_default_vs_econmt_full_training_validation_bleu('T')
