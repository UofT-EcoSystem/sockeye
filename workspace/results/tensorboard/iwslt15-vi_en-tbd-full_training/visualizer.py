#!/usr/bin/python

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/..")
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)) + "/../..")

from visualizer_helper import plt_rc_setup
from tensorboard_visualizer_helper import gen_from_txt


plt_rc_setup()

# ==================================================================================================


def plt_default_vs_econmt_full_training_validation_bleu(xscale, par_rev):

    metric, metric_unit = 'validation_bleu', None

    if xscale != 'N' and xscale != 'T':
        assert False, "Invalid xlabel %s. It must be either \'N\' or \'T\'."
    
    title ='default_vs_econmt%s-%s-%s' % ('-par_rev' if par_rev else '', xscale, metric)

    default_128_metric = gen_from_txt("default-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_128_metric  = gen_from_txt( "econmt-B_128%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)
    econmt_256_metric  = gen_from_txt( "econmt-B_256%s/csv/%s.csv" % ('-par_rev' if par_rev else '', metric), metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    first_k_ckpts = 7 if par_rev else 8

    plt.plot(default_128_metric[:first_k_ckpts,1] if xscale == 'N' else default_128_metric[:first_k_ckpts,0], 
             default_128_metric[:first_k_ckpts,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10,
             color='black', label=r'Default$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_128_metric [:first_k_ckpts,1] if xscale == 'N' else econmt_128_metric [:first_k_ckpts,0], 
             econmt_128_metric [:first_k_ckpts,2], linewidth=2, linestyle='--', 
             marker='.', markersize=10,
             color='black', label= r'EcoNMT$_{B=128}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))
    plt.plot(econmt_256_metric [:6,1] if xscale == 'N' else econmt_256_metric [:6,0], 
             econmt_256_metric [:6,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10,
             color='black', label= r'EcoNMT$_{B=256}%s$' % ('^{\mathrm{par\_rev}}' if par_rev else ''))

    plt.axhline(y=22.6, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Training Checkpoint Number' if xscale == 'N' else 'Time (min)')
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

def plt_default_vs_econmt_full_training_end2end():
    metric, metric_unit = 'validation_bleu', None
    
    title ='default_vs_econmt-end2end-%s' % metric

    default_128_metric = gen_from_txt(       "default-B_128/csv/%s.csv" % metric, metric, metric_unit)
    econmt_256_metric  = gen_from_txt("econmt-B_256-par_rev/csv/%s.csv" % metric, metric, metric_unit)

    # ==============================================================================================

    plt.figure()

    plt.plot(default_128_metric[:7,0],
             default_128_metric[:7,2], linewidth=2, linestyle='-', 
             marker='X', markersize=10,
             color='black', label=r'Default$_{B=128}$')
    plt.plot(econmt_256_metric [:6,0], 
             econmt_256_metric [:6,2], linewidth=2, linestyle='-', 
             marker='^', markersize=10,
             color='black', label= r'EcoNMT$_{B=256}^{\mathrm{par\_rev}}$')

    plt.axhline(y=22.6, color='r', linewidth=2, linestyle='-.')

    plt.xlabel('Time (min)')
    plt.ylabel("Validation BLEU Score")
    plt.yticks(fontsize=20)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


plt_default_vs_econmt_full_training_validation_bleu('N', False)
plt_default_vs_econmt_full_training_validation_bleu('T', False)
plt_default_vs_econmt_full_training_validation_bleu('N', True)
plt_default_vs_econmt_full_training_validation_bleu('T', True)

plt_default_vs_econmt_full_training_end2end()