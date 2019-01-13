#!/usr/bin/python

import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--dram-transactions-profile", type=str)


def parse_dram_transactions_profile(fname):
    import csv
    
    dram_transactions, dram_read_transactions, dram_write_transactions = {}, {}, {}

    with open(fname, 'r') as csv_file:
        csv_fin = csv.reader(csv_file)
        
        for row in csv_fin:
            if row[0][0] == "=" or row[0] == "Device":
                continue
            # The parsed `row` is a list of metrics that consist of the following information.
            # "Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
            assert len(row) == 8, "Length of `row` must be 8."
            
            if row[3] == "dram_read_transactions":
                dram_read_transactions [row[1]] = int(row[2]) * int(row[7]) / 1e6
            if row[3] == "dram_write_transactions":
                dram_write_transactions[row[1]] = int(row[2]) * int(row[7]) / 1e6
            
            if row[1] not in dram_transactions:
                dram_transactions[row[1]]  = int(row[2]) * int(row[7]) / 1e6
            else:
                dram_transactions[row[1]] += int(row[2]) * int(row[7]) / 1e6
    
    return dram_transactions, dram_read_transactions, dram_write_transactions


args = parser.parse_args()

dram_transactions, dram_read_transactions, dram_write_transactions = parse_dram_transactions_profile(args.dram_transactions_profile)

import pprint, operator 

sorted_dram_transactions = sorted(dram_transactions.items(), 
                                  key=lambda kv: kv[1], reverse=True)
pprint.pprint(sorted_dram_transactions[:10])
