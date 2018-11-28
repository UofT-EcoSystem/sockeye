import numpy as np
import matplotlib.pyplot as plt


def gen_from_txt(fname, metric, metric_unit=None, skip=None):
    data = np.genfromtxt(fname=fname, delimiter=',').astype(np.float64)[1:,:]

    if metric == 'throughput' and skip != None:
        data = data[np.mod(np.arange(data.shape[0])+1,skip)!=0,:]
    if metric == 'memory_usage' and metric_unit == 'GB':
        data[:,2] = data[:,2] / 1000
    if metric == 'validation_bleu':
        data[:,2] = data[:,2] * 100
        starting_walltime = np.genfromtxt(fname=fname.replace(metric, 'perplexity'), 
                                          delimiter=',').astype(np.float64)[1,0]
        data = np.insert(data, 0, [starting_walltime, 0, 0], axis=0)

    # normalize the time axis to minutes
    data[:,0] = (data[:,0] - data[0, 0]) / 60.0

    return data


def plt_default_vs_econmt_preliminary(metric, metric_unit=None):
    """
    Plot the comparison between legacy backpropagation and 
    partial forward propagation (First 500 Updates, Preliminary Ver.).
    """
    ylabel = metric.title().replace('_', ' ')
    title ='default_vs_econmt-%s' % metric

    default = gen_from_txt(fname='default/csv/%s.csv' % metric,
                           metric=metric, metric_unit=metric_unit)
    econmt  = gen_from_txt(fname= 'econmt/csv/%s.csv' % metric,
                           metric=metric, metric_unit=metric_unit)
    
    # ==============================================================================================

    plt.figure()

    plt.plot(default[:,1], default[:,2], linewidth=2, linestyle='--', 
             color='black', label='Default')
    plt.plot(econmt [:,1], econmt [:,2], linewidth=2, linestyle='-',
             color='black', label='EcoNMT')

    plt.xlabel('Global Step (Number of Training Batches)')
    plt.ylabel("%s (%s)" % (ylabel, metric_unit) if metric_unit is not None else ylabel)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if metric == 'memory_usage':
        plt.yticks(np.arange(0, 13, 4))
    if metric == 'perplexity':
        plt.yticks(np.arange(0, 1300, 400))

    plt.legend(fontsize=20)
    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig(title + ".png")


def plt_throughput_vs_batch_size():
    B = [4, 8, 16, 32, 64, 128]

    resnet50_throughput = [99.36, 137.38, 172.26, 197.28, 200.02, 206.91]

    sockeye_throughput   = []
    sockeye_memory_usage = []

    for batch_size in B:
        sockeye_throughput  .append(gen_from_txt("iwslt15-vi_en-tbd-500-default-B_%d/csv/throughput.csv"   % batch_size,
                                    metric="throughput")[0, 2])
        sockeye_memory_usage.append(gen_from_txt("iwslt15-vi_en-tbd-500-default-B_%d/csv/memory_usage.csv" % batch_size,
                                    metric="memory_usage", metric_unit="GB")[-1, 2])

    # ==============================================================================================

    plt.figure()

    plt.plot(B, resnet50_throughput, linewidth=2, linestyle='-', 
             color='black', marker='o', markersize=5)

    plt.xlabel("Batch Size")
    plt.xlim(xmin=0, xmax=140)
    plt.xticks(B, ['%d' % batch_size if batch_size != 8 else '' \
        for batch_size in B], fontsize=20)
    plt.ylabel("Throughput (samples/s)")
    plt.yticks(np.arange(0, 251, 50), fontsize=20)

    plt.grid(linestyle='-.', linewidth=1)

    plt.tight_layout()
    plt.savefig("throughput_vs_batch_size-resnet_50.png")

    # ==============================================================================================

    fig, axes = plt.subplots()

    throughput_plot = axes.plot(B, sockeye_throughput, linewidth=2, linestyle='-',
                                color='black', marker='o', markersize=5, label="Throughput")
    axes.set_xlabel("Batch Size")
    axes.set_xlim(xmin=0, xmax=140)
    axes.set_xticks(B)
    axes.set_xticklabels(['%d' % batch_size if batch_size != 8 else '' \
        for batch_size in B])
    axes.set_ylabel("Throughput (samples/s)")
    axes.set_yticks(np.arange(0, 501, 100))
    
    for ticklabel in axes.get_xticklabels() + axes.get_yticklabels():
        ticklabel.set_fontsize(20)

    axes.grid(linestyle='-.', linewidth=1)

    # ==============================================================================================

    axes = axes.twinx()

    memory_usage_plot = axes.plot(B, sockeye_memory_usage, linewidth=2, linestyle='--',
                                  color='black', marker='X', markersize=5, label="Memory Usage")
    
    axes.set_ylabel("Memory Consumption (GB)")
    axes.set_yticks(np.arange(0, 11, 2))

    legends = throughput_plot + memory_usage_plot
    axes.legend(legends, [legend.get_label() for legend in legends], fontsize=20)

    plt.tight_layout()
    plt.savefig("throughput_vs_batch_size-sockeye.png")
