"""
    Author: Abhishek Tiwari, Bojian Zheng (ArmageddonKnight)
    Description: This file does analysis of the collected memory profile.
    Acknowledgement: This file is just a slightly modified version of the memory analysis tool
                     developed by Abhishek Tiwari from the University of Toronto.
"""


FUNCTION_REGEX_DICT = {
    'forward_features'  : 'Feature Map',
    'in_arg'            : 'Weight (Bias)',
    'arg_grad'          : 'Weight (Bias) Gradient',
    'aux_grad'          : 'Auxiliary State',
    'workspace'         : 'Workspace',
    'optimizer'         : 'Optimizer state',
    '(source)'          : 'Placeholder (SRC)',
    '(target)'          : 'Placeholder (TGT)',
    '(target_label)'    : 'Placeholder (TGT Label)',
    'untagged'          : 'Unknown (From Python Side)',
    'warning!,ctx_source_unclear' : 'Unknown (From C++ side)',
}


OPERATOR_REGEX_DICT = {
    'encoder_birnn'     : 'Encoder RNN',
    'decoder_rnn'       : 'Decoder RNN',
    'source_embed'      : 'Source Embedding',
    'target_embed'      : 'Target Embedding',
    'mul'               : 'Multiplication',
    'rsqrt'             : 'Square Root',
    'rminus'            : 'Minus',
    ':mean'             : 'Mean',
    'att'               : 'Attention',
    'split'             : 'Split',
    'logit'             : 'Logit',
    'swapaxes'          : 'SwapAxes',
    'square'            : 'Square',
    'softmax'           : 'SoftMax',
    'sequencereverse'   : 'SequenceReverse',
    'dot'               : 'Dot',
    'broadcast'         : 'Broadcast',
    'zeros'             : 'Zero',
    'sum'               : 'Sum',
    'transpose'         : 'Transpose',
    ':dropout'          : 'Dropout',
    ':slice'            : 'Slice',
    'cnn'               : 'CNN Layer',
    'arange'            : 'Arange',
    'fullyconnected'    : 'FullyConnected',
    'sequencemask'      : 'SequenceMask',
    'activation'        : 'Activation',
    'reshape'           : 'Reshape',
    'transformer'       : 'Transformer',
    'equal_scalar'      : 'Equal Scalar',
    'aux_state'         : 'Auxiliary State',
    'relu'              : 'Relu',
    'conv'              : 'Convolutional Unit',
    'pool'              : 'Pooling',
    'bn'                : 'Batch Norm',
    ':tile'             : 'Tile',
    ':id'               : 'Identity',
    ':fc'               : 'Fully Connected',
    ':indexing'         : 'Indexing',
    '(data)'            : 'Data',
    '(source)'          : 'Source',
    '(target)'          : 'Target',
    '(target_label)'    : 'Target Label',
    # '_optimizer'        : 'Optimizer',
    'untagged'          : 'Unknown (From Python Side)',
    'warning!,ctx_source_unclear' : 'Unknown (From C++ side)',
}


def parse_memory_profile(memory_profile, regex_dict=OPERATOR_REGEX_DICT):
    """
    Parse the memory profile

    :param memory_profile: Path to Memory Profile
    :param regex_dict    : Dictionary of Regular Expression
    
    :return Sorted Dictionary of Statistics
    """
    stats_dict = {}

    with open(memory_profile, 'r') as fin:
        for line in fin:
            line = line.rstrip()
            if 'Allocate' in line:
                words = line.split(' ')
                regex_matched = False

                for regex in regex_dict:
                    if regex in words[6]:
                        if regex_matched is True:
                            print("[WARNING]: " "%30s is considered match for another regex." % words[6])

                        regex_matched = True

                        if regex in stats_dict.keys():
                            stats_dict[regex][0] += float(words[2])
                            stats_dict[regex][1].append(words[6])
                        else:
                            stats_dict[regex] = [float(words[2]), [words[6]]]
                            break
                if regex_matched is False:
                    print("[INFO]: " "[Memory Profile Analyzer] " "Unknown Tag: %s" % words[6])

    sorted_stats_list = sorted(stats_dict.items(), key=lambda kv: kv[1][0], reverse=True)
    
    for stats in sorted_stats_list:
        print("Regex: %20s, Memory Consumption: %7.2f MB, Entries: %5d" % \
            (stats[0], stats[1][0] * 1.0 / (1024 * 1024), len(stats[1][1])))

    return sorted_stats_list
