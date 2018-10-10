"""
    Author: Abhishek Tiwari, Bojian Zheng (ArmageddonKnight)
    Description: This file does analysis of the collected memory profile.
    Acknowledgement: This file is just a slightly modified version of the memory analysis tool
                     developed by Abhishek Tiwari from the University of Toronto.
"""

regex_dict = {
    'rnn'               : 'RNN',
    'embed'             : 'Embedding',
    'mul'               : 'Multiplication',
    'rsqrt'             : 'Sqrt',
    'mean'              : 'Mean',
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
    'dropout'           : 'Dropout',
    'slice'             : 'Slice',
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
    'id'                : 'Identity',
    'fc'                : 'Fully Connected',
    '(data)'            : 'Data',
    '(source)'          : 'Source',
    '(target)'          : 'Target',
    '(target_label)'    : 'Target Label',
    'workspace'         : 'Workspace',
    'untagged'          : 'Unknown (From Python Side)',
    'warning!,ctx_source_unclear' : 'Unknown (From C++ side)',
}


def parse_memory_profile(memory_profile):
    """
    Parse the memory profile

    :param memory_profile: Path to Memory Profile
    
    :return None
    """
    stats_dict = {}

    with open(memory_profile, 'r') as fin:
        for line in fin.readline():
            if 'Allocate' in line:
                words = line.split(' ')
                regex_matched = False

                for regex in regex_dict:
                    if regex in words[6]:
                        regex_matched = True

                        if regex in stats_dict.keys():
                            stats_dict[regex][0] += float(words[2])
                            stats_dict[regex][1].append(words[6])
                        else:
                            stats_dict[regex] = [ float(words[2]), [words[6]] ]
                            break
                    if regex_matched is False:
                        print('UNKNOWN TAG:', words[6])    
    print(stats_dict)
