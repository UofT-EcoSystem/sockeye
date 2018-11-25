"""
    Author: Abhishek Tiwari, Bojian Zheng (ArmageddonKnight)
    Description: This file does analysis of the collected memory profile.
    Acknowledgement: This file is just a slightly modified version of the memory analysis tool
                     developed by Abhishek Tiwari from the University of Toronto.
"""

"""
FUNCTION_REGEX_DICT maps Function Descriptions to List of Regular Expressions.
"""
SOCKEYE_FUNCTION_REGEX_DICT = {
    'Feature Maps'       : ['forward_features'],
    'Weights$'           : ['in_arg', 'arg_grad', 'optimizer'],
    'Others'             : ['aux_state',
                            'workspace',
                            '(data)', '(label)',
                            '(source)', 
                            '(target)',
                            '(sum)', 
                            '(target_label)',
                            '_equal_scalar', 
                            '_rminus_scalar',
                             'untagged', 
                            'warning!,ctx_source_unclear'],
}


"""
LAYER_REGEX_DICT maps Layer Descriptions to Regular Expressions.
"""
SOCKEYE_LAYER_REGEX_DICT = {
    # 'Embedding'          : ['target_embed',
    #                         'source_embed'],
    'Attention'          : ['att'],
    # 'Attention_Query_Plus_Input'    : ['att_query_plus_input'],
    # 'Attention_NormInp_Minus_Mean'  : ['att_norminp_minus_mean'],
    # 'Attention_NormInp_Norm_Scaled' : ['att_norminp_norm_scaled'],
    # 'Attention_Hidden'              : ['att_hidden'],
    'RNN'                : ['encoder_rnn', 
                            'encoder_birnn',
                            'decoder_rnn', 
                            'decoder_birnn'],
    # 'Square'             : ['square'],
    'Loss'               : ['logit',
                            'softmax'],
    'Square'             : ['square'],
    'Others'             : ['target_embed',
                            'source_embed',
                            'mul',
                            'rsqrt',
                            'rminus',
                            ':mean',
                            'split',
                            'swapaxes',
                            'sequencereverse',
                            'dot',
                            'broadcast',
                            'zeros',
                            'sum',
                            'transpose',
                            ':dropout',
                            ':slice',
                            'cnn',
                            'arange',
                            'fullyconnected',
                            'sequencemask',
                            'activation',
                            'reshape',
                            'transformer',
                            '_equal_scalar',
                            'aux_state',
                            'relu',
                            'conv',
                            'pool',
                            'bn',
                            ':tile',
                            ':id',
                            ':fc',
                            ':indexing',
                            '(data)',
                            '(source)',
                            '(target)',
                            '(target_label)',
                            'untagged',
                            'warning!,ctx_source_unclear',],
}


def parse_memory_profile(memory_profile, regex_dict):
    """
    Parse the memory profile

    :param memory_profile: Path to Memory Profile
    :param regex_dict    : Dictionary of Regular Expressions
    
    :return Sorted Dictionary of Statistics
    """
    stats_dict = {}

    with open(memory_profile, 'r') as fin:
        for line in fin:
            line = line.rstrip()
            if 'Allocate' in line:
                words = line.split(' ')
                regex_matched = False

                for key, regex_list in regex_dict.items():
                    for regex in regex_list:
                        if regex in words[6]:
                            if regex_matched is True:
                                print("[WARNING]: " "[Memory Profile Analyzer] " "%-30s is considered "
                                    "match for another regex." % words[6])
                                continue

                            regex_matched = True

                            if key in stats_dict.keys():
                                stats_dict[key][0] += float(words[2])
                                stats_dict[key][1].append(words[6])
                            else:
                                stats_dict[key] = [float(words[2]), [words[6]]]
                                break
                if regex_matched is False:
                    print("[WARNING]: " "[Memory Profile Analyzer] " "Unknown Tag: %s, " "Allocation Size: %5.2f" % \
                        (words[6], float(words[2]) * 1.0 / 1e6))

    sorted_stats_list = sorted(stats_dict.items(), key=lambda kv: kv[1][0], reverse=True)
    
    for stats in sorted_stats_list:
        if 'Other' in stats[0]:
            stats_kv = [stats[0], stats[1]]
            sorted_stats_list.remove(stats)
            sorted_stats_list.append(stats_kv)
            break

    for stats in sorted_stats_list:
        print("Keyword: %-30s, Memory Consumption: %7.2f GB, Entries: %5d" % \
            (stats[0], stats[1][0] * 1.0 / 1e9, len(stats[1][1])))

    return sorted_stats_list
