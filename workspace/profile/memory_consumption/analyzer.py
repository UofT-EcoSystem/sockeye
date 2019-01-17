#!/usr/bin/python

FUNCTION_REGEX_DICT = {
    'Feature Maps'       : ['forward_features'],
    'Weights'            : ['in_arg', 'arg_grad', 'optimizer'],
    'Workspace'          : ['workspace'],
    'Others'             : ['aux_state', '(data)', '(label)', '(source)',  '(target)', '(sum)', '(target_label)',
                            '_equal_scalar', '_rminus_scalar', 'untagged', 'warning!,ctx_source_unclear'],
}

LAYER_REGEX_DICT = {
    'Attention'          : ['att'],
    'LSTM Feature Maps'  : ['reserved_space'],
    'LSTM Cell State'    : ['lstmnonlinblock'],
    # 'RNN'                : ['rnn'],
    'RNN'                : ['in_arg:encoder', 'arg_grad:encoder', 
                            'in_arg:decoder', 'arg_grad:decoder',
                            '_optimizer_weight_update_encoder',
                            '_optimizer_weight_update_decoder'],
    'Decoder Concat'     : ['hidden_concat', 'concat_target'],
    'Decoder MLP'        : ['hidden_norminp', 'hidden_fc', 'next_hidden'],
    'Square'             : ['square'],
    'Output'             : ['logit', 'softmax'],
    'Others'             : ['target_embed', 'source_embed', 'mul', 'rsqrt', 'rminus', ':mean', 'split',
                            'swapaxes', 'sequencereverse', 'dot', 'broadcast', 'zeros', 'sum', 'transpose', 
                            ':dropout', ':slice', 'cnn', 'arange', 'fullyconnected', 'sequencemask',
                            'activation', 'reshape', 'transformer', '_equal_scalar', 'aux_state', 'relu',
                            'conv', 'pool', 'bn', ':tile', ':id', ':fc', ':indexing', '(data)', '(source)',
                            '(target)', '(target_label)', 'untagged', 'warning!,ctx_source_unclear',
                            ':concat', ':sequencelast'],
}

def parse_memory_profile(memory_profile, regex_dict):
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

    import pprint

    pprint.pprint(stats_dict['RNN'])

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


parse_memory_profile('iwslt15-vi_en-groundhog-010.log', LAYER_REGEX_DICT)
parse_memory_profile('iwslt15-vi_en-groundhog-110.log', LAYER_REGEX_DICT)
parse_memory_profile('iwslt15-vi_en-groundhog-2:2-110.log', LAYER_REGEX_DICT)
parse_memory_profile('iwslt15-vi_en-groundhog-3:3-110.log', LAYER_REGEX_DICT)