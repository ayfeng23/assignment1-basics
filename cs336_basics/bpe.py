import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
corpus = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
lines = corpus.split("\n")
pretokens = []
for line in lines:
    pretokens.extend(line.split(" "))
pretokens_dict = {}
for pretoken in pretokens:
    pretoken = tuple(pretoken)
    if pretoken not in pretokens_dict.keys():
        pretokens_dict[pretoken] = 1
    else: 
        pretokens_dict[pretoken] += 1

from collections import Counter
def merge_tokens(pretokens_dict):
    pair_counter = Counter()
    for pretoken in pretokens_dict.keys():
        pretoken_len = len(pretoken)
        num_occurrences = pretokens_dict[pretoken]
        if pretoken_len == 1: continue
        for i in range(pretoken_len - 1):
            token_pair = pretoken[i:i+2]
            if token_pair in pair_counter.keys():
                pair_counter[token_pair] += num_occurrences
            else:
                pair_counter[token_pair] = num_occurrences
    max_count = max(pair_counter.values())
    max_keys = [k for k, v in pair_counter.items() if v == max_count]
    max_keys_merged = [a + b for (a,b) in max_keys]
    keys_to_merge = max_keys[max_keys_merged.index(sorted(max_keys_merged, reverse=True)[0])]
    edits = []
    for pretoken in pretokens_dict:
        old_token = pretoken
        pretoken = list(pretoken)
        mergeable_tokens = []
        for i in range(len(pretoken) - 1):
            if (pretoken[i], pretoken[i + 1]) == keys_to_merge:
                mergeable_tokens.append(i)

        for i in range(len(mergeable_tokens)):
            if mergeable_tokens[i] == -2:
                continue
            pretoken[mergeable_tokens[i]] = keys_to_merge[0] + keys_to_merge[1]
            if i != len(mergeable_tokens) - 1:
                if mergeable_tokens[i+1] == mergeable_tokens[i] + 1:
                    mergeable_tokens[i+1] = -2
            if mergeable_tokens[i] != len(pretoken) - 1:
                pretoken[mergeable_tokens[i] + 1] = None
        pretoken = [item for item in pretoken if item != None]
        if pretoken != list(old_token):
            edits.append((pretoken, old_token))
    for pretoken, old_token in edits:
        pretokens_dict[tuple(pretoken)] = pretokens_dict.pop(old_token)
    return pretokens_dict, keys_to_merge[0] + keys_to_merge[1]
print(merge_tokens(pretokens_dict))
print(merge_tokens(pretokens_dict))
print(merge_tokens(pretokens_dict))
print(merge_tokens(pretokens_dict))
print(merge_tokens(pretokens_dict))
print(merge_tokens(pretokens_dict))
        