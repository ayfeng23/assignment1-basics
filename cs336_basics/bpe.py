import os
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex as re
from collections import Counter

def process_chunk(input_path: str, start: int, end: int, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # split_chunk = re.split(f"({'|'.join(special_tokens)})", chunk)
    split_chunk = re.split(f"{'|'.join(special_tokens)}", chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = Counter()
    for chunk in split_chunk:
        pretoken_counts.update(m.group(0) for m in re.finditer(PAT, chunk))

    return pretoken_counts 

def update_counter(counter, merge_token1, merge_token2):
    merged_token = merge_token1 + merge_token2
    edits = []
    for item in counter.keys():
        if merge_token1 not in item:
            continue
        elif merge_token2 not in item:
            continue

        old_item = item
        item = list(item)
        mergeable_tokens = []
        for i in range(len(item) - 1):
            if (item[i], item[i + 1]) == (merge_token1, merge_token2):
                mergeable_tokens.append(i)
        
        for i in range(len(mergeable_tokens)):
            if mergeable_tokens[i] == -2:
                continue
            item[mergeable_tokens[i]] = merged_token
            if i != len(mergeable_tokens) - 1:
                if mergeable_tokens[i + 1] == mergeable_tokens[i] + 1:
                    mergeable_tokens[i+1] = -2
            if mergeable_tokens[i] != len(item) - 1:
                item[mergeable_tokens[i] + 1] = None
        
        item = [item for item in item if item != None]
        if item != list(old_item):
            edits.append((item, old_item))

    for item, old_item in edits:
        if len(item) > 1:
            counter[tuple(item)] = counter.pop(old_item)
        else:
            counter.pop(old_item)
    return counter
    
def compute_vocab(vocab_dict, pretoken_counter, pair_counter = None, ordered_list = []):
    if pair_counter == None:
        pair_counter = Counter()
        for pretoken, num_occurrences in pretoken_counter.items():
            for i in range(len(pretoken) - 1):
                token_pair = pretoken[i:i+2]
                pair_counter[token_pair] += num_occurrences

    two_most_common = pair_counter.most_common(2)

    if two_most_common[0][1] != two_most_common[1][1]:
        merge_token1, merge_token2 = two_most_common[0][0]
    else:
        max_count = two_most_common[0][1]
        max_keys = [k for k, v in pair_counter.items() if v == max_count]
        merge_token1, merge_token2 = sorted(max_keys, reverse=True)[0]
        
    merged_token = merge_token1 + merge_token2
    vocab_dict[len(vocab_dict)] = merged_token

    pretoken_counter = update_counter(pretoken_counter, merge_token1, merge_token2)

    ordered_list.append((merge_token1, merge_token2))

    ###update pair counter
    del pair_counter[(merge_token1, merge_token2)]
    #pair_counter = update_counter(pair_counter, merge_token1, merge_token2)

    return vocab_dict, pretoken_counter, pair_counter, ordered_list

def run_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = None):
    #Get pretoken counts
    if num_processes is None:
        num_processes = mp.cpu_count()

    special_tokens = [re.escape(special_token) for special_token in special_tokens]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        jobs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            jobs.append((input_path, start, end, vocab_size, special_tokens))

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(process_chunk, jobs)
    results = sum(results, Counter()) 

    pretoken_counter = Counter({tuple(bytes([b]) for b in k.encode("utf-8")): v for k, v in results.items() if len(k) > 1})
    pretoken_counter = Counter({
        tuple(bytes([b]) for b in k.encode("utf-8")): v
        for k, v in results.items()
        if len(k.encode("utf-8")) > 1
    })

    vocab_dict = {}
    for key in list(range(33, 127)):
        vocab_dict[key - 32] = chr(key)
    for key in list(range(161, 173)):
        vocab_dict[key - 66] = chr(key)
    for key in list(range(174, 256)):
        vocab_dict[key - 67] = chr(key)
    used_chars = set(vocab_dict.values())
    current_id = max(vocab_dict.keys()) + 1

    for cp in range(0x00A0, 0x0180):
        c = chr(cp)
        if c not in used_chars and c.isprintable() and current_id <= 256:
            vocab_dict[current_id] = c
            current_id += 1
    vocab_dict[0] = "<|endoftext|>"

    pair_counter = None

    ordered_list = []
    for _ in range(vocab_size - 257):
        vocab_dict, pretoken_counter, pair_counter, ordered_list = compute_vocab(vocab_dict, pretoken_counter, None, ordered_list)

    return vocab_dict, pretoken_counter, pair_counter, ordered_list

def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def run_bpe_list(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = None):
    vocab_dict, _, _, ordered_list = run_bpe(input_path, vocab_size, special_tokens, num_processes)
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    del vocab_dict[0]
    vocab_dict = {
        k: bytes([gpt2_byte_decoder[c] for c in v]) if isinstance(v, str) else v
        for k, v in vocab_dict.items()
    }
    vocab_dict[0] = b'<|endoftext|>'
    #print(vocab_dict)
    return vocab_dict, ordered_list

special_tokens = [
    "<|bos|>",   # begin-of-sequence
    "<|eos|>",   # end-of-sequence
    "<|pad|>",   # padding token
    "<|sep|>",   # separator
    "<|cls|>",   # classification token
]

if __name__ == "__main__":
    import cProfile, pstats
    import multiprocessing as mp

    profiler = cProfile.Profile()
    profiler.enable()

    run_bpe_list("data/TinyStoriesV2-GPT4-valid.txt", 1000, special_tokens, num_processes=4)

    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)