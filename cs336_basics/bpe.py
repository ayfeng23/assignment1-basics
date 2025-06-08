import os
from typing import BinaryIO
from pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex as re
from collections import Counter

def process_chunk(input_path: str, start: int, end: int, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    split_chunk = re.split("|".join(special_tokens), chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_counts = Counter()
    for chunk in split_chunk:
        pretoken_counts.update(m.group(0) for m in re.finditer(PAT, chunk))

    return pretoken_counts 
    
def compute_vocab(vocab_dict, pretoken_counter):
    pair_counter = Counter()
    for pretoken, num_occurrences in pretoken_counter.items():
        for i in range(len(pretoken) - 1):
            token_pair = pretoken[i:i+2]
            pair_counter[token_pair] += num_occurrences

    two_most_common = pair_counter.most_common(2)
    if two_most_common[0][1] != two_most_common[1][1]:
        merge_token1, merge_token2 = two_most_common[0][0]
    else:
        max_count = two_most_common[0][0]
        max_keys = [k for k, v in pair_counter.items() if v == max_count]
        max_keys_merged = [a + b for (a,b) in max_keys]
        merge_token1, merge_token2 = max_keys[max_keys_merged.index(sorted(max_keys_merged, reverse=True)[0])]
    merged_token = merge_token1 + merge_token2
    

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
    results = Counter({tuple(k): v for k, v in results.items() if len(k) > 1})
    
    vocab_dict = {key: chr(key) for key in range(256)}
    
    compute_vocab(vocab_dict, results)
    return results

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

    run_bpe("data/TinyStoriesV2-GPT4-valid.txt", 1000, special_tokens, num_processes=4)

    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)