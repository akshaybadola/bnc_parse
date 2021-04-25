from typing import Optional, Tuple, Dict, List
import argparse
from collections import Counter
import numpy as np
import json
import time
import sys
import os
import logging
import stanza
from common_pyutil.log import get_file_and_stream_logger
from common_pyutil.monitor import Timer


def build_vocab(corpus) -> Dict[str, int]:
    vocab: Counter = Counter()
    for line in corpus:
        vocab.update(line.split())
    return {k: v for k, v in vocab.items()}


def split_corpus(corpus: List[str], ratios: List[int]):
    if sum(ratios) != 1:
        ratios = np.cumsum([x/sum(ratios) for x in ratios])
    else:
        ratios = np.cumsum(ratios)
    size = len(corpus)
    indices = np.int64(np.floor(ratios * size))
    splits = []
    for x, y in zip([0, *indices[:-1]], indices):
        splits.append(corpus[x:y])
    return splits


def load_corpus(filename):
    corpus = []
    with open(filename) as f:
        for line in f:
            corpus.append(line.strip())
    return corpus


def get_dep_context(filename: str, chunk_size: int, out_dir: str, scan: bool = False):
    _, logger = get_file_and_stream_logger("logs", "parser", "parser")
    logger.info(f"Chunk size is {chunk_size}")
    with open("vocab.json") as f:
        vocab = json.load(f)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    corpus = load_corpus(filename)
    split = filename.split(".")[0].split("_")[1]
    times = []
    timer = Timer(True)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

    # NOTE: continue from previous
    deps_prefix = "deps"
    pairs_prefix = "pairs"
    pairs_files = [x for x in os.listdir(out_dir)
                   if x.startswith(f"{pairs_prefix}_") and f"_{split}_" in x
                   and x.endswith("json")]
    if not pairs_files or scan:
        indx = 0
    else:
        indx = max(int(x.split(".")[0].split("_")[-1]) for x in pairs_files) + 1

    loops = len(corpus)//chunk_size
    for i in range(indx, loops):
        out_deps_file = os.path.join(out_dir, f"{deps_prefix}_{split}_{i:06}.json")
        out_pairs_file = os.path.join(out_dir, f"{pairs_prefix}_{split}_{i:06}.json")
        if os.path.exists(out_deps_file) and os.path.exists(out_pairs_file):
            logger.debug(f"Files for index {i} already exist for this iteration")
            continue
        lines = corpus[chunk_size*i:chunk_size*(i+1)]
        with timer:
            doc = nlp(". ".join(lines))
        pairs = []
        with timer:
            for sent in doc.sentences:
                for word in sent.words:
                    if word.head == 0:
                        continue
                    if word.id < word.head:
                        left = word.text
                        right = sent.words[word.head - 1].text
                    else:
                        right = word.text
                        left = sent.words[word.head - 1].text
                    if left in vocab and right in vocab and\
                       vocab[left] >= 100 and vocab[right] >= 100:
                        pairs.append((left, right, word.deprel))
        if i % 10 == 9:
            logger.info(f"{i} out of {loops} iterations done")
            logger.info(f"Time taken = {timer.time} seconds")
            times.append(timer.time)
            timer.clear()
        if i % 20 == 19:
            loops_done = (i - indx)
            loops_remaining = (loops - i)
            avg_time_per_loop = sum(times) / loops_done
            estimated_time = avg_time_per_loop * loops_remaining
            logger.info(f"Avg time per loop = {avg_time_per_loop} seconds\n" +
                        f"Estimated time remaining = {estimated_time / 3600} hours")
        with open(out_deps_file, "w") as f:
            json.dump(doc.to_dict(), f)
        with open(out_pairs_file, "w") as f:
            json.dump(pairs, f)
        logger.debug(f"Dumped files [{deps_prefix},{pairs_prefix}]_{split}_{i:06}.json")


def main(args):
    get_dep_context(args.filename, args.chunk_size, args.out_dir, args.scan)


def load_json_files(files):
    data = []
    for fl in files:
        with open(fl) as f:
            temp = json.load(f)
            data.append("\n".join([" ".join(x) for x in temp]))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dependency Parser")
    parser.add_argument("filename")
    parser.add_argument("--chunk-size", "-c", type=int, default=100)
    parser.add_argument("--out-dir", "-o", type=str, default="out")
    parser.add_argument("--scan", "-s", action="store_true", help="Scann for missing ouput files")
    args = parser.parse_args()
    main(args)
