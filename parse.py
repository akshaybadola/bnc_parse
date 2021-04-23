from typing import Optional, Tuple, Dict
import argparse
from collections import Counter
import json
import time
import sys
import os
import logging
import stanza


def get_file_and_stream_logger(logdir: str, logger_name: str,
                               log_file_name: str,
                               file_loglevel: Optional[str] = "debug",
                               stream_loglevel: Optional[str] = "info",
                               logger_level: Optional[str] = "debug") -> Tuple:
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    if stream_loglevel is not None and hasattr(logging, stream_loglevel.upper()):
        stream_handler.setLevel(getattr(logging, stream_loglevel.upper()))
    else:
        stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if file_loglevel is not None and hasattr(logging, file_loglevel.upper()):
        file_handler.setLevel(getattr(logging, file_loglevel.upper()))
    else:
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if logger_level is not None and hasattr(logging, logger_level.upper()):
        logger.setLevel(getattr(logging, logger_level.upper()))
    else:
        logger.setLevel(logging.DEBUG)
    return log_file, logger


class Timer:
    def __init__(self, accumulate=False):
        self._accumulate = accumulate
        self._time = 0

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        if self.accumulate:
            self._time += time.time() - self._start
        else:
            self._time = time.time() - self._start

    def clear(self):
        self._time = 0

    @property
    def time(self):
        return self._time

    @property
    def accumulate(self):
        return self._accumulate

    @property
    def as_dict(self):
        return {"time": self._time}


def build_vocab(corpus) -> Dict[str, int]:
    vocab: Counter = Counter()
    for line in corpus:
        vocab.update(line.split())
    return {k: v for k, v in vocab.items()}


def load_corpus(filename):
    corpus = []
    with open(filename) as f:
        for line in f:
            corpus.append(line.strip())
    return corpus


def get_dep_context(filename: str, chunk_size: int):
    _, logger = get_file_and_stream_logger("logs", "parser", "parser")
    logger.info(f"Chunk size is {chunk_size}")
    with open("vocab.json") as f:
        vocab = json.load(f)
    corpus = load_corpus(filename)
    split = filename.split(".")[0].split("_")[1]
    times = []
    timer = Timer(True)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
    # doc = nlp("this virus affects the body 's defence system so that it can not fight infection")

    # NOTE: continue from previous
    output_files = [x for x in os.listdir(".") if x.startswith("output_")]
    if output_files:
        indx = max(int(x.split(".")[0].split("_")[-1]) for x in output_files) + 1
    else:
        indx = 0

    loops = len(corpus)//chunk_size
    for i in range(indx, loops):
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
                        pairs.append((left, right))
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
        with open(f"output_{split}_{i:06}.json", "w") as f:
            json.dump(pairs, f)
        logger.debug(f"Dumped file output_{split}_{i:06}.json")


def main(args):
    get_dep_context(args.filename, args.chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dependency Parser")
    parser.add_argument("filename")
    parser.add_argument("--chunk-size", "-c", type=int, default=100)
    args = parser.parse_args()
    main(args)