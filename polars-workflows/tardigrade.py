from __future__ import annotations 

import functools
import itertools
import heapq
import json
import os
import subprocess
import tempfile
import weakref


from collections.abc import Iterable
from itertools import groupby
from typing import Any, Callable

from collections import deque, namedtuple

ReduceSpec = namedtuple('ReduceSpec', ['input_col', 'output_col', 'aggregator'])

DEFAULT_BATCH_SIZE = 100


#AGGREGATORS = {
#    len:  (lambda vs: sum(1 for _ in vs),        pl.Int64),
#    sum:  (lambda vs: sum(float(v) for v in vs), pl.Float64),
#    list: (lambda vs: list(vs),                  pl.List(pl.String)),
#}


class LazyFrame:

    def __init__(self, iter: Iterable):
        self.iter = iter
    
    def map(self, fn: Callable[Any, Any]) -> LazyFrame:
        return LazyFrame(map(fn, self.iter))

    def head(self, k=20) -> LazyFrame:
        return LazyFrame(itertools.islice(self.iter, k))

    def tail(self, k=20) -> LazyFrame:
        """Last k rows.  Consumes the whole frame, but holds only k rows.
        """
        def generator():
            yield from deque(self.iter, maxlen=k)
        return LazyFrame(generator())

    def reduce(self, operator: Callable[Any, Any]) -> LazyFrame:
        """Reduces a series and returns result in a singleton series.
        """
        def generator():
            yield functools.reduce(operator, self.iter)
        return LazyFrame(generator()) 

    def collect(self) -> list:
        return list(self.iter)

    def explode(self) -> LazyFrame:
        return LazyFrame(itertools.chain.from_iterable(self.iter))

    def filter(self, fn: Callable[Any,Any]) -> LazyFrame:
        return LazyFrame(filter(fn, self.iter))

    def sort(self, 
             batch_size: int = DEFAULT_BATCH_SIZE, 
             key: Callable[Any,Any] | None = None,
             reverse: bool = False) -> LazyFrame:
        """External merge sort: spill sorted runs to disk, then k-way merge them.

        Peak memory is one batch rather than the whole frame, so this sorts
        inputs larger than RAM.  Rows must be JSON-serializable.
        """
        # sort each batch and spill it to its own run file
        def spill(batch):
            # TemporaryFile is unlinked at creation, so a run is reclaimed on close
            fp = tempfile.TemporaryFile(mode='w+')
            for x in sorted(batch, key=key, reverse=reverse):
                fp.write(json.dumps(x) + '\n')
            fp.seek(0)
            return fp

        def read_run(fp) -> Iterable:
            with fp:
                for line in fp:
                    yield json.loads(line)

        # one heap over all runs: O(n log k), vs O(n*k) for repeated pairwise merges
        def generator():
            runs = [spill(batch) for batch in itertools.batched(self.iter, batch_size)]
            yield from heapq.merge(*map(read_run, runs), key=key, reverse=reverse)

        return LazyFrame(generator())


    # unique, group_by_key, join
    def sink(self, filename, *open_args, **open_kw):
        with open(filename, *open_args, **open_kw) as fp:
            for x in self.iter:
                fp.write(x)
            

# zip, concat

def scan_lines(filename: str, *open_args, **open_kw):
    for line in open(filename, *open_args, **open_kw):
        yield line

def scan_gz_lines(filename: str, *open_args, **open_kw):
    process = subprocess.Popen(
        [f"gunzip -c {filename} | grep -v '#'"],
        shell=True,
        text=False,
        stdout=subprocess.PIPE,
    )
    for line in process.stdout:
        yield line
