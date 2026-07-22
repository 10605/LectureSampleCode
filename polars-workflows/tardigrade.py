from __future__ import annotations 

import functools
import itertools
import heapq
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

# merging f runs holds f files open at once; beyond this we cascade instead
DEFAULT_MERGE_FANIN = 256


class _SpillFile:
    """One sorted run, written to a temp file and deleted when unreferenced.

    Iterating opens the file, so a run costs a filename until it is read --
    only the runs actively being merged hold a file descriptor.
    """

    def __init__(self, rows: Iterable):
        fd, self.name = tempfile.mkstemp(suffix='.run')
        with os.fdopen(fd, 'w') as fp:
            fp.writelines(map(_as_line, rows))
        weakref.finalize(self, os.unlink, self.name)

    def __iter__(self) -> Iterable:
        with open(self.name) as fp:
            yield from fp


def _as_line(row: Any) -> str:
    """Rows are spilled verbatim, one per line -- so they must be text."""
    if not isinstance(row, str):
        raise TypeError(
            f'sort() spills rows as lines of text, but got {type(row).__name__}. '
            'Serialize rows yourself first, e.g. .map(json.dumps) after importing json')
    return row if row.endswith('\n') else row + '\n'


def _spill_batches(batches: Iterable, key=None, reverse=False) -> tuple[Iterable, int]:
    """Sorts each batch and spills it; returns the runs and how many there are."""
    runs = [_SpillFile(sorted(batch, key=key, reverse=reverse)) for batch in batches]
    return iter(runs), len(runs)


def _merge_spills(runs: Iterable, fanin: int, key=None, reverse=False) -> tuple[Iterable, int]:
    """One merge pass: combines runs in groups of fanin, respilling each group."""
    merged = [_SpillFile(heapq.merge(*group, key=key, reverse=reverse))
              for group in itertools.batched(runs, fanin)]
    return iter(merged), len(merged)


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

    def batched(self, batch_size: int = DEFAULT_BATCH_SIZE) -> LazyFrame:
        """Groups rows into tuples of batch_size (the last one may be short).
        """
        return LazyFrame(itertools.batched(self.iter, batch_size))

    def sort(self,
             batch_size: int = DEFAULT_BATCH_SIZE,
             key: Callable[Any,Any] | None = None,
             reverse: bool = False,
             max_fanin: int = DEFAULT_MERGE_FANIN) -> LazyFrame:
        """External merge sort: spill sorted runs to disk, then merge them.

        Peak memory is one batch plus at most max_fanin open runs, so this
        sorts inputs larger than RAM.  Rows are written verbatim as lines, so
        pick your own serialization if they aren't already text -- e.g. with
        the stdlib json module,
            frame.map(json.dumps).sort(key=...).map(json.loads)
        """
        def generator():
            runs, n_runs = _spill_batches(
                self.batched(batch_size).iter, key=key, reverse=reverse)
            # merging n runs at once holds n files open, so cascade if n is large.
            # Each pass rewrites everything, so prefer one pass: log_fanin(n) of them.
            while n_runs > max_fanin:
                runs, n_runs = _merge_spills(
                    runs, max_fanin, key=key, reverse=reverse)
            yield from heapq.merge(*runs, key=key, reverse=reverse)

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
