from __future__ import annotations 

import functools
import itertools
import heapq
import os
import subprocess
import tempfile
import weakref


from collections.abc import Iterable
from typing import Any, Callable

from collections import deque, namedtuple

ReduceSpec = namedtuple('ReduceSpec', ['input_col', 'output_col', 'aggregator'])

DEFAULT_BATCH_SIZE = 1024

# merging f runs holds f files open at once; beyond this we cascade instead
DEFAULT_MERGE_FANIN = 16

# Aggregators are handed an *iterator* over a group, so that reducing a group
# needn't hold it in memory.  len() is the one common aggregator that insists on
# a sized object, so count without materializing; sum/max/min/list are fine.
ITERATOR_AGGREGATORS = {len: lambda vs: sum(1 for _ in vs)}


#
# helpers
#

class _SpillFile:
    """One sorted run, written to a temp file and deleted when unreferenced.

    Iterating opens the file, so a run costs a filename until it is read --
    only the runs actively being merged hold a file descriptor.
    """

    def __init__(self, rows: Iterable):
        def _as_line(row: Any) -> str:
            if not isinstance(row, str):
                raise ValueError('spilled objects must be strings')
            return row.rstrip() + '\n'
        fd, self.name = tempfile.mkstemp(suffix='.run')
        with os.fdopen(fd, 'w') as fp:
            fp.writelines(map(_as_line, rows))
        weakref.finalize(self, os.unlink, self.name)

    def __iter__(self) -> Iterable:
        with open(self.name) as fp:
            yield from fp

def _spill_batches(batches: Iterable, key=None, reverse=False) -> tuple[Iterable, int]:
    """Sorts each batch and spills it; returns the runs and how many there are."""
    runs = [_SpillFile(sorted(batch, key=key, reverse=reverse)) for batch in batches]
    return iter(runs), len(runs)


def _merge_spills(runs: Iterable, fanin: int, key=None, reverse=False) -> tuple[Iterable, int]:
    """One merge pass: combines runs in groups of fanin, respilling each group."""
    merged = [_SpillFile(heapq.merge(*group, key=key, reverse=reverse))
              for group in itertools.batched(runs, fanin)]
    return iter(merged), len(merged)


#
# The LazyFrame class
#

class LazyFrame:

    def __init__(self, iter: Iterable):
        """Wraps any iterable of rows.  Nothing is read until collect().

        A frame is ONE-SHOT, like the iterator inside it: collecting or
        iterating consumes it, so re-read the source to go round again.
        """
        self.iter = iter

    #
    # chaining API
    #

    def map(self, fn: Callable[Any, Any]) -> LazyFrame:
        """Applies fn to every row.
        """
        return LazyFrame(map(fn, self.iter))

    def head(self, k=20) -> LazyFrame:
        """First k rows, or all of them if the frame is shorter.

        islice rather than a next() loop: next() past the end raises
        StopIteration, which PEP 479 turns into RuntimeError inside a generator.
        """
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
        """Runs the pipeline and returns every row as a list.

        This is the point where the work actually happens -- and where the
        result stops being streaming, since the whole thing lands in memory.
        """
        return list(self.iter)

    def explode(self) -> LazyFrame:
        """Flattens one level: each row must itself be iterable.
        """
        return LazyFrame(itertools.chain.from_iterable(self.iter))

    def filter(self, fn: Callable[Any,Any]) -> LazyFrame:
        """Keeps the rows for which fn is true.
        """
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

    def unique(self, is_sorted: bool = False, *sort_args, **sort_kw) -> LazyFrame:
        """Drops duplicate rows.  Sorts first unless the frame already is.

        groupby only collapses *adjacent* equal rows, which is why sorting is
        what makes this O(1) in the number of distinct rows rather than O(n).
        """
        iter_to_group = self.iter if is_sorted else self.sort(*sort_args, **sort_kw).iter
        return LazyFrame(k for k, _ in itertools.groupby(iter_to_group))

    def group_by_key(self, key: Callable[Any,Any], agg: Callable[Any,Any] = list,
                     value: Callable[Any,Any] | None = None,
                     is_sorted: bool = False, **sort_kw) -> LazyFrame:
        """Groups rows by key and reduces each group, yielding (key, aggregate).

            frame.group_by_key(key=first_field, agg=len)
            frame.group_by_key(key=first_field, value=second_field, agg=sum)

        Sorting brings equal keys together so groupby can collapse each run,
        so memory is one group rather than one entry per distinct key -- only
        agg=list holds a whole group, while len/sum/max stay O(1).  Sorting is
        stable, so a key's values arrive in input order.

        agg is passed an iterator, and must consume it before the next key is
        reached (list, sum, len, max all do); returning a lazy generator from agg
        yields empty groups.
        """
        agg = ITERATOR_AGGREGATORS.get(agg, agg)
        def generator():
            rows = self.iter if is_sorted else self.sort(key=key, **sort_kw).iter
            for k, grp in itertools.groupby(rows, key=key):
                yield k, agg(grp if value is None else map(value, grp))
        return LazyFrame(generator())

    def merge_join(self, right: LazyFrame, key: Callable[Any,Any],
                   right_key: Callable[Any,Any] | None = None, **sort_kw) -> LazyFrame:
        """Inner join on a key, yielding (key, left_row, right_row) triples.

            edges.merge_join(n_outlinks, key=first_field)

        A join is a group-by on the join key where each key emits the cross
        product of its left and right rows, so this is group_by_key's skeleton:
        tag each side, sort the two together, and walk one key's run at a time.

        Only the LEFT side of a key is buffered -- the right streams past it --
        because sorting is stable and left rows are concatenated first, so a
        key's left rows always arrive before its right ones.  Put the 'one'
        side of a many-to-one join on the left and memory is O(1) in the
        'many' side; otherwise it is O(left rows per key).

        right_key defaults to key, for when the two sides are shaped alike.
        """
        right_key = right_key or key
        # tag rows so the two sides stay distinguishable once interleaved;
        # the tag is a prefix, so the key functions see the original row
        tagged_key = lambda row: (key if row[0] == 'L' else right_key)(row[1:])

        def generator():
            both = itertools.chain(map('L'.__add__, self.iter),
                                   map('R'.__add__, right.iter))
            rows = LazyFrame(both).sort(key=tagged_key, **sort_kw).iter
            for k, grp in itertools.groupby(rows, key=tagged_key):
                lefts = []
                for row in grp:
                    if row[0] == 'L':
                        lefts.append(row[1:])
                    else:
                        # no lefts => key is right-only, so the inner gate drops it
                        yield from ((k, l, row[1:]) for l in lefts)

        return LazyFrame(generator())


    def sink(self, filename, *open_args, **open_kw):
        """Writes every row to a file, one row per write, holding none of them.

        Rows are written verbatim, so they must be text and carry their own
        newline -- as scan_lines produces them.  The streaming counterpart to
        collect(): the terminal operation that does NOT build a list.
        """
        with open(filename, *open_args, **open_kw) as fp:
            for x in self.iter:
                fp.write(x)
            

#
# functions that create LazyFrames
#


def concat(lfs: list[LazyFrame], how: str) -> LazyFrame:
    """Combines frames, as polars' pl.concat does.

    vertical   -- one frame's rows after another's.
    horizontal -- rows side by side, as tuples.  zip_longest, so a short frame
                  is padded with None rather than truncating the others, which
                  is what polars does with nulls.

    Both stream: a row is pulled from a source only when one is asked for.
    """
    if how == 'vertical':
        return LazyFrame(itertools.chain.from_iterable(lf.iter for lf in lfs))
    elif how == 'horizontal':
        return LazyFrame(itertools.zip_longest(*(lf.iter for lf in lfs)))
    else:
        raise ValueError(f'unimplemented concat "how" method "{how}"')


def scan_lines(filename: str, *open_args, **open_kw) -> LazyFrame:
    """A frame of the lines of a file, read lazily.  Rows keep their newline.
    """
    return LazyFrame(_scan_lines(filename, *open_args, **open_kw))

def scan_gz_lines(filename: str, *open_args, **open_kw) -> LazyFrame:
    """The same, for a gzipped file, decompressed by a gunzip subprocess.
    """
    return LazyFrame(_scan_gz_lines(filename, *open_args, **open_kw))

def _scan_lines(filename: str, *open_args, **open_kw):
    """Returns an iterator over lines in a file.
    """
    with open(filename, *open_args, **open_kw) as fp:
        yield from fp

def _scan_gz_lines(filename: str):
    """Returns an iterator over lines in a zipped file.
    """
    process = subprocess.Popen(
        ['gunzip', '-c', filename],
        text=True,
        stdout=subprocess.PIPE,
    )
    with process.stdout as fp:
        yield from fp
    # only reached if the reader drained us: a truncated .gz would otherwise
    # look like a short file.  Abandoning early (head) skips this by design,
    # since gunzip is then killed by SIGPIPE.
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, process.args)

