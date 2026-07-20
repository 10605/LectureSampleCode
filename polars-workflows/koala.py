"""koala.py -- guaranteed memory-efficient group-by / join / unique for polars,
built map-reduce style (guineapig lineage, hence the name) *without* the
distribution.

The trick: never rely on polars' streaming group_by/join, which buffer their
input proportional to the number of rows (see MEMORY_DIAGNOSIS.md). Instead

  1. sink the (key, value) columns to a TSV on disk,

  2. sort that file with something that is memory-efficient: unix
     `sort` (external, spills to disk, with env var LC_ALL=C)

  3. stream through the sorted file collapsing contiguous runs of equal keys
     into (key, aggregate) rows, and feed those back into polars via an IO
     plugin (register_io_source).

Memory is O(1) in rows and in number of groups for sum/len; for `list` a single
group's aggregate must fit in memory (inherent to group-by-key semantics).

Note on eager/lazy: group_by_key returns a genuinely lazy node -- the sink and
the external sort are deferred until the first collect()/sink() and then cached,
so building a plan does no I/O. The sorted temp file lives as long as the
source's callback is referenced by the query plan, and is removed when it is
garbage-collected (whether or not the sink ever ran).

pagerank.py equivalents this is meant to replace:
  edges.group_by('src').len()                       -> group_by_key(edges, 'src', reduce('dst', via=len))
  ...group_by('dst').agg(pl.col('delta').sum()...)  -> group_by_key(msgs,  'dst', reduce('delta', via=sum))

"""

from __future__ import annotations 

import json
import os
import subprocess
import tempfile
import weakref
from itertools import groupby
from typing import Callable

import polars as pl
from polars.io.plugins import register_io_source

from collections import namedtuple

# package arguments to a 'reduce' action

ReduceSpec = namedtuple('ReduceSpec', ['input_col', 'output_col', 'aggregator'])

# How many rows koala accumulates before emitting a DataFrame from its IO
# source. Smaller batches cap the resident memory of an operator (esp. the
# list-gathering in inner_join) at the cost of more, smaller DataFrames.
DEFAULT_BATCH_SIZE = 100_000

# aggregator -> (fn over a stream of string values, output dtype).
# Each aggregator owns its own parsing so `list` keeps whole strings intact
# (list("42") would split into ['4','2'] -- do the parse per-aggregator).

AGGREGATORS = {
    len:  (lambda vs: sum(1 for _ in vs),        pl.Int64),
    sum:  (lambda vs: sum(float(v) for v in vs), pl.Float64),
    list: (lambda vs: list(vs),                  pl.List(pl.String)),
}


class KoalaFrame:

    def __init__(self, ldf: pl.LazyFrame):
        self.ldf = ldf

    def group_by_key(self, key_col: str, reducer: ReduceSpec,
                     batch_size: int = DEFAULT_BATCH_SIZE) -> KoalaFrame:
        return KoalaFrame(group_by_key(self.ldf, key_col, reducer, batch_size))

    def inner_join(self, right_kf: KoalaFrame, on: str,
                   batch_size: int = DEFAULT_BATCH_SIZE) -> KoalaFrame:
        return KoalaFrame(inner_join(self.ldf, right_kf.ldf, on, batch_size))

    def unique(self) -> KoalaFrame:
        return KoalaFrame(unique(self.ldf))

    def __getattr__(self, name):
        """Delegate anything not defined here to the wrapped LazyFrame.

        A method returning a LazyFrame is re-wrapped so chains stay in
        KoalaFrame-land; terminal methods (collect/sink_csv -> DataFrame/None)
        and plain attributes (columns, schema, ...) pass through unwrapped.
        KoalaFrame arguments are unwrapped to their ldf automatically.
        """
        if name == 'ldf':                       # not set yet (e.g. during unpickle)
            raise AttributeError(name)          # -> avoids infinite recursion
        attr = getattr(self.ldf, name)          # AttributeError if ldf lacks it too
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            args = [a.ldf if isinstance(a, KoalaFrame) else a for a in args]
            kwargs = {k: v.ldf if isinstance(v, KoalaFrame) else v
                      for k, v in kwargs.items()}
            result = attr(*args, **kwargs)
            return KoalaFrame(result) if isinstance(result, pl.LazyFrame) else result

        return wrapper

#
# External API as LazyFrame functions
#

def group_by_key(ldf: pl.LazyFrame, key_col: str, reducer: ReduceSpec,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> pl.LazyFrame:
    """Memory-efficient group-by-then-reduce.

        group_by_key(edges, 'src', reduce('dst', to='n_outlinks', via=len))
    """
    agg_fn, output_type = AGGREGATORS[reducer.aggregator]
    schema = {key_col: pl.String, reducer.output_col: output_type}
    return _streaming_source(
        schema,
        sink_sort=lambda: _sink_sorted(ldf, [key_col, reducer.input_col]),
        iter_records=lambda path: _reduced_values(path, agg_fn),
        to_row=lambda kv: {key_col: kv[0], reducer.output_col: kv[1]},
        batch_size=batch_size,
    )


def inner_join(left: pl.LazyFrame, right: pl.LazyFrame, on: str,
               batch_size: int = DEFAULT_BATCH_SIZE) -> pl.LazyFrame:
    """Inner join, built on ONE group_by_key call.

    The map-reduce insight: a join is a group-by on the join key where, instead
    of summing, each key emits the cross-product of its left rows and its right
    rows. So we

      1. pack each side's row as JSON, prefixed with a side tag 'L'/'R', keyed
         by the join column,
      2. group_by_key(..., via=list) -- ONE call -- to gather both sides of a
         key into a single list,
      3. split that list back into a left-list and a right-list (by the tag),
      4. keep only keys present on BOTH sides (the inner-join gate),
      5. cross-product via two *sequential* explodes (a single parallel
         explode would only zip, and errors on many-to-one), then json_decode.

    Only left carries the join key, so unnest doesn't collide. Right's non-key
    column names are assumed distinct from left's (add a suffix rule to
    generalize).
    """
    lschema = left.collect_schema()
    rschema = right.collect_schema()
    left_dtype = pl.Struct(lschema)                                   # incl. key
    right_dtype = pl.Struct({c: t for c, t in rschema.items() if c != on})

    def pack(ldf, side, cols):
        return ldf.select(
            pl.col(on).cast(pl.String).alias('_key_'),               # sort/group key
            (pl.lit(side) + pl.struct(cols).struct.json_encode()).alias('_row_'))

    combined = pl.concat(
        [pack(left, 'L', list(lschema)),
         pack(right, 'R', [c for c in rschema if c != on])],
        how='vertical')

    # the one group_by_key: co-locate both sides of each key into one list
    grouped = group_by_key(combined, '_key_', reduce('_row_', to='_rows', via=list),
                           batch_size=batch_size)

    def _side(tag):     # elements of _rows starting with tag, with the tag sliced off
        return pl.col('_rows').list.eval(
            pl.element().filter(pl.element().str.starts_with(tag)).str.slice(1))

    return (
        grouped
        .with_columns(_L=_side('L'), _R=_side('R'))
        .filter((pl.col('_L').list.len() > 0) & (pl.col('_R').list.len() > 0))  # inner gate
        .explode('_L').explode('_R')                                 # sequential => cross-product
        .with_columns(pl.col('_L').str.json_decode(left_dtype),
                      pl.col('_R').str.json_decode(right_dtype))
        .drop('_key_', '_rows')                                      # drop group_by_key scaffolding
        .unnest('_L').unnest('_R')
    )


def unique(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop duplicate rows, memory-efficiently, as a thin wrapper over `sort -u`.

    Pack each whole row as a single JSON column, external-sort with `sort -u`
    (so identical rows -- identical JSON -- collapse), then stream the survivors
    back, parsing each JSON line into a row. Same sink -> sort -> stream skeleton
    as group_by_key; the reduce is just "keep one row per identical line".

    Like group_by_key, the sink+sort is deferred to the first collect().
    """
    schema = ldf.collect_schema()
    packed = ldf.select(pl.struct(pl.all()).struct.json_encode().alias('_row_'))
    return _streaming_source(
        schema,
        sink_sort=lambda: _sink_sorted(packed, ['_row_'], dedup=True),
        iter_records=_iter_lines,
        to_row=json.loads,          # each surviving JSON line -> a row dict
    )




def reduce(input_col: str, to: str | None = None, via: Callable = list) -> ReduceSpec:
    """Describe an aggregation.

        kl.reduce('dst', to='n_outlinks', via=len)
        kl.reduce('delta', via=sum)        # 'to' defaults to 'delta_sum'
    """
    if via not in AGGREGATORS:
        raise ValueError(f'unsupported aggregator {via!r}; use len, sum, or list')
    if to is None:
        to = f'{input_col}_{via.__name__}'
    return ReduceSpec(input_col=input_col, output_col=to, aggregator=via)


def _cleanup(state):
    """Remove the sorted temp file if the lazy sink+sort ever ran."""
    path = state.get('sorted_path')
    if path is not None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _sink_sorted(ldf: pl.LazyFrame, cols: list[str], dedup: bool = False) -> str:
    """Execute `ldf`, write `cols` as headerless TSV, external-sort by the first
    column (byte order), and return the path to the sorted file.

    dedup=True adds `sort -u`, which drops duplicate lines by the sort key. Only
    used by unique(), which packs the whole row into a single column so the key
    spans the entire line and dedup is whole-row. (Do NOT combine dedup=True
    with multiple columns -- it would dedup on the first field only.)

    The caller owns the returned file and is responsible for deleting it.
    """
    fd, raw = tempfile.mkstemp(suffix='.tsv')
    os.close(fd)
    fd, srt = tempfile.mkstemp(suffix='.sorted.tsv')
    os.close(fd)

    # sink runs the upstream plan; keep only the key/value columns.
    # quote_style='never' writes values raw so the split-based reader below sees
    # exactly what was written -- important once values are JSON (which contains
    # the '"' and ',' that CSV quoting would otherwise escape). The contract is:
    # keys and values contain no tab or newline.
    ldf.select(cols).sink_csv(raw, include_header=False, separator='\t',
                              quote_style='never')

    # External sort. -k1,1 groups on the key only (preserving value order within
    # a key); -t '\t' so the field separator is the tab; LC_ALL=C for fast,
    # deterministic byte ordering (grouping only needs contiguity, not a
    # numeric/locale collation).
    cmd = ['sort', '-t', '\t', '-k1,1', raw, '-o', srt]
    if dedup:
        cmd.insert(1, '-u')
    subprocess.run(cmd, env={**os.environ, 'LC_ALL': 'C'}, check=True)

    os.remove(raw)          # no longer needed once sorted
    return srt


def _reduced_values(filepath, agg_fn):
    """Stream (key, aggregate) pairs from a file sorted by key.

    Because the file is sorted, equal keys are contiguous, so can use
    itertools.groupby to collapse each run to a single iterator
    """
    with open(filepath) as f:
        rows = (line.rstrip('\n').split('\t', 1) for line in f)
        for key, grp in groupby(rows, key=lambda kv: kv[0]):
            yield key, agg_fn(v for _, v in grp)


def _iter_lines(filepath):
    """Yield the raw (newline-stripped) lines of a file."""
    with open(filepath) as f:
        for line in f:
            yield line.rstrip('\n')


def _streaming_source(schema, sink_sort, iter_records, to_row, batch_size=DEFAULT_BATCH_SIZE):
    """The shared sink -> sort -> stream skeleton for every koala operator.

    Builds a lazy IO source that, on its first execution,

      * calls sink_sort() -> path (deferred and cached, so the returned frame is
        a genuinely lazy node and the sink+sort runs at most once),
      * streams iter_records(path), maps each record to a row dict via to_row,
      * batches those rows into DataFrames of `schema`, honoring the engine's
        projection/predicate pushdown hints, and
      * deletes the temp file when the plan that captured this callback is GC'd.
    """
    state = {'sorted_path': None}

    def batches(with_columns, predicate, n_rows, size):
        if state['sorted_path'] is None:
            state['sorted_path'] = sink_sort()

        def _make_df(rows):
            df = pl.DataFrame(rows, schema=schema)
            if with_columns is not None:            # projection pushdown
                df = df.select(with_columns)
            if predicate is not None:               # predicate pushdown
                df = df.filter(predicate)
            return df

        # Our configured batch_size controls koala's own batching; the engine's
        # `size` hint is only a fallback (the engine passes 100k by default,
        # which would otherwise override a smaller caller-chosen batch_size).
        size = batch_size or size
        batch = []
        for rec in iter_records(state['sorted_path']):
            batch.append(to_row(rec))
            if len(batch) >= size:
                yield _make_df(batch)
                batch = []
        if batch:
            yield _make_df(batch)

    # Tie the temp file's lifetime to the callback the plan captures, so it
    # survives repeated collect()s and is removed when the plan is dropped.
    # `state` is read at finalize time, so it sees the path if the sink ever ran.
    weakref.finalize(batches, _cleanup, state)
    return register_io_source(batches, schema=schema)


if __name__ == '__main__':
    print('RUNNING SMOKE TESTS')

    edges = (
        pl.scan_lines('../data/citeseer-graph.txt')
        .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)'))
        .with_columns(src=pl.col('edge').struct['1'], dst=pl.col('edge').struct['2'])
        .select('src', 'dst')
    )

    def check(name, got, want):
        g = got.sort(got.collect_schema().names()).collect(engine='streaming')
        w = want.sort(want.collect_schema().names()).collect(engine='streaming')
        ok = g.equals(w)
        print(f'[{"ok" if ok else "FAIL"}] {name}: {g.height} rows')
        if not ok:
            print(' koala:', g.head(5))
            print(' polars:', w.head(5))
        return ok

    all_ok = True

    # 1. outdegree: group_by('src').len()  ==  group_by_key(edges,'src',len)
    all_ok &= check(
        'len / n_outlinks',
        group_by_key(edges, 'src', reduce('dst', to='n_outlinks', via=len)),
        edges.group_by('src').len().rename({'len': 'n_outlinks'}),
    )

    # 2. sum: give every edge a weight, sum per dst
    weighted = edges.with_columns(w=(pl.col('src').str.len_chars() % 5 + 1).cast(pl.Float64))
    all_ok &= check(
        'sum / incoming',
        group_by_key(weighted, 'dst', reduce('w', to='incoming', via=sum)),
        weighted.group_by('dst').agg(pl.col('w').sum().alias('incoming')),
    )

    # 3. list: collect all dsts per src (order-independent compare via sort)
    koala_lists = (
        group_by_key(edges, 'src', reduce('dst', to='dsts', via=list))
        .with_columns(dsts=pl.col('dsts').list.sort())
    )
    polars_lists = (
        edges.group_by('src').agg(pl.col('dst').sort().alias('dsts'))
    )
    all_ok &= check('list / dsts', koala_lists, polars_lists)

    # 4. inner_join: the actual pagerank.py join  edges.join(n_outlinks, on='src')
    #    (many-to-one: src repeats in edges, is unique in n_outlinks)
    n_outlinks = group_by_key(edges, 'src', reduce('dst', to='n_outlinks', via=len))
    all_ok &= check(
        'inner_join / edges+n_outlinks (many-to-one)',
        inner_join(edges, n_outlinks, on='src'),
        edges.join(n_outlinks, on='src', how='inner'),
    )

    # 5. inner_join: a made-up many-to-many case + a non-matching key on each side
    L = pl.LazyFrame({'k': ['a', 'a', 'b', 'c'], 'lv': [1, 2, 3, 4]})
    R = pl.LazyFrame({'k': ['a', 'a', 'b', 'd'], 'rv': [10, 20, 30, 40]})
    all_ok &= check(
        'inner_join / many-to-many',
        inner_join(L, R, on='k'),
        L.join(R, on='k', how='inner'),
    )

    # 6. unique: whole-row dedup on a frame with deliberate duplicate rows
    dup = pl.concat([edges, edges.head(100)])          # 100 rows now appear twice
    all_ok &= check('unique / edges+dups', unique(dup), dup.unique())

    print('ALL OK' if all_ok else 'SOME CHECKS FAILED')
