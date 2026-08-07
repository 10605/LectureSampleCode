"""Memory-efficient group_by_key operations, used for off-loading
inefficient operations in Polars.
"""

import ast
import itertools
import heapq
import operator
import os
import shutil
import tempfile
import polars as pl

from math import sqrt, ceil
from tqdm import tqdm
from typing import Any, Callable, Iterable, Optional


EncoderFn =  Optional[Callable[[Any],str]]
DecoderFn = Optional[Callable[[str],Any]]

DEFAULT_ENCODER = repr
DEFAULT_DECODER = ast.literal_eval

# itemgetter rather than a lambda: these run once per record per pass, and the
# C-level version skips a Python frame each time.
_BY_KEY = operator.itemgetter(0)
_BY_KEY_THEN_SOURCE = operator.itemgetter(0, 1)
_PAIR = operator.itemgetter(1)

#
# useful aggregator names
#

def count(group: Iterable):
    """Lazy way to count elements in an iterable.
    """
    return sum(1 for _ in group)

def first(group: Iterable):
    """First element.
    """
    return next(group)

def _pairs_from(input_tsv_path, decode):
    """Iterator over the key-value pairs in a key-value tsv file.
    """
    with open(input_tsv_path) as fp:
        for line in fp:
            key, val = line.rstrip('\n').split('\t', 1)
            yield decode(key), decode(val)

def _keyed_lines_from(input_tsv_path, decode):
    """Iterator over (key, line) for a key-value tsv file.

    Only the key is decoded; the line is carried along verbatim, so a sort can
    move a record from file to file without ever decoding its value or
    re-encoding either field.  A missing final newline is supplied here, so a
    line is always safe to write back out as-is.
    """
    with open(input_tsv_path) as fp:
        for line in fp:
            key = decode(line[:line.index('\t')])
            yield key, line if line[-1] == '\n' else line + '\n'

class _SpillFileIndex:
    """One generation of sorted runs: a temp directory, deleted as a unit.

    Runs hold the (key, line) records of _keyed_lines_from, and are iterated in
    the order they were added -- which is what keeps the sort stable, since
    heapq.merge breaks ties by iterator order.  Use it as a context manager, or
    call close() yourself, to remove the directory.
    """
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix='spill-')
        self.names = []

    def add(self, records: Iterable):
        """Write one run.  The records must ALREADY be in key order: either
        sorted() in memory, or heapq.merge()d off of earlier runs.
        """
        name = os.path.join(self.dir, f'run-{len(self.names):05d}.tsv')
        with open(name, 'w') as fp:
            fp.writelines(map(_PAIR, records))
        self.names.append(name)

    def __iter__(self) -> Iterable:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.names = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class PolarBar:
    
    # a merge costs only O(log fanin) per record, so a wide fan-in is nearly
    # free and saves whole read/write passes over the data; the cap is open
    # file descriptors, one per run in the final merge.
    def __init__(self, encode: EncoderFn = None, decode: DecoderFn = None, sort_batch_size=1024, sort_fanin=64):
        self.encode = encode or DEFAULT_ENCODER
        self.decode = decode or DEFAULT_DECODER
        self.sort_batch_size = sort_batch_size
        self.sort_fanin = sort_fanin
        self.sorted_kv_pair_paths = set()

    def _check_sorted(self, path):
        if not path in self.sorted_kv_pair_paths:
            raise ValueError(f'{path} is not known to be sorted')

    def _mark_sorted(self, path):
        self.sorted_kv_pair_paths.add(path)

    def sink_kv_pairs(self, ldf: pl.LazyFrame, output_tsv_path, key_col: str, val_col: str, is_sorted=False):
        kv_df = ldf.select([key_col,val_col])
        kv_df.sink_csv(output_tsv_path, separator='\t', include_header=False)
        # only the caller knows if the frame was already sorted by key
        if is_sorted:
            self._mark_sorted(output_tsv_path)

    def sort_kv_pairs(self, input_tsv_path, output_tsv_path):
        """ Merge-sort key value pairs in the file, only holding sort_batch_size items in memory.

        Records move as whole lines, so each pass costs one key decode per
        record and no encoding at all.  heapq.merge breaks ties by iterator
        order, which keeps the sort stable.
        """
        def streams(run_names):
            return [_keyed_lines_from(name, self.decode) for name in run_names]

        print('spilling...')
        runs = _SpillFileIndex()
        try:
            for batch in tqdm(itertools.batched(
                    _keyed_lines_from(input_tsv_path, self.decode), self.sort_batch_size)):
                # sorted() is stable, so equal keys keep their input order
                runs.add(sorted(batch, key=_BY_KEY))

            if self.sort_fanin < 1:
                # scale fan_in to finish in one pass
                self.sort_fanin = ceil(sqrt(len(runs)))

            while len(runs) > self.sort_fanin:
                print(f'merging {len(runs)} files {len(runs)/self.sort_fanin} merges')
                merged = _SpillFileIndex()
                for run_batch in tqdm(itertools.batched(iter(runs), self.sort_fanin)):
                    merged.add(heapq.merge(*streams(run_batch), key=_BY_KEY))
                # add() consumed each merge as it went, so this generation is done
                runs.close()
                runs = merged

            print(f'merging {len(runs)} files....')
            final_sort = heapq.merge(*streams(runs), key=_BY_KEY)

            # write to the output file
            with open(output_tsv_path, 'w') as fp:
                fp.writelines(map(_PAIR, final_sort))
        finally:
            runs.close()
        self._mark_sorted(output_tsv_path)

    def group_and_aggregate_by_key(self, input_tsv_path, output_tsv_path, agg: Callable[[Iterable],Any]):
        """Group the items in a kv tsv by key, and aggregate the results.  Items must be sorted by key.

        Aggregator should consume an iterable, so things sum, list, min, max are ok.
        To count pass in "lambda values: sum(1 for _ in values)".
        """
        self._check_sorted(input_tsv_path)
        with open(output_tsv_path, 'w') as fp:
            for key, group in itertools.groupby(_pairs_from(input_tsv_path, self.decode), key=_BY_KEY):
                group_values = (v for _, v in group)
                fp.write(self.encode(key) + '\t' + self.encode(agg(group_values)) + '\n')
        self._mark_sorted(output_tsv_path)

    def join_by_key(self, input_tsv_path1, input_tsv_path2, output_tsv_path, combine_vals: Callable[[Any, Any],Any]):
        self._check_sorted(input_tsv_path1)
        self._check_sorted(input_tsv_path2)

        def tag_values(tag, path):
            for key, val in _pairs_from(path, self.decode):
                yield key, tag, val

        merged_pairs = heapq.merge(
            tag_values('L', input_tsv_path1),
            tag_values('R', input_tsv_path2),
            key=_BY_KEY_THEN_SOURCE)

        with open(output_tsv_path, 'w') as fp:
            for key, group in itertools.groupby(merged_pairs, key=_BY_KEY):
                # values from the left come first
                left = []
                for _, val_src, val in group:
                    if val_src == 'L': 
                        left.append(val)
                    else:
                        # values from the right needs to paired with all the left
                        # values
                        for left_val in left:
                            combined_val = combine_vals(left_val, val)
                            fp.write(self.encode(key) + '\t' + self.encode(combined_val) + '\n')

        self._mark_sorted(output_tsv_path)

