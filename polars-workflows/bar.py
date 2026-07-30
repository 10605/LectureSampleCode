"""Memory-efficient group_by_key operations, used for off-loading
inefficient operations in Polars.
"""

import ast
import itertools
import heapq
import os
import tempfile
import weakref

from typing import Any, Callable, Iterable, Optional

EncoderFn =  Optional[Callable[[Any],str]]
DecoderFn = Optional[Callable[[str],Any]]

DEFAULT_ENCODER = repr
DEFAULT_DECODER = ast.literal_eval

BY_KEY=lambda kv:kv[0]

def _pairs_from(input_tsv_path, decode):
    """Iterator over the key-value pairs in a key-value tsv file.
    """
    with open(input_tsv_path) as fp:
        for line in fp:
            key, val = line.rstrip('\n').split('\t', 1)
            yield decode(key), decode(val)

class _SortSpillFile:
    """One sorted subset of a set of key-value pairs, written to a temp file.

    Temp file is deleted when unreferenced.  
    """
    def __init__(self, batch: Iterable, encode: EncoderFn, decode: DecoderFn):
        fd, self.name = tempfile.mkstemp(suffix='.spill')
        self.encode = encode
        self.decode = decode
        with os.fdopen(fd, 'w') as fp:
            for key, value in sorted(batch, key=BY_KEY):
                fp.write(self.encode(key) + '\t' + self.encode(value) + '\n')
        weakref.finalize(self, os.unlink, self.name)

    def __iter__(self) -> Iterable:
        yield from _pairs_from(self.name, self.decode)


class PolarBar:
    
    def __init__(self, encode: EncoderFn = None, decode: DecoderFn = None, sort_batch_size=1024, sort_fanin=16):
        self.encode = encode or DEFAULT_ENCODER
        self.decode = decode or DEFAULT_DECODER
        self.sort_batch_size = sort_batch_size
        self.sort_fanin = sort_fanin

    def sort_kv_pairs(self, input_tsv_path, output_tsv_path):
        """ Merge-sort key value pairs in the file, only holding sort_batch_size items in memory.
        """
        spill_files = [_SortSpillFile(batch, self.encode, self.decode)
                       for batch in itertools.batched(_pairs_from(input_tsv_path, self.decode), self.sort_batch_size)]
        while len(spill_files) > self.sort_fanin:
            spill_files = [_SortSpillFile(heapq.merge(*file_batch, key=BY_KEY), self.encode, self.decode)
                           for file_batch in itertools.batched(iter(spill_files), self.sort_fanin)]
        final_sort = heapq.merge(*spill_files, key=BY_KEY)

        # write to the output file
        with open(output_tsv_path, 'w') as fp:
            for key, val in final_sort:
                fp.write(self.encode(key) + '\t' + self.encode(val) + '\n') 

    def group_and_aggregate_by_key(self, input_tsv_path, output_tsv_path, agg: Callable[[Iterable],Any]):
        """ Group the items by key, and aggregate the results.  Items must be sorted by key.

        Aggregator should consume an iterable, so things sum, list, min, max are ok.
        To count pass in "lambda values: sum(1 for _ in values)".
        """
        with open(output_tsv_path, 'w') as fp:
            for key, group in itertools.groupby(_pairs_from(input_tsv_path, self.decode), key=BY_KEY):
                group_values = (v for _, v in group)
                fp.write(self.encode(key) + '\t' + self.encode(agg(group_values)) + '\n')
            
