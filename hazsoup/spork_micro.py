from __future__ import annotations


import collections
from dataclasses import dataclass
from functools import reduce
import json
import subprocess
from typing import Callable, Iterable

import reduce_util as ru

class Context:
    """Like a SparkContext.
    """


    def textFile(self, fileName: str) -> RDD:
        """A local text file as an RDD.

        The RDD has one string per line.
        """
        return TextFile(self, fileName)

    def asRDD(self, xs: list) -> RDD:
        """Any python list as an RDD.
        """

        return ListRDD(self, xs)

class RDD:
    context: Context

    def _contents(self):
        """Generator producing each thing in the RDD.
        """
        assert False, 'abstract method called'

    def map(self, fn: Callable) -> RDD:
        """Transform each element of an RDD.
        """
        return Mapper(context=self.context, upstream=self, map_fn=fn)

    def flatMap(self, fn: Callable) -> RDD:
        """Transform elements into lists and append the lists.
        """
        return FlatMapper(context=self.context, upstream=self, map_fn=fn)

    def reduceByKey(self, fn: Callable, in_memory=False) -> RDD:
        """Reduce a key-value RDD.
        """
        if in_memory:
            return MemoryKVReducer(
                context=self.context, upstream=self, reduce_fn=fn)        
        else:
            return DiskKVReducer(
                context=self.context, upstream=self, reduce_fn=fn)

    def collect(self) -> list:
        """An action that converts an RDD to a list.
        """
        return list(self._contents())
    
    def take(self, n: int) -> list:
        """An action that returns the first n elements of an RDD.
        """
        buffer = []
        for i, x in enumerate(self._contents()):
            if i > n: break
            buffer.append(x)
        return buffer

# generally not constructed by end users

@dataclass
class TextFile(RDD):
    context: Context
    fileName: str

    def _contents(self):
        for line in open(self.fileName):
            yield line
    
@dataclass
class ListRDD(RDD):
    context: Context
    contents: list

    def _contents(self):
        for x in self.contents:
            yield x

@dataclass
class Mapper(RDD):
    context: Context
    upstream: RDD
    map_fn: Callable

    def _contents(self):
        for x in self.upstream._contents():
            yield self.map_fn(x)

@dataclass
class FlatMapper(RDD):
    context: Context
    upstream: RDD
    map_fn: Callable

    def _contents(self):
        for x in self.upstream._contents():
            for y in self.map_fn(x):
                yield y

@dataclass
class MemoryKVReducer(RDD):
    context: Context
    upstream: RDD
    reduce_fn: Callable

    def _contents(self):

        grouped_by_key = collections.defaultdict(list)
        for key, value in self.upstream._contents():
            grouped_by_key[str(key)].append(value)
        
        for key, values in grouped_by_key.items():
            yield key, reduce(self.reduce_fn, values)


@dataclass
class DiskKVReducer(RDD):
    context: Context
    upstream: RDD
    reduce_fn: Callable

    def _contents(self):
        """Return the contents of the upstream RDD, sorted by key.

        Upstream RDD must produce pairs.

        """
        sort_input_buf = f'/tmp/mapout.tsv'
        sort_output_buf = f'/tmp/sortout.tsv'

        with open(sort_input_buf, 'w') as fp:
            for (key, value) in self.upstream._contents():
                fp.write(ru.kv_to_line(key, value))

        sort_cmd = f'LC_ALL=C sort -k1 -o {sort_output_buf} {sort_input_buf}'
        subprocess.check_call(sort_cmd, shell=True)

        def pair_generator():
            for line in open(sort_output_buf):
                yield ru.kv_from_line(line)

        for key, values in ru.ReduceReady(pair_generator()):
            yield key, reduce(self.reduce_fn, values)
