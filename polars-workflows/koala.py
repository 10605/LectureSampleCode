"""
pagerank.py

  .group_by('src').len()
  .unique()
  .join(pagerank_scores.lazy(), left_on='src', right_on='node', how='...')
  .group_by('dst').agg(pl.col('delta').sum().alias('incoming_pr'))
  edges = edges.join(n_outlinks, on='src')


pagerank_mapper.py
  .with_columns(pl.col('edge').map_elements(delta_fn, return_dtype=pl.Float64).alias('delta'))

phrases.py
  .group_by('token').len()
  .group_by('bigram').len()

guineapig.py uses
   Group(inner=None,by=lambda x:x,reducingTo=ReduceToList(),combiningTo=None,retaining=None)
   Join(*joinInputs) # list of Jin(view,by=(lambda x:x),outer=False)
   ReduceTo(baseType,by=lambda accum,val:accum+val)
   - ReduceToCount, ReduceToList, ReduceToSum


proposal:
 group_by_key(ldf, reduce(expr), retain=[c1,...]) -> ldf
 join(left_ldf, right_ldf, on=, left_on= right_on=, how=..., suffix=, retain=[....]) -> ldf
 unique(ldf) # do I need this?

expr is
  reduce=pl.col(name:str).([sum|len]().|)?alias('incoming_pr')
  or kl.reduce(input_col: str, to=output_col: str, via=[sum, len, list | lambda], initial=1...)
"""

import itertools
import tempfile
import subprocess
import os

from collections import namedtuple
from typing import Any, Callable

import polars as pl
from polars.io.plugins import register_io_source


ReduceSpec = namedtuple('ReduceSpec', ['input_col', 'output_col', 'aggregator', 'base_value'])

def reduce(input_col: str, to: str | None = None, via: Callable = list) -> ReduceSpec:
    """Syntax:
         kl.reduce('dst', to='n_outlinks', via='len')
         kl.reduce('value', via=sum)  # 'to' defaults to value_sum
    """
    if to is None:
        to = f'{input_col}_{via}'
    return ReduceSpec(input_col=input_col, output_col=to, aggregator=via, base_value=base_value)

def group_by_key(ldf: pl.LazyFrame, key_col: str, reducer: ReduceSpec) -> pl.LazyFrame:
    """Syntax: 
         group_by(ldf, 'src', kl.reduce('dst', to='n_outlinks', via=len)) 
    """

    # create some temp filenames
    fd, tsv_filepath = tempfile.mkstemp(suffix='.tsv')
    os.close(fd) # will overwrite this below
    fd, sorted_tsv_filepath = tempfile.mkstemp(suffix='.tsv')
    os.close(fd) # will overwrite this below

    # keep the key/value columns
    ldf.select(key_col, reducer.input_col)
    ldf.sink_csv(tsv_filepath, include_header=False, separator='\t')

    # sort the file offline so it's definitely memory efficient
    subprocess.run(
        ['sort', '-k1', tsv_filepath, '-o', sorted_tsv_filepath],
        env={**os.environ, 'LC_ALL': 'C'},
        check=True)

    match reducer.aggregator:
        case len:
            base_type = int
            reduce_fn = lambda accum, _: accum + 1
            output_type = pl.Integer
        case sum:
            base_type = float
            reduce_fn = lambda accum, x: accum + x
            output_type = pl.Float64
        case list:
            base_type = list
            reduce_fn = lambda accum, x: accum + [x]
            output_type = pl.List
        case _:
            raise ValueError('unknown aggregation function: len, sum, and list are supported')
    
    schema = {key_col:pl.String, reducer.output_col: output_type}

    def reduced_value_batches(with_columns, predicate, n_rows, batch_size):
        def _make_df(rows):
            df = pl.DataFrame(rows, schema=schema)
            # handle the pushdown hints
            if with_columns is not None:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)
            return df

        batch_size = batch_size or 100_000
        batch = []
        for key, reduced_value in _reduced_values(sorted_tsv_filepath, base_type, reduce_fn):
            batch.append({key_col: key, reducer.output_col: reduced_value})
            if len(batch) >= batch_size:
                yield _make_df(batch)
        if batch:
            yield _make_df(batch)

        .......

    return register_io_source(


def _reduced_values(filepath, base_type, reduce_fn):
    pairs = _key_value_pairs(filepath)
    accum = base_type()
    last_key = None
    for key, value in pairs:
        if last_key is None or key == last_key:
            accum = reduce_fn(accum, base_type(valye))
            last_key = key
        else:
            # output the last run's reduced value
            yield last_key, accum
            # reset the accumulator and last_key
            accum = base_type()
            last_key = key
            # restart the loop with this value 
            pairs = itertools.chain([(key, value)], pairs)
    # yield the final run, if there was one
    if last_key != None:
        yield key, accum

def _key_value_pairs(filepath):
    for line in open(filepath):
        key, value = line.rstrip().split('\t')
        yield key, value

       

if __name__ == '__main__':

    edges = (pl.scan_lines('../data/citeseer-graph.txt')
           .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)'))
           .with_columns(
               src=pl.col('edge').struct['1'],
               dst=pl.col('edge').struct['2'])
           .select('src', 'dst')
           )
    print(edges.collect(engine='streaming'))
    print(group_by(edges, 'src', reducer=reduce('dst', len)))
