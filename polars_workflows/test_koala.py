"""pytest suite for koala.py -- the checks that used to live in koala's
__main__ smoke block, plus guards for merge_join's contract.

Every join/reduce is validated against the equivalent polars operation as the
oracle, so any graph works as input. Run with:  pytest test_koala.py
"""
import pathlib

import polars as pl
import pytest

import koala as kl

DATA = pathlib.Path(__file__).parent.parent / 'data' / 'citeseer-graph.txt'


@pytest.fixture(scope='module')
def edges():
    """(src, dst) edges from the citeseer graph, lazy."""
    if not DATA.exists():
        pytest.skip(f'missing test graph {DATA}')
    return (pl.scan_lines(str(DATA))
            .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)'))
            .with_columns(src=pl.col('edge').struct['1'], dst=pl.col('edge').struct['2'])
            .select('src', 'dst'))


@pytest.fixture(scope='module')
def n_outlinks(edges):
    """(src, n_outlinks) -- unique keys, used as the 'one' side of joins."""
    return kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len))


def assert_frames_equal(got, want):
    """Order-independent equality of two lazy frames (sort on all columns)."""
    g = got.sort(got.collect_schema().names()).collect(engine='streaming')
    w = want.sort(want.collect_schema().names()).collect(engine='streaming')
    assert g.equals(w), f'\nkoala:\n{g.head()}\npolars:\n{w.head()}'


# --- group_by_key reducers ---

def test_group_by_key_len(edges):
    assert_frames_equal(
        kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len)),
        edges.group_by('src').len().rename({'len': 'n_outlinks'}))


def test_group_by_key_sum(edges):
    w = edges.with_columns(w=(pl.col('src').str.len_chars() % 5 + 1).cast(pl.Float64))
    assert_frames_equal(
        kl.group_by_key(w, 'dst', kl.reduce('w', to='incoming', via=sum)),
        w.group_by('dst').agg(pl.col('w').sum().alias('incoming')))


def test_group_by_key_list(edges):
    assert_frames_equal(
        kl.group_by_key(edges, 'src', kl.reduce('dst', to='dsts', via=list))
          .with_columns(dsts=pl.col('dsts').list.sort()),
        edges.group_by('src').agg(pl.col('dst').sort().alias('dsts')))


# --- inner_join (general, JSON list-gather) ---

def test_inner_join_many_to_one(edges, n_outlinks):
    assert_frames_equal(
        kl.inner_join(edges, n_outlinks, on='src'),
        edges.join(n_outlinks, on='src', how='inner'))


def test_inner_join_many_to_many():
    L = pl.LazyFrame({'k': ['a', 'a', 'b', 'c'], 'lv': [1, 2, 3, 4]})
    R = pl.LazyFrame({'k': ['a', 'a', 'b', 'd'], 'rv': [10, 20, 30, 40]})
    assert_frames_equal(kl.inner_join(L, R, on='k'), L.join(R, on='k', how='inner'))


# --- merge_join (single-value, no JSON; values come back as strings) ---

def test_merge_join_many_to_one(edges, n_outlinks):
    assert_frames_equal(
        kl.merge_join(edges, n_outlinks, on='src').with_columns(pl.col('n_outlinks').cast(pl.Int64)),
        edges.join(n_outlinks, on='src', how='inner'))


def test_merge_join_many_to_many():
    L = pl.LazyFrame({'k': ['a', 'a', 'b', 'c'], 'lv': [1, 2, 3, 4]})
    R = pl.LazyFrame({'k': ['a', 'a', 'b', 'd'], 'rv': [10, 20, 30, 40]})
    assert_frames_equal(
        kl.merge_join(L, R, on='k').with_columns(pl.col('lv').cast(pl.Int64), pl.col('rv').cast(pl.Int64)),
        L.join(R, on='k', how='inner'))


def test_merge_join_float_roundtrip():
    """The property pagerank relies on: f64 values survive str <-> f64."""
    L = pl.LazyFrame({'k': ['a', 'b'], 'x': [0.1, 2.0 / 3.0]})
    R = pl.LazyFrame({'k': ['a', 'b'], 'y': [1.5, -3.25]})
    assert_frames_equal(
        kl.merge_join(L, R, on='k').with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64)),
        L.join(R, on='k', how='inner'))


def test_merge_join_rejects_multi_value(edges, n_outlinks):
    """merge_join requires exactly one non-key column per side."""
    two_value = edges.with_columns(extra=pl.lit(1))      # left now has dst AND extra
    with pytest.raises(ValueError, match='one non-key column'):
        kl.merge_join(two_value, n_outlinks, on='src')


# --- unique ---

def test_unique(edges):
    dup = pl.concat([edges, edges.head(100)])            # 100 rows appear twice
    assert_frames_equal(kl.unique(dup), dup.unique())
