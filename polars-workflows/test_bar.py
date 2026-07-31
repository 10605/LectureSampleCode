"""pytest suite for bar.py -- external sort and group-by over kv tsv files.

Both operations are checked against the obvious in-memory version (sorted, and
a dict of lists) as the oracle, so the tests say what the methods *mean* rather
than restating the merge-sort.  Run with:  pytest test_bar.py

The interesting cases are the ones where the spilling actually happens: batch
sizes that don't divide the input, enough spill files to force several fan-in
rounds, duplicate keys straddling a batch boundary, empty input, a file with no
trailing newline, and values whose encoding contains a tab or a newline.
"""
import builtins
import glob
import gc
import itertools
import os
import random
import tempfile

import polars as pl
import pytest

import bar

# (batch_size, fanin): 1/2 forces many rounds of merging, 10_000 keeps
# everything in a single spill file and never enters the fan-in loop.
CONFIGS = [(1, 2), (2, 2), (3, 4), (7, 3), (100, 16), (10_000, 16)]


def write_pairs(path, pairs, encode=bar.DEFAULT_ENCODER):
    with open(path, 'w') as fp:
        for key, val in pairs:
            fp.write(encode(key) + '\t' + encode(val) + '\n')
    return path


def read_pairs(path, decode=bar.DEFAULT_DECODER):
    return list(bar._pairs_from(path, decode))


@pytest.fixture
def paths(tmp_path):
    """An (input, output) pair of paths in a per-test directory."""
    return str(tmp_path / 'in.tsv'), str(tmp_path / 'out.tsv')


def grouping_bar(sorted_path, **kwargs):
    """A PolarBar that trusts sorted_path -- which the caller wrote in key order.

    group_and_aggregate_by_key only accepts paths this PolarBar knows to be
    sorted, and these tests build their input by hand rather than via
    sort_kv_pairs, so the promise has to be made explicitly.
    """
    polar_bar = bar.PolarBar(**kwargs)
    polar_bar._mark_sorted(sorted_path)
    return polar_bar


def sample_pairs(n, n_keys=None, seed=0):
    """n pairs in shuffled order; few keys => lots of duplicates to group."""
    rng = random.Random(seed)
    n_keys = n_keys if n_keys is not None else n
    return [(f'k{rng.randrange(n_keys):04d}', rng.randrange(1000)) for _ in range(n)]


# ---------------------------------------------------------------- sorting

@pytest.mark.parametrize('batch_size,fanin', CONFIGS)
@pytest.mark.parametrize('n', [0, 1, 2, 5, 100])
def test_sort_matches_sorted(paths, batch_size, fanin, n):
    in_path, out_path = paths
    pairs = sample_pairs(n)
    write_pairs(in_path, pairs)

    bar.PolarBar(sort_batch_size=batch_size, sort_fanin=fanin).sort_kv_pairs(in_path, out_path)

    assert read_pairs(out_path) == sorted(pairs, key=bar._BY_KEY)


@pytest.mark.parametrize('batch_size,fanin', CONFIGS)
def test_sort_is_stable_across_duplicate_keys(paths, batch_size, fanin):
    """Equal keys keep their input order -- so a value can be used as a tiebreak.

    Duplicates here span batch boundaries, which is where a naive merge loses
    stability.
    """
    in_path, out_path = paths
    pairs = [('dup', i) for i in range(50)]
    random.Random(1).shuffle(pairs)
    pairs = [('a', -1)] + pairs + [('z', -2)]
    write_pairs(in_path, pairs)

    bar.PolarBar(sort_batch_size=batch_size, sort_fanin=fanin).sort_kv_pairs(in_path, out_path)

    assert read_pairs(out_path) == sorted(pairs, key=bar._BY_KEY)


def test_sort_of_sorted_input_is_a_noop(paths):
    in_path, out_path = paths
    pairs = sorted(sample_pairs(60), key=bar._BY_KEY)
    write_pairs(in_path, pairs)

    bar.PolarBar(sort_batch_size=7, sort_fanin=2).sort_kv_pairs(in_path, out_path)

    assert read_pairs(out_path) == pairs


@pytest.mark.parametrize('value', [
    'plain',
    'has\ttab',            # a literal tab would break the split on read
    'has\nnewline',        # a literal newline would look like a record boundary
    "quote'and\"quote",
    '',
    ('tuple', 1, (2, 3)),
    [1, 2, 3],
    {'a': 1},
    3.5,
    None,
    True,
])
def test_values_round_trip(paths, value):
    in_path, out_path = paths
    pairs = [('k', value), ('j', value)]
    write_pairs(in_path, pairs)

    bar.PolarBar(sort_batch_size=1, sort_fanin=2).sort_kv_pairs(in_path, out_path)

    assert read_pairs(out_path) == [('j', value), ('k', value)]


def test_custom_encoder_decoder(paths):
    """With str/int, keys decode to ints and so sort numerically, not as text."""
    in_path, out_path = paths
    pairs = [(k, k * 10) for k in [2, 100, 30, 4]]
    write_pairs(in_path, pairs, encode=str)

    bar.PolarBar(encode=str, decode=int, sort_batch_size=2, sort_fanin=2).sort_kv_pairs(
        in_path, out_path)

    assert read_pairs(out_path, decode=int) == [(2, 20), (4, 40), (30, 300), (100, 1000)]


def test_final_line_without_newline_is_not_truncated(paths):
    in_path, out_path = paths
    with open(in_path, 'w') as fp:
        fp.write("'b'\t2\n'a'\t1")       # no trailing newline

    bar.PolarBar(sort_batch_size=1, sort_fanin=2).sort_kv_pairs(in_path, out_path)

    assert read_pairs(out_path) == [('a', 1), ('b', 2)]


# ------------------------------------------------- grouping / aggregation

@pytest.mark.parametrize('n,n_keys', [(0, 1), (1, 1), (100, 1), (100, 7), (100, 100)])
def test_group_and_aggregate_matches_dict_oracle(paths, n, n_keys):
    in_path, out_path = paths
    pairs = sorted(sample_pairs(n, n_keys), key=bar._BY_KEY)
    write_pairs(in_path, pairs)

    grouping_bar(in_path).group_and_aggregate_by_key(in_path, out_path, list)

    expected = [(k, [v for _, v in g])
                for k, g in itertools.groupby(pairs, key=bar._BY_KEY)]
    assert read_pairs(out_path) == expected


@pytest.mark.parametrize('agg', [sum, min, max, list, sorted,
                                 lambda values: sum(1 for _ in values)])
def test_aggregators(paths, agg):
    in_path, out_path = paths
    pairs = sorted(sample_pairs(50, n_keys=5), key=bar._BY_KEY)
    write_pairs(in_path, pairs)

    grouping_bar(in_path).group_and_aggregate_by_key(in_path, out_path, agg)

    groups = {k: [v for _, v in g] for k, g in itertools.groupby(pairs, key=bar._BY_KEY)}
    assert dict(read_pairs(out_path)) == {k: agg(vs) for k, vs in groups.items()}


def test_sort_then_count_is_a_word_count(tmp_path):
    """The two operations composed: the intended end-to-end use."""
    words = 'the quick brown fox jumps over the lazy dog the fox'.split()
    unsorted, srt, counts = (str(tmp_path / n) for n in ['u.tsv', 's.tsv', 'c.tsv'])
    write_pairs(unsorted, [(w, 1) for w in words])

    polar_bar = bar.PolarBar(sort_batch_size=2, sort_fanin=2)
    polar_bar.sort_kv_pairs(unsorted, srt)
    polar_bar.group_and_aggregate_by_key(srt, counts, sum)

    assert dict(read_pairs(counts)) == {'the': 3, 'fox': 2, 'quick': 1, 'brown': 1,
                                        'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}


# ------------------------------------------------------ the sorted guard

def test_grouping_rejects_a_file_not_known_to_be_sorted(paths):
    in_path, out_path = paths
    write_pairs(in_path, sample_pairs(20))

    with pytest.raises(ValueError, match='not known to be sorted'):
        bar.PolarBar().group_and_aggregate_by_key(in_path, out_path, list)


def test_sink_does_not_claim_sortedness_it_was_not_promised(paths):
    """An unsorted sink must not satisfy the guard -- else group_by silently
    collapses only *adjacent* duplicates and dedup leaves duplicates behind.
    """
    in_path, out_path = paths
    unsorted = pl.LazyFrame({'k': ['b', 'a', 'b'], 'v': ['1', '2', '3']})
    polar_bar = bar.PolarBar(encode=str, decode=lambda x: x)
    polar_bar.sink_kv_pairs(unsorted, in_path, key_col='k', val_col='v')

    with pytest.raises(ValueError, match='not known to be sorted'):
        polar_bar.group_and_aggregate_by_key(in_path, out_path, bar.first)


def test_sink_of_an_already_sorted_frame_can_be_grouped(paths):
    in_path, out_path = paths
    srt = pl.LazyFrame({'k': ['a', 'b', 'b'], 'v': ['1', '2', '3']})
    polar_bar = bar.PolarBar(encode=str, decode=lambda x: x)
    polar_bar.sink_kv_pairs(srt, in_path, key_col='k', val_col='v', is_sorted=True)

    polar_bar.group_and_aggregate_by_key(in_path, out_path, bar.first)

    assert read_pairs(out_path, decode=str) == [('a', '1'), ('b', '2')]


def test_sort_then_dedup_keeps_one_row_per_key(tmp_path):
    """The dedup path from pagerank_bar: sink -> sort -> group with `first`."""
    nodes = ['c', 'a', 'b', 'a', 'c', 'a']
    sunk, srt, out = (str(tmp_path / n) for n in ['n.tsv', 's.tsv', 'o.tsv'])
    polar_bar = bar.PolarBar(encode=str, decode=lambda x: x, sort_batch_size=2, sort_fanin=2)
    polar_bar.sink_kv_pairs(pl.LazyFrame({'node': nodes, 'score': ['1.0'] * len(nodes)}),
                            sunk, key_col='node', val_col='score')
    polar_bar.sort_kv_pairs(sunk, srt)
    polar_bar.group_and_aggregate_by_key(srt, out, bar.first)

    assert read_pairs(out, decode=str) == [('a', '1.0'), ('b', '1.0'), ('c', '1.0')]


# --------------------------------------------------------- resource bounds

def test_spill_files_are_deleted(paths):
    """Every temp file the sort creates is unlinked once it is unreferenced."""
    in_path, out_path = paths
    write_pairs(in_path, sample_pairs(200))
    spills = os.path.join(tempfile.gettempdir(), '*.spill')
    before = set(glob.glob(spills))

    bar.PolarBar(sort_batch_size=3, sort_fanin=2).sort_kv_pairs(in_path, out_path)
    gc.collect()

    assert set(glob.glob(spills)) == before


@pytest.mark.parametrize('fanin', [2, 4, 8])
def test_at_most_fanin_files_open_at_once(paths, monkeypatch, fanin):
    """sort_fanin caps concurrently open files -- the point of the fan-in loop.

    A merge that opened every spill file at once would still produce the right
    answer, so only a resource check can catch it.  Allowance is fanin spill
    files plus the input and the output.
    """
    in_path, out_path = paths
    write_pairs(in_path, sample_pairs(500))

    live, high_water = [], 0
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        nonlocal live, high_water
        fp = real_open(*args, **kwargs)
        live = [f for f in live if not f.closed] + [fp]
        high_water = max(high_water, len(live))
        return fp

    monkeypatch.setattr(builtins, 'open', counting_open)
    bar.PolarBar(sort_batch_size=5, sort_fanin=fanin).sort_kv_pairs(in_path, out_path)

    assert high_water <= fanin + 2
