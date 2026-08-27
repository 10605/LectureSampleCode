"""pytest suite for tardigrade.py -- the streaming LazyFrame.

Every operation is validated against the equivalent builtin (sorted, slicing,
map/filter) as the oracle, so the tests say what the method *means* rather than
restating its implementation.  Run with:  pytest test_tardigrade.py

The interesting cases here are the ones that broke before: empty frames, frames
shorter than k, batch boundaries in the external sort, the promise that nothing
is consumed until collect(), and the resource bounds sort() claims -- memory
proportional to batch_size, and at most max_fanin files open at once.

Rows are lines of text, as produced by scan_lines and consumed by sink.
"""
import collections
import gc
from collections import deque
import os
import random
import tracemalloc

import pytest

import tardigrade as tg

BATCH_SIZES = [1, 2, 3, 7, 100, 10_000]      # 1 and 2 stress the merge;
                                             # 10_000 puts everything in one run


def peak_bytes(fn):
    """Peak allocation while running fn().

    Stops tracing even if fn raises -- otherwise one failing memory test leaks
    tracing state into the next one, whose peak then includes this test's.
    """
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.fixture
def rows():
    """1000 lines of digits, with many duplicates so ties exercise stability."""
    rng = random.Random(42)
    return [f'{rng.randrange(50)}\n' for _ in range(1000)]


# --- sort ---

@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_sort(rows, batch_size):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=batch_size).collect() == sorted(rows)


@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_sort_reverse(rows, batch_size):
    assert (tg.LazyFrame(iter(rows)).sort(batch_size=batch_size, reverse=True).collect()
            == sorted(rows, reverse=True))


def test_sort_key(rows):
    """Numeric order, not lexical -- the reason key= exists."""
    assert (tg.LazyFrame(iter(rows)).sort(batch_size=13, key=int).collect()
            == sorted(rows, key=int))


@pytest.mark.parametrize('n', [0, 1, 9, 10, 11])
def test_sort_batch_boundaries(n):
    """n just below / at / above a batch multiple -- the classic off-by-one."""
    lines = [f'{i}\n' for i in reversed(range(n))]
    assert tg.LazyFrame(iter(lines)).sort(batch_size=5).collect() == sorted(lines)


def test_sort_empty():
    assert tg.LazyFrame(iter([])).sort().collect() == []


def test_sort_is_stable(rows):
    """Ties keep input order: sorted() is stable and merge prefers earlier runs."""
    tagged = [f'{v.strip()} {i}\n' for i, v in enumerate(rows)]
    key = lambda line: int(line.split()[0])
    assert (tg.LazyFrame(iter(tagged)).sort(batch_size=7, key=key).collect()
            == sorted(tagged, key=key))


def test_sort_adds_missing_newline():
    """Rows are records; an unterminated one would silently merge with the next."""
    assert tg.LazyFrame(iter(['b', 'a'])).sort(batch_size=1).collect() == ['a\n', 'b\n']


def test_sort_rejects_non_text_rows():
    """Serializing is the caller's job -- spilling only handles text."""
    with pytest.raises(ValueError, match='must be strings'):
        tg.LazyFrame(iter([3, 1, 2])).sort(batch_size=2).collect()


# --- sort: resource bounds ---

def test_sort_peak_memory_tracks_batch_size():
    """The point of spilling: RAM follows batch_size, not frame size."""
    def drain(batch_size):
        lines = (f'{i}\n' for i in range(200_000))
        return lambda: deque(tg.LazyFrame(lines).sort(batch_size=batch_size).iter, maxlen=0)

    assert peak_bytes(drain(1_000)) < peak_bytes(drain(200_000)) / 2


@pytest.mark.parametrize('max_fanin', [2, 4, 16])
def test_sort_cascades_when_runs_exceed_fanin(rows, max_fanin):
    """batch_size=3 over 1000 rows is 334 runs, so this needs several passes."""
    assert (tg.LazyFrame(iter(rows)).sort(batch_size=3, max_fanin=max_fanin).collect()
            == sorted(rows))


@pytest.mark.skipif(not os.path.isdir('/dev/fd'), reason='needs /dev/fd to count fds')
def test_merge_holds_at_most_fanin_files_open(rows):
    """Why the cascade exists: a single pass over N runs would open N files."""
    fanin = 4
    base = len(os.listdir('/dev/fd'))
    peak = 0
    for _ in tg.LazyFrame(iter(rows)).sort(batch_size=3, max_fanin=fanin).iter:
        peak = max(peak, len(os.listdir('/dev/fd')) - base)
    assert peak <= fanin + 3, f'{peak} files open at once, expected <= {fanin}'


def test_spill_files_are_deleted(rows, monkeypatch):
    """Runs are temp files; nothing should survive the sort."""
    names = []
    init = tg._SpillFile.__init__

    def spy(self, rows):
        init(self, rows)
        names.append(self.name)

    monkeypatch.setattr(tg._SpillFile, '__init__', spy)
    tg.LazyFrame(iter(rows)).sort(batch_size=3, max_fanin=4).collect()
    gc.collect()
    assert names, 'no runs were spilled'
    assert not [n for n in names if os.path.exists(n)], 'spill files leaked'


# --- unique ---

def test_unique(rows):
    assert tg.LazyFrame(iter(rows)).unique().collect() == sorted(set(rows))


def test_unique_presorted():
    """is_sorted=True skips the sort -- and only collapses *adjacent* dups."""
    lines = ['a\n', 'a\n', 'b\n', 'a\n']
    assert tg.LazyFrame(iter(lines)).unique(is_sorted=True).collect() == ['a\n', 'b\n', 'a\n']


def test_unique_empty():
    assert tg.LazyFrame(iter([])).unique().collect() == []


# --- group_by_key ---

@pytest.fixture
def pairs():
    """key<TAB>value lines, keys repeating, in deliberately unsorted order."""
    rng = random.Random(7)
    return [f'k{rng.randrange(20)}\t{i}\n' for i in range(500)]


KEY = lambda line: line.split('\t')[0]
VALUE = lambda line: int(line.split('\t')[1])


def test_group_by_key_len(pairs):
    want = collections.Counter(map(KEY, pairs))
    assert dict(tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, agg=len).collect()) == want


def test_group_by_key_sum(pairs):
    want = collections.defaultdict(int)
    for line in pairs:
        want[KEY(line)] += VALUE(line)
    got = tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, value=VALUE, agg=sum).collect()
    assert dict(got) == dict(want)


def test_group_by_key_list_preserves_input_order(pairs):
    """sort() is stable, so a key's values arrive in the order they were read."""
    want = collections.defaultdict(list)
    for line in pairs:
        want[KEY(line)].append(VALUE(line))
    got = tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, value=VALUE, agg=list).collect()
    assert dict(got) == dict(want)


def test_group_by_key_emits_each_key_once(pairs):
    keys = [k for k, _ in tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, agg=len).collect()]
    assert len(keys) == len(set(keys)) == len(set(map(KEY, pairs)))
    assert keys == sorted(keys)


def test_group_by_key_empty():
    assert tg.LazyFrame(iter([])).group_by_key(key=KEY, agg=len).collect() == []


def test_group_by_key_presorted_skips_sort(pairs):
    """is_sorted=True must not re-sort -- and needs genuinely sorted input."""
    assert (tg.LazyFrame(iter(sorted(pairs))).group_by_key(key=KEY, agg=len, is_sorted=True).collect()
            == tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, agg=len).collect())


def test_group_by_key_spills(pairs):
    """The whole point: grouping inherits sort()'s external, batched behavior."""
    got = tg.LazyFrame(iter(pairs)).group_by_key(key=KEY, agg=len, batch_size=7, max_fanin=4)
    assert dict(got.collect()) == collections.Counter(map(KEY, pairs))


def test_group_by_key_holds_only_one_group():
    """agg gets an iterator, so len never materializes a group -- even though
    these 200k rows are only 3 keys, i.e. ~67k rows per group."""
    def run():
        lines = (f'k{i % 3}\t{i}\n' for i in range(200_000))
        return tg.LazyFrame(lines).group_by_key(key=KEY, agg=len, batch_size=1_000).collect()

    assert peak_bytes(run) < 5_000_000, 'group_by_key buffered a group'


# --- merge_join ---

def nested_loop_join(left, right, key=KEY, right_key=None):
    """Oracle: the O(n*m) definition of an inner join."""
    right_key = right_key or key
    return sorted((key(l), l, r) for l in left for r in right if key(l) == right_key(r))


def test_merge_join_many_to_one():
    left = [f'k{i % 4}\tedge{i}\n' for i in range(20)]
    right = [f'k{i}\t{i * 10}\n' for i in range(4)]
    got = tg.LazyFrame(iter(left)).merge_join(tg.LazyFrame(iter(right)), key=KEY).collect()
    assert sorted(got) == nested_loop_join(left, right)


def test_merge_join_many_to_many():
    left = ['a\t1\n', 'a\t2\n', 'b\t3\n', 'c\t4\n']
    right = ['a\t10\n', 'a\t20\n', 'b\t30\n', 'd\t40\n']
    got = tg.LazyFrame(iter(left)).merge_join(tg.LazyFrame(iter(right)), key=KEY).collect()
    assert sorted(got) == nested_loop_join(left, right)
    assert len(got) == 5, 'a x a should be 2x2, plus b'


def test_merge_join_inner_gate_drops_one_sided_keys():
    """Keys on only one side vanish -- including a right-only key, which the
    'no lefts buffered' branch must handle rather than emitting."""
    left = ['a\t1\n', 'only_left\t2\n']
    right = ['a\t10\n', 'only_right\t20\n']
    got = tg.LazyFrame(iter(left)).merge_join(tg.LazyFrame(iter(right)), key=KEY).collect()
    assert got == [('a', 'a\t1\n', 'a\t10\n')]


@pytest.mark.parametrize('left,right', [
    ([], []),
    (['a\t1\n'], []),
    ([], ['a\t1\n']),
])
def test_merge_join_empty(left, right):
    assert tg.LazyFrame(iter(left)).merge_join(tg.LazyFrame(iter(right)), key=KEY).collect() == []


def test_merge_join_distinct_key_functions():
    """The two sides need not be shaped alike."""
    left = ['a,1\n', 'b,2\n']
    right = ['a\t10\n', 'b\t20\n']
    got = tg.LazyFrame(iter(left)).merge_join(
        tg.LazyFrame(iter(right)), key=lambda r: r.split(',')[0], right_key=KEY).collect()
    assert sorted(got) == nested_loop_join(left, right, key=lambda r: r.split(',')[0], right_key=KEY)


def test_merge_join_spills(pairs):
    """Joining inherits sort()'s external behavior, cascade included."""
    right = [f'k{i}\tr{i}\n' for i in range(20)]
    got = tg.LazyFrame(iter(pairs)).merge_join(
        tg.LazyFrame(iter(right)), key=KEY, batch_size=7, max_fanin=4).collect()
    assert sorted(got) == nested_loop_join(pairs, right)


def test_merge_join_streams_the_many_side():
    """With the 'one' side on the left, a high-degree key
    does not buffer the 'many' side (100k rows all sharing one key)."""
    def run():
        left = iter(['k\tone\n'])
        right = (f'k\tmany{i}\n' for i in range(100_000))
        return sum(1 for _ in tg.LazyFrame(left).merge_join(
            tg.LazyFrame(right), key=KEY, batch_size=1_000).iter)

    assert peak_bytes(run) < 5_000_000, 'merge_join buffered the streaming side'


# --- concat ---

def test_concat_vertical():
    frames = [tg.LazyFrame(iter(['a\n', 'b\n'])), tg.LazyFrame(iter(['c\n']))]
    assert tg.concat(frames, 'vertical').collect() == ['a\n', 'b\n', 'c\n']


def test_concat_vertical_empty():
    assert tg.concat([], 'vertical').collect() == []
    assert tg.concat([tg.LazyFrame(iter([]))], 'vertical').collect() == []


def test_concat_horizontal():
    frames = [tg.LazyFrame(iter(['a\n', 'b\n'])), tg.LazyFrame(iter(['c\n', 'd\n']))]
    assert tg.concat(frames, 'horizontal').collect() == [('a\n', 'c\n'), ('b\n', 'd\n')]


def test_concat_horizontal_pads_ragged():
    """zip_longest, so the shorter side is padded -- polars' horizontal fills null."""
    frames = [tg.LazyFrame(iter(['a\n', 'b\n'])), tg.LazyFrame(iter(['c\n']))]
    assert tg.concat(frames, 'horizontal').collect() == [('a\n', 'c\n'), ('b\n', None)]


def test_concat_rejects_unknown_how():
    with pytest.raises(ValueError, match='unimplemented'):
        tg.concat([tg.LazyFrame(iter([]))], 'diagonal')


def test_concat_is_lazy():
    it = iter(['a\n'])
    tg.concat([tg.LazyFrame(it)], 'vertical')
    assert next(it) == 'a\n', 'concat ran at construction time'


# --- batched ---

@pytest.mark.parametrize('batch_size,want', [
    (1, [(0,), (1,), (2,), (3,), (4,)]),
    (2, [(0, 1), (2, 3), (4,)]),          # last batch short
    (5, [(0, 1, 2, 3, 4)]),
    (10, [(0, 1, 2, 3, 4)]),              # batch larger than the frame
])
def test_batched(batch_size, want):
    assert tg.LazyFrame(iter(range(5))).batched(batch_size).collect() == want


def test_batched_empty():
    assert tg.LazyFrame(iter([])).batched(3).collect() == []


# --- head / tail ---

@pytest.mark.parametrize('k', [0, 1, 4, 10, 25])
def test_head(k):
    xs = list(range(10))
    assert tg.LazyFrame(iter(xs)).head(k).collect() == xs[:k]


@pytest.mark.parametrize('k', [0, 1, 4, 10, 25])
def test_tail(k):
    xs = list(range(10))
    assert tg.LazyFrame(iter(xs)).tail(k).collect() == (xs[-k:] if k else [])


def test_head_shorter_than_k_does_not_raise():
    """Regression: next() past the end used to become RuntimeError (PEP 479)."""
    assert tg.LazyFrame(iter([1, 2])).head(5).collect() == [1, 2]


@pytest.mark.parametrize('method', ['head', 'tail'])
def test_head_tail_empty(method):
    assert getattr(tg.LazyFrame(iter([])), method)(3).collect() == []


@pytest.mark.parametrize('method', ['head', 'tail', 'batched'])
def test_methods_accept_plain_iterable(method):
    """__init__ is typed Iterable, so a list must work -- not just an iterator."""
    assert getattr(tg.LazyFrame([1, 2, 3]), method)(2).collect()


def test_head_does_not_over_consume():
    it = iter(range(10))
    assert tg.LazyFrame(it).head(3).collect() == [0, 1, 2]
    assert next(it) == 3, 'head consumed more of the source than it returned'


def test_tail_holds_only_k_rows():
    got = []
    peak = peak_bytes(lambda: got.extend(tg.LazyFrame(iter(range(500_000))).tail(3).collect()))
    assert got == [499_997, 499_998, 499_999]
    assert peak < 100_000, f'tail buffered the frame ({peak} bytes)'


# --- laziness ---

@pytest.mark.parametrize('build', [
    lambda lf: lf.map(str),
    lambda lf: lf.filter(bool),
    lambda lf: lf.head(3),
    lambda lf: lf.tail(3),
    lambda lf: lf.batched(2),
    lambda lf: lf.map(lambda i: f'{i}\n').sort(batch_size=2),
])
def test_nothing_is_consumed_before_collect(build):
    it = iter(range(10))
    build(tg.LazyFrame(it))                  # built but never collected
    assert next(it) == 0, 'pipeline ran at construction time'


# --- map / filter / explode / reduce ---

def test_map(rows):
    assert tg.LazyFrame(iter(rows)).map(str.strip).collect() == [r.strip() for r in rows]


def test_filter(rows):
    pred = lambda line: int(line) % 3 == 0
    assert tg.LazyFrame(iter(rows)).filter(pred).collect() == [r for r in rows if pred(r)]


def test_explode():
    nested = [[1, 2], [], [3], [4, 5]]
    assert tg.LazyFrame(iter(nested)).explode().collect() == [1, 2, 3, 4, 5]


def test_reduce(rows):
    assert (tg.LazyFrame(iter(rows)).reduce(lambda a, b: a + b).collect()
            == [''.join(rows)])


# --- composition ---

def test_sort_then_head(rows):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=8).head(5).collect() == sorted(rows)[:5]


def test_sort_then_tail(rows):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=8).tail(5).collect() == sorted(rows)[-5:]


def test_pipeline(rows):
    """Round-trip through the serialization students would write themselves."""
    got = (tg.LazyFrame(iter(rows))
           .map(int)
           .filter(lambda v: v % 2 == 0)
           .map(lambda v: f'{v * 10}\n')
           .sort(batch_size=16, key=int)
           .head(4)
           .map(int)
           .collect())
    assert got == sorted(int(r) * 10 for r in rows if int(r) % 2 == 0)[:4]


# --- sink ---

def test_sink(tmp_path):
    out = tmp_path / 'out.txt'
    tg.LazyFrame(iter(['a\n', 'b\n'])).sink(str(out), 'w')
    assert out.read_text() == 'a\nb\n'
