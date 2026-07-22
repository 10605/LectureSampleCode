"""pytest suite for tardigrade.py -- the streaming LazyFrame.

Every operation is validated against the equivalent builtin (sorted, slicing,
map/filter) as the oracle, so the tests say what the method *means* rather than
restating its implementation.  Run with:  pytest test_tardigrade.py

The interesting cases here are the ones that broke before: empty frames, frames
shorter than k, batch boundaries in the external sort, and the promise that
nothing is consumed until collect().
"""
import random
import tracemalloc

import pytest

import tardigrade as tg

BATCH_SIZES = [1, 2, 3, 7, 100, 10_000]      # 1 and 2 stress the k-way merge;
                                             # 10_000 puts everything in one run


@pytest.fixture
def rows():
    """1000 ints with many duplicates, so ties exercise merge stability."""
    rng = random.Random(42)
    return [rng.randrange(50) for _ in range(1000)]


# --- sort ---

@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_sort(rows, batch_size):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=batch_size).collect() == sorted(rows)


@pytest.mark.parametrize('batch_size', BATCH_SIZES)
def test_sort_reverse(rows, batch_size):
    assert (tg.LazyFrame(iter(rows)).sort(batch_size=batch_size, reverse=True).collect()
            == sorted(rows, reverse=True))


def test_sort_key(rows):
    key = lambda v: (v % 7, -v)
    assert (tg.LazyFrame(iter(rows)).sort(batch_size=13, key=key).collect()
            == sorted(rows, key=key))


@pytest.mark.parametrize('n', [0, 1, 9, 10, 11])
def test_sort_batch_boundaries(n):
    """n just below / at / above a batch multiple -- the classic off-by-one."""
    xs = list(range(n))[::-1]
    assert tg.LazyFrame(iter(xs)).sort(batch_size=5).collect() == sorted(xs)


def test_sort_empty():
    assert tg.LazyFrame(iter([])).sort().collect() == []


def test_sort_is_stable():
    """Ties keep input order: sorted() is stable and merge prefers earlier runs."""
    rows = [{'k': i % 5, 'i': i} for i in range(60)]
    key = lambda r: r['k']
    assert tg.LazyFrame(iter(rows)).sort(batch_size=7, key=key).collect() == sorted(rows, key=key)


def test_sort_roundtrips_rows_through_json():
    """Runs are spilled as JSON, so dict/str/float rows must survive intact."""
    rows = [{'w': w, 'n': n} for w, n in zip('delta alpha charlie bravo'.split(), [1.5, 2.0, -3.25, 0.1])]
    key = lambda r: r['w']
    assert tg.LazyFrame(iter(rows)).sort(batch_size=2, key=key).collect() == sorted(rows, key=key)


def test_sort_peak_memory_tracks_batch_size():
    """The point of spilling: RAM follows batch_size, not frame size."""
    def peak(batch_size):
        tracemalloc.start()
        for _ in tg.LazyFrame(iter(range(200_000))).sort(batch_size=batch_size).iter:
            pass
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    assert peak(1_000) < peak(200_000) / 2


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


@pytest.mark.parametrize('method', ['head', 'tail'])
def test_head_tail_accept_plain_iterable(method):
    """__init__ is typed Iterable, so a list must work -- not just an iterator."""
    assert getattr(tg.LazyFrame([1, 2, 3]), method)(2).collect()


def test_head_does_not_over_consume():
    it = iter(range(10))
    assert tg.LazyFrame(it).head(3).collect() == [0, 1, 2]
    assert next(it) == 3, 'head consumed more of the source than it returned'


def test_tail_holds_only_k_rows():
    tracemalloc.start()
    got = tg.LazyFrame(iter(range(500_000))).tail(3).collect()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert got == [499_997, 499_998, 499_999]
    assert peak < 100_000, f'tail buffered the frame ({peak} bytes)'


# --- laziness ---

@pytest.mark.parametrize('build', [
    lambda lf: lf.map(lambda x: x),
    lambda lf: lf.filter(bool),
    lambda lf: lf.head(3),
    lambda lf: lf.tail(3),
    lambda lf: lf.sort(batch_size=2),
])
def test_nothing_is_consumed_before_collect(build):
    it = iter(range(10))
    build(tg.LazyFrame(it))                  # built but never collected
    assert next(it) == 0, 'pipeline ran at construction time'


# --- map / filter / explode / reduce ---

def test_map(rows):
    assert tg.LazyFrame(iter(rows)).map(str).collect() == [str(r) for r in rows]


def test_filter(rows):
    pred = lambda v: v % 3 == 0
    assert tg.LazyFrame(iter(rows)).filter(pred).collect() == [r for r in rows if pred(r)]


def test_explode():
    nested = [[1, 2], [], [3], [4, 5]]
    assert tg.LazyFrame(iter(nested)).explode().collect() == [1, 2, 3, 4, 5]


def test_reduce(rows):
    assert tg.LazyFrame(iter(rows)).reduce(lambda a, b: a + b).collect() == [sum(rows)]


# --- composition ---

def test_sort_then_head(rows):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=8).head(5).collect() == sorted(rows)[:5]


def test_sort_then_tail(rows):
    assert tg.LazyFrame(iter(rows)).sort(batch_size=8).tail(5).collect() == sorted(rows)[-5:]


def test_pipeline(rows):
    got = (tg.LazyFrame(iter(rows))
           .filter(lambda v: v % 2 == 0)
           .map(lambda v: v * 10)
           .sort(batch_size=16)
           .head(4)
           .collect())
    assert got == sorted(v * 10 for v in rows if v % 2 == 0)[:4]


# --- sink ---

def test_sink(tmp_path):
    out = tmp_path / 'out.txt'
    tg.LazyFrame(iter(['a\n', 'b\n'])).sink(str(out), 'w')
    assert out.read_text() == 'a\nb\n'
