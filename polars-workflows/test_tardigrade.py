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
import gc
import os
import random
import tracemalloc

import pytest

import tardigrade as tg

BATCH_SIZES = [1, 2, 3, 7, 100, 10_000]      # 1 and 2 stress the merge;
                                             # 10_000 puts everything in one run


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
    """Serializing is the caller's job, but the error should say so."""
    with pytest.raises(TypeError, match='json.dumps'):
        tg.LazyFrame(iter([3, 1, 2])).sort(batch_size=2).collect()


# --- sort: resource bounds ---

def test_sort_peak_memory_tracks_batch_size():
    """The point of spilling: RAM follows batch_size, not frame size."""
    def peak(batch_size):
        lines = (f'{i}\n' for i in range(200_000))
        tracemalloc.start()
        for _ in tg.LazyFrame(lines).sort(batch_size=batch_size).iter:
            pass
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    assert peak(1_000) < peak(200_000) / 2


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
    tracemalloc.start()
    got = tg.LazyFrame(iter(range(500_000))).tail(3).collect()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
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
