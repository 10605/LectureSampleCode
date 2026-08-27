# polars_workflows

Lecture sample code for 10-605: map-reduce-style workflows expressed as
dataframe pipelines, and what happens to *memory* when the data no longer fits
in RAM.

Everything here is a variation on one theme: a computation is written once as a
lazy pipeline over a stream of lines, then run through progressively more
memory-frugal engines — polars' streaming engine, polars with the group-by/join
steps offloaded to disk (`bar.py`), and pure Python generators with external
sorting (`tardigrade.py`).

Most scripts read corpora and graphs from `../data`.

## Small polars examples

| file | what it shows |
| --- | --- |
| `redvsblue.py` | word counts over two corpora, joined and scored — the basic lazy `scan_lines → explode → group_by` pipeline |
| `phrases.py` | same idea extended to bigrams, with foreground/background phrase scoring |
| `matmul.py` | sparse matrix multiply as a self-join plus a windowed sum, checked against a gold product in `../data/matmul.json` |

## Pagerank, four ways

Same algorithm, same default graph (`../data/citeseer-graph.txt`), different
execution engines. Each takes `--filename` and can read a `.gz` graph.

| file | engine |
| --- | --- |
| `pagerank.py` | plain polars lazy/streaming; joins scores onto edges each iteration |
| `pagerank_mapper.py` | keeps scores in a Python dict and maps over edges instead of joining (see its note on dangling nodes — it uses a different convention) |
| `pagerank_bar.py` | every group-by/join offloaded to `bar.PolarBar`, i.e. to sorted key-value files on disk |
| `pagerank_tg.py` | no polars at all: `tardigrade` iterators and external sort |

## The low-memory machinery

- **`merge_sort.py`** — external merge sort for line-oriented files. Only the
  sort key is decoded; lines are carried verbatim. Stable, with configurable
  batch size and merge fan-in.
- **`bar.py`** — `PolarBar`: sink a polars `LazyFrame` to a key-value TSV, sort
  it, then `group_and_aggregate_by_key` / `join_by_key` over the sorted file.
  Lets a polars pipeline keep its shape while the memory-hungry operators run
  on disk.
- **`tardigrade.py`** — a miniature lazy dataframe built entirely from Python
  generators: `scan_lines`, `map`, `filter`, `explode`, `sort`, `unique`,
  `group_by_key`, `merge_join`, `sink`. Spills sorted runs to temp files that
  delete themselves when unreferenced, so peak memory tracks the batch size,
  not the data size.

## Measurement drivers

- **`demo_memory.py`** — the main experiment harness. Sweeps graph sizes across
  engines and `lazy`/`eager` input modes, each configuration in its own
  subprocess so peak RSS (`getrusage`) is clean. RSS, not `tracemalloc`: polars
  keeps data in Arrow buffers outside the Python heap.

  ```
  python3 demo_memory.py --nodes 20000 --edges 1000000 2000000 4000000 --iterations 10
  ```

- **`wordcount_probe.py`** — stress test on the RCV1 corpus: tokenize, emit
  `(docid##label, token)` pairs, sort and group them through `PolarBar`.
  Flags for corpus size/split, sort batch size, and fan-in (`--fanin 0`
  autoscales to `sqrt(num_batches)`).
## Findings

- `RESULTS.md` — measured comparisons (polars vs. tardigrade) and RCV1
  wordcount timings. Short version: tardigrade holds peak RSS ~flat at ~80 MB
  while polars climbs into the gigabytes, at roughly 100× the wall-clock cost.
- `MEMORY_DIAGNOSIS.md` — why the lazy pagerank demo does *not* show flat
  memory on polars 1.42.1: the streaming `group_by` and `join` buffer their
  input proportional to the number of edges. Common-subplan elimination was
  ruled out as the cause.

## Tests

```
python3 -m pytest        # 162 tests
```

`test_bar.py` and `test_tardigrade.py` check sort correctness and stability
against in-memory oracles, and — importantly — that spill files are deleted and
that no more than `fanin` files are held open during a merge.
