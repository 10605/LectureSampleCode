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

## Clustering

`kmeans.py` — spherical k-means over the documents of a corpus. Documents
arrive as sparse tf-idf vectors, precomputed offline and shipped alongside each
corpus as `data/<corpus>.tfidf.tsv`: one `(doc, token, w)` row per token per
document, each document L2-normalized, `doc` being its line number in the
corpus.

Held that way, both halves of the algorithm are joins: assignment is
`docs ⋈ centroids ON token` summed per `(doc, cluster)` — a sparse matrix
product, and since both sides are unit length that dot product *is* the cosine
similarity — and the update is `docs ⋈ assignment ON doc` summed per
`(cluster, token)`. Only the O(k × vocab) centroid table is materialized, the
same shape as pagerank's score table.

```
python3 kmeans.py                                    # 8 clusters over bluecorpus
python3 kmeans.py --k 12 --num_iterations 30 --seed 7
python3 kmeans.py --corpus ../data/redcorpus.txt
```

Prints per-iteration mean cosine and the number of documents that moved, then
each cluster's size, heaviest terms, and closest members.

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

### Running under a memory cap

On a large-RAM machine these experiments cannot show what they are meant to
show: polars' out-of-core budget defaults to unlimited, so it never spills, and
the low-memory engines never get to prove anything. The `Dockerfile` at the
repo root builds a Linux image with the locked dependencies, to be run under a
hard memory limit:

```
container build -t pr-demo .                          # or: docker build -t pr-demo .
container run --rm -m 512m -v "$PWD:/work" -w /work pr-demo \
    python polars_workflows/demo_memory.py --nodes 20000 --edges 8000000 \
    --iterations 10 --configs polars-lazy tardigrade-lazy
```

Under the cap the kernel OOM-kills a run that will not fit. `demo_memory.py`
treats that as a result rather than a crash — the cell reads `OOM` and the
sweep carries on, so one engine dying does not hide the others:

```
self RSS (MB)     8,020,000
---------------------------
polars-lazy             OOM
tardigrade-lazy        66.4
```

Any other nonzero exit is still raised as a real failure. Measured numbers are
in `RESULTS.md`.

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
  ruled out as the cause. Later sections read the polars 1.44.1 engine source
  for what its lazy `group_by` actually does (two-level hash aggregation, never
  a sort) and what it writes to disk, and measure the spilling it can be forced
  into.

## Tests

```
python3 -m pytest        # 193 passed, 1 xfailed
```

`test_bar.py` and `test_tardigrade.py` check sort correctness and stability
against in-memory oracles, and — importantly — that spill files are deleted and
that no more than `fanin` files are held open during a merge.

`test_kmeans.py` checks each join against the dense in-memory version:
assignment against a dict-of-dicts argmax over cosines, the update against a
plain mean. The cases that matter are the ones a sparse formulation can quietly
drop — a document sharing no token with any centroid, a cluster that loses
every member — plus determinism under a fixed seed.
