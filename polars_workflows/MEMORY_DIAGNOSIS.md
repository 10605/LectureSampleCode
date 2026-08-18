# Why `demo_memory.py` shows lazy peak RSS climbing with edge count

**Environment:** polars 1.42.1, Python 3.13, macOS (peak RSS via `resource.getrusage`).

## The expectation

`demo_memory.py` is built to demonstrate the central claim in `pagerank.py`:
the lazy/streaming pipeline never holds the edge list in memory. Edges are
supposed to *stream* from disk on every `collect()`, so peak resident memory
(RSS) should track the number of **nodes** (materialized as the in-memory
`pagerank_scores` DataFrame) and stay **flat** as the number of **edges** grows.
Only the `eager` contrast case — which pins all raw lines in RAM — should climb.

## The observed result

Fixed nodes = 20,000, iterations = 10, sweeping edges:

```
   total lines   lazy RSS MB   eager RSS MB   lazy s   eager s
--------------------------------------------------------------
     1,020,000         474.4          408.9      0.4       0.3
     2,020,000         614.2          590.8      0.6       0.5
     4,020,000        1009.7         1091.9      1.0       0.9
```

The `lazy` peak RSS climbs almost identically to `eager` (474 → 614 → 1010 MB).
The lazy path is *not* staying flat — it tracks the edge count, defeating the
whole point of the demo.

## Hypothesis that turned out wrong: CSE caching

A natural first guess: the `edges` LazyFrame is referenced multiple times (by
`n_outlinks`, by the two-branch `concat`, and again every loop iteration), so
polars' **common-subplan elimination (CSE)** inserts a CACHE node that
materializes the whole edge scan in RAM.

Disabling `comm_subplan_elim` barely moved the number:

```
CSE_ON  peak_rss 1042.2 MB   (4M-edge graph)
CSE_OFF peak_rss  929.0 MB
```

~110 MB, not the ~600 MB the demo is trying to show. **CSE is not the cause.**

(Reproduce with the `--no-cse` flag on `demo_memory.py` or `pagerank.py`, e.g.
`python demo_memory.py --nodes 20000 --edges 4000000 --no-cse`.)

## Actual cause: streaming group-by / join buffer their input

Isolating each pipeline stage on the same fixed 20k-node graph, sweeping edges
250k → 4M:

| stage                          | 250k edges | 4M edges | growth        |
| ------------------------------ | ---------: | -------: | ------------- |
| `scan_lines` + parse + `count` |     79 MB  |  140 MB  | +61 MB (~flat)|
| `group_by('dst').len()`        |    124 MB  |  378 MB  | **+254 MB**   |
| self-join for `n_outlinks`     |    135 MB  |  373 MB  | **+238 MB**   |

- A pure `scan_lines → parse → count` genuinely streams: RSS stays essentially
  flat as edges grow (79 → 140 MB). The scan half of the pipeline is fine.
- The moment a `group_by('dst')` is added, RSS grows *linearly with edge count*
  (124 → 378 MB) even though the output is only 20,000 rows. A hash aggregation
  keyed on 20k node values should be node-bounded; instead the polars 1.42.1
  streaming engine materializes the edge rows rather than streaming them through
  the grouping.
- The `n_outlinks` self-join shows the same growth.

So in `pagerank.py`, every iteration's
`edges.join(pagerank_scores).group_by('dst')` pulls the whole edge list into
memory. **The `group_by` / `join` operators under `engine='streaming'` in polars
1.42.1 are not fully streaming — they buffer their input proportional to the
number of edges.** That is why lazy peak RSS climbs almost identically to eager.

## Bottom line

The demo's premise (edges only ever stream, so lazy peak RSS stays flat while
eager climbs) does not hold on polars 1.42.1 because its streaming aggregations
and joins still buffer their input. This is a polars-version limitation, not a
bug in the pagerank formulation.

## Possible next steps

1. **Document only** — note in `demo_memory.py` that on polars ≤ 1.42 the
   streaming group-by/join buffer input, so the lazy/eager gap is small; the
   demo becomes meaningful once polars' streaming aggregations spill/stream.
2. **Try to make it actually stream** — integer-encode node ids (intern strings
   → `UInt32` up front) to shrink the per-edge buffered footprint, and/or test
   the older `streaming=True` path vs the new `engine='streaming'`.
3. **Verify version-specificity** — re-run on a newer polars to see whether the
   streaming group-by flattens the curve.

## How to reproduce

```
cd polars_workflows
python demo_memory.py --nodes 20000 --edges 1000000 2000000 4000000 --iterations 10
```
