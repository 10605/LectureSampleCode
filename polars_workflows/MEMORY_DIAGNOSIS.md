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

(This was measured with a `--no-cse` flag on `demo_memory.py` / `pagerank.py`
that plumbed `comm_subplan_elim` through to `collect()`. Since CSE was ruled out
as the cause, that plumbing has been **removed** -- the flag no longer exists. To
re-run the experiment, pass
`optimizations=pl.QueryOptFlags(comm_subplan_elim=False)` to the `collect()`
calls in `pagerank.py` directly.)

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

---

# What the polars group-by actually does (source reading, 1.44.1)

The sections above infer polars' behaviour from RSS curves on **1.42.1**. This
section reads the engine source directly to explain the mechanism. It is based
on the upstream tree cloned at tag `py-1.44.1` (commit `e4ae5ea`), matching the
polars now pinned in `pyproject.toml`:

```
git clone --depth 1 --branch py-1.44.1 https://github.com/pola-rs/polars.git polars_src
```

`polars_src/` is gitignored — re-clone it if you want to follow the file
references below. Note this is **1.44.1, a newer engine than the 1.42.1 the
measurements above were taken on**; the numbers in the earlier sections have not
been re-collected against it.

## Algorithm: two-level hash aggregation, never a sort

The streaming sink is `crates/polars-stream/src/nodes/group_by.rs`. Per morsel,
per worker thread:

1. **Hash the keys** — `HashKeys::from_df`.
2. **Probe a small fixed-size "hot" table** — `new_hash_hot_grouper`, sized
   `DEFAULT_HOT_TABLE_SIZE = 4096` groups (`group_by.rs:31`). Hits are folded
   straight into `GroupedReduction` accumulators. This pre-aggregation is what
   makes high-skew group-bys cheap.
3. **Misses are "cold"** (`group_by.rs:218-235`): gathered out, assigned a
   partition by `HashPartitioner`, and their payload rows wrapped in a
   `SpillFrame`.
4. **Hot-table evictions** become `PreAgg` records — already partially reduced
   state, also partitioned.

`combine_locals` (`group_by.rs:331+`) then runs **one task per partition**, each
replaying that partition's cold rows and pre-aggregates into a fresh grouper.
Each partition sizes its hash table up front from a `CardinalitySketch`
(HyperLogLog-style), combined across threads:

```rust
let est_num_groups = sketch.estimate() * 5 / 4;   // group_by.rs:335
```

`num_partitions = num_pipelines` (`group_by.rs:553`) — one per thread, fixed,
*not* adaptive to data size.

So the shape is **hash aggregation with a hot/cold split plus hash
partitioning**. polars only sorts in `sorted_group_by.rs`, a separate node for
input already known to be sorted. This is the substantive contrast with
`tardigrade.py`, whose `group_by` is `sort()` + `itertools.groupby` — O(n log n)
and unconditionally on disk.

## What reaches disk

Spilling is **not** group-by-specific; it is a global memory manager in
`crates/polars-ooc/`.

- `SpillFrame::new(cold_df, ctx)` marks a DataFrame *spillable*. It stays in RAM
  until evicted; `sf.get().await` transparently reads it back.
- The decision is global (`memory_manager.rs:45`):

  ```rust
  fn should_spill(&self) -> bool {
      let usage = crate::estimate_memory_usage();
      let likely_dealt_with = self.est_spill_in_progress.load(Ordering::Relaxed);
      usage.saturating_sub(likely_dealt_with) > config().ooc_memory_budget_bytes()
  }
  ```

- **Budget: unlimited by default** — `DEFAULT_OOC_MEMORY_BUDGET_MB = u64::MAX`
  (`polars-config/src/lib.rs:79`), so `should_spill()` is never true unless you
  set `POLARS_OOC_MEMORY_BUDGET_MB` yourself. There *is* a
  `DEFAULT_OOC_MEMORY_BUDGET_FRACTION = 0.8` (`lib.rs:76`), but its getter is
  **never called anywhere outside the config module** — it is dead in 1.44.1.
  Confirmed empirically below: at the default budget the spill directory is
  never even created.
- **Format: Arrow IPC** (`DEFAULT_OOC_SPILL_FORMAT = SpillFormat::Ipc`,
  `lib.rs:69`), optionally compressed.
- **Location:** `$TMPDIR/polars-$USER/spill` on macOS, `/var/tmp/polars-$USER/spill`
  on Linux (`spill_path.rs:14-29`). A background thread cleans stale files.

Three things stay resident regardless: the **hash keys** (the `HashKeys` in
`cold_morsels` is not wrapped in a `SpillFrame`), the **cardinality sketches**,
and the **per-partition index vectors**. Only payload columns spill.

## Consequence for this demo

Because the default budget is unlimited, this demo **never spills at all** out
of the box — it is in-memory hash aggregation throughout, at any edge count.
That is exactly the buffering diagnosed above: cold payload rows accumulate in
`SpillFrame`s that are never evicted, so RSS tracks edge count.

This sharpens the lecture framing:

- polars' lazy group-by is *in-memory hash aggregation that can spill under
  pressure*;
- tardigrade's is *an external merge sort that always spills*.

To make the comparison apples-to-apples the budget must be forced down
explicitly. Results of doing so are in the next section.

Two observability knobs are *not* usable in 1.44.1, despite appearances:
`POLARS_OOC_LOG_METRICS` exists and is read (`spill_context/stats.rs:45`), but
the `[ooc] ...` verbose lines and `POLARS_OOC_SPILL_POLICY` referenced by
upstream's own `py-polars/tests/unit/ooc/test_ooc.py` do not exist in this
version at all -- and that test is `@pytest.mark.skip`-ped. Measure spilling by
watching the spill directory instead.

---

# Measured: forcing polars to spill (`POLARS_OOC_MEMORY_BUDGET_MB=256`)

**Environment:** polars 1.44.1, Python 3.13.3, macOS. `polars-lazy` config only,
fixed nodes = 20,000, iterations = 10, CSE on. Peak RSS is `ru_maxrss`, i.e. a
**high-water mark** for the process, not live memory.

Spill activity was measured by sampling `$TMPDIR/polars-$USER/spill` every 2 s
for peak bytes and peak file count.

## Results

| total lines | default budget RSS (MB) | 256 MB budget RSS (MB) | default s | 256 MB s |
| ----------: | ----------------------: | ---------------------: | --------: | -------: |
|   1,020,000 |                   484.4 |                  483.5 |       0.4 |      0.4 |
|   2,020,000 |                   599.5 |                  605.5 |       0.6 |      0.6 |
|   4,020,000 |                   925.9 |                 1217.9 |       0.9 |      2.2 |
|   8,020,000 |                  1492.9 |                 1824.4 |       1.5 |      3.9 |
|  16,020,000 |                  2072.2 |                 2238.3 |       2.6 |      7.7 |
|  32,020,000 |                  3248.9 |                 3001.1 |       5.1 |     12.2 |

Spill directory over the whole sweep:

| run              | dir created? | peak spill bytes | peak files |
| ---------------- | ------------ | ---------------: | ---------: |
| default budget   | **no**       |                0 |          0 |
| `BUDGET_MB=256`  | yes          |      **1.28 GB** |   **1373** |

## What this shows

1. **The default really is "never spill."** At the default budget the spill
   directory is never created — confirming the `u64::MAX` reading above. Every
   earlier measurement in this document was of a pure in-memory engine.

2. **The env var works.** With a 256 MB budget polars spills hard: 1.28 GB peak
   across 1373 Arrow IPC files.

3. **But spilling does not reduce peak RSS.** This is the surprise. At 4M-8M
   lines the budgeted run is *worse* (1218 vs 926 MB; 1824 vs 1493 MB), and only
   at 32M lines does it come out ahead, by ~8% (3001 vs 3249 MB). Spilling costs
   ~2.4x in wall time throughout.

The most likely reasons, consistent with the source reading above:

- **Peak RSS is a high-water mark.** The rows must be allocated and populated
  *before* the memory manager can notice the budget is exceeded and evict them,
  so the peak is set before spilling helps. `should_spill()` is reactive, not
  admission-controlled.
- **The non-spillable state dominates.** `HashKeys`, `CardinalitySketch`es and
  the per-partition index vectors are never wrapped in a `SpillFrame`. Only
  payload columns spill, and this pagerank's payload is narrow (two id columns).
- **Freed pages are not necessarily returned to the OS**, so RSS does not fall
  when a `SpillFrame` is evicted.

**Caveat:** these are the mechanisms the source makes plausible; they have not
been isolated experimentally. Distinguishing them would need live-memory
sampling rather than `ru_maxrss`, which this harness does not currently do.

## Consequence for the demo

Forcing the budget down does **not** rescue the flat-lazy-RSS story: polars'
peak RSS still grows roughly linearly with edge count either way. The honest
contrast for the lecture is therefore about *algorithm*, not peak RSS:

- polars: **in-memory hash aggregation**, spilling only reactively and only
  payload columns, with an unlimited budget by default;
- tardigrade: **external merge sort**, unconditionally on disk, bounded by
  construction.

## How to reproduce

```
# spilling forced
POLARS_OOC_MEMORY_BUDGET_MB=256 uv run polars_workflows/demo_memory.py \
    --nodes 20000 --edges 1000000 2000000 4000000 8000000 16000000 32000000 \
    --iterations 10 --configs polars-lazy

# control (default budget)
uv run polars_workflows/demo_memory.py \
    --nodes 20000 --edges 1000000 2000000 4000000 8000000 16000000 32000000 \
    --iterations 10 --configs polars-lazy

# watch spill activity in another shell
watch -n1 'du -sh $TMPDIR/polars-$USER/spill 2>/dev/null; \
           find $TMPDIR/polars-$USER/spill -type f 2>/dev/null | wc -l'
```
