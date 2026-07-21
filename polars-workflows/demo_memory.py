"""
Driver that illustrates the central advantage of the lazy/streaming
implementation in pagerank.py: it does NOT hold the edge list in memory.

Peak resident memory (RSS) should track the number of NODES -- which get
materialized as the in-memory `pagerank_scores` DataFrame -- and stay flat as
the number of EDGES grows, because edges only ever *stream* through the lazy
pipeline (they are re-scanned from disk each iteration, never cached).

To make that visible we run two ways of feeding the same graph to pagerank():

  lazy  : `pl.scan_lines(path)` -- edges stream from disk, nothing cached.
  eager : the raw lines are collected into an in-memory DataFrame first, then
          re-wrapped as lazy. This pins all the lines in RAM and is the
          contrast case: its peak RSS grows with the number of edges.

Because peak RSS is a per-process high-water mark, each configuration is run in
its own subprocess so the numbers don't contaminate each other.

IMPORTANT: measure RSS, not tracemalloc/getsizeof -- polars keeps edge data in
Arrow buffers allocated outside the Python heap, which those tools cannot see.

Usage:
    # Full comparison table (spawns one subprocess per config):
    python demo_memory.py

    # tune it:
    python demo_memory.py --nodes 20000 --edges 200000 1000000 4000000 --iterations 10

    # rule out common-subplan-elimination caching (see MEMORY_DIAGNOSIS.md):
    python demo_memory.py --no-cse

    # compare implementations in one table (default runs both):
    python demo_memory.py --engines polars koala
    python demo_memory.py --engines koala          # koala only

    # A single run (this is what the table driver spawns internally):
    python demo_memory.py --single --mode lazy --graph /tmp/g.txt --iterations 10
"""
import argparse
import os
import random
import resource
import subprocess
import sys
import time

import polars as pl

import koala as kl
import pagerank
import pagerank_kl

# The two pagerank implementations, keyed by --engine. Both expose the same
# pagerank(edge_lines, num_iterations=, cse=) signature; koala additionally
# wants its edge_lines wrapped as a KoalaFrame (its heavy group_by/join spill to
# disk instead of buffering in RAM -- see koala.py / MEMORY_DIAGNOSIS.md).
ENGINES = {
    'polars': (pagerank,    lambda lines: lines),
    'koala':  (pagerank_kl, kl.KoalaFrame),
}

# A table config is an engine paired with an edge-feeding mode. The table sweeps
# a chosen subset of these (see --configs) across a range of edge counts.
MODES = ('lazy', 'eager')
ALL_CONFIGS = [f'{e}-{m}' for e in ENGINES for m in MODES]


def peak_rss_mb(who=resource.RUSAGE_SELF):
    """Peak resident set size, in MiB.

    who=RUSAGE_SELF  -> this Python process only.
    who=RUSAGE_CHILDREN -> largest waited-for child (e.g. koala's external
        `sort`); this is invisible to RUSAGE_SELF, so measure it separately.
    """
    r = resource.getrusage(who).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    return r / 1024**2 if sys.platform == 'darwin' else r / 1024


def generate_graph(path, n_nodes, n_edges, seed=1):
    """Write an edge-list file with a fixed node count and a tunable edge count.

    A self-loop `i i` is emitted for EVERY node so that indegree >= 1 and
    outdegree >= 1 hold for all nodes -- the correctness assumption stated in
    pagerank.py. On top of that we add n_edges random edges. Node count stays
    fixed at n_nodes; grow n_edges to stress memory.
    """
    rng = random.Random(seed)
    with open(path, 'w') as f:
        for i in range(n_nodes):                      # self-loop on every node
            f.write(f'{i} {i}\n')
        for _ in range(n_edges):                      # random extra edges
            f.write(f'{rng.randrange(n_nodes)} {rng.randrange(n_nodes)}\n')
    return n_nodes + n_edges                          # total lines written


def make_lines(path, mode):
    """Build the `lines` LazyFrame handed to pagerank().

    lazy : scan the file lazily; edges stream from disk on every collect().
    eager: force the raw lines into an in-memory DataFrame, then re-wrap as
           lazy -- this keeps every line resident in RAM (the contrast case).
    """
    scan = pl.scan_lines(path)
    if mode == 'eager':
        return scan.collect(engine='streaming').lazy()
    return scan


def run_single(mode, path, iterations, engine='polars', cse=True, batch_size=None):
    """Run pagerank once and report this process's peak RSS."""
    module, wrap = ENGINES[engine]
    module.VERBOSE = 0                               # silence show() output
    lines = wrap(make_lines(path, mode))
    kwargs = dict(num_iterations=iterations, cse=cse)
    if engine == 'koala' and batch_size is not None:  # koala-only knob
        kwargs['batch_size'] = batch_size
    start = time.time()
    module.pagerank(lines, **kwargs)
    elapsed = time.time() - start
    # markers parsed by the table driver
    print(f'PEAK_RSS_MB {peak_rss_mb():.1f}')
    print(f'PEAK_CHILD_RSS_MB {peak_rss_mb(resource.RUSAGE_CHILDREN):.1f}')
    print(f'ELAPSED_S {elapsed:.2f}')


def _parse_marker(text, marker):
    for line in text.splitlines():
        if line.startswith(marker):
            return float(line.split()[1])
    return float('nan')


def run_matrix(nodes, edge_sizes, iterations, configs, cse=True, batch_size=None):
    """Spawn a fresh subprocess per (config, edges) and tabulate peak RSS.

    `configs` is a list of (engine, mode) pairs. The printed table puts one
    config per row and one edge count per column. lazy self RSS should stay
    ~flat as edges grow while eager climbs; koala spills its group_by/join to
    disk so it stays bounded too, at a fixed per-iteration overhead.
    """
    total_runs = len(edge_sizes) * len(configs)
    done = 0

    rows = []
    for n_edges in edge_sizes:
        path = f'/tmp/prdemo_{nodes}_{n_edges}.txt'
        total_lines = generate_graph(path, nodes, n_edges)
        row = {'lines': total_lines, 'rss': {}, 'child': {}, 's': {}}
        for engine, mode in configs:
            done += 1
            print(f'[{done}/{total_runs}] {total_lines:,} lines  {engine}/{mode} ...',
                  end='', flush=True, file=sys.stderr)
            cmd = [sys.executable, __file__, '--single',
                   '--mode', mode, '--engine', engine, '--graph', path,
                   '--iterations', str(iterations)]
            if not cse:
                cmd.append('--no-cse')
            if batch_size is not None:
                cmd += ['--batch_size', str(batch_size)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(' FAILED', file=sys.stderr)
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            row['rss'][(engine, mode)] = _parse_marker(proc.stdout, 'PEAK_RSS_MB')
            row['child'][(engine, mode)] = _parse_marker(proc.stdout, 'PEAK_CHILD_RSS_MB')
            row['s'][(engine, mode)] = _parse_marker(proc.stdout, 'ELAPSED_S')
            print(f' {row["rss"][(engine, mode)]:.0f} MB, '
                  f'{row["s"][(engine, mode)]:.1f} s', file=sys.stderr)
        rows.append(row)
        os.remove(path)

    _print_matrix(rows, configs, nodes, iterations, cse)


def _print_matrix(rows, configs, nodes, iterations, cse):
    """Render one sub-table per metric: config per row, edge count per column.

    child RSS is the largest external `sort` koala spawns (invisible to the
    process's own RSS); it stays ~0 for polars, which spawns no such helper.
    """
    label = lambda c: f'{c[0]}-{c[1]}'                        # ('koala','lazy') -> 'koala-lazy'
    sizes = [r['lines'] for r in rows]
    metrics = [('self RSS (MB)', 'rss'),
               ('child RSS (MB)', 'child'),
               ('elapsed (s)', 's')]

    name_w = max([len(t) for t, _ in metrics] + [len(label(c)) for c in configs])
    col_w = max(12, len(f'{max(sizes):,}') + 2)

    print()
    print(f'Fixed nodes = {nodes:,}   iterations = {iterations}   '
          f'CSE = {"on" if cse else "off"}   (columns = total lines)')
    print("(self RSS = this process; child RSS = koala's external sort, ~0 for "
          'polars. lazy self RSS ~flat as edges grow; eager climbs.)')

    for title, key in metrics:
        print()
        hdr = f'{title:<{name_w}}' + ''.join(f'{s:>{col_w},}' for s in sizes)
        print(hdr)
        print('-' * len(hdr))
        for c in configs:
            vals = ''.join(f'{r[key][c]:>{col_w}.1f}' for r in rows)
            print(f'{label(c):<{name_w}}{vals}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--single', action='store_true',
                        help='run one configuration and print its peak RSS')
    parser.add_argument('--mode', choices=('lazy', 'eager'), default='lazy',
                        help='how to feed edges to pagerank (single-run mode)')
    parser.add_argument('--engine', choices=tuple(ENGINES), default='polars',
                        help='which pagerank implementation to run in single-run '
                             'mode: polars (pagerank.py) or koala (pagerank_kl.py)')
    parser.add_argument('--configs', nargs='+', choices=ALL_CONFIGS,
                        default=ALL_CONFIGS, metavar='ENGINE-MODE',
                        help='engine-mode configs to compare in the table, a '
                             'subset of {' + ', '.join(ALL_CONFIGS) + '} '
                             '(default: all)')
    parser.add_argument('--graph', help='edge-list file to use (single-run mode)')
    parser.add_argument('--nodes', type=int, default=20000,
                        help='fixed node count for the comparison table')
    parser.add_argument('--edges', type=int, nargs='+',
                        default=[250_000, 1_000_000, 4_000_000],
                        help='edge counts to sweep for the comparison table')
    parser.add_argument('--iterations', type=int, default=10,
                        help='pagerank iterations per run')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='koala-only: rows per DataFrame emitted by '
                             "inner_join's IO source (default: koala's "
                             f'{kl.DEFAULT_BATCH_SIZE:,}). Smaller caps memory.')
    parser.add_argument('--no-cse', action='store_true',
                        help='disable common-subplan elimination in pagerank '
                             '(memory experiment; see MEMORY_DIAGNOSIS.md)')
    args = parser.parse_args()

    if args.single:
        run_single(args.mode, args.graph, args.iterations,
                   engine=args.engine, cse=not args.no_cse,
                   batch_size=args.batch_size)
    else:
        configs = [tuple(c.split('-')) for c in args.configs]
        run_matrix(args.nodes, args.edges, args.iterations,
                   configs=configs, cse=not args.no_cse,
                   batch_size=args.batch_size)
