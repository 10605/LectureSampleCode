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

import pagerank


def peak_rss_mb():
    """Peak resident set size of THIS process, in MiB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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


def run_single(mode, path, iterations, cse=True):
    """Run pagerank once and report this process's peak RSS."""
    pagerank.VERBOSE = 0                              # silence show() output
    lines = make_lines(path, mode)
    start = time.time()
    pagerank.pagerank(lines, num_iterations=iterations, cse=cse)
    elapsed = time.time() - start
    # markers parsed by the table driver
    print(f'PEAK_RSS_MB {peak_rss_mb():.1f}')
    print(f'ELAPSED_S {elapsed:.2f}')


def _parse_marker(text, marker):
    for line in text.splitlines():
        if line.startswith(marker):
            return float(line.split()[1])
    return float('nan')


def run_matrix(nodes, edge_sizes, iterations, cse=True):
    """Spawn a fresh subprocess per (edges, mode) and tabulate peak RSS."""
    rows = []
    for m in edge_sizes:
        path = f'/tmp/prdemo_{nodes}_{m}.txt'
        total_lines = generate_graph(path, nodes, m)
        row = {'lines': total_lines}
        for mode in ('lazy', 'eager'):
            cmd = [sys.executable, __file__, '--single',
                   '--mode', mode, '--graph', path,
                   '--iterations', str(iterations)]
            if not cse:
                cmd.append('--no-cse')
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            row[f'{mode}_rss'] = _parse_marker(proc.stdout, 'PEAK_RSS_MB')
            row[f'{mode}_s'] = _parse_marker(proc.stdout, 'ELAPSED_S')
        rows.append(row)
        os.remove(path)

    print()
    print(f'Fixed nodes = {nodes:,}   iterations = {iterations}   CSE = {"on" if cse else "off"}')
    print('(peak RSS should stay ~flat for lazy as edges grow; eager climbs)')
    print()
    hdr = f'{"total lines":>14}  {"lazy RSS MB":>12}  {"eager RSS MB":>13}  {"lazy s":>7}  {"eager s":>8}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["lines"]:>14,}  {r["lazy_rss"]:>12.1f}  {r["eager_rss"]:>13.1f}'
              f'  {r["lazy_s"]:>7.1f}  {r["eager_s"]:>8.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--single', action='store_true',
                        help='run one configuration and print its peak RSS')
    parser.add_argument('--mode', choices=('lazy', 'eager'), default='lazy',
                        help='how to feed edges to pagerank (single-run mode)')
    parser.add_argument('--graph', help='edge-list file to use (single-run mode)')
    parser.add_argument('--nodes', type=int, default=20000,
                        help='fixed node count for the comparison table')
    parser.add_argument('--edges', type=int, nargs='+',
                        default=[200_000, 1_000_000, 4_000_000],
                        help='edge counts to sweep for the comparison table')
    parser.add_argument('--iterations', type=int, default=10,
                        help='pagerank iterations per run')
    parser.add_argument('--no-cse', action='store_true',
                        help='disable common-subplan elimination in pagerank '
                             '(memory experiment; see MEMORY_DIAGNOSIS.md)')
    args = parser.parse_args()

    if args.single:
        run_single(args.mode, args.graph, args.iterations, cse=not args.no_cse)
    else:
        run_matrix(args.nodes, args.edges, args.iterations, cse=not args.no_cse)
