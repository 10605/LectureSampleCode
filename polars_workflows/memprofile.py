"""memprofile.py -- localize where koala's peak RSS goes.

Runs each koala operation that appears in a pagerank iteration on its own, at a
range of edge counts, each in a fresh subprocess so peak RSS (RUSAGE_SELF) is
clean. Ops that produce an edge-proportional stream are sunk to /dev/null so the
number measured is the PIPELINE's resident memory, not a held result.

Finding this was written to capture (see TODO.md #4): `inner_join` is the sink.
It batches by KEY but each key carries all its rows, so peak memory is
O(min(batch_size, #keys) x rows-per-key) -- one batch = the whole join when
#keys <= batch_size. sum/len stay ~O(nodes); the external sort stays bounded.

Usage:
    python memprofile.py                          # full op x edges table
    python memprofile.py --edges 500_000 4_000_000 --ops join iter
    python memprofile.py --op join --batch_size 1000 --graph /tmp/g.txt  # one run

    # sweep just inner_join's batch_size on a fixed graph to see it scale:
    for B in 100 1000 20000 100000; do
        python memprofile.py --op join --batch_size $B --graph /tmp/g.txt
    done
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

DEVNULL = os.devnull


def rss(who=resource.RUSAGE_SELF):
    """Peak resident set size in MiB (macOS: bytes; Linux: KiB)."""
    r = resource.getrusage(who).ru_maxrss
    return r / 1024**2 if sys.platform == 'darwin' else r / 1024


def generate_graph(path, n_nodes, n_edges, seed=1):
    """Edge list with a self-loop per node (indegree/outdegree >= 1) plus
    n_edges random edges. Node count fixed; grow n_edges to stress memory."""
    rng = random.Random(seed)
    with open(path, 'w') as f:
        for i in range(n_nodes):
            f.write(f'{i} {i}\n')
        for _ in range(n_edges):
            f.write(f'{rng.randrange(n_nodes)} {rng.randrange(n_nodes)}\n')
    return n_nodes + n_edges


def edges_lf(path):
    return (pl.scan_lines(path)
            .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)'))
            .with_columns(src=pl.col('edge').struct['1'], dst=pl.col('edge').struct['2'])
            .select('src', 'dst'))


# --- the ops: each takes (edges, batch_size); only `join` uses batch_size ---

def op_scan(edges, batch_size):                    # baseline read/parse cost
    edges.sink_csv(DEVNULL)

def op_gbk_len(edges, batch_size):                 # outdegree reduce -> O(nodes)
    kl.group_by_key(edges, 'src', kl.reduce('dst', to='n', via=len)).sink_csv(DEVNULL)

def op_gbk_sum(edges, batch_size):                 # sum reduce -> O(nodes)
    w = edges.with_columns(w=(pl.col('src').str.len_chars() % 5 + 1).cast(pl.Float64))
    kl.group_by_key(w, 'dst', kl.reduce('w', to='incoming', via=sum)).sink_csv(DEVNULL)

def op_join(edges, batch_size):                    # inner_join (list-gather), streamed out
    n_out = kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len))
    kl.inner_join(edges, n_out, on='src', batch_size=batch_size).sink_csv(DEVNULL)

def op_merge(edges, batch_size):                   # merge_join (streaming), streamed out
    n_out = kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len))
    kl.merge_join(edges, n_out, on='src', batch_size=batch_size).sink_csv(DEVNULL)

def op_iter(edges, batch_size):                    # inner_join + sum per dst -> O(nodes)
    n_out = kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len))
    joined = kl.inner_join(edges, n_out, on='src', batch_size=batch_size)
    msgs = joined.with_columns(delta=(1.0 / pl.col('n_outlinks')).cast(pl.Float64)) \
                 .select('dst', 'delta')
    kl.group_by_key(msgs, 'dst', kl.reduce('delta', to='pr', via=sum)) \
      .collect(engine='streaming')

def op_iter_merge(edges, batch_size):              # merge_join + sum per dst -> O(nodes)
    n_out = kl.group_by_key(edges, 'src', kl.reduce('dst', to='n_outlinks', via=len))
    joined = kl.merge_join(edges, n_out, on='src', batch_size=batch_size)   # values are strings
    msgs = joined.with_columns(delta=(1.0 / pl.col('n_outlinks').cast(pl.Float64))) \
                 .select('dst', 'delta')
    kl.group_by_key(msgs, 'dst', kl.reduce('delta', to='pr', via=sum)) \
      .collect(engine='streaming')

OPS = {'scan': op_scan, 'gbk_len': op_gbk_len, 'gbk_sum': op_gbk_sum,
       'join': op_join, 'merge': op_merge, 'iter': op_iter, 'iter_merge': op_iter_merge}


def run_single(op, graph, batch_size):
    t = time.time()
    OPS[op](edges_lf(graph), batch_size)
    print(f'SELF_MB {rss():.1f}')
    print(f'CHILD_MB {rss(resource.RUSAGE_CHILDREN):.1f}')
    print(f'ELAPSED_S {time.time() - t:.1f}')


def _marker(text, name):
    for line in text.splitlines():
        if line.startswith(name):
            return float(line.split()[1])
    return float('nan')


def run_matrix(nodes, edge_sizes, ops, batch_size):
    total, done = len(edge_sizes) * len(ops), 0
    results = {}                                   # (op, n_edges) -> {self, child, s}
    for n_edges in edge_sizes:
        path = f'/tmp/memprof_{nodes}_{n_edges}.txt'
        lines = generate_graph(path, nodes, n_edges)
        for op in ops:
            done += 1
            print(f'[{done}/{total}] {lines:,} lines  {op} ...',
                  end='', flush=True, file=sys.stderr)
            cmd = [sys.executable, __file__, '--op', op, '--graph', path,
                   '--batch_size', str(batch_size)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(' FAILED', file=sys.stderr)
                sys.stderr.write(proc.stderr)
                proc.check_returncode()
            results[(op, n_edges)] = {
                'self': _marker(proc.stdout, 'SELF_MB'),
                'child': _marker(proc.stdout, 'CHILD_MB'),
                's': _marker(proc.stdout, 'ELAPSED_S')}
            print(f' {results[(op, n_edges)]["self"]:.0f} MB', file=sys.stderr)
        os.remove(path)

    _print(results, ops, edge_sizes, nodes, batch_size)


def _print(results, ops, edge_sizes, nodes, batch_size):
    sizes = [nodes + e for e in edge_sizes]        # total lines per column
    metrics = [('self RSS (MB)', 'self'), ('child RSS (MB)', 'child'),
               ('elapsed (s)', 's')]
    name_w = max([len(t) for t, _ in metrics] + [len(o) for o in ops])
    col_w = max(12, len(f'{max(sizes):,}') + 2)

    print()
    print(f'Fixed nodes = {nodes:,}   batch_size = {batch_size:,}   '
          '(columns = total lines)')
    print('(op run alone; edge-proportional ops sunk to /dev/null so the number '
          'is pipeline memory, not a held result)')
    for title, key in metrics:
        print()
        hdr = f'{title:<{name_w}}' + ''.join(f'{s:>{col_w},}' for s in sizes)
        print(hdr)
        print('-' * len(hdr))
        for op, n_edges in ((o, e) for o in ops for e in [None]):  # rows = ops
            vals = ''.join(f'{results[(op, e)][key]:>{col_w}.1f}' for e in edge_sizes)
            print(f'{op:<{name_w}}{vals}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--op', choices=OPS, help='run one op (single-run worker mode)')
    ap.add_argument('--graph', help='edge-list file (single-run mode)')
    ap.add_argument('--ops', nargs='+', choices=OPS, default=list(OPS),
                    help='ops to profile in the table (default: all)')
    ap.add_argument('--nodes', type=int, default=20_000)
    ap.add_argument('--edges', type=int, nargs='+',
                    default=[500_000, 2_000_000, 4_000_000])
    ap.add_argument('--batch_size', type=int, default=kl.DEFAULT_BATCH_SIZE)
    args = ap.parse_args()

    if args.op:
        run_single(args.op, args.graph, args.batch_size)
    else:
        run_matrix(args.nodes, args.edges, args.ops, args.batch_size)
