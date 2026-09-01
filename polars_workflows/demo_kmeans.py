"""Benchmark spherical k-means on a corpus, sweeping the number of clusters.

kmeans.py holds only the O(k x vocab) centroid table in memory and streams the
document vectors past it, so cost should grow with k rather than with the
corpus.  This sweeps k to see whether that holds.

Each k runs in its own subprocess, so peak RSS is that run's alone -- the same
arrangement demo_memory.py uses, and its helpers are reused here.

Usage:
    # the corpora shipped with the repo
    python3 demo_kmeans.py --vectors ../data/bluecorpus.tfidf.tsv

    # RCV1: 724k documents, 77M nonzeros (vectors built offline)
    python3 demo_kmeans.py --vectors ../data/RCV1.full_train.tfidf.tsv \
        --k 2 4 8 16 32 64
"""
import argparse
import re
import subprocess
import sys
import time

import kmeans
from demo_memory import peak_rss_mb, _parse_marker


def run_single(vectors, k, max_iterations, seed):
    """Cluster once and report this process's peak RSS."""
    kmeans.VERBOSE = 0                               # silence show() output
    start = time.time()
    assignment_df, _ = kmeans.kmeans(
        kmeans.scan_vectors(vectors), k=k,
        max_iterations=max_iterations, seed=seed)
    elapsed = time.time() - start
    # markers parsed by the sweep driver
    print(f'PEAK_RSS_MB {peak_rss_mb():.1f}')
    print(f'ELAPSED_S {elapsed:.2f}')
    print(f'MEAN_COSINE {assignment_df["sim"].mean():.4f}')


def run_sweep(vectors, ks, max_iterations, seed):
    rows = []
    for i, k in enumerate(ks, 1):
        print(f'[{i}/{len(ks)}] k={k} ...', end='', flush=True, file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, __file__, '--single', '--vectors', vectors,
             '--k', str(k), '--max_iterations', str(max_iterations),
             '--seed', str(seed)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(' FAILED', file=sys.stderr)
            sys.stderr.write(proc.stderr)
            proc.check_returncode()
        # kmeans.py prints one 'iteration N of M' line per pass; it stops early
        # once nothing moves, so the count is a result, not a setting.
        done = re.findall(r'^iteration\s+(\d+) of', proc.stdout, re.MULTILINE)
        rows.append(dict(
            k=k,
            iters=int(done[-1]) if done else 0,
            rss=_parse_marker(proc.stdout, 'PEAK_RSS_MB'),
            s=_parse_marker(proc.stdout, 'ELAPSED_S'),
            cos=_parse_marker(proc.stdout, 'MEAN_COSINE')))
        print(f' {rows[-1]["rss"]:.0f} MB, {rows[-1]["s"]:.1f} s', file=sys.stderr)
    _print_table(vectors, rows, max_iterations, seed)


def _print_table(vectors, rows, max_iterations, seed):
    print(f'\n{vectors}   max_iterations = {max_iterations}   seed = {seed}')
    hdr = f'{"k":>5}{"iters":>8}{"peak RSS (MB)":>16}{"elapsed (s)":>14}{"mean cosine":>14}'
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["k"]:>5}{r["iters"]:>8}{r["rss"]:>16.1f}'
              f'{r["s"]:>14.1f}{r["cos"]:>14.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--single', action='store_true',
                        help='run one k and print its peak RSS')
    parser.add_argument('--vectors', default='../data/bluecorpus.tfidf.tsv',
                        help='tf-idf vectors as a (doc, token, w) tsv')
    parser.add_argument('--k', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                        help='cluster counts to sweep (one value with --single)')
    parser.add_argument('--max_iterations', type=int, default=20,
                        help='cap on iterations; k-means stops early if it '
                             'converges first')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for the initial choice of centroids')
    args = parser.parse_args()

    if args.single:
        run_single(args.vectors, args.k[0], args.max_iterations, args.seed)
    else:
        run_sweep(args.vectors, args.k, args.max_iterations, args.seed)
