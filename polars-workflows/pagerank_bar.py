"""Pagerank with the group_by and join steps offloaded to disk.

Same algorithm as pagerank.py, but every group_by/join goes through PolarBar,
which sorts and merges key-value tsv files instead of building hash tables in
memory.  Everything else -- parsing edges, arithmetic on the messages, re-keying
them from src to dst -- stays in polars, streaming.

The kv files are plain text (encode=str), so a value may itself hold several
tab-separated fields: `_pairs_from` splits only on the first tab, and polars
reads the rest back as extra columns.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import polars as pl

import bar as pb

VERBOSE = 1

def show(msg, df, k=10):
    """Give a quick view of the contents of a dataframe.
    """
    if VERBOSE < 1:
        return
    print(msg)
    if isinstance(df, pl.LazyFrame):
        df = df.head(k).collect(engine='streaming')
    print(df)

def scan_scores(scores_tsv_path):
    """Lazily read a node/score kv file back into polars.
    """
    return pl.scan_csv(
        scores_tsv_path, separator='\t', has_header=False, quote_char=None,
        schema={'node': pl.String, 'score': pl.Float64})

def pagerank(edge_lines, reset=0.15, num_iterations=30, workdir=None, batch_size=None):
    """Pagerank, with every group_by/join spilled to kv files under workdir.

    workdir=None uses a temp directory, removed on the way out.  batch_size is
    PolarBar's sort_batch_size: the number of pairs held in memory per run.
    """
    own_workdir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix='pagerank-bar-')
    try:
        return _pagerank(edge_lines, reset, num_iterations, workdir, batch_size)
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

def _pagerank(edge_lines, reset, num_iterations, workdir, batch_size):
    bar = pb.PolarBar(encode=str, decode=lambda x: x,
                      **({} if batch_size is None else dict(sort_batch_size=batch_size)))
    path = lambda name: os.path.join(workdir, name)

    # construct edges
    edges = (
        edge_lines
        .with_columns(edge=pl.col('line').str.extract_groups(r'(\S+)\s+(\S+)'))
        .with_columns(
            src=pl.col('edge').struct['1'],
            dst=pl.col('edge').struct['2'])
        .select('src', 'dst')
    )

    # spill the edges to disk as src -> dst, sorted by src
    bar.sink_kv_pairs(edges, path('edges.tsv'), key_col='src', val_col='dst')
    bar.sort_kv_pairs(path('edges.tsv'), path('edges-sorted.tsv'))

    # augment edges with number of outlinks from the src
    #   n_outlinks = edges.group_by('src').len()
    bar.group_and_aggregate_by_key(
        path('edges-sorted.tsv'), path('n_outlinks.tsv'), agg=pb.count)
    #   edges = edges.join(n_outlinks, on='src')
    bar.join_by_key(
        path('n_outlinks.tsv'), path('edges-sorted.tsv'), path('edges-with-outlinks.tsv'),
        combine_vals=lambda n_outlinks, dst: f'{n_outlinks}\t{dst}')
    show('edges', pl.scan_csv(
        path('edges-with-outlinks.tsv'), separator='\t', has_header=False, quote_char=None,
        schema={'src': pl.String, 'n_outlinks': pl.Int64, 'dst': pl.String}))

    # initialize a score for each node
    #   the .unique() is the dedup, so it is a group_by: keep the first of each
    #   run of equal keys
    nodes_with_dups = (
        pl.concat(
            [edges.with_columns(node='src').select('node'),
             edges.with_columns(node='dst').select('node')],
            how='vertical')
        .with_columns(score=pl.lit(1.0).cast(pl.Float64))
    )
    bar.sink_kv_pairs(
        nodes_with_dups, path('nodes-with-dups.tsv'), key_col='node', val_col='score')
    bar.sort_kv_pairs(path('nodes-with-dups.tsv'), path('nodes-sorted.tsv'))
    bar.group_and_aggregate_by_key(
        path('nodes-sorted.tsv'), path('scores-init.tsv'), agg=pb.first)
    scores_path = path('scores-init.tsv')
    show('pagerank_scores', scan_scores(scores_path))

    #
    # main loop - this assumes all nodes have indegree >=1 and outdegree >= 1
    #
    start = time.time()
    for t in range(num_iterations):

        # monitor progress -- a global aggregate, so polars streams it
        stats = scan_scores(scores_path).select(
            num_scores=pl.len(),
            min_score=pl.col('score').min(),
            max_score=pl.col('score').max(),
            mean_score=pl.col('score').mean(),
        ).collect(engine='streaming').row(0, named=True)
        print(f'iteration {t + 1:2d} of {num_iterations} time {time.time() - start:.4f}',
              f'len {stats["num_scores"]} max {stats["max_score"]:.2f}',
              f'min {stats["min_score"]:.2f} mean {stats["mean_score"]:.2f}')

        # distribute scores from src to destinations - each msg is the
        # part of the src's score that will be sent to the dst via a 'hop'
        #   edges.join(pagerank_scores, left_on='src', right_on='node')
        # scores are on the left so only one value per key is buffered
        bar.join_by_key(
            scores_path, path('edges-with-outlinks.tsv'), path('msgs-by-src.tsv'),
            combine_vals=lambda score, n_and_dst: f'{score}\t{n_and_dst}')

        # the delta arithmetic and the re-key from src to dst are not groupings,
        # so polars does them, streaming, on the way back out to disk
        pr_messages = (
            pl.scan_csv(
                path('msgs-by-src.tsv'), separator='\t', has_header=False, quote_char=None,
                schema={'src': pl.String, 'score': pl.Float64,
                        'n_outlinks': pl.Int64, 'dst': pl.String})
            .with_columns(delta=((pl.col('score') / pl.col('n_outlinks')) * (1 - reset)).cast(pl.Float64))
            .select('dst', 'delta')
        )
        bar.sink_kv_pairs(pr_messages, path('msgs.tsv'), key_col='dst', val_col='delta')

        # create the new scores: add up the incoming pagerank messages and add
        # the reset
        #   pr_messages.group_by('dst').agg(pl.col('delta').sum())
        bar.sort_kv_pairs(path('msgs.tsv'), path('msgs-sorted.tsv'))
        next_scores_path = path(f'scores-{t % 2}.tsv')
        bar.group_and_aggregate_by_key(
            path('msgs-sorted.tsv'), next_scores_path,
            agg=lambda deltas: sum(map(float, deltas)) + reset)
        scores_path = next_scores_path

    elapsed = time.time() - start
    pagerank_scores = scan_scores(scores_path).sort('score', descending=True).collect(
        engine='streaming')
    print('top pagerank_scores:')
    print(pagerank_scores.head(10))
    print('bottom pagerank_scores:')
    print(pagerank_scores.tail(10))
    print('scores collected [polar + bar]:', elapsed, 'sec')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute pagerank scores for a graph.')
    parser.add_argument('--filename', default='../data/citeseer-graph.txt',
                        help='edge list file to load (may be gzipped)')
    parser.add_argument('--reset', type=float, default=0.15,
                        help='reset (teleport) probability')
    parser.add_argument('--num_iterations', type=int, default=30,
                        help='number of pagerank iterations')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level: 1 shows show() output, 0 suppresses it')
    parser.add_argument('--workdir', default=None,
                        help='where to keep the spilled kv files; default is a '
                             'temp directory, deleted on exit')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='pairs held in memory per sorted run '
                             "(PolarBar's sort_batch_size)")
    args = parser.parse_args()

    VERBOSE = args.verbose

    filename = args.filename
    print(f'loading from {filename}')
    if filename.endswith('.gz'):
        import subprocess
        process = subprocess.Popen(
            [f"gunzip -c {filename} | grep -v '#'"],
            shell=True,
            text=False,
            stdout=subprocess.PIPE,
        )
        lines = pl.scan_lines(process.stdout)
    else:
        lines = pl.scan_lines(filename)

    if args.workdir is not None:
        os.makedirs(args.workdir, exist_ok=True)
    pagerank(lines, reset=args.reset, num_iterations=args.num_iterations,
             workdir=args.workdir, batch_size=args.batch_size)
