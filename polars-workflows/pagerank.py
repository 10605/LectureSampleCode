import argparse
import sys
import time

import polars as pl
        
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

def pagerank(edge_lines, reset=0.15, num_iterations=30):

    # construct edges
    edges = (
        edge_lines
        .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)'))
        .with_columns(
            src=pl.col('edge').struct['1'],
            dst=pl.col('edge').struct['2'])
        .select('src', 'dst')
    )
    # augment edges with number of outlinks from the src 
    n_outlinks = (
        edges.group_by('src').len()
        .rename(dict(len='n_outlinks'))
    )
    edges = edges.join(n_outlinks, on='src')
    show('edges', edges)

    # initialize a score for each node 
    # these are kept in memory - a non-lazy DataFrame
    pagerank_scores = (
        pl.concat(
            [edges.with_columns(node='src').select('node'),
             edges.with_columns(node='dst').select('node')],
            how='vertical')
        .unique()
        .with_columns(score=pl.lit(1.0).cast(pl.Float64))
        .collect(engine='streaming')
    )
    show('pagerank_scores', pagerank_scores)

    #
    # main loop
    #
    start = time.time()
    for t in range(num_iterations):
    
        # monitor progress
        num_scores = pagerank_scores.select(pl.len()).item()
        min_score = pagerank_scores.select('score').min().item()    
        max_score = pagerank_scores.select('score').max().item()    
        mean_score = pagerank_scores.select('score').mean().item()    
        sum_score = pagerank_scores.select('score').sum().item()    
        print(f'iteration {t + 1:2d} of {num_iterations} time {time.time() - start:.4f}', 
              f'len {num_scores} max {max_score:.2f} min {min_score:.2f} mean {mean_score:.2f}')

        # distribute scores from src to destinations - each msg is the
        # part of the src's score that will be sent to the dst via a 'hop'
        pr_messages = (
            edges
            # note we have to make the scores lazy to join them
            .join(pagerank_scores.lazy(), left_on='src', right_on='node')
            .with_columns(delta=((pl.col('score') / pl.col('n_outlinks')) * (1 - reset)).cast(pl.Float64))
            .select('dst', 'delta')
        )

        # create the new scores: add up the incoming pagerank messages and add the reset
        pagerank_scores = (
            pr_messages
            .group_by('dst').agg(pl.col('delta').sum().alias('incoming_pr'))
            .with_columns(score=(pl.col('incoming_pr') + reset))
            # fix up the names and collect
            .rename(dict(dst='node'))
            .select('node', 'score')
            # and put in memory
            .collect(engine='streaming')
        )

    elapsed = time.time() - start
    pagerank_scores=pagerank_scores.sort('score', descending=True)
    print('top pagerank_scores:')
    print(pagerank_scores.head(10))
    print('bottom pagerank_scores:')
    print(pagerank_scores.tail(10))
    print('scores collected [pure polar]:',elapsed,'sec')

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
    pagerank(lines, reset=args.reset, num_iterations=args.num_iterations)
