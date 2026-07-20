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
        .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)').alias('edge'))
        .with_columns(pl.col('edge').struct.rename_fields(['src', 'dst']))
        .select('edge')
    )
    # a variant with two distinct columns for src, dstinstead of a struct
    edges_two_cols = (
        edges
        .with_columns(
            src=pl.col('edge').struct['src'],
            dst=pl.col('edge').struct['dst'])
        .drop('edge')
    )
    show('edges', edges)
    show('edges_two_cols', edges_two_cols)

    # score for each node is kept in memory as a dict
    nodes = (
        pl.concat(
            [edges_two_cols.with_columns(node='src').select('node'),
             edges_two_cols.with_columns(node='dst').select('node')],
            how='vertical')
        .unique()
        .collect(engine='streaming')
    )
    # NOTE ON DANGLING NODES: score_dict is seeded from ALL nodes (src union
    # dst). A node with no indegree is never a 'dst', so it is never updated in
    # the loop below and keeps emitting messages from its stale 1.0 score every
    # iteration. This differs from the join-based pagerank.py / pagerank_kl.py,
    # where such a node is dropped after iteration 1 (its scores row disappears,
    # and the inner join on 'src' then drops its out-edges) so it stops
    # contributing. The two conventions therefore give different scores whenever
    # the "every node has indegree >= 1" assumption is violated -- e.g. the
    # citeseer graph has 11 no-indegree nodes, so this mapper's top scores run
    # slightly higher. On a graph with a self-loop per node (indegree >= 1 for
    # all, as in demo_memory.py) all three implementations agree exactly.
    score_dict = {node:1.0 for node in nodes['node']}

    # also in memory: number of outlinks from the src 
    n_outlinks = (
        edges_two_cols
        .group_by('src').len()
        .rename(dict(len='n_outlinks'))
        .collect(engine='streaming')
    )
    outlink_dict = {src:n for src,n in zip(n_outlinks['src'], n_outlinks['n_outlinks'])}

    if VERBOSE:
        print('pagerank scores:')
        print(list(score_dict.items())[0:4])
        print('num_outlinks:')
        print(list(outlink_dict.items())[0:4])

    # a mapping function that compute page rank message from an edge
    def makemapper(score_dict, outlink_dict):
        def delta(edge) -> float:
            src = edge['src']
            n_outs = outlink_dict[src]
            src_score = score_dict[src]
            return ((1.0 - reset) * src_score  / n_outs)
        return delta
    delta_fn = makemapper(score_dict, outlink_dict)

    # the main pagerank loop

    start = time.time()
    for t in range(num_iterations):
    
        # monitor progress
        num_scores = len(score_dict)
        max_score = max(score_dict.values())
        min_score = min(score_dict.values())
        mean_score = sum(score_dict.values()) / num_scores
        print(f'iteration {t + 1:2d} of {num_iterations} time {time.time() - start:.4f}', 
              f'len {num_scores} max {max_score:.2f} min {min_score:.2f} mean {mean_score:.2f}')

        # distribute scores from src to destinations - each msg is the
        # part of the src's score that will be sent to the dst via a 'hop'
        pr_messages = (
            edges
            # apply mapper that uses in-memory dicts - a 'map-side join'
            .with_columns(pl.col('edge').map_elements(delta_fn, return_dtype=pl.Float64).alias('delta'))
            # structure dataframe for the reduce stage below
            .with_columns(dst=pl.col('edge').struct['dst'])
            .select('dst', 'delta')
        )

        # new scores: add up the incoming pagerank messages and add the reset
        score_df = (
            pr_messages
            # add up the deltas
            .group_by('dst').agg(pl.col('delta').sum().cast(pl.Float64).alias('incoming_pr'))
            # add in the reset
            .with_columns(score=(pl.col('incoming_pr') + reset))
            # normalize the names
            .rename(dict(dst='node'))
            .select('node', 'score')
            # and put in memory
            .collect(engine='streaming')
        )
        # then load into the score_dict used by the mapper
        for node, score in zip(score_df['node'], score_df['score']): 
            score_dict[node] = score 

    elapsed = time.time() - start
    score_df=score_df.sort('score', descending=True)
    print('top pagerank_scores:')
    print(score_df.head(10))
    print('bottom pagerank_scores:')
    print(score_df.tail(10))
    print('scores collected [python mapper]:',elapsed,'sec')

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
    
