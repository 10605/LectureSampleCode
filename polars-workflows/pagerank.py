import time

import polars as pl
        
RESET = 0.15
NUM_ITERATIONS = 20

def show(msg, ldf, k=10):
    """Give a quick view of the contents of a lazy dataframe.
    """
    print(msg)
    print(ldf.head(k).collect(engine='streaming'))

# read in edges
lines = pl.scan_lines("../data/citeseer-graph.txt")
edges = (
    lines
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

# score for each node is kept in memory - an non-lazy DataFrame
pagerank_scores = (
    pl.concat(
        [edges.with_columns(node='src').select('node'),
         edges.with_columns(node='dst').select('node')],
        how='vertical')
    .unique()
    .with_columns(score=pl.lit(1.0))
)
pagerank_scores = pagerank_scores.collect(engine='streaming')

print('pagerank_scores:')
print(pagerank_scores)

start = time.time()
for t in range(NUM_ITERATIONS):
    
    # monitor progress
    num_scores = pagerank_scores.select(pl.len()).item()
    min_score = pagerank_scores.select('score').min().item()    
    max_score = pagerank_scores.select('score').max().item()    
    mean_score = pagerank_scores.select('score').mean().item()    
    sum_score = pagerank_scores.select('score').sum().item()    
    print(f'iteration {t + 1:2d} of {NUM_ITERATIONS} time {time.time() - start:.4f}', 
          f'len {num_scores} max {max_score:.2f} min {min_score:.2f} mean {mean_score:.2f}')

    # distribute scores from src to destinations - each msg is the
    # part of the src's score that will be sent to the dst via a 'hop'
    pr_messages = (
        edges
        # note we have to make the scores lazy to join them
        .join(pagerank_scores.lazy(), left_on='src', right_on='node')
        .with_columns(delta=((pl.col('score') / pl.col('n_outlinks')) * (1 - RESET)))
        .select('dst', 'delta')
    )

    # new scores: add up the incoming pagerank messages and add the RESET
    pagerank_scores = (
        pr_messages
        .group_by('dst').agg(pl.col('delta').sum().alias('incoming_pr'))
        .with_columns(score=(pl.col('incoming_pr') + RESET))
        # fix up the names to be like pagerank_scores
        .rename(dict(dst='node'))
        .select('node', 'score')
        # and put in memory
        .collect(engine='streaming')
    )


print('scores collected [pure polar]:',time.time() - start,'sec')

pagerank_scores=pagerank_scores.sort('score', descending=True)

print('top pagerank_scores:')
print(pagerank_scores.head(10))

print('bottom pagerank_scores:')
print(pagerank_scores.tail(10))
