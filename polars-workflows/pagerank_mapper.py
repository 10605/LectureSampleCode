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
    .with_columns(edge=pl.col('line').str.extract_groups(r'(\w+)\s+(\w+)').alias('edge'))
    .with_columns(pl.col('edge').struct.rename_fields(['src', 'dst']))
    .drop('line')
)

edges_two_cols = (
    edges
    .with_columns(
        src=pl.col('edge').struct['src'],
        dst=pl.col('edge').struct['dst'])
    .drop('edge')
)

n_outlinks = (
    edges_two_cols
    .group_by('src').len()
    .rename(dict(len='n_outlinks'))
    .collect(engine='streaming')
)

# score for each node is kept in memory - an non-lazy DataFrame
nodes = (
    pl.concat(
        [edges_two_cols.with_columns(node='src').select('node'),
         edges_two_cols.with_columns(node='dst').select('node')],
        how='vertical')
    .unique()
    .collect(engine='streaming')
)
# also number of outlinks from the src 

score_dict = {node:1.0 for node in nodes['node']}
outlink_dict = {src:n for src,n in zip(n_outlinks['src'], n_outlinks['n_outlinks'])}

print('pagerank scores:')
print(list(score_dict.items())[0:4])

print('num_outlinks:')
print(list(outlink_dict.items())[0:4])

def makemapper(score_dict, outlink_dict):
    def delta(edge) -> float:
        src = edge['src']
        dst = edge['src']
        n_outs = outlink_dict[src]
        src_score = score_dict[src]
        return (1.0 - RESET) * src_score  / n_outs
    return delta

delta_fn = makemapper(score_dict, outlink_dict)

start = time.time()
for t in range(NUM_ITERATIONS):
    
    # monitor progress
    num_scores = len(score_dict)
    max_score = max(score_dict.values())
    min_score = min(score_dict.values())
    mean_score = sum(score_dict.values()) / num_scores
    print(f'iteration {t + 1:2d} of {NUM_ITERATIONS} time {time.time() - start:.4f}', 
          f'len {num_scores} max {max_score:.2f} min {min_score:.2f} mean {mean_score:.2f}')

    # distribute scores from src to destinations - each msg is the
    # part of the src's score that will be sent to the dst via a 'hop'
    pr_messages = (
        edges
        # apply mapper that uses in-memory dicts
        .with_columns(pl.col('edge').map_elements(delta_fn, return_dtype=pl.Float64).alias('delta'))
        # structure dataframe for the reduce stage 
        .with_columns(dst=pl.col('edge').struct['dst'])
        .select('dst', 'delta')
    )

    # new scores: add up the incoming pagerank messages and add the RESET
    score_df = (
        pr_messages
        # add up the deltas
        .group_by('dst').agg(pl.col('delta').sum().alias('incoming_pr'))
        # add in the RESET
        .with_columns(score=(pl.col('incoming_pr') + RESET))
        # normalize the names
        .rename(dict(dst='node'))
        .select('node', 'score')
        # and put in memory
        .collect(engine='streaming')
    )
    # then load into the score_dict used by the mapper
    for node, score in zip(score_df['node'], score_df['score']): 
        score_dict[node] = score 

print('scores collected [python mapper]:',time.time() - start,'sec')

score_df=score_df.sort('score', descending=True)

print('top pagerank_scores:')
print(score_df.head(10))

print('bottom pagerank_scores:')
print(score_df.tail(10))
