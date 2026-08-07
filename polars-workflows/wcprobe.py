import resource
import sys
import time

import polars as pl
import bar as pb

def peak_rss_gb(who=resource.RUSAGE_SELF):
    """Peak resident set size, in MiB.

    who=RUSAGE_SELF  -> this Python process only.
    who=RUSAGE_CHILDREN -> largest waited-for child (e.g. koala's external
        `sort`); this is invisible to RUSAGE_SELF, so measure it separately.
    """
    r = resource.getrusage(who).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    return r / 1024**3 if sys.platform == 'darwin' else r / 1024**2

if __name__ == '__main__':
    size = '10x'
    split = 'train'
    file_name = f'../../bigml/data/RCV1/RCV1.{size}_{split}.txt'
    
    print('loading and tokenizing....')
    line_df = (
        pl.scan_csv(
            file_name,
            separator='\t',
            has_header=False,
            new_columns=['labels', 'text'],
            schema_overrides={'labels': pl.String, 'text': pl.String},
            encoding='utf8-lossy')
        .with_row_index('id')
    )
    tokenized_df = (
        line_df.select(
            pl.col('id'),
            pl.col('labels').str.split(','),
            pl.col('text').str.extract_all(r'\w+').alias('tokens'))
    )
    labeled_word_df = (
        tokenized_df
        .explode('tokens')
        .explode('labels')
    )
    df = (
        labeled_word_df
        .with_columns(
            key=pl.format('{}##{}', pl.col('id'), pl.col('labels')),
            val=pl.col('tokens'))
        .select('key', 'val')
    )
    sort_batch_size=1024*1024
    if False:
        # just the count uses >40Gb here which is quite inefficient
        print(f'counting - {peak_rss_gb()} peak rss gb')
        start = time.time()
        n_tokens = df.select(pl.len()).collect().item()
        print(f'df {n_tokens} tokens, {n_tokens/sort_batch_size} batches:\n', df.head(3).collect())
        print(f'elapsed time {time.time()-start}')


    bar = pb.PolarBar(encode=str, decode=lambda x:x, sort_batch_size=sort_batch_size, sort_fanin=0)

    print(f'sinking - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.sink_kv_pairs(df, 'unsort.csv', 'key', 'val')
    print(f'sink {time.time() - start:.4f} sec elapsed')

    # on full:
    #
    # (1) 1024*1024 sort_batch_size means about 4 spills/sec and 558
    # batches so spilling is ~ 2:30 

    # (2) using sqrt(#batches) as fan-in gives 24 merges in first pass
    # at about 10s/merge so first pass is ~ 3:30, last pass is ~ 6:30
    # total sort is ~ 9:10

    # didn't track peak_rss_gb - maybe 10Gb? the full files are about
    # 10Gb. unix sort > 100Gb and is not faster (~ 16:30 +)

    # 10x: should be ~ 5580 spills ~ 30min, ~75 merges 75-way
    # merges are ~ 30s/merge, run up to 20Gb, first pass was
    # ~ 40min, second ~ 65min, peaked at ~ 22 Gb.

    print(f'sorting - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.sort_kv_pairs('unsort.csv', 'sort.csv')
    print(f'sort {time.time() - start:.4f} sec elapsed')

    # grouping is ~ 2min on full, ... ~19m on 10x

    print(f'grouping - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.group_and_aggregate_by_key('sort.csv', 'group.csv', pb.count)
    print(f'group {time.time() - start:.4f} sec elapsed')    
