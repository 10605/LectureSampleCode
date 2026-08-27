import argparse
import os
import resource
import sys
import time

import polars as pl
import bar as pb

DEFAULT_DATA_DIR = '../../bigml/data/RCV1'

# the corpora are RCV1.<size>_<split>.txt, except for the combined split,
# which has no suffix at all: RCV1.<size>.txt
SIZES = ['very_small', 'small', 'full', '10x']
SPLITS = ['train', 'test', 'all']

def corpus_path(data_dir, size, split):
    """Locate one RCV1 corpus.  Not every size/split pair exists (10x is
    train-only), so fail here rather than deep inside polars.
    """
    stem = f'RCV1.{size}' if split == 'all' else f'RCV1.{size}_{split}'
    path = os.path.join(data_dir, stem + '.txt')
    if not os.path.exists(path):
        raise SystemExit(f'no such corpus: {path}')
    return path

def peak_rss_gb(who=resource.RUSAGE_SELF):
    """Peak resident set size, in MiB.

    who=RUSAGE_SELF  -> this Python process only.
    who=RUSAGE_CHILDREN -> largest waited-for child (e.g. an external
        `sort`); this is invisible to RUSAGE_SELF, so measure it separately.
    """
    r = resource.getrusage(who).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    return r / 1024**3 if sys.platform == 'darwin' else r / 1024**2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Time and measure sink/sort/group on an RCV1 corpus.')
    parser.add_argument('--size', choices=SIZES, default='full',
                        help='which corpus to read (default: full)')
    parser.add_argument('--split', choices=SPLITS, default='train',
                        help="which split to read; 'all' is the combined file "
                             '(default: train)')
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                        help=f'where the RCV1 corpora live (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--batch_size', type=int, default=1024 * 1024,
                        help='pairs held in memory per sorted run '
                             "(PolarBar's sort_batch_size, default: 1048576)")
    parser.add_argument('--fanin', type=int, default=0,
                        help='runs merged at once; 0 scales it to sqrt(#runs) '
                             "(PolarBar's sort_fanin, default: 0)")
    parser.add_argument('--show_counts', action='store_true',
                        help='first count the tokens in polars, for comparison; '
                             'this is the memory-hungry path the rest of the '
                             'script exists to avoid (>40Gb on full)')
    args = parser.parse_args()

    file_name = corpus_path(args.data_dir, args.size, args.split)

    print(f'loading and tokenizing {file_name} ....')
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
    sort_batch_size = args.batch_size
    if args.show_counts:
        # just the count uses >40Gb here which is quite inefficient
        print(f'counting - {peak_rss_gb()} peak rss gb')
        start = time.time()
        n_tokens = df.select(pl.len()).collect().item()
        print(f'df {n_tokens} tokens, {n_tokens/sort_batch_size} batches:\n', df.head(3).collect())
        print(f'elapsed time {time.time()-start}')


    bar = pb.PolarBar(encode=str, decode=lambda x:x, sort_batch_size=sort_batch_size,
                      sort_fanin=args.fanin)

    print(f'sinking - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.sink_kv_pairs(df, 'unsort.csv', 'key', 'val')
    print(f'sink {time.time() - start:.4f} sec elapsed')

    print(f'sorting - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.sort_kv_pairs('unsort.csv', 'sort.csv')
    print(f'sort {time.time() - start:.4f} sec elapsed')

    print(f'grouping - {peak_rss_gb():.4f} peak rss gb')
    start = time.time()
    bar.group_and_aggregate_by_key('sort.csv', 'group.csv', pb.count)
    print(f'group {time.time() - start:.4f} sec elapsed')    
