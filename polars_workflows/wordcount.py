import polars as pl

CORPUS = "../data/brown_nolines.txt"

word_counts = (
    # scan the corpus
    pl.scan_lines(CORPUS)
    # add tokenized line
    .with_columns(
        tokens=pl.col('line').str.extract_all(r'\b\w+\b')
    )
    # complete the flatmap
    .explode('tokens', empty_as_null=True)
    # a column with just words, labeled 'tokens'
    .select(token=pl.col('tokens'))
    # count and aggregate by length
    .group_by('token').len()
    .sort('len', descending=True)
)

print(word_counts.head(20).collect(engine='streaming'))
