import polars as pl

fg_lines = pl.scan_lines("../data/bluecorpus.txt")
bg_lines = pl.scan_lines("../data/redcorpus.txt")

def wc_pipe(ldf):
    lc = ldf.with_columns(
        pl.col("line").str.to_lowercase().alias('lowercase_line')).select('lowercase_line')
    tokenized = lc.with_columns(
        pl.col('lowercase_line').str.extract_all(r"\b\w+\b").alias('token')).select('token')
    words = tokenized.explode('token')
    wc = words.group_by('token').len()
    return wc

fg_word_count = wc_pipe(fg_lines)
bg_word_count = wc_pipe(bg_lines)

def row_count(ldf):
    return ldf.select(pl.len()).collect(engine='streaming').item()

fg_vocab_n = row_count(fg_word_count)
bg_vocab_n = row_count(bg_word_count)

wc_pairs = fg_word_count.join(bg_word_count, on='token', how='inner', suffix='_bg')
wc_pairs = wc_pairs.with_columns(len_fg=pl.col('len')).drop('len')

def show(msg, ldf, k=10):
    print(msg)
    print(ldf.head(k).collect(engine='streaming'))

score_counted_pairs = (
    wc_pairs.with_columns(
        fg_vocab_n = pl.lit(fg_vocab_n),
        bg_vocab_n = pl.lit(bg_vocab_n),
    ).with_columns(
        p_fg=((pl.col('len_fg') + 1.0 / pl.col('fg_vocab_n')) / (pl.col('fg_vocab_n') + 1)),
        p_bg=((pl.col('len_bg') + 1.0 / pl.col('bg_vocab_n')) / (pl.col('bg_vocab_n') + 1))
    ).with_columns(
        score=(pl.col('p_fg') / pl.col('p_bg')).log())
)

show('score_counted_pairs', score_counted_pairs)

reds = score_counted_pairs.sort('score', descending=False)
blues = score_counted_pairs.sort('score', descending=True)

show('most red', reds, 20)
show('most blue', blues, 20)

#reds = result.sortBy(lambda ws: ws[1], ascending=True)
#blues = result.sortBy(lambda ws: ws[1], ascending=False)
#
#n = 20
#print(f'top {n} most red:')
#for word, score in reds.take(20):
#    print(word, score)
#print()
#
#print(f'top {n} most blue:')
#for word, score in blues.take(20):
#    print(word, score)
#print()

