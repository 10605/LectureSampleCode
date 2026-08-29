import polars as pl

def show(msg, ldf, k=10):
    """Give a quick view of the contents of a lazy dataframe.
    """
    print(msg)
    print(ldf.head(k).collect(engine='streaming'))

# two corpora
fg_lines = pl.scan_lines("../data/bluecorpus.txt")
bg_lines = pl.scan_lines("../data/redcorpus.txt")

def lowercase(expr):
    return expr.str.to_lowercase()

def wordcount_pipe(ldf):
    return(
        ldf
        .with_columns(
            token=(
                pl.col('line')
                .pipe(lowercase)
                .str.extract_all(r'\b\w+\b')))
        .explode('token')
        .group_by('token').len()
    )


def shift(expr):
    return expr.str.extract(r'\W*\w+\W+(.*)')

def bigrams(expr):
    return expr.str.replace_all(r'\W+', ' ').str.extract_all(r'\b\w+ \w+')


def bigram_pipe(ldf):
    return (
        ldf
        .with_columns(line=pl.col('line').pipe(lowercase))
        .with_columns(bigrams=pl.col('line').pipe(bigrams))
        .with_columns(shifted_line=pl.col('line').pipe(shift))
        .with_columns(shifted_bigrams=pl.col('shifted_line').pipe(bigrams))
        .with_columns(bigrams=pl.col('bigrams').list.concat('shifted_bigrams'))
        .select('bigrams')
    )

def bigram_count(ldf):
    bigram_ldf = bigram_pipe(ldf)
    return bigram_ldf.with_columns(bigram='bigrams').explode('bigram').group_by('bigram').len()

def voc_size(ldf):
    """Get vocabulary size from a wordcount-like lazy dataframe."""
    return ldf.select(pl.len()).collect(engine='streaming').item()

def total_count(ldf):
    """Get total number of word/bigram occurrences from a wordcount-like lazy dataframe."""
    return ldf.select('len').sum().collect(engine='streaming').item()

# wordcount-like dfs for phrases and words

fg_phrase_count = bigram_count(fg_lines)
bg_phrase_count = bigram_count(bg_lines)

fg_word_count = wordcount_pipe(fg_lines)
bg_word_count = wordcount_pipe(bg_lines)

def prob(k_expr_name, n_col_name):
    n = pl.col(n_col_name)
    return (pl.col(k_expr_name) + (1.0 / n)) / ( n + 1.0)

def pair_score_pipe(key, fg_df, bg_df, fg_n, bg_n):
    return (
        fg_df.join(bg_df, on=key, how='inner')
        .with_columns(
            fg_k='len', 
            bg_k='len_right',
            fg_n=pl.lit(fg_n),
            bg_n=pl.lit(bg_n))
        ## score by smoothed log odds of Pr(x|corpus1) / Pr(x|corpus2)
        .select(key, 'fg_k', 'bg_k', 'fg_n', 'bg_n')
        .with_columns(
            p_fg=prob('fg_k', 'fg_n'),
            p_bg=prob('bg_k', 'bg_n'))
        .with_columns(
            fg_score=(pl.col('p_fg') / pl.col('p_bg')).log())
        )


# 1: look at most informative words and phrases in the foreground vs
# background

word_pairs = pair_score_pipe(
    'token', fg_word_count, bg_word_count,
    voc_size(fg_word_count), voc_size(bg_word_count))

def show_extreme_scores(ldf, key, n=10):
    sorted = ldf.sort(key, descending=True)
    print(f'highest {n} by {key}:'.center(80))
    print(sorted.head(n).collect(engine='streaming'))
    print(f'lowest {n} by {key}:'.center(80))
    print(sorted.tail(n).collect(engine='streaming'))

print(' words '.center(80, '='))
show_extreme_scores(word_pairs, 'fg_score')

phrase_stats = pair_score_pipe(
    'bigram', fg_phrase_count, bg_phrase_count,
    voc_size(fg_phrase_count), voc_size(bg_phrase_count))

print(' phrases '.center(80, '='))
show_extreme_scores(phrase_stats, 'fg_score')

########### experiment 2: adding 'phraseness'

# bigram is two words "x y" - extract them

phrase_stats = (
    phrase_stats
    .with_columns(xy=pl.col('bigram').str.extract_groups(r'(\w+) (\w+)'))
    .with_columns(x=pl.col('xy').struct['1'], y=pl.col('xy').struct['2'])
    .drop('xy')
)

# join the foreground counts of x and y and save as x_k, y_k

phrase_stats = (
    phrase_stats
    .join(fg_word_count, left_on='x', right_on='token', how='inner')
    .with_columns(x_k=pl.col('len')).drop('len')
    .join(fg_word_count, left_on='y', right_on='token', how='inner')
    .with_columns(y_k=pl.col('len')).drop('len')
)

# add in the phrase vocabulary size, total number of phrase
# occurrences, and same for the words, all taken from the vocabulary

phrase_stats =(
    phrase_stats
    .with_columns(
        word_voc_n=pl.lit(voc_size(fg_word_count)),
        word_tot_n=pl.lit(total_count(fg_word_count)),
        phrase_voc_n=pl.lit(voc_size(fg_phrase_count)),
        phrase_tot_n=pl.lit(total_count(fg_phrase_count)))
)

# smoothed estimate of p(x) = ( freq(x) + 1/V ) / ( N + 1)
# where 
#   V is number of words (or phrases if x is a phrase)
#   N is total number of occurrences of words (or phrases)

def prob_smooth(k_expr_name, n_expr_name, denom_n_expr_name):
    return (pl.col(k_expr_name) + 1.0/pl.col(n_expr_name)) / (pl.col(denom_n_expr_name) + 1.0)

# compute phraseness score for xy = log( p(xy) / p(x)*p(y) )

phrase_stats = (
    phrase_stats
    .with_columns(
        px=prob_smooth('x_k', 'word_voc_n', 'word_tot_n'),
        py=prob_smooth('y_k', 'word_voc_n', 'word_tot_n'),
        pxy=prob_smooth('fg_k', 'phrase_voc_n', 'phrase_tot_n')
    )
    .with_columns(
        phraseness=(pl.col('pxy') / (pl.col('px') * pl.col('py'))).log()
    )
    # drop all but the important columns
    .select('bigram', 'p_fg', 'p_bg', 'fg_score', 'px', 'py', 'pxy', 'phraseness')
)

########### experiment 2: adding 'phraseness'

print(' phrases '.center(80, '='))
show_extreme_scores(phrase_stats, 'phraseness') 

good_phrases = phrase_stats.filter(pl.col('phraseness') > 1.25)

show_extreme_scores(good_phrases, 'fg_score') 
