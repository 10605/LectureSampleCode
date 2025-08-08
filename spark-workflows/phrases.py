import collections
import functools
import math
import operator
import re

from pprint import pprint

from pyspark import SparkConf, SparkContext 
        
conf = SparkConf().setAppName("PhraseFindingExample").setMaster("local") 
sc = SparkContext(conf=conf) 

# two corpora
bg_lines = sc.textFile("data/redcorpus.txt")
fg_lines = sc.textFile("data/bluecorpus.txt")

# extract 'phrases' and tokens
def bigrams(line):
    tokens = re.findall('\w+', line.lower())
    for i in range(0, len(tokens) - 1):
        yield (tokens[i], tokens[i + 1])

def tokens(line):
    return re.findall('\w+', line.lower())

# generalized version of the 'wordcount' sequence of transformations 
def wc_pipe(lines, preprocess):
    return lines.flatMap(preprocess).map(lambda x:(x, 1)).reduceByKey(operator.add)
    
# generalized code to get vocabulary size from wordcount-like RDD
def voc_size_pipe(sc, wc):
    n = wc.keys().count()
    return sc.broadcast(n)
    
# generalized code to get sum of counts from a wordcount-like RDD
def total_count_pipe(sc, wc):
    n = wc.values().reduce(operator.add)
    return sc.broadcast(n)

# wordcount-like RDDs for phrases and words in foreground and background corpora
fg_phrase_count = wc_pipe(fg_lines, bigrams)
bg_phrase_count = wc_pipe(bg_lines, bigrams)
fg_word_count = wc_pipe(fg_lines, tokens)
bg_word_count = wc_pipe(bg_lines, tokens)

# score by smoothed log odds of Pr(x|corpus1) / Pr(x|corpus2)
def score_counted_pair(join_output, n1, n2):
    x, (fg_k, bg_k) = join_output
    p1 = (fg_k + 1.0/n1.value) /  (n1.value + 1.0)
    p2 = (bg_k + 1.0/n2.value) /  (n2.value + 1.0)
    return x, math.log( p1 / p2 )

########## experiment 1: look at most informative words for foreground vs background

word_pairs = fg_word_count.join(bg_word_count)
fg_word_v = voc_size_pipe(sc, fg_word_count)
bg_word_v = voc_size_pipe(sc, bg_word_count)

word_result = word_pairs.map(
    lambda p: score_counted_pair(p, fg_word_v, bg_word_v))

def inspect(rdd, n, msg):
    print(f'top {n} {msg}:')
    for x, score in rdd.sortBy(lambda kv: kv[1], ascending=False).take(n):
        print(x, score)
    print()
    print(f'bottom {n} {msg}:')
    for x, score in rdd.sortBy(lambda kv: kv[1], ascending=True).take(n):
        print(x, score)
    print()

inspect(word_result, 10, 'foreground words')

phrase_pairs = fg_phrase_count.join(bg_phrase_count)
fg_phrase_v = voc_size_pipe(sc, fg_phrase_count)
bg_phrase_v = voc_size_pipe(sc, bg_phrase_count)

phrase_result = phrase_pairs.map(
    lambda p: score_counted_pair(p, fg_phrase_v, bg_phrase_v))

inspect(phrase_result, 10, 'foreground phrases')

#
# incorporating 'phraseness'
#

phrase_dicts = phrase_result.map(
    lambda pair: dict(
        phrase=pair[0],
        x=pair[0][0],
        y=pair[0][1],
        redness=pair[1]))

def join_in(dict_rdd, join_slot, join_rdd, new_slot):
    by_join_key = dict_rdd.map(lambda d: (d[join_slot], d))
    with_joined_info = by_join_key.join(join_rdd)
    return with_joined_info.values().map(
        lambda join_pair: {new_slot:join_pair[1], **join_pair[0]})

phrase_dicts = join_in(phrase_dicts, 'phrase', fg_phrase_count, 'phrase_count')
phrase_dicts = join_in(phrase_dicts, 'x', fg_word_count, 'x_count')
phrase_dicts = join_in(phrase_dicts, 'y', fg_word_count, 'y_count')

pprint(phrase_dicts.take(10))

def add_computed_field(dict_rdd, new_slot, fn):
    return dict_rdd.map(
        lambda d: {new_slot:fn(d), **d})

fg_word_v = voc_size_pipe(sc, fg_word_count)
fg_word_n = total_count_pipe(sc, fg_word_count)
fg_phrase_v = voc_size_pipe(sc, fg_phrase_count)
fg_phrase_n = total_count_pipe(sc, fg_phrase_count)

def score_phrasiness(d, word_v, word_n, phrase_v, phrase_n):
    fx = d['x_count']
    fy = d['y_count']
    fxy = d['phrase_count']
    px = (fx + 1.0/word_v.value) / (word_n.value + 1.0)
    py = (fy + 1.0/word_v.value) / (word_n.value + 1.0)
    pxy = (fxy + 1.0/phrase_v.value) / (phrase_n.value + 1.0)
    return math.log(pxy / (px * py))

phrase_dicts = add_computed_field(
    phrase_dicts,
    'phraseness',
    lambda d: score_phrasiness(d, fg_word_v, fg_word_n, fg_phrase_v, fg_phrase_n))

good_phrase_dicts = (
    phrase_dicts
    .map(lambda d: (d['phrase'], d['phraseness']))
    .filter(lambda kv: kv[1]>2.5)
)
good_fg_phrase_count = fg_phrase_count.join(good_phrase_dicts).map(lambda kv:(kv[0],kv[1][0]))
good_phrase_pairs = good_fg_phrase_count.join(bg_phrase_count)
good_phrase_result = good_phrase_pairs.map(lambda p: score_counted_pair(p, fg_phrase_v, bg_phrase_v))

inspect(good_phrase_result, 10, 'foreground phrases')


