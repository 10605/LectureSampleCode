import collections
import functools
import math
import operator
import re

from pyspark import SparkConf, SparkContext 
        
conf = SparkConf().setAppName("PhraseFindingExample").setMaster("local") 
sc = SparkContext(conf=conf) 

bg_lines = sc.textFile("data/redcorpus.txt")
fg_lines = sc.textFile("data/bluecorpus.txt")

def tokenize(line):
    return re.findall('\w+', line.lower())

# generalized version of the 'wordcount' sequence of transformations 
def wc_pipe(lines):
    return lines.flatMap(tokenize).map(lambda x:(x, 1)).reduceByKey(operator.add)
    
fg_word_count = wc_pipe(fg_lines)
bg_word_count = wc_pipe(bg_lines)

fg_vocab = fg_word_count.keys().count()
bg_vocab = fg_word_count.keys().count()

wc_pairs = fg_word_count.join(bg_word_count)

def score_counted_pair(join_output):
    word, (fg_n, bg_n) = join_output
    p1 = (fg_n + 1.0/fg_vocab) /  (fg_vocab + 1.0)
    p2 = (bg_n + 1.0/bg_vocab) /  (bg_vocab + 1.0)
    return word, math.log( p1 / p2 )

result = wc_pairs.map(score_counted_pair)
reds = result.sortBy(lambda ws: ws[1], ascending=True)
blues = result.sortBy(lambda ws: ws[1], ascending=False)

n = 20
print(f'top {n} most red:')
for word, score in reds.take(20):
    print(word, score)
print()

print(f'top {n} most blue:')
for word, score in blues.take(20):
    print(word, score)
print()

