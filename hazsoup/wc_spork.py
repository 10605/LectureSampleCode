import re

import spork_micro as spork

sc = spork.Context()

wc = sc.textFile('../data/brown_nolines.txt') \
       .map(lambda line: line.lower()) \
       .flatMap(lambda line: re.findall(r'\w+', line)) \
       .map(lambda word: (word, 1)) \
       .reduceByKey(lambda a,b: a+b)

freqs = dict(wc.collect())
for w in 'the|of|and|to|in|that|is|was|he'.split('|'):
    print(f'freq of "{w}":\t{freqs[w]}')


