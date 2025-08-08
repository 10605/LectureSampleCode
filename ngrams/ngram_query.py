import argparse
import collections
import pprint
import random
import sys

# Examples
#
# what's the bias? 
# py ngram_query.py --C _
# 103,385,894 matching ngrams (100.0000%)
# 1    95.1358 C=effect
# 2     4.2454 C=affect
# 3     0.5547 C=Effect
# 4     0.0593 C=EFFECT
# 5     0.0023 C=Affect
# 6     0.0020 C=effecT
# 7     0.0005 C=AFFECT
#
# what generally precedes 'affect'?
# py ngram_query.py --B _ --C affect
# 4,389,162 matching ngrams (4.2454%)
# 1    61.0687 B=not,C=affect
# 2    11.5791 B=to,C=affect
# 3     3.2641 B=may,C=affect
# 4     2.8882 B=they,C=affect
# 5     1.4957 B=which,C=affect
# 6     1.3919 B=way,C=affect
# 7     1.3159 B=shall,C=affect
# 8     1.1096 B=that,C=affect
# 9     1.1021 B=or,C=affect
# 10    1.0325 B=materially,C=affect
# 
# what generally precedes 'effect'? 
# py ngram_query.py --B _ --C effect
# 98,356,978 matching ngrams (95.1358%)
# 1    30.4189% B=the,C=effect
# 2     5.6861% B=The,C=effect
# 3     3.7761% B=into,C=effect
# 4     3.4638% B=same,C=effect
# 5     3.2066% B=no,C=effect
# 6     3.2037% B=this,C=effect
# 7     2.9828% B=and,C=effect
# 8     2.9079% B=that,C=effect
# 9     2.7214% B=an,C=effect
# 10    2.5262% B=its,C=effect
# 
# which is more common - 'no effect' or 'no affect'?
# py ngram_query.py --B no --C _
# 3,157,011 matching ngrams (3.0536%)
# 1    99.9024 B=no,C=effect
# 2     0.0976 B=no,C=affect
#
# when do people use 'no affect'?
# py ngram_query.py --B no --C affect --D _ --E _
# 3,082 matching ngrams (0.0030%)
# 1    44.7112 A=has,B=no,C=affect,D=on,E=the
# 2    38.1570 A=have,B=no,C=affect,D=on,E=the
# 3    17.1317 A=had,B=no,C=affect,D=on,E=the
#
# and is that right?
# py ngram_query.py --A has --B no --C _ --D on --E the
# 23,956 matching ngrams (0.0232%)
# 1    94.2478% A=has,B=no,C=effect,D=on,E=the
# 2     5.7522% A=has,B=no,C=affect,D=on,E=the

def match(query, ngram):
    """Match an n-gram query to an n-gram.  A query has n parts, where
    each part is either a named variable (starting with an
    underscore), an anonymous variable (just the underscore), or a
    constant word (anything else).  Returns a 'binding', i.e., a
    dictionary that maps non-anonymous variable names to the constant
    values in the n-gram they match.
    """
    bindings = {}
    for event_name, word in zip("ABCDE", ngram):
        if event_name in query:
            required_value = query[event_name]
            if word == required_value or required_value == '_':
                bindings[event_name] = word
            else:
                return None
    return bindings

def as_string(bindings):
    """Convert bindings to a string."""
    return ",".join([f'{k}={v}' for k, v in sorted(bindings.items())])

def count(filename, query):
    """Count the number of n-grams in a file that match every binding
    to a query.  In particular, returns a mapping C from string
    representations of 'bindings' of a query to the probability of
    seeing that binding, given that the query matches."""

    ctr = collections.Counter()
    grand_total = 0
    for line in open(filename):
        ngram_str, n_str = line.rstrip().split("\t")
        n = int(n_str)
        grand_total += n
        ngram = ngram_str.split(" ")
        bindings = match(query, ngram)
        if bindings is not None:
            ctr[as_string(bindings)] += n
    return ctr, grand_total

def as_distribution(ctr, top=10):
    total = sum(ctr.values())
    return total, [(k, v/total) for k, v in ctr.most_common(top)]

QUERY_USAGE="""
"--A foo" means the first word in the ngram must be "foo", and "--A _" is a wildcard.
The options --B, --C, ..., --E are similar.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(epilog=QUERY_USAGE)
    parser.add_argument('--ngram_file', default='../data/aeffect-train.txt')
    parser.add_argument('--top', default=10, type=int)
    for v in "ABCED":
        parser.add_argument(f'--{v}')
    args = parser.parse_args()

    # create a query, ie a dict mapping events A,...,E
    # to required values for that event, where '_' is a wildcard
    query = {event:value for event, value in vars(args).items()
             if event in "ABCDE" and value is not None}

    ctr, total_ngrams = count(args.ngram_file, query)
    total_matching, dist = as_distribution(ctr, top=args.top)
    pct_matching = 100 * total_matching / total_ngrams
    print(f'{total_matching:,} matching ngrams ({pct_matching:.04f}%)')
    for i, (event, freq) in enumerate(dist):
        print(f'{i + 1:<5d}{freq * 100:7.04f}% {event}')
