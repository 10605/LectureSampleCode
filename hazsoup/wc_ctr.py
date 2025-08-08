from collections import Counter
from pprint import pprint
import re


if __name__ == '__main__':
    
    # compute the frequency of each word in a corpus

    freq = Counter()
    for line in open('../data/brown_nolines.txt'):
        for word in re.findall('\w+', line.strip().lower()):
            freq[word] += 1


    # print a few counts
    pprint(freq.most_common(10))
