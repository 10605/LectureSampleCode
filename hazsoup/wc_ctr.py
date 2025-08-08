from collections import Counter
import re

if __name__ == '__main__':
    
    # compute the frequency of each word in a corpus

    freq = Counter()
    for line in open('../data/brown_nolines.txt'):
        for word in re.findall('\w+', line.strip().lower()):
            freq[word] += 1


