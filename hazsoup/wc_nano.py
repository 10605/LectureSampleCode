import re

import hz_nano as hz

class WordCount(hz.Worker):
    
    def map(self, x):
        for word in re.findall(r'\w+', x.lower()):
            yield (word, 1)

    def reduce(self, word, counts):
        yield sum(counts)
