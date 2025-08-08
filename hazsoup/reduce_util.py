"""Utilities for map-reduce.
"""

from collections.abc import Iterator
import hashlib
import json

def kv_to_line(key, value) -> str:
    """Convert a key-value pair into a sortable line.

    Specifically, running sort -k1 with LC_ALL=C option on these lines
    will group common keys together.
    """
    return json.dumps(key) + '\t' + json.dumps(value) + '\n'

def kv_from_line(line: str) -> tuple:
    """Convert a sortable line back a key-value pair.
    """
    str_key, str_value = line.rstrip().split('\t')
    return json.loads(str_key), json.loads(str_value)
    
def kv_keyhash(key) -> int:
    """A hash code for the key.

    This code should be appropriate for sharding key/value pairs by
    key, and then sorting them.  Note python's builtin hash() is not
    good enough in a distributed context, as it is not always stable
    across machines.
    """
    encoded_key = json.dumps(key).encode('utf8')
    return int(hashlib.md5(encoded_key).hexdigest(), base=16)

class PushBackIterator(Iterator):
    """Extended Iterator that supports peek, has_more(), and pushback()
    """

    def __init__(self, inner: Iterator, max_pushbacks: int = 1):
        self.inner = inner
        self.max_pushbacks = max_pushbacks
        self.stack = []  # this is the whole trick

    def peek(self):
        """Return the next thing to be produced without changing the iterator.
        """
        x = self.__next__()
        self.pushback(x)
        return x

    def has_more(self):
        """Return True iff the iterator has more objects to produce.
        """
        try:
            x = self.__next__()
            self.pushback(x)
            return True
        except StopIteration:
            return False

    def pushback(self, x):
        """Make x be the next thing the iterator will produce.
        """
        assert len(self.stack) < self.max_pushbacks, f'more than {self.max_pushbacks} pushbacks'
        self.stack.append(x)

    # implement Iterator interface

    def __next__(self):
        if self.stack:
            x = self.stack.pop()
            return x
        else:
            return self.inner.__next__()

    def __iter__(self):
        return self

class ReduceReady(Iterator):
    """An iterator over pairs (key, values) where value is a generator.

    Formed from an iterator over pairs (key, value) that are sorted by
    key.
    """

    def __init__(self, pairs: Iterator):
        self.pairs = PushBackIterator(pairs, 1)

    def __next__(self):
        def value_generator(key, first_value):
            """Generate the values associated with this key in self.pairs.
            """
            yield first_value
            while (self.pairs.has_more() and self.pairs.peek()[0] == key):
                value = next(self.pairs)[1]
                yield value
        key, value = next(self.pairs)
        return key, value_generator(key, value)

    def __iter__(self):
        return self

if __name__ == "__main__":
    test = ((key, ch) for key in 'william w cohen'.split() for ch in key)
    rr = ReduceReady(test)
    print([(key, list(gen)) for key, gen in rr])
