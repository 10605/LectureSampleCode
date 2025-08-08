import collections
from collections.abc import Iterator

class Worker:

    # abstract routines to be implemented by subclasses

    def map(self, x):
        """Yield one or more items. 
        """
        assert False, unimplemented

    def reduce(self, key, values: Iterator):
        """Yield one or more items to be associated with the key.
        """
        assert False, 'unimplemented'


    # higher-level commands

    def map_only(self, src: str, dst: str):
        """Run a map-only process

        This reads the data in src and stores the converted items in
        dst.
        """
        with open(dst, 'w') as fp:
            for line in open(src):
                for x in self.map(line):
                    fp.write(str(x) + '\n')


    def map_reduce(self, src: str, dst: str):
        """Run a map-reduce process.

        This reads the data in src and stores the converted items in
        dst.
        """

        grouped_by_key = collections.defaultdict(list)
        for line in open(src):
            for key, val in self.map(line):
                grouped_by_key[str(key)].append(val)
        with open(dst, 'w') as fp:
            for key, values in grouped_by_key.items():
                for reduced_value in self.reduce(key, values):
                    pair = (key, reduced_value)
                    fp.write(str(pair) + '\n')
