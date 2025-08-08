import collections
from collections.abc import Iterator
import subprocess
import os

import reduce_util as ru

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

        stem = os.path.basename(src)
        sort_input_buf = f'/tmp/mapout-{stem}.tsv'
        sort_output_buf = f'/tmp/sortout-{stem}.tsv'

        # save the map output 
        with open(sort_input_buf, 'w') as fp:
            for line in open(src):
                for key, value in self.map(line):
                    fp.write(ru.kv_to_line(key, value))

        # sort the map output
        command = (f'LC_ALL=C sort -k1'
                   + f' -o {sort_output_buf} {sort_input_buf}')
        subprocess.check_call(command, shell=True)

        # create a generator for the sorted pairs 
        def pair_generator():
            for line in open(sort_output_buf):
                yield ru.kv_from_line(line)

        # convert pair_generator and invoke and save reduce outputs
        with open(dst, 'w') as fp:
            for key, values in ru.ReduceReady(pair_generator()):
                for reduced_value in self.reduce(key, values):
                    pair = (key, reduced_value)
                    fp.write(str(pair) + '\n')
        

