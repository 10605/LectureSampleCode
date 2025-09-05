"""Base class for Hazsoup worker programs.

To implement a Worker, subclass the Worker class and provide
implementations of map and reduce.
"""

from collections.abc import Iterator
import json
import logging
import shlex
import socket
from subprocess import Popen, PIPE, check_call
import sys
import os

import reduce_util as ru

WORKER_FILENAME = 'workers.json'
CLOUD_USERNAME = 'ec2-user'
KEYPAIR_FILE = 'hazsoup.pem'

class CloudBase:
    """Base class for using a set of ec2 workers.
    """
    def __init__(self):
        """Loads worker names and sets other defaults.
        """
        self.worker_file = WORKER_FILENAME
        self.cloud_username = CLOUD_USERNAME
        self.keypair_file = 'hazsoup.pem'
        try:
            internal_external_pairs = json.load(open(self.worker_file))
            # remove pairs without external DNS
            internal_external_pairs = [
                (internal, external) for (internal, external) in internal_external_pairs
                if external != ""]
            self.workers = [
                external_name
                for _internal_name, external_name in internal_external_pairs]
            # internal2external is used only to figure out the
            # external name of the worker node that a process is
            # running on
            self.internal2external = dict(internal_external_pairs)
        except FileNotFoundError:
            print(f'warning: could not open {self.worker_file}')
            self.workers = None
            self.internal2external = None

    def ssh_args(self) -> str:
        """Arguments for an ssh command invoking a worker.
        """
        return (f'-i {self.keypair_file} -o StrictHostKeyChecking=no'
                + f' -l {self.cloud_username}')

    def scp_args(self) -> str:
        """Arguments for an scp command invoking the worker.
        """
        return (f'-i {self.keypair_file} -o StrictHostKeyChecking=no')


class Worker(CloudBase):
    """An abstract worker for map-reduce tasks.
    """

    # abstract routines to be implemented by subclasses

    def map(self, x):
        """Yield one or more items. 
        """
        assert False, unimplemented

    def reduce(self, key, values: Iterator):
        """Yield one or more items to be associated with the key.
        """
        assert False, 'unimplemented'


    def do_map(self, src, dst):
        """Invoked from command line for map-only jobs.
        """
        with open(dst, 'w') as fp:
            for line in open(src):
                for x in self.map(line):
                    fp.write(str(x) + '\n')

    def _shard_bufname(self, src, worker_idx):
        """Name of a single output shard from a worker.
        """
        stem = os.path.basename(src)
        return f'sortout-{stem}-from-w{worker_idx:02d}.tsv'

    def do_map_and_shuffle(self, src):
        """Run mapper on src and distribute shards to co-workers. 

        Distributed shards are sorted by key.
        """
        this_worker = self.internal2external[socket.gethostname()]
        coworkers = self.workers

        # first stage of a map-reduce: run the map process, shard the
        # outputs, and send the shards to an appropriate worker.

        # set up a process on each co-worker machine to accept the
        # appropriate shard of data from this worker.  These processes
        # will sort the incoming data by key - that needs to be done
        # anyway for the reduce
        def sort_pipe_command_str(worker):
            dst = self._shard_bufname(src, coworkers.index(this_worker))
            result = f'ssh {self.ssh_args()} {worker} LC_ALL=C sort -k1 -o {dst}'
            return result
        sample_command = sort_pipe_command_str(coworkers[0])
        logging.info(f'sample pipe: {sample_command}')
        coworker_processes = [
            Popen(
                sort_pipe_command_str(worker),
                text=True, stderr=PIPE, stdin=PIPE, shell=True)
            for worker in coworkers
        ]
        # run the map and distribute the data to the processes,
        # choosing the destination based on the key
        for line in open(src):
            for key, val in self.map(line):
                # convert the pair to a sortable line
                kv_line = ru.kv_to_line(key, val)
                # figure out where to send this line
                key_worker_idx = ru.kv_keyhash(key) % len(coworkers)                
                # and send it to the correct coworker
                try:
                    coworker_processes[key_worker_idx].stdin.write(kv_line)
                except BrokenPipeError as ex:
                    # do_map_and_shuffle is normally called via ssh and errors
                    # are passed back via Cloud._report which looks as stderr
                    # This exception indicates a worker error - so pass the error
                    # messages back to stderr
                    failing_coworker = coworkers[key_worker_idx]
                    print(
                        f'Error raised when {this_worker} wrote to {failing_coworker}'
                        + f' index={key_worker_idx}:', file=sys.stderr)
                    print(f'{ex}', file=sys.stderr)
                    print(f'stderr from {failing_coworker}:',
                          file=sys.stderr)
                    for line in coworker_processes[key_worker_idx].stderr:
                        print(line, end='', file=sys.stderr)
                    return
        # close the coworker processes and report any errors
        for proc, worker in zip(coworker_processes, coworkers):
            proc.stdin.close()
            proc.wait()
            if proc.returncode:
                print(f'returncode {worker}: {proc.returncode}')
            error_log = proc.stderr.read()
            if error_log:
                print(f'stderr {worker}'.center(60, '='))
                print(error_log, end='')
        
    def do_gather_reduce(self, src, dst):
        """Merge shards generated by do_map_and_shuffle and then reduce.
        """
        coworkers = self.workers
        
        # second stage of the map-reduce - gather shards sent by the
        # other workers in the do_map_and_shuffle stage and run reduce
        # on them

        incoming_shards = [
            self._shard_bufname(src, i) for i in range(len(coworkers))
        ]
        stem = os.path.basename(src)
        merge_dst =  f'mergeout-{stem}.tsv'
        merge_sort_cmd = (f'LC_ALL=C sort -k1 -m -o {merge_dst} '
                          + ' '.join(incoming_shards))
        # TODO: work out error handling
        check_call(merge_sort_cmd, shell=True)

        # TODO: use itertools.group_by

        # create a generator for the sorted pairs so we can reduce
        def pair_generator():
            for line in open(merge_dst):
                yield ru.kv_from_line(line)

        # convert pair_generator and invoke and save output of reduce
        with open(dst, 'w') as fp:
            for key, values in ru.ReduceReady(pair_generator()):
                for reduced_value in self.reduce(key, values):
                    pair = (key, reduced_value)
                    fp.write(str(pair) + '\n')

