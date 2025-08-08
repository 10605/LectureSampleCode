"""Very simple map-reduce system.
"""

import ast
import collections
import fire
import json
import logging
import os
from pprint import pprint
from subprocess import Popen, PIPE
from subprocess import run as run_subproc
import shlex
import sys
import time
from tqdm import tqdm

from hz_worker import CloudBase

class Cloud(CloudBase):
    """Base class for working with an ec2 cloud.
    """
    # utilities

    def _completion_progress(self, processes, delay=0.25):
        """A progress bar for process completion.
        """
        for n in tqdm(range(len(processes))):
            finished = []
            while len(finished) < n:
                finished = [p for p in processes if p.poll() is not None]
                time.sleep(delay)

    def _report(self, proc, worker):
        """Echo outputs of processes from a worker.
        """
        def report_stdx(what, proc_stdx, worker):
            # report proc.stdout or proc.stderr
            outp = proc_stdx if isinstance(proc_stdx, str) else proc_stdx.read()
            if outp:
                print(f'{what} {worker}'.center(60, '='))
                print(outp, end='')
        if proc.returncode:
            print(f'returncode {worker}: {proc.returncode}')
        report_stdx('stdout', proc.stdout, worker) 
        report_stdx('stderr', proc.stderr, worker)

    # user commands

    def ssh(self, command):
        """Run a shell command on all workers sequentially.
        """
        sample_command = f'ssh {self.ssh_args()} {self.workers[0]} {command}'
        logging.info(f'sample command: {sample_command}')
        for worker in self.workers:
            proc = run_subproc(
                shlex.split(f'ssh {self.ssh_args()} {worker} {command}'),
                text=True, capture_output=True)
            self._report(proc, worker)
                
    def sshp(self, command):
        """Run a shell command on all workers in parallel.
        """
        sample_command = f'ssh {self.ssh_args()} {self.workers[0]} {command}'
        logging.info(f'sample command: {sample_command}')
        processes = [
            Popen(
                shlex.split(f'ssh {self.ssh_args()} {worker} {command}'),
                text=True, stderr=PIPE, stdout=PIPE)
            for worker in self.workers]
        self._completion_progress(processes)
        for proc, worker in zip(processes, self.workers):
            self._report(proc, worker)
            proc.wait()

    def ssh1(self, command):
        """Run a shell command on one worker.
        """
        sample_command = f'ssh {self.ssh_args()} {self.workers[0]} {command}'
        logging.info(f'sample command: {sample_command}')
        worker = self.workers[0]
        proc = run_subproc(
            shlex.split(f'ssh {self.ssh_args()} {worker} {command}'),
            text=True, capture_output=True)
        self._report(proc, worker)

    def upload(self, filenames):
        """Copy a file to all workers.
        """
        sample_command = (
            f'scp {self.scp_args()} {filenames}' 
            + f' {self.cloud_username}@{self.workers[0]}:')
        logging.info(f'sample command: {sample_command}')
        for worker in tqdm(self.workers):
            scp_toks = shlex.split(
                f'scp {self.scp_args()} {filenames}' 
                + f' {self.cloud_username}@{worker}:')
            proc = run_subproc(
                scp_toks, text=True, capture_output=True)
            self._report(proc, worker)

    def setup(self, local_files=None):
        """Commands needed to initialize an ec2 cluster.
        """
        logging.info('installing pip')
        self.sshp("sudo yum install python3-pip -y")
        logging.info('installing fire')
        self.sshp("pip3 install fire")
        logging.info('uploading core')
        self.upload(
            f"hz_worker.py reduce_util.py "
            + f"{self.worker_file} {self.keypair_file}")
        self.sshp(f"chmod 400 {self.keypair_file}")
        if local_files is not None:
            logging.info(f'uploading local files {local_files}')
            self.upload(" ".join(local_files.split(",")))

class FileSystem(Cloud):
    """A simple distributed filesystem.
    """

    def put(self, src, dst):
        """Shard a local file and upload shards to the workers.
        """
        sample_command = f'cat {src} | ssh {self.ssh_args()} {self.workers[0]} cat > {dst}'
        logging.info(f'sample command: {sample_command}')
        line_ctr = collections.Counter()
        # create a process for each worker that can accept text lines
        # via its stdin
        worker_processes = [
            Popen(
                shlex.split(
                    f'ssh {self.ssh_args()} {worker} cat > {dst}'),
                text=True, stdin=PIPE)
            for worker in self.workers
        ]
        # upload the local file
        for line in tqdm(open(src)):
            # send line to the worker with this index
            worker_idx = hash(line) % len(self.workers)
            worker_processes[worker_idx].stdin.write(line)
            # record some statistics
            line_ctr[src] += 1
            line_ctr[f'{self.workers[worker_idx]}:{dst}'] += 1
        # close the worker processes and wait for them to finish
        for proc in tqdm(worker_processes):
            proc.stdin.close()
            proc.wait()
        # echo statistics
        pprint(line_ctr)
        
    def get_merge(self, src, dst):
        """Get remote shards and collect them into a local file.
        """
        line_ctr = collections.Counter()
        sample_command = f'ssh {self.ssh_args()} {self.workers[0]} cat < {src} | cat > {dst}'
        logging.info(f'sample command: {sample_command}')
        with open(dst, 'w') as fp:
            for worker in tqdm(self.workers):
                # download the data from that worker to local file
                proc = Popen(
                    shlex.split(
                        f'ssh {self.ssh_args()} {worker} cat < {src}'),
                    text=True, stdout=PIPE)
                for line in proc.stdout:
                    fp.write(line)
                    # record some statistics
                    line_ctr[f'{worker}:{src}'] += 1
                    line_ctr[dst] += 1
                # wait for process to end
                proc.wait()
        # echo statistics
        pprint(line_ctr)

    def head(self, src):
        """Print the head of a sharded file on the first worker.
        """
        worker = self.workers[0]
        proc = run_subproc(
            ['ssh', '-i', self.keypair_file, 
             f'{self.cloud_username}@{worker}',
             'head', src],
            capture_output=True, text=True)
        print(proc.stdout, end='')

    def tail(self, src):
        """Print the head of a sharded file on the last worker.
        """
        worker = self.workers[-1]
        proc = run_subproc(
            ['ssh', '-i', self.keypair_file, 
             f'{self.cloud_username}@{worker}',
             'tail', src],
            capture_output=True, text=True)
        print(proc.stdout, end='')

class Driver(Cloud):
    """Invokes workers appropriately to complete a task.
    """

    def map_only(self, main_script, main_class, src, dst):
        """A map-only job.
        """
        self.sshp(f'python3 -m fire {main_script} {main_class}'
                  + f' do_map --src {src} --dst {dst}')

    def map_reduce(self, main_script, main_class, src, dst):
        """A map-reduce job.
        """

        print('map and shuffle phase')
        self.sshp(f' python3 -m fire {main_script} {main_class}'
                  + f' do_map_and_shuffle -src {src}')

        print('gather and reduce phase')
        self.sshp(f'python3 -m fire {main_script} {main_class}'
                  + f' do_gather_reduce --src {src} --dst {dst}')
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        fire.Fire(dict(
            fs = FileSystem,
            run = Driver,
            cloud = Cloud))
