Sample code to illustrate map-reduce concepts.  Examples of how to
invoke the code are in Makefile.  Most of these use the 'fire' package
instead of argparse: for example

```
python3 -m fire wc_nano.py WordCount map_reduce --src ../data/brown_nolines.txt --dst /tmp/brown_counts.txt 
```

is equivalent to running `python3 -i wc_nano.py` and typing in the command 

```
WordCount().map_reduce(src="../data/brown_nolines.txt", dst="/tmp/brown_counts.txt")
```

Contents:
 * `Makefile`: targets show how to run wordcount with various
   backends, and also some AWS operations like launching/stopping
   clusters.

 * `hz_nano.py`: implements the map-reduce interface with one process
   that groups map outputs by key with Python data structures, so not
   memory efficient.
 * `wc_nano.py`: wordcount using `hz_nano.py` for the back end.
 * `wc_ctr.py`: wordcount using Python data structures directly, not
   map-reduce.

 * `hz_micro.py`, `merge_sort.py`: implements the map-reduce interface
   with one process that groups map outputs by key using a
   memory-efficient merge sort.  There's an older version on git that
   uses Unix sort instead.
 * `wc_micro.py`: wordcount using `hz_micro.py` for the back end.

 * `hazsoup.py`, `hz_worker.py`, `reduce_util.py`: parallel
   memory-efficient map-reduce.  This still uses Unix sort, and worked
   on AWS in Fall 2025.
 * `wc_hz.py`: wordcount using full hazsoup for the back end.

 * `spork_micro.py`, `reduce_util.py` - a spark-like backend, which like the old
   hz_micro uses Unix sort for grouping.  Mostly just illustrates
   how to set up a chaining API.
 * `wc_spork.py`: using `spork_micro` as the backend.
