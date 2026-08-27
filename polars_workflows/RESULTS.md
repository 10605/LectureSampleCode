# STATUS

## Tardigrade (water bears) vs Polars

Tardigrade is all based on Python generators and on-disk sorting.
This is actually low memory.

self RSS (MB)     1,020,000   2,020,000   4,020,000   8,020,000  16,020,000 32,020,000
--------------------------------------------------------------------------------------
polars-eager          355.3       539.8       827.6      1278.3      2000.9     3502.5
polars-lazy           374.8       491.9       743.5      1177.5      1931.5     3093.7
tardigrade-lazy        79.2        80.4        83.8        86.8        88.5       89.5
									                               
elapsed (s)       1,020,000   2,020,000   4,020,000   8,020,000  16,020,000 32,020,000
--------------------------------------------------------------------------------------
polars-eager            0.1         0.1         0.3         0.5         0.8        1.5
polars-lazy             0.1         0.2         0.3         0.5         0.8        1.5
tardigrade-lazy         5.0        11.7        23.9        50.1       106.5      240.5


## PolarBar

An attempt to localize polars' memory use by pulling the joins and
group_by operations out into sorted key-value files on disk.

## Wordcount stress test

wordcount_probe.py takes an RCV1 corpus, tokenizes it, and creates
keyvalue pairs where key is f"{docid}##{label}" and value is a token.

With autoscaling sort fanin, this tracks linearly for time and
maybe linearly with fanin for space (fanin is ~3x, space is ~5x)

### Full corpus 

On RCV1 full train (~ 1 Gb) with `sort_batch_size=1024**2` and
autoscaling `sort_fanin=sqrt(num_batches)`
 performance is
 * sink ~ 30sec
 * 558 batches => fanin 24
 * ~4 spills/sec => spilling ~ 2:30
 * ~10sec/merge => pass1 is ~ 3:30
 * pass2 ~ 6:30
 * overall sort is ~ 9:10 and ~ 4.7 Gb

 * groupby ~ 2min

Unix sort uses > 100 Gb and ~ 16:30 sec

### 10x full corpus 

On RCV1 10x full train (~ 10 Gb) with same params
 * 5576 batches => fanin 75
 * ~4 spills/sec => spilling ~ 2:30
 * ~30sec/merge => pass1 is ~ 40min
 * pass2 ~ 65min
 * overall sort is ~ 104min and ~ 22 Gb

 * groupby is ~ 19min

# Logs

## full

```
bash-3.2$ time py wcprobe.py
time py wcprobe.py
loading and tokenizing....
/Users/wcohen/Documents/code/LectureSampleCode/polars-workflows/wcprobe.py:43: DeprecationWarning: In Polars 2.0, the default behavior for `empty_as_null` will change to `False`. To keep the current behavior, explicitly set `empty_as_null=True`.
  .explode('tokens')
/Users/wcohen/Documents/code/LectureSampleCode/polars-workflows/wcprobe.py:44: DeprecationWarning: In Polars 2.0, the default behavior for `empty_as_null` will change to `False`. To keep the current behavior, explicitly set `empty_as_null=True`.
  .explode('labels')
sinking - 0.0593 peak rss gb
sink 4.2892 sec elapsed
sorting - 4.7254 peak rss gb
spilling...
558it [02:34,  3.60it/s]
merging 558 files 23.25 merges
24it [03:28,  8.69s/it]
merging 24 files....
sort 551.5618 sec elapsed
grouping - 9.6319 peak rss gb
group 112.9995 sec elapsed

real	11m9.067s
user	11m48.332s
sys	0m33.634s
```

## 10x full

```
bash-3.2$ time py wcprobe.py
time py wcprobe.py
loading and tokenizing....
/Users/wcohen/Documents/code/LectureSampleCode/polars-workflows/wcprobe.py:43: DeprecationWarning: In Polars 2.0, the default behavior for `empty_as_null` will change to `False`. To keep the current behavior, explicitly set `empty_as_null=True`.
  .explode('tokens')
/Users/wcohen/Documents/code/LectureSampleCode/polars-workflows/wcprobe.py:44: DeprecationWarning: In Polars 2.0, the default behavior for `empty_as_null` will change to `False`. To keep the current behavior, explicitly set `empty_as_null=True`.
  .explode('labels')
sinking - 0.0594 peak rss gb
sink 27.6543 sec elapsed
sorting - 15.1154 peak rss gb
spilling...
5576it [30:45,  3.02it/s]
merging 5576 files 74.34666666666666 merges
75it [39:38, 31.72s/it]
merging 75 files....
sort 6282.1918 sec elapsed
grouping - 21.8147 peak rss gb
group 1135.2172 sec elapsed

real	124m6.238s
user	128m42.659s
sys	5m44.929s
```


# TODO

 * write mapside joins for tardigrade?
