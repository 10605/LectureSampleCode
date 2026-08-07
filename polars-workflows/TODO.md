# STATUS

## Koala vs Polars

Koala was an effort to rewrite *just the memory-intensive parts* of
polars, using external unix sorts.  That didn't work - polars seems to
really like using memory!

Fixed nodes = 20,000   iterations = 2   CSE = on   (columns = total lines)

self RSS (MB)    1,020,000   2,020,000   4,020,000   8,020,000  16,020,000  32,020,000
--------------------------------------------------------------------------------------
koala-lazy           305.7       385.1       590.5       951.5      1454.5      2151.3
polars-lazy          379.1       494.8       781.4      1200.0      1945.7      3120.1
															                          
child RSS (MB)   1,020,000   2,020,000   4,020,000   8,020,000  16,020,000  32,020,000
--------------------------------------------------------------------------------------
koala-lazy           118.0       137.5       168.3       203.0       314.2       515.0
polars-lazy            0.0         0.0         0.0         0.0         0.0         0.0
															                          
elapsed (s)      1,020,000   2,020,000   4,020,000   8,020,000  16,020,000  32,020,000
--------------------------------------------------------------------------------------
koala-lazy             4.5         9.8        19.6        38.9       100.0       203.2
polars-lazy            0.1         0.1         0.3         0.5         1.0         1.7


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

This is a second koala-like attempt to localize use of polar's memory
by pulling out joins and group_by operations.  It also doesn't seem to
keep memory capped properly.  There may be some simple optimizations
around the sorting involving use of decode/encode that explain why
it's slow.

self RSS (MB)    1,020,000   2,020,000   4,020,000
--------------------------------------------------
polarbar-lazy        448.2       641.4       850.7

elapsed (s)      1,020,000   2,020,000   4,020,000
--------------------------------------------------
polarbar-lazy          8.3        18.7        37.8


# Logs

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
