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


# TODO

 * write mapside joins for tardigrade?
 * could try koala/polars on my laptop I guess to see if polars is
   using memory "because it can"
