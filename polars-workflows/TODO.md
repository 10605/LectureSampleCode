# STATUS

polars seems to really like using memory.  I should try on a smaller
machine to see what it does.  and maybe write a 'real' merge sort

koala now uses the streaming merge_join (no JSON) in the pagerank loop; matches
pagerank.py exactly. Per-iteration join+sum dropped ~4.9x vs inner_join (see

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


# TODO

 * [done] drilldown where the memory goes in koala -> inner_join list-gather (#4)
 * [done] convert smoke tests to unit tests -> test_koala.py (pytest, 10 tests)
 * [done] optimized inner_join with key-value pairs (no packing) -> merge_join
   * sort left, right together with key, lval, rval (tagged L/R)
   * per-key combine in the streaming layer; inner gate = both sides present
 * write inner_join(..., how=) to switch implementations
 * write mapside joins for polar/koala
   * as a koala method, not special code like in pagerank_mapper
