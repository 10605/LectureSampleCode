# STATUS

Engines = polars, koala   Fixed nodes = 20,000   iterations = 2   CSE = on
(self RSS = this process; child RSS = koala's external sort, ~0 for polars. lazy self RSS ~flat as edges grow; eager climbs.)

                              self RSS (MB)                              child RSS (MB)                              elapsed (s)               
   total lines      pl-lazy  pl-eager   ko-lazy  ko-eager      pl-lazy  pl-eager   ko-lazy  ko-eager      pl-lazy  pl-eager   ko-lazy  ko-eager
-----------------------------------------------------------------------------------------------------------------------------------------------
     2,020,000        487.4     539.1     550.6     576.3          0.0       0.0     257.4     253.4          0.2       0.1       7.5       7.8
     4,020,000        852.5     838.4     849.2     872.1          0.0       0.0     501.2     502.4          0.3       0.2      14.6      14.5

# TODO

demo_memory.pl
 * chose --modes as subset of polar-lazy polar-eager koala ...
 * print table as method (row) size (column)
 * limit child subprocess size
 * drilldown and see where the memory use is happening in koala
 * write mapside joins for polar/koala
   * as a method, not special code like in pagerank_mapper
 * write optimized inner_join with key-value pairs (no packing)
   * sort left, right together with key, lval, rval (one val is always null)
   * groupby with a special aggregator that (combines lval, rval in a row)
   * filter by lval != None and rval != None
