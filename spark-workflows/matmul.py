import collections
import json
import operator
import numpy as np
from pprint import pprint

from pyspark import SparkConf, SparkContext 
        
conf = SparkConf().setAppName("MatmulExample").setMaster("local") 
sc = SparkContext(conf=conf) 

# an entry in a sparse matrix 
Mat = collections.namedtuple('Mat', ['row', 'col', 'w'])

with open('data/matmul.json') as fp:
    d = json.loads(fp.read())

def asMat(xs):
    return 

a = sc.parallelize([Mat(*e) for e in d['a']])
b = sc.parallelize([Mat(*e) for e in d['b']])

# compute C = A.dot(B.T)
# ie c[i,j] = sum_k a[i,k] b^T[k,j]
#           = = sum_k a[i,k] b[j,k]

# first step, find all ((i,j), a[i,k]*b[j,k]) for any i,j,k

a_by_col = a.map(lambda e: (e.col, e))
b_by_col = b.map(lambda e: (e.col, e))
def aik_bjk_product(join_result):
    k, (aik, bjk) = join_result
    return ((aik.row, bjk.row), aik.w * bjk.w)
ak_bk_prods = (
    #pairs a[i,k]>0 with all j: b[j,k]>0
    a_by_col
    .join(b_by_col)
    # then convert to structures ((i,j), a[i,k]*b[j,k])
    .map(aik_bjk_product)
)

# now add these up to get non-zero dot products from C
dotprods = ak_bk_prods.reduceByKey(operator.add)

# and convert back to a 'sparse matrix'
def asMatEntry(dotprod):
    (row, col), weight = dotprod
    return Mat(row, col, weight)
c = dotprods.map(asMatEntry)

spark_ans = sorted(c.collect())
reference_ans = sorted([Mat(*e) for e in d['r']])

print('correct?', spark_ans == reference_ans)
