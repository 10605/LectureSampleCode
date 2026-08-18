import json
import numpy as np

import polars as pl
        
# load two sparse matrices A, B, and a gold product R
with open('../data/matmul.json') as fp:
    d = json.loads(fp.read())

A = pl.DataFrame(
    dict(
        i=[row[0] for row in d['a']],
        j=[row[1] for row in d['a']],
        w=[row[2] for row in d['a']]))

# actually the json stores B^T for some reason
B = pl.DataFrame(
    dict(
        i=[row[1] for row in d['b']],
        j=[row[0] for row in d['b']],
        w=[row[2] for row in d['b']]))

R = pl.DataFrame(
    dict(
        i=[row[0] for row in d['r']],
        j=[row[1] for row in d['r']],
        w=[row[2] for row in d['r']]))

# compute C = A.dot(B.T)
# ie c[i,j] = sum_k a[i,k] b^T[k,j]
#           = sum_k a[i,k] b[j,k]

AB = (
    A.join(B, left_on='j', right_on='i')  # match column i in A, j in B
    .rename(dict(
        i='i',
        j='k',
        j_right='j',  #also B['i']
        w='Aw',
        w_right='Bw'))
    .with_columns(prod=(pl.col('Aw') * pl.col('Bw')))
    .select('i', 'j', 'k', 'prod')
    .with_columns(pl.col('prod').sum().over(['i', 'j']).alias('w'))
    .drop('prod', 'k')
    .unique()
)
print('A:\n', A)
print('B\n', B)
print('AB:\n',AB.sort(['i','j']))

# check result

J = (
    AB.join(R, on=['i', 'j']).sort(['i', 'j'])
    .rename(dict(w_right='Rw', w='ABw'))
    .with_columns(different=pl.col('Rw')!=pl.col('ABw'))
)
print(J)
print(J.filter('different'))
