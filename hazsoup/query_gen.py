# figure out possible queries to find public and private Dns addresses

import json
from pprint import pprint

with open('tmp.json') as fp:
    x = json.load(fp)

def query(target_key, x, from_root=None):
    if from_root is None:
        from_root = []
    if isinstance(x, dict):
        if target_key in x and x.get(target_key):
            yield '.'.join(from_root + [target_key])
        for k, v in x.items():
            if k != target_key:
                for result in query(target_key, v, from_root + [k]):
                    yield result
    elif isinstance(x, list):
        for y in x:
            for result in query(target_key, y, from_root[:-1] + [from_root[-1]+'[]']):
                yield result
            
pprint('PublicDnsName:')
for q in query('PublicDnsName', x):
    print(q)
pprint('PrivateDnsName:')
for q in query('PrivateDnsName', x):
    print(q)
