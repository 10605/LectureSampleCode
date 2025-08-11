import argparse

import sys
import math
import random
import hashlib

BITS_PER_INT = 31

#example:
#  py bloomfilter.py --n 7000 --p 0.01 --file1 ../data/bloom-train.txt --file2 ../data/bloom-test.txt

class BloomFilter(object):

    def __init__(self,seed=0,maxInserts=10000,falsePosProb=0.01):
        self.num_bits = int(2 * maxInserts*math.log(1.0/falsePosProb) + 0.5)
        self.num_hashes = int(1.5 * math.log(1.0/falsePosProb) + 0.5)
        self.bits = []
        for i in range(round(self.num_bits/BITS_PER_INT + 1)):
            self.bits.append(int(0))

        rnd = random.Random()
        rnd.seed(seed)
        self.ithHashSeed = []
        for i in range(self.num_hashes):
            self.ithHashSeed.append(rnd.getrandbits(BITS_PER_INT))

    def ithHash(self,i,value):
        return (self.ithHashSeed[i] ^ hash(value)) % self.num_bits

    def insert(self,value):
        for i in range(self.num_hashes):
            hi = self.ithHash(i,value)
            self.setbit(hi)

    def contains(self,value):
        for i in range(self.num_hashes):
            hi = self.ithHash(i,value)
            if not self.testbit(hi): return False
        return True

    def setbit(self,index):
        self.bits[index//BITS_PER_INT] = int(self.bits[index//BITS_PER_INT] | (1 << (index % BITS_PER_INT)))

    def testbit(self,index):
        return self.bits[index//BITS_PER_INT] & (1 << (index % BITS_PER_INT))

    def density(self):
        num_bits = 0.0
        num_bitsSet = 0.0
        for bi in self.bits:
            for j in range(BITS_PER_INT):
                num_bits += 1
                if bi & (1 << j):
                    num_bitsSet += 1
        return num_bitsSet/num_bits

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', required=True, help='first file')
    parser.add_argument('--file2', required=True, help='second file')
    parser.add_argument('--n', type=int, required=True, help='n >= number of items')
    parser.add_argument('--p', type=float, required=True, help='false positive rate <= p')

    args = parser.parse_args()

    bf = BloomFilter(maxInserts=args.n,falsePosProb=args.p)
    num_lines1 = num_lines2 = overlap = 0
    for line in open(args.file1):
        line = line.strip()
        if not line: continue
        bf.insert(line)
        num_lines1 += 1
    print('bit size',bf.num_bits,'kb size',bf.num_bits/(8.0*1024))
    print('numhashes',bf.num_hashes)
    print('density',bf.density())
    for line in open(args.file2):
        line = line.strip()
        if not line: continue
        if bf.contains(line.strip()): 
            overlap += 1
        num_lines2 += 1
    print('lines in', args.file1, num_lines1)
    print('lines in', args.file2, num_lines2)
    print('overlap estimated by Bloom filter', overlap)
    print('expected false positives', round(num_lines2 * args.p))
