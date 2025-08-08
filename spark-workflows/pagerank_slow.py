import time

from pyspark import SparkConf, SparkContext 
        
# 5 iterations in 30sec
# 10 iterations in 9:23s
# same output values as the gpig version

RESET = 0.15
NUM_ITERATIONS = 5

conf = SparkConf().setAppName("PageRank Example").setMaster("local") 
sc = SparkContext(conf=conf) 
lines = sc.textFile("data/citeseer-graph.txt")

edges = lines.map(lambda line:tuple(line.split()))
nodes = edges.keys().distinct()
outlinks_by_node = edges.groupByKey().mapValues(lambda iter:list(iter)).cache()

pagerank_scores = nodes.map(lambda node:(node,1.0))

def messages(page_info):
    src, (current_score, outlinks) = page_info
    delta_from_src = (1 - RESET) * current_score/len(outlinks)
    return [(dst, delta_from_src) for dst in outlinks]

for t in range(NUM_ITERATIONS):
    
    print(f'iteration {t + 1} of {NUM_ITERATIONS}')

    page_hops = pagerank_scores.join(outlinks_by_node).flatMap(messages)
    resets = pagerank_scores.mapValues(lambda _: RESET)
    pagerank_scores = page_hops.union(resets).reduceByKey(lambda a, b:a+b)

start = time.time()
pr_items = pagerank_scores.collect()
print('scores collected:',time.time() - start,'sec')
pr_items = sorted(pr_items)
print(pr_items[:3])
print(pr_items[-3:])


