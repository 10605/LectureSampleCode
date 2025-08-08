import time
import operator

from pyspark import SparkConf, SparkContext 
        
RESET = 0.15
NUM_ITERATIONS = 10

conf = SparkConf().setAppName("PageRank Example").setMaster("local") 
sc = SparkContext(conf=conf) 
lines = sc.textFile("data/citeseer-graph.txt")

edges = lines.map(lambda line:tuple(line.split()))
nodes = edges.keys().distinct()
outlinks_by_node = edges.groupByKey().mapValues(lambda iter:list(iter)).cache()

# keep these in memory
pagerank_scores = {node:1.0 for node in nodes.collect()}

# pass the pagerank_scores to the workers
# with an explicit closure
def make_message_mapper(pagerank_scores):
    def outgoing_msgs_for_page(node_outlink_pair):
        node, outlinks = node_outlink_pair
        current_score = pagerank_scores[node]
        delta = (1 - RESET) * current_score/len(outlinks)
        return [(dst, delta) for dst in outlinks]
    return outgoing_msgs_for_page

start = time.time()
for t in range(NUM_ITERATIONS):
    
    # monitor progress
    scores = pagerank_scores.values()
    print(f'iteration {t + 1} of {NUM_ITERATIONS} time {time.time() - start}', 
          f'len {len(scores)} max {max(scores)} min {min(scores)} mean {sum(scores)/len(scores)}')

    page_hops = outlinks_by_node.flatMap(make_message_mapper(pagerank_scores))
    new_scores = page_hops.combineByKey(
        createCombiner=lambda delta: delta + RESET,
        mergeValue=operator.add,
        mergeCombiners=operator.add)
    pagerank_scores = dict(new_scores.collect())

print('scores collected:',time.time() - start,'sec')
pr_items = sorted(pagerank_scores.items())
print(pr_items[:3])
print(pr_items[-3:])
