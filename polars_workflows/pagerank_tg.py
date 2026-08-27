"""pagerank_tg.py -- pagerank over tardigrade's streaming LazyFrame.

The same algorithm as pagerank.py, with polars swapped out for plain Python
iterators. The shape is unchanged because it was map-reduce all along:

  * the O(nodes) tables -- outdegree per src, and the scores -- are ordinary
    dicts, exactly as pagerank.py keeps them as in-memory DataFrames;

  * the O(edges) step, which emits one message per edge, is streamed: the edge
    file is re-scanned each iteration and the messages are grouped by dst
    through tardigrade's external sort, so a graph with far more edges than RAM
    still runs.

Assumes, as pagerank.py does, that every node has indegree >= 1 and
outdegree >= 1: a node with no incoming edge gets no message and so drops out
of the score table.
"""

import argparse
import time

import tardigrade as tg

VERBOSE = 1

# rows are lines throughout: edges are 'src dst', messages are 'dst<TAB>mass'
src_of = lambda line: line.split()[0]
key_of = lambda line: line.split('\t')[0]
mass_of = lambda line: float(line.split('\t')[1])


def show(msg, rows, k=10):
    """Give a quick view of the first few rows of a frame."""
    if VERBOSE < 1:
        return
    print(msg)
    for row in rows[:k]:
        print(row.rstrip('\n') if isinstance(row, str) else row)


def pagerank(scan_edges, reset=0.15, num_iterations=30,
             batch_size=tg.DEFAULT_BATCH_SIZE):
    """scan_edges() must return a FRESH LazyFrame over the edge lines -- the main
    loop re-reads the graph every iteration rather than holding it in memory.
    """
    def edges():
        return scan_edges().filter(
            lambda line: line.strip() and not line.startswith('#'))

    # outdegree per src.  O(nodes) in memory, but computed by external sort,
    # so the edge list itself never has to fit.
    n_outlinks = dict(edges().group_by_key(key=src_of, agg=len, batch_size=batch_size)
                      .collect())
    show('n_outlinks', list(n_outlinks.items()))

    # every src and every dst, deduped -- pagerank.py's concat + unique
    nodes = (edges()
             .map(str.split).explode()
             .unique(batch_size=batch_size)
             .map(str.rstrip))
    scores = {node: 1.0 for node in nodes.collect()}
    show('scores', list(scores.items()))

    start = time.time()
    for t in range(num_iterations):
        print(f'iteration {t + 1:2d} of {num_iterations} time {time.time() - start:.4f}',
              f'len {len(scores)} max {max(scores.values()):.2f}',
              f'min {min(scores.values()):.2f}',
              f'mean {sum(scores.values()) / len(scores):.2f}')

        # per-src mass = (1-reset) * score / outdegree: the scalar each out-edge
        # carries.  Both operands are O(nodes), so this stays in memory.
        mass = {src: (1 - reset) * scores[src] / n for src, n in n_outlinks.items()}

        # the edge-proportional step: one message per edge, keyed by dst.  This
        # is where pagerank.py does its join; here mass is already in
        # memory, so a dict lookup does the same work without the second sort.
        def message(line):
            src, dst = line.split()
            return f'{dst}\t{mass[src]}\n'

        scores = {node: reset + incoming for node, incoming in
                  edges().map(message)
                  .group_by_key(key=key_of, value=mass_of, agg=sum,
                                batch_size=batch_size).collect()}

    elapsed = time.time() - start
    ranked = sorted(scores.items(), key=lambda ns: ns[1], reverse=True)
    print('top pagerank_scores:')
    for node, score in ranked[:10]:
        print(f'{node}\t{score:.6f}')
    print('bottom pagerank_scores:')
    for node, score in ranked[-10:]:
        print(f'{node}\t{score:.6f}')
    print('scores collected [tardigrade]:', elapsed, 'sec')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute pagerank scores for a graph.')
    parser.add_argument('--filename', default='../data/citeseer-graph.txt',
                        help='edge list file to load (may be gzipped)')
    parser.add_argument('--reset', type=float, default=0.15,
                        help='reset (teleport) probability')
    parser.add_argument('--num_iterations', type=int, default=30,
                        help='number of pagerank iterations')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level: 1 shows show() output, 0 suppresses it')
    parser.add_argument('--batch_size', type=int, default=tg.DEFAULT_BATCH_SIZE,
                        help='rows per sorted run in the external sort')
    args = parser.parse_args()

    VERBOSE = args.verbose

    filename = args.filename
    print(f'loading from {filename}')
    scan = tg.scan_gz_lines if filename.endswith('.gz') else tg.scan_lines
    pagerank(lambda: scan(filename), reset=args.reset,
             num_iterations=args.num_iterations, batch_size=args.batch_size)
