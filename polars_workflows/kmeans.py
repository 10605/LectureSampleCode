"""kmeans.py -- spherical k-means over a corpus of documents, in polars.

Documents arrive as sparse tf-idf vectors, precomputed by
data/<corpus>.tfidf.tsv and held long-form as (doc, token, w) rows -- the same
representation matmul.py uses for a sparse matrix.  Every step of the algorithm
is then a group_by or a join, which is to say it is map-reduce:

  * assignment  -- docs JOIN centroids ON token, summed per (doc, cluster), is
    a sparse matrix product.  Both vectors are L2-normalized, so that dot
    product IS the cosine similarity, and the nearest centroid is the largest.

  * update      -- docs JOIN assignment ON doc, summed per (cluster, token), is
    the mean of each cluster's members, renormalized.

Only the O(k x vocab) centroid table is materialized in memory, exactly as
pagerank.py materializes its O(nodes) score table; the docs stay a LazyFrame
and stream on each iteration.

The vectors are already L2-normalized, so every document has unit length and a
dot product between two of them is their cosine similarity.

Usage:
    python3 kmeans.py                          # 8 clusters over bluecorpus
    python3 kmeans.py --k 12 --num_iterations 30 --seed 7
    python3 kmeans.py --corpus ../data/redcorpus.txt
"""

import argparse
import time

import polars as pl

VERBOSE = 1


def show(msg, df, k=10):
    """Give a quick view of the contents of a dataframe.
    """
    if VERBOSE < 1:
        return
    print(msg)
    if isinstance(df, pl.LazyFrame):
        df = df.head(k).collect(engine='streaming')
    print(df)


def scan_vectors(path):
    """The precomputed tf-idf vectors: one (doc, token, w) row per token per doc.

    `doc` is the document's line number in the corpus it came from, so the
    vectors join back to the text.  A document whose tokens were all pruned
    away has no rows here at all, and simply is not clustered.
    """
    return pl.scan_csv(path, separator='\t')


def seed_centroids(docs, doc_ids, k, seed):
    """initialize with k distinct documents, taken as the first centroids.
    """
    chosen = (
        doc_ids.sample(n=k, shuffle=True, seed=seed)
        .with_row_index('cluster')
    )
    return (
        docs.join(chosen.lazy(), on='doc')
        .select('cluster', 'token', 'w')
        .collect(engine='streaming')
    )


def assign(docs, centroids):
    """Nearest centroid per document, by cosine similarity.

    The join is the sparse product docs x centroids^T: a (doc, cluster) pair
    accumulates one term per token the two share, and unshared tokens
    contribute nothing, so they are never enumerated.  Ties break toward the
    lower cluster id, which keeps a run reproducible.
    """
    return (
        docs.join(centroids.lazy(), on='token')
        .with_columns(prod=pl.col('w') * pl.col('w_right'))
        .group_by('doc', 'cluster').agg(pl.col('prod').sum().alias('sim'))
        .group_by('doc').agg(
            pl.col('cluster').sort_by(['sim', 'cluster'],
                                      descending=[True, False]).first(),
            pl.col('sim').max().alias('sim'))
    )


def update(docs, assignment):
    """The mean of each cluster's members, renormalized to unit length.

    Summing rather than averaging is deliberate: a cluster's mean and its sum
    differ by a per-cluster constant, which the normalization then divides out.

    The documents arrive normalized, but a centroid cannot: it is rebuilt from
    a different membership every iteration, so this is the one normalization
    the algorithm has to do for itself -- without it the dot product in
    assign() would not be a cosine.
    """
    return (
        docs.join(assignment.lazy(), on='doc')
        .group_by('cluster', 'token').agg(pl.col('w').sum().alias('w'))
        .with_columns(
            w=pl.col('w') / pl.col('w').pow(2).sum().sqrt().over('cluster'))
        .collect(engine='streaming')
    )


def kmeans(docs, k=8, num_iterations=20, seed=1):
    """Cluster the tf-idf vectors, returning the assignment and the centroids.
    """
    start = time.time()
    # sorted, not merely unique: unique() does not promise an order, and
    # seed_centroids samples positionally, so without this the same seed picks
    # different documents from one run to the next
    doc_ids = docs.select('doc').unique().sort('doc').collect(engine='streaming')
    n_docs = doc_ids.height
    if n_docs < k:
        raise SystemExit(f'only {n_docs} documents for k={k}')

    centroids = seed_centroids(docs, doc_ids, k, seed)
    show('initial centroids', centroids)

    # every doc starts in cluster 0, a placeholder the first assignment
    # overwrites; carrying it lets a doc that matches no centroid at all keep
    # the cluster it had rather than drop out of the table.
    assignment = doc_ids.with_columns(cluster=pl.lit(0, dtype=pl.UInt32),
                                      sim=pl.lit(0.0))
    n_clusters = k

    for t in range(num_iterations):
        # the left join keeps a doc that matched no centroid at all: its
        # proposed cluster is null, and coalesce leaves it where it was
        previous = assignment
        assignment = (
            previous.lazy()
            .join(assign(docs, centroids), on='doc', how='left', suffix='_new')
            .with_columns(
                cluster=pl.coalesce('cluster_new', 'cluster'),
                sim=pl.coalesce('sim_new', pl.lit(0.0)))
            .select('doc', 'cluster', 'sim')
            .collect(engine='streaming')
        )
        moved = (
            previous.join(assignment, on='doc', suffix='_new')
            .filter(pl.col('cluster') != pl.col('cluster_new')).height
        )
        mean_sim = assignment.select('sim').mean().item()
        print(f'iteration {t + 1:2d} of {num_iterations} '
              f'time {time.time() - start:.4f} '
              f'mean cosine {mean_sim:.4f} moved {moved}')

        centroids = update(docs, assignment)
        # a cluster that lost every member has nothing to average and so drops
        # out of the centroid table: k shrinks rather than the run failing.
        # Only happens when k approaches the number of documents.
        n_clusters = centroids['cluster'].n_unique()
        if n_clusters < k:
            print(f'  note: down to {n_clusters} clusters -- '
                  f'{k - n_clusters} lost every member')

        if moved == 0:
            print(f'converged after {t + 1} iterations')
            break

    print(f'clustered {n_docs} documents into {n_clusters} clusters: '
          f'{time.time() - start:.4f} sec')
    return assignment, centroids


def report(assignment, centroids, lines, n_terms=8, n_titles=2):
    """Per cluster: how many documents, its heaviest terms, a couple of members.
    """
    sizes = (
        assignment.group_by('cluster').agg(
            pl.len().alias('n_docs'),
            pl.col('sim').mean().alias('mean_cosine'))
        .sort('n_docs', descending=True)
    )
    top_terms = (
        centroids.lazy()
        .group_by('cluster').agg(
            pl.col('token').sort_by('w', descending=True).head(n_terms))
        .collect(engine='streaming')
    )
    # a document's first few words stand in for a title
    snippets = (
        lines.with_row_index('doc')
        .with_columns(snippet=pl.col('line').str.strip_chars().str.slice(0, 70))
        .select('doc', 'snippet')
        .collect(engine='streaming')
    )

    print()
    print(' clusters '.center(80, '='))
    terms_of = dict(zip(top_terms['cluster'], top_terms['token']))
    for row in sizes.iter_rows(named=True):
        c = row['cluster']
        print(f"\ncluster {c}: {row['n_docs']} docs, "
              f"mean cosine {row['mean_cosine']:.3f}")
        print(f"  terms: {' '.join(terms_of.get(c, []))}")
        members = (
            assignment.filter(pl.col('cluster') == c)
            .sort('sim', descending=True).head(n_titles)
            .join(snippets, on='doc')
        )
        for m in members.iter_rows(named=True):
            print(f"    [{m['sim']:.3f}] {m['snippet']}...")


def vectors_for(corpus):
    """The precomputed vectors that go with a corpus: foo.txt -> foo.tfidf.tsv."""
    return corpus.rsplit('.', 1)[0] + '.tfidf.tsv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster a corpus with spherical k-means.')
    parser.add_argument('--corpus', default='../data/bluecorpus.txt',
                        help='corpus to cluster, one document per line')
    parser.add_argument('--vectors', default=None,
                        help='precomputed tf-idf vectors as a (doc, token, w) '
                             'tsv (default: the corpus with a .tfidf.tsv suffix)')
    parser.add_argument('--k', type=int, default=8,
                        help='number of clusters')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='maximum number of iterations')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for the initial choice of centroids')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level: 1 shows show() output, 0 suppresses it')
    args = parser.parse_args()

    VERBOSE = args.verbose

    vectors = args.vectors or vectors_for(args.corpus)
    print(f'loading vectors from {vectors}')
    docs = scan_vectors(vectors)
    show('docs (sparse tf-idf)', docs)

    assignment, centroids = kmeans(docs, k=args.k,
                                   num_iterations=args.num_iterations,
                                   seed=args.seed)
    # the corpus itself is read only to show what landed in each cluster
    report(assignment, centroids, pl.scan_lines(args.corpus))
