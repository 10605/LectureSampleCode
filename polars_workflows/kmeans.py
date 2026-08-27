"""kmeans.py -- spherical k-means over a corpus of documents, in polars.

Documents arrive as sparse tf-idf vectors, precomputed by
data/<corpus>.tfidf.tsv and stored as (doc, token, weight) rows -- the same
representation matmul.py uses for a sparse matrix.  Every step of the
algorithm is then a group_by or a join, which is to say it is
map-reduce.

  * assign_to_closest_centroid -- docs JOIN centroids ON token, summed
    per (doc_id, cluster_id), is a sparse matrix product.  Both
    vectors are L2-normalized, so that dot product IS the cosine
    similarity, and the nearest centroid is the largest.

  * recompute_centroids      -- docs JOIN assignment ON doc, summed per (cluster, token), is
    the mean of each cluster's members, renormalized.

Only the O(k x vocab) centroid table is materialized in memory, exactly as
pagerank.py materializes its O(nodes) score table; the docs stay a LazyFrame
and stream on each iteration.

The vectors are already L2-normalized, so every document has unit length and a
dot product between two of them is their cosine similarity.

Usage:
    python3 kmeans.py                          # 8 clusters over bluecorpus
    python3 kmeans.py --k 12 --max_iterations 30 --seed 7
    python3 kmeans.py --corpus ../data/redcorpus.txt

"""

import argparse
import time

import polars as pl

VERBOSE = 1


def show(msg, df: pl.LazyFrame | pl.DataFrame, k=10):
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

    `doc_id` is the document's line number in the corpus it came from, so the
    vectors join back to the text.  A document whose tokens were all pruned
    away has no rows here at all, and simply is not clustered.
    """
    return pl.scan_csv(path, separator='\t').rename({'doc': 'doc_id'})


def initial_centroids(
        docs: pl.LazyFrame, k, seed) -> pl.DataFrame:
    """Initialize with k distinct documents, taken as the first centroids.
    """

    # get the distinct docids. also sort the distinct doc_ids so that
    # sampling will always produce the same results given a fixed seed

    doc_ids = docs.select('doc_id').unique().sort('doc_id').collect(engine='streaming')

    # sample k distinct docids and add a 'cluster_id' field
    chosen = (
        doc_ids.sample(n=k, shuffle=True, seed=seed)
        .with_row_index('cluster_id')
    )
    # 
    return (
        # find the documents that were chosen
        docs.join(chosen.lazy(), on='doc_id')
        # output a DataFrame with the fields (cluster_id, token, w)
        .select('cluster_id', 'token', 'w')
        .collect(engine='streaming')
    )


def assign_to_closest_centroid(
        docs: pl.LazyFrame, centroids: pl.DataFrame) -> pl.LazyFrame:
    """Nearest centroid to each document by cosine similarity.
    """

    return (
        # join on tokens
        docs.join(centroids.lazy(), on='token')

        # compute product of weights of shared tokens
        .with_columns(prod=pl.col('w') * pl.col('w_right'))

        # aggregate to get doc-cluster dot product
        .group_by('doc_id', 'cluster_id').agg(pl.col('prod').sum().alias('sim'))

        # get best cluster for each document, breaking ties by cluster
        # number so that documents can't move back and forth between
        # equally-good clusters
        .group_by('doc_id').agg(
            pl.col('cluster_id').sort_by(
                ['sim', 'cluster_id'],
                descending=[True, False]).first(),
            pl.col('sim').max().alias('sim'))
    )


def recompute_centroids(
        docs: pl.LazyFrame, assignment: pl.DataFrame) -> pl.DataFrame:
    """The L2-normalized mean of each cluster's members.

    The docs are L2-normalized, so renormalizing the centroid to unit
    length means that a dot product computes cosine similarity.
    """

    return (
        # get the weighted tokens for the documents in each cluster
        docs.join(assignment.lazy(), on='doc_id')

        # sum the weights of each token in each cluster
        .group_by('cluster_id', 'token').agg(pl.col('w').sum().alias('w'))

        # normalize to unit length
        .with_columns(
            w=pl.col('w') / pl.col('w').pow(2).sum().sqrt().over('cluster_id'))
        .collect(engine='streaming')
    )


def kmeans(docs: pl.LazyFrame, k=8, max_iterations=20, seed=1):
    """Cluster the tf-idf vectors, returning the assignment and the centroids.
    """
    start = time.time()

    # generate the initial centroids

    centroids = initial_centroids(docs, k, seed)
    show('initial centroids', centroids)

    # initial assignment every doc
    assignment = (
        # uniq doc ids
        docs.select('doc_id').unique()

        # add default cluster assignment and similarity
        .with_columns(
            cluster_id=pl.lit(0, dtype=pl.UInt32),
            sim=pl.lit(0.0))
        .collect(engine='streaming')
    )

    for t in range(max_iterations):

        previous_assignment = assignment
        closest = assign_to_closest_centroid(docs, centroids)

        #TODO: can we make this step conditional on t==0 ?
        assignment = (
            previous_assignment.lazy()
            .join(closest, on='doc_id', how='left', suffix='_new')
            # coalesce takes the first non-null value across its arguments
            # e.g., it sets cluster_id to the _new one if it is non-null
            .with_columns(
                cluster_id=pl.coalesce('cluster_id_new', 'cluster_id'),
                sim=pl.coalesce('sim_new', pl.lit(0.0)))
            .select('doc_id', 'cluster_id', 'sim')
            .collect(engine='streaming')
        )

        # see how many docs changed clusters and stop if none did
        # TODO: move this up earlier?
        moved = (
            previous_assignment.join(assignment, on='doc_id', suffix='_new')
            .filter(pl.col('cluster_id') != pl.col('cluster_id_new')).height
        )
        if moved == 0:
            print(f'converged after {t + 1} iterations')
            break

        # report progress
        mean_sim = assignment.select('sim').mean().item()
        print(f'iteration {t + 1:2d} of {max_iterations} '
              f'time {time.time() - start:.4f} '
              f'mean cosine {mean_sim:.4f} moved {moved}')

        # recompute the centroids
        centroids = recompute_centroids(docs, assignment)


    print(f'clustered into {k} clusters: '
          f'{time.time() - start:.4f} sec')
    return assignment, centroids


def report(assignment: pl.DataFrame, centroids: pl.DataFrame,
           lines: pl.LazyFrame, n_terms=8, n_titles=2):
    """Per cluster: how many documents, its heaviest terms, a couple of members.
    """
    sizes = (
        assignment.group_by('cluster_id').agg(
            pl.len().alias('n_docs'),
            pl.col('sim').mean().alias('mean_cosine'))
        .sort('n_docs', descending=True)
    )
    top_terms = (
        centroids.lazy()
        .group_by('cluster_id').agg(
            pl.col('token').sort_by('w', descending=True).head(n_terms))
        .collect(engine='streaming')
    )
    # a document's first few words stand in for a title
    snippets = (
        lines.with_row_index('doc_id')
        .with_columns(snippet=pl.col('line').str.strip_chars().str.slice(0, 70))
        .select('doc_id', 'snippet')
        .collect(engine='streaming')
    )

    print()
    print(' clusters '.center(80, '='))
    terms_of = dict(zip(top_terms['cluster_id'], top_terms['token']))
    for row in sizes.iter_rows(named=True):
        c = row['cluster_id']
        print(f"\ncluster {c}: {row['n_docs']} docs, "
              f"mean cosine {row['mean_cosine']:.3f}")
        print(f"  terms: {' '.join(terms_of.get(c, []))}")
        members = (
            assignment.filter(pl.col('cluster_id') == c)
            .sort('sim', descending=True).head(n_titles)
            .join(snippets, on='doc_id')
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
    parser.add_argument('--max_iterations', type=int, default=20,
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

    assignment, centroids = kmeans(
        docs, k=args.k,
        max_iterations=args.max_iterations,
        seed=args.seed)

    # the corpus itself is read only to show what landed in each cluster
    report(assignment, centroids, pl.scan_lines(args.corpus))
