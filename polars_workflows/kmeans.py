"""kmeans.py -- spherical k-means over a corpus of documents, in polars.

Documents arrive as sparse tf-idf vectors, precomputed by
data/<corpus>.tfidf.tsv and stored as (doc, token, w) rows, which are
immediately renamed as (doc_id, token, w).

┌────────┬────────────┬──────────┐
│ doc_id ┆ token      ┆ w        │
│ ---    ┆ ---        ┆ ---      │
│ i64    ┆ str        ┆ f64      │
╞════════╪════════════╪══════════╡
│ 0      ┆ all        ┆ 0.1822   │
│ 0      ┆ anything   ┆ 0.363998 │
│ 0      ┆ coke       ┆ 0.581559 │
│ 0      ┆ difference ┆ 0.419413 │
...

Centroids are stored in the same format except 'doc_id' is renamed to
'cluster_id'.

┌────────────┬────────────┬──────────┐
│ cluster_id ┆ token      ┆ w        │
│ ---        ┆ ---        ┆ ---      │
│ u32        ┆ str        ┆ f64      │
╞════════════╪════════════╪══════════╡
│ 2          ┆ about      ┆ 0.048988 │
│ 2          ┆ an         ┆ 0.050782 │
│ …          ┆ …          ┆ …        │
│ 7          ┆ their      ┆ 0.089382 │
│ 7          ┆ you        ┆ 0.067531 │
...


One key routine is assign_to_closest_centroid(docs, centroids), which
joins docs with centroids on tokens so as to compute a sparse matrix
product. Both document and centroid vectors are L2-normalized, so that
dot product IS the cosine similarity, and the nearest centroid is the
largest. It returns a df like this


┌────────┬────────────┬──────────┐
│ doc_id ┆ cluster_id ┆ sim      │
│ ---    ┆ ---        ┆ ---      │
│ i64    ┆ u32        ┆ f64      │
╞════════╪════════════╪══════════╡
│ 922    ┆ 3          ┆ 0.081892 │
│ 247    ┆ 0          ┆ 0.067043 │
│ 758    ┆ 2          ┆ 0.070868 │
...

The second key routine is recompute_centroids(docs, assignment) which
joins docs and assignments and aggregates the document weights by
(cluster,token) to compute the mean token weight (actually the summed
token weight, which is then renormalized).

Assignments and centroids are materialized in memory, and the the docs
remain a LazyFrame, which are streamed through on each iteration.

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
        print(f'LazyFrame top {k}:')
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
        docs_lf: pl.LazyFrame, k, seed) -> pl.DataFrame:
    """Initialize with k distinct documents, taken as the first centroids.
    """

    # get the distinct docids. also sort the distinct doc_ids so that
    # sampling will always produce the same results given a fixed seed

    doc_ids_df = docs_lf.select('doc_id').unique().sort('doc_id').collect(engine='streaming')

    # sample k distinct docids and add a 'cluster_id' field
    chosen_df = (
        doc_ids_df.sample(n=k, shuffle=True, seed=seed)
        .with_row_index('cluster_id')
    )
    return (
        # find the documents that were chosen
        docs_lf.join(chosen_df.lazy(), on='doc_id')
        # output a DataFrame with the fields (cluster_id, token, w)
        .select('cluster_id', 'token', 'w')
        .collect(engine='streaming')
    )


def assign_to_closest_centroid(
        docs_lf: pl.LazyFrame, centroids_df: pl.DataFrame) -> pl.LazyFrame:
    """Nearest centroid to each document by cosine similarity.
    """

    return (
        # join on tokens
        docs_lf.join(centroids_df.lazy(), on='token')

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
        docs_lf: pl.LazyFrame, assignment_df: pl.DataFrame) -> pl.DataFrame:
    """The L2-normalized mean of each cluster's members.

    The docs are L2-normalized, so renormalizing the centroid to unit
    length means that a dot product computes cosine similarity.
    """

    return (
        # get the weighted tokens for the documents in each cluster
        docs_lf.join(assignment_df.lazy(), on='doc_id')

        # sum the weights of each token in each cluster
        .group_by('cluster_id', 'token').agg(pl.col('w').sum().alias('w'))

        # normalize to unit length
        .with_columns(
            w=pl.col('w') / pl.col('w').pow(2).sum().sqrt().over('cluster_id'))
        .collect(engine='streaming')
    )


def kmeans(docs_lf: pl.LazyFrame, k=8, max_iterations=20, seed=1):
    """Cluster the tf-idf vectors, returning the assignment and the centroids.
    """
    start = time.time()

    def handle_sparse_centroids(assignment_lf, t):
        """Helper function.

        The first set of centroids are sparse, because they are single
        docs, so assign cluster 0 to any docs sharing no tokens with
        any centroid.  This only need be done for the first iteration.
        """

        if t > 0:
            return assignment_lf
        return (
            # anchor against every doc: a doc that matched no centroid is
            # absent from `assignment_lf`, so left-join it back as a null row...
            docs_lf.select('doc_id').unique()
            .join(assignment_lf, on='doc_id', how='left')
            # ...then default those docs into cluster 0
            .with_columns(
                cluster_id=pl.col('cluster_id').fill_null(0),
                sim=pl.col('sim').fill_null(0.0),
            )
        )

    # generate the initial centroids
    centroids_df = initial_centroids(docs_lf, k, seed)
    show('initial centroids', centroids_df)

    previous_assignment_df = None

    for t in range(max_iterations):

        # assign docs to centroids, materializing once so the sparse join
        # runs a single time rather than being recomputed by each consumer
        assignment_df = (
            assign_to_closest_centroid(docs_lf, centroids_df)
            .pipe(handle_sparse_centroids, t)
            .collect(engine='streaming')
        )
        if t == 0:
            show(f'assignment {t + 1}', assignment_df)

        # recompute the centroids
        centroids_df = recompute_centroids(docs_lf, assignment_df)

        # check convergence
        num_moved = None
        if previous_assignment_df is not None:
            num_moved = (
                previous_assignment_df
                .join(assignment_df, on='doc_id', suffix='_new')
                .filter(pl.col('cluster_id') != pl.col('cluster_id_new'))
                .height
            )

        # report
        mean_sim = assignment_df.select('sim').mean().item()
        print(f'iteration {t + 1:2d} of {max_iterations} '
              f'time {time.time() - start:.4f} '
              f'mean cosine {mean_sim:.4f} '
              f'moved {num_moved}')

        previous_assignment_df = assignment_df

        if num_moved == 0:
            break

    print(f'clustered into {k} clusters: '
          f'{time.time() - start:.4f} sec')
    return assignment_df, centroids_df



def report(assignment_df: pl.DataFrame, centroids_df: pl.DataFrame,
           lines_lf: pl.LazyFrame, n_terms=8, n_titles=2):
    """Per cluster: how many documents, its heaviest terms, a couple of members.
    """
    sizes_df = (
        assignment_df.group_by('cluster_id').agg(
            pl.len().alias('n_docs'),
            pl.col('sim').mean().alias('mean_cosine'))
        .sort('n_docs', descending=True)
    )
    top_terms_df = (
        centroids_df.lazy()
        .group_by('cluster_id').agg(
            pl.col('token').sort_by('w', descending=True).head(n_terms))
        .collect(engine='streaming')
    )
    # a document's first few words stand in for a title
    snippets_df = (
        lines_lf.with_row_index('doc_id')
        .with_columns(snippet=pl.col('line').str.strip_chars().str.slice(0, 70))
        .select('doc_id', 'snippet')
        .collect(engine='streaming')
    )

    print()
    print(' clusters '.center(80, '='))
    terms_of = dict(zip(top_terms_df['cluster_id'], top_terms_df['token']))
    for row in sizes_df.iter_rows(named=True):
        c = row['cluster_id']
        print(f"\ncluster {c}: {row['n_docs']} docs, "
              f"mean cosine {row['mean_cosine']:.3f}")
        print(f"  terms: {' '.join(terms_of.get(c, []))}")
        members_df = (
            assignment_df.filter(pl.col('cluster_id') == c)
            .sort('sim', descending=True).head(n_titles)
            .join(snippets_df, on='doc_id')
        )
        for m in members_df.iter_rows(named=True):
            print(f"    [{m['sim']:.3f}] {m['snippet']}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster a corpus with spherical k-means.')
    parser.add_argument(
        '--corpus', default='../data/bluecorpus.txt',
        help='corpus to cluster, one document per line')
    parser.add_argument(
        '--vectors', default=None,
        help='precomputed tf-idf vectors as a (doc, token, w) '
        'tsv (default: the corpus with a .tfidf.tsv suffix)')
    parser.add_argument(
        '--k', type=int, default=8,
        help='number of clusters')
    parser.add_argument(
        '--max_iterations', type=int, default=20,
        help='maximum number of iterations')
    parser.add_argument(
        '--seed', type=int, default=1,
        help='seed for the initial choice of centroids')
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='verbosity level: 1 shows show() output, 0 suppresses it')
    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error('--max_iterations must be at least 1')

    VERBOSE = args.verbose

    vectors = args.vectors or args.corpus.rsplit('.', 1)[0] + '.tfidf.tsv'

    print(f'loading vectors from {vectors}')
    docs_lf = scan_vectors(vectors)
    show('docs with TFIDF token weights', docs_lf)

    assignment_df, centroids_df = kmeans(
        docs_lf, k=args.k,
        max_iterations=args.max_iterations,
        seed=args.seed)

    if VERBOSE:
        # the corpus itself is read only to show what landed in each cluster
        report(assignment_df, centroids_df, pl.scan_lines(args.corpus))
