"""pytest suite for kmeans.py -- spherical k-means as sparse polars joins.

Each step is checked against the obvious dense in-memory version as the oracle:
assignment against a dict-of-dicts argmax over cosines, the update against a
plain mean of its members.  So the tests say what the joins *mean* rather than
restating them.  Run with:  pytest test_kmeans.py

Documents here are unit vectors built by hand, standing in for the tf-idf
vectors kmeans.py reads from a .tfidf.tsv file: the weights are invented, only
their normalization matters.

The interesting cases are the ones where a sparse formulation can quietly lose
a row: a document sharing no token with any centroid, a cluster that loses
every member and would vanish from the centroid table, and ties between equally
near centroids.  Determinism is tested too -- a fixed seed must give the same
clustering twice, which is not free when the seeding samples from an unordered
frame.
"""
import math

import polars as pl
import pytest

import kmeans as km

# Three groups with disjoint vocabularies, so a document in one has cosine 0
# with every document in another and the true partition is the best clustering
# at k=3.  Weights vary within a group so no two documents are the same vector.
GROUPS = [
    [dict(alpha=3, beta=1), dict(alpha=1, beta=2, gamma=1),
     dict(beta=1, gamma=3), dict(alpha=2, gamma=1)],
    [dict(delta=3, epsilon=1), dict(delta=1, epsilon=2, zeta=1),
     dict(epsilon=1, zeta=3), dict(delta=2, zeta=1)],
    [dict(eta=3, theta=1), dict(eta=1, theta=2, iota=1),
     dict(theta=1, iota=3), dict(eta=2, iota=1)],
]
SEPARABLE = [counts for group in GROUPS for counts in group]
TRUE_LABELS = [i for i, group in enumerate(GROUPS) for _ in group]


def vectors(documents):
    """(doc_id, token, w) rows for a list of token->weight dicts, each normalized.

    A stand-in for what scan_vectors returns: invented weights, but unit
    length, which is what the algorithm actually relies on.  Note the *file*
    on disk names the column 'doc'; scan_vectors renames it to 'doc_id', so
    that is the name every kmeans.py entry point expects.
    """
    rows = []
    for doc, counts in enumerate(documents):
        length = math.sqrt(sum(w * w for w in counts.values()))
        rows += [(doc, token, w / length) for token, w in counts.items()]
    doc, token, w = zip(*rows)
    return pl.LazyFrame(dict(doc_id=doc, token=token, w=w))


def dense(docs, key='doc_id'):
    """The sparse (key, token, w) frame as {key: {token: w}} -- the oracle's form."""
    if isinstance(docs, pl.LazyFrame):
        docs = docs.collect()
    out = {}
    for k, token, w in docs.select(key, 'token', 'w').iter_rows():
        out.setdefault(k, {})[token] = w
    return out


def norm(vec):
    return math.sqrt(sum(w * w for w in vec.values()))


def cosine(u, v):
    """Dot product; on unit vectors, which is what kmeans.py builds, it is the cosine."""
    return sum(w * v.get(token, 0.0) for token, w in u.items())


def nearest(vec, centroids):
    """Oracle for assign_to_closest_centroid: argmax cosine, ties to the lower id."""
    return min(centroids, key=lambda c: (-cosine(vec, centroids[c]), c))


def as_centroids(rows):
    """A centroid table, cluster ids typed as kmeans.py produces them."""
    cluster, token, w = zip(*rows)
    return pl.DataFrame(dict(cluster_id=cluster, token=token, w=w),
                        schema_overrides=dict(cluster_id=pl.UInt32))


def as_assignment(rows):
    cluster, sim = zip(*((c, s) for _, c, s in rows))
    return pl.DataFrame(
        dict(doc_id=[d for d, _, _ in rows], cluster_id=cluster, sim=sim),
        schema_overrides=dict(doc_id=pl.UInt32, cluster_id=pl.UInt32))


def partition(assignment):
    """The clustering as a set of frozensets, so cluster *labels* don't matter."""
    by_cluster = {}
    for doc, cluster in assignment.select('doc_id', 'cluster_id').iter_rows():
        by_cluster.setdefault(cluster, set()).add(doc)
    return {frozenset(docs) for docs in by_cluster.values()}


@pytest.fixture
def separable():
    """Unit vectors for the three disjoint-vocabulary groups."""
    return vectors(SEPARABLE)


# ------------------------------------------------------------- input file

def write_tsv(documents, path):
    """The vectors as the .tfidf.tsv converter writes them: column named 'doc'."""
    vectors(documents).rename({'doc_id': 'doc'}).sink_csv(path, separator='\t')
    return path


def test_scan_vectors_reads_what_the_converter_writes(tmp_path):
    """The integration point: a (doc, token, w) tsv, as __main__ reads it.

    scan_vectors renames the file's 'doc' column to 'doc_id' on the way in.
    """
    path = write_tsv(SEPARABLE, str(tmp_path / 'vectors.tsv'))
    got = km.scan_vectors(path)
    assert got.collect().columns == ['doc_id', 'token', 'w']
    read = dense(got)
    for doc, vec in dense(vectors(SEPARABLE)).items():
        assert read[doc] == pytest.approx(vec)


def test_kmeans_runs_from_a_file(tmp_path):
    """End to end from the tsv, the way the script is actually invoked."""
    path = write_tsv(SEPARABLE, str(tmp_path / 'vectors.tsv'))
    assignment, _ = km.kmeans(km.scan_vectors(path), k=3,
                              max_iterations=20, seed=0)
    assert partition(assignment) == true_partition()


# ------------------------------------------------------------ assignment

def test_assign_picks_the_nearest_centroid(separable):
    docs = separable
    centroids = km.initial_centroids(docs, 3, seed=0)
    got = km.assign_to_closest_centroid(docs, centroids).collect()
    oracle = dense(centroids, key='cluster_id')
    vectors = dense(docs)
    for doc, cluster, sim in got.select('doc_id', 'cluster_id', 'sim').iter_rows():
        assert cluster == nearest(vectors[doc], oracle)
        assert sim == pytest.approx(cosine(vectors[doc], oracle[cluster]))


def test_assign_breaks_ties_toward_the_lower_cluster():
    """Two centroids equally near: the run has to be reproducible."""
    docs = pl.LazyFrame(dict(doc_id=[0], token=['aa'], w=[1.0]),
                        schema_overrides=dict(doc_id=pl.UInt32))
    centroids = as_centroids([(1, 'aa', 1.0), (0, 'aa', 1.0), (2, 'aa', 1.0)])
    assert (km.assign_to_closest_centroid(docs, centroids)
            .collect()['cluster_id'].to_list()) == [0]


def test_assign_omits_a_document_sharing_no_token():
    """The join cannot produce a row for it, so the caller must cope -- which
    is what the left join in kmeans() is for."""
    docs = pl.LazyFrame(dict(doc_id=[0, 1], token=['aa', 'zz'], w=[1.0, 1.0]),
                        schema_overrides=dict(doc_id=pl.UInt32))
    centroids = as_centroids([(0, 'aa', 1.0)])
    assert (km.assign_to_closest_centroid(docs, centroids)
            .collect()['doc_id'].to_list()) == [0]


def test_a_document_matching_nothing_keeps_its_cluster():
    """End to end: the orphan stays put rather than dropping out of the table."""
    docs = vectors([dict(aa=1, bb=1), dict(aa=1, bb=2), dict(zz=1)])
    assignment, _ = km.kmeans(docs, k=2, max_iterations=5, seed=0)
    assert sorted(assignment['doc_id'].to_list()) == [0, 1, 2]


# ---------------------------------------------------------------- update

def test_update_is_the_normalized_mean_of_its_members(separable):
    docs = separable
    assignment = as_assignment(
        [(doc, label, 1.0) for doc, label in enumerate(TRUE_LABELS)])
    got = dense(km.recompute_centroids(docs, assignment), key='cluster_id')

    vectors = dense(docs)
    for cluster in set(TRUE_LABELS):
        members = [vectors[d] for d, c in enumerate(TRUE_LABELS) if c == cluster]
        mean = {}
        for vec in members:
            for token, w in vec.items():
                mean[token] = mean.get(token, 0.0) + w / len(members)
        length = norm(mean)
        assert got[cluster] == pytest.approx({t: w / length for t, w in mean.items()})


def test_update_centroids_are_unit_length(separable):
    docs = separable
    assignment = as_assignment(
        [(doc, label, 1.0) for doc, label in enumerate(TRUE_LABELS)])
    for vec in dense(km.recompute_centroids(docs, assignment),
                     key='cluster_id').values():
        assert norm(vec) == pytest.approx(1.0)


# ------------------------------------------------------ vanishing clusters

# three identical documents plus one other: two seeds land on the same vector,
# ties go to the lower cluster id, and the higher one is left with no members
DUPLICATES = [dict(aa=1), dict(aa=1), dict(aa=1), dict(zz=1)]


def test_update_loses_a_cluster_with_no_members(separable):
    """A cluster with nothing to average simply has no rows to produce."""
    docs = separable
    assignment = as_assignment([(doc, 0, 1.0) for doc in range(len(SEPARABLE))])
    assert (km.recompute_centroids(docs, assignment)['cluster_id']
            .unique().to_list()) == [0]


@pytest.mark.parametrize('seed', range(6))
def test_k_shrinks_rather_than_the_run_failing(seed):
    """Nothing re-seeds an emptied cluster, so k is an upper bound, not a
    promise -- but every document still ends up somewhere."""
    docs = vectors(DUPLICATES)
    assignment, centroids = km.kmeans(docs, k=3, max_iterations=5, seed=seed)
    assert centroids['cluster_id'].n_unique() <= 3
    assert sorted(assignment['doc_id'].to_list()) == list(range(len(DUPLICATES)))


@pytest.mark.xfail(strict=True, reason=
                   "kmeans.py no longer reports emptied clusters; the "
                   "'lost every member' message was dropped in the refactor. "
                   "Kept as a marker: if the report is restored this XPASSes "
                   "and the marker should be removed.")
def test_a_shrinking_k_is_reported(capsys):
    """Silently returning fewer clusters than asked for would be a trap."""
    km.kmeans(vectors(DUPLICATES), k=3, max_iterations=5, seed=1)
    assert 'lost every member' in capsys.readouterr().out


def test_no_note_when_every_cluster_keeps_a_member(separable, capsys):
    """Vacuous while the report above is missing; meaningful again if restored."""
    km.kmeans(separable, k=3, max_iterations=20, seed=0)
    assert 'lost every member' not in capsys.readouterr().out


# ------------------------------------------------------------- clustering

def true_partition():
    return {frozenset(d for d, c in enumerate(TRUE_LABELS) if c == label)
            for label in set(TRUE_LABELS)}


# seeds 2 and 6 start with all three centroids inside one group; see
# test_a_degenerate_seeding_finds_a_worse_local_optimum
@pytest.mark.parametrize('seed', [0, 1, 3, 4, 5, 7])
def test_kmeans_recovers_disjoint_groups(separable, seed):
    """Three groups with no vocabulary in common: k=3 finds exactly them."""
    docs = separable
    assignment, _ = km.kmeans(docs, k=3, max_iterations=20, seed=seed)
    assert partition(assignment) == true_partition()


@pytest.mark.parametrize('seed', [2, 6])
def test_a_degenerate_seeding_finds_a_worse_local_optimum(separable, seed):
    """Forgy initialization can draw every centroid from one group, and k-means
    only ever improves the objective it starts from -- so it converges, to
    something strictly worse than the true partition.  Not a defect in the
    implementation: the documented cost of seeding from random documents."""
    docs = separable
    assignment, _ = km.kmeans(docs, k=3, max_iterations=20, seed=seed)
    assert partition(assignment) != true_partition()
    assert assignment['sim'].mean() < objective_of(docs, TRUE_LABELS)


def objective_of(docs, labels):
    """Mean cosine to its own centroid, for a clustering imposed from outside."""
    assignment = as_assignment([(doc, c, 1.0) for doc, c in enumerate(labels)])
    centroids = km.recompute_centroids(docs, assignment)
    return km.assign_to_closest_centroid(docs, centroids).collect()['sim'].mean()


def test_every_document_is_assigned_exactly_once(separable):
    docs = separable
    assignment, _ = km.kmeans(docs, k=3, max_iterations=20, seed=0)
    assert sorted(assignment['doc_id'].to_list()) == list(range(len(SEPARABLE)))


def test_kmeans_keeps_k_clusters_when_none_go_empty(separable):
    """At a sane k every cluster keeps members, and k comes back intact."""
    docs = separable
    _, centroids = km.kmeans(docs, k=3, max_iterations=20, seed=3)
    assert sorted(centroids['cluster_id'].unique().to_list()) == list(range(3))


def test_assignment_is_the_argmax_at_convergence(separable):
    """The fixed-point property: no document would rather be somewhere else."""
    docs = separable
    assignment, centroids = km.kmeans(docs, k=3, max_iterations=20, seed=0)
    oracle = dense(centroids, key='cluster_id')
    vectors = dense(docs)
    for doc, cluster, sim in assignment.select(
            'doc_id', 'cluster_id', 'sim').iter_rows():
        assert cluster == nearest(vectors[doc], oracle)
        assert sim == pytest.approx(cosine(vectors[doc], oracle[cluster]))


def test_objective_never_decreases(separable):
    """Each iteration of k-means can only improve the mean cosine.  Running to
    successively more iterations is the cheapest way to watch it.

    Compared with a tolerance: once converged the objective merely repeats, and
    a threaded group_by need not sum a group the same way twice.
    """
    docs = separable
    objectives = [
        km.kmeans(docs, k=3, max_iterations=t, seed=5)[0]['sim'].mean()
        for t in range(1, 6)
    ]
    for before, after in zip(objectives, objectives[1:]):
        assert after >= before - 1e-12, objectives


def test_the_same_seed_gives_the_same_clustering(separable):
    """Regression: seeding samples positionally and unique() has no order, so
    this only holds because doc ids are sorted first.

    The clustering must match exactly; the similarities only to within
    floating-point noise, since a threaded group_by is free to sum a group in
    any order.
    """
    docs = separable
    runs = [km.kmeans(docs, k=3, max_iterations=20, seed=11)[0].sort('doc_id')
            for _ in range(2)]
    assert runs[0]['doc_id'].to_list() == runs[1]['doc_id'].to_list()
    assert runs[0]['cluster_id'].to_list() == runs[1]['cluster_id'].to_list()
    assert runs[0]['sim'].to_list() == pytest.approx(runs[1]['sim'].to_list())


def test_the_seed_reaches_the_initial_centroids(separable):
    """Different seeds have to actually pick different documents."""
    docs = separable
    seeded = [dense(km.initial_centroids(docs, 3, seed=s), key='cluster_id')
              for s in range(8)]
    assert any(a != b for a, b in zip(seeded, seeded[1:]))


def test_kmeans_refuses_more_clusters_than_documents(separable):
    """Asking for more clusters than documents fails rather than silently
    returning fewer.

    It used to exit with a diagnostic; initial_centroids now samples without
    replacement and lets polars raise, so this asserts the weaker property
    that the run does not succeed.  A friendlier error would be an
    improvement, not a behaviour change.
    """
    docs = separable
    with pytest.raises(pl.exceptions.ShapeError):
        km.kmeans(docs, k=len(SEPARABLE) + 1, max_iterations=5, seed=0)
