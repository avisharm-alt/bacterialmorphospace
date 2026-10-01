"""The two-stage Pfam filter (cmh_rank) and its use inside lopo_evaluate: no leakage, nulls refit, old models untouched."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("joblib")

from src import embed_core as ec  # noqa: E402


def _cmh_z(x, y, strata):
    """Independent, loop-based Cochran-Mantel-Haenszel z for ONE binary feature (the textbook formula)."""
    num = var = 0.0
    for s in sorted(set(strata)):
        idx = [i for i, t in enumerate(strata) if t == s]
        N = len(idx)
        n1 = sum(y[i] for i in idx)
        n0 = N - n1
        m1 = sum(x[i] for i in idx)
        a = sum(1 for i in idx if x[i] and y[i])
        if N < 2 or n1 == 0 or n0 == 0:
            continue
        num += a - n1 * m1 / N
        var += n1 * n0 * m1 * (N - m1) / (N * N * (N - 1))
    return abs(num) / var ** 0.5 if var > 0 else 0.0


def _toy(seed=0, per=60, d=40):
    rng = np.random.default_rng(seed)
    strata = np.repeat(["A", "B"], per)
    y = np.concatenate([rng.random(per) < 0.8, rng.random(per) < 0.2])  # prevalence differs a lot between strata
    X = (rng.random((2 * per, d)) < 0.5).astype(np.uint8)
    X[:, 0] = (strata == "A").astype(np.uint8)  # feature 0 is only the phylum label
    flip = rng.random(2 * per) < 0.1
    X[:, 1] = np.where(flip, 1 - y, y).astype(np.uint8)  # feature 1 tracks the trait inside both strata
    return X, y, strata


def test_cmh_ranking_matches_textbook_formula():
    X, y, strata = _toy()
    got = ec.cmh_rank(X, y, strata, min_prev=0.0)
    z = np.array([_cmh_z(X[:, j].tolist(), y.tolist(), strata.tolist()) for j in range(X.shape[1])])
    expected = np.argsort(-z, kind="stable")
    assert got.tolist() == expected.tolist()


def test_stratification_demotes_phylum_marker_and_promotes_within_phylum_signal():
    X, y, strata = _toy()
    order = ec.cmh_rank(X, y, strata, min_prev=0.0).tolist()
    assert order[0] == 1  # the within-phylum family wins
    assert order.index(0) > 10  # the pure phylum marker is not near the top
    naive = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    assert int(np.argmax(naive)) in (0, 1) and naive[0] > 0.3  # unstratified ranking WOULD rate the phylum marker highly


def test_stage1_prevalence_filter_uses_only_the_rows_it_is_given():
    n = 400
    rng = np.random.default_rng(1)
    X = (rng.random((n, 6)) < 0.5).astype(np.uint8)
    X[:, 0] = 0
    X[:3, 0] = 1  # 0.75% -> kept at 0.5%
    X[:, 1] = 0
    X[:1, 1] = 1  # 0.25% -> dropped
    X[:, 2] = 1  # ubiquitous -> dropped
    X[:, 3] = 1
    X[:1, 3] = 0  # 99.75% -> dropped
    y = rng.random(n) < 0.5
    cols = set(ec.cmh_rank(X, y, np.repeat(["A", "B"], n // 2), min_prev=0.005).tolist())
    assert 0 in cols and not ({1, 2, 3} & cols) and {4, 5} <= cols


def test_degenerate_strata_and_all_constant_features_do_not_crash():
    X = np.zeros((6, 3), dtype=np.uint8)
    X[:3, 0] = 1
    y = np.array([1, 0, 1, 1, 1, 1], dtype=bool)  # second stratum is single-class: contributes nothing
    out = ec.cmh_rank(X, y, np.array(list("AAABBB")), min_prev=0.0)
    assert sorted(out.tolist()) == [0, 1, 2]


def _lopo_data(seed=3, per=45, d=70, phyla=("P1", "P2", "P3", "P4")):
    rng = np.random.default_rng(seed)
    ph = np.repeat(list(phyla), per)
    y = np.concatenate([rng.random(per) < p for p in (0.3, 0.5, 0.7, 0.4)])
    genus = np.array([f"{p}g{i % 9}" for p in ph for i in range(per)][: len(ph)])
    X = (rng.random((len(ph), d)) < 0.4).astype(np.float32)
    for j in range(1, 6):  # within-phylum signal
        X[:, j] = np.where(rng.random(len(ph)) < 0.85, y, 1 - y)
    ids = np.arange(len(ph), dtype=np.float32)[:, None]
    return {"pfam": np.hstack([ids, X]), "plain": X[:, :20].copy()}, y, ph, genus


class _Recorder:
    """A selector that records which rows (by the id in column 0), strata and labels each call was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, Xtr, ytr, strata):
        self.calls.append((Xtr[:, 0].astype(int), np.asarray(ytr).copy(), np.asarray(strata).copy()))
        return ec.cmh_rank(Xtr[:, 1:], ytr, strata, min_prev=0.0) + 1  # drop the id column from the candidates


def _run(models, features, y, ph, genus, n_perm=6):
    return ec.lopo_evaluate(features, y, ph, genus, models, [], n_perm=n_perm, n_boot=10, seed=5, n_jobs=1)


def test_selector_never_sees_the_held_out_phylum_and_null_refits_it_on_permuted_labels():
    feats, y, ph, genus = _lopo_data()
    rec = _Recorder()
    res = _run({"pfam": {"features": "pfam", "grid": [0.01, 1.0], "ks": [5, None], "select": rec, "null": True}},
               feats, y, ph, genus)
    phyla = set(ph.tolist())
    assert rec.calls
    for ids, ytr, strata in rec.calls:
        assert set(strata.tolist()) < phyla  # every call is missing at least one phylum: the held-out one
        assert set(ph[ids].tolist()) == set(strata.tolist())  # and the rows are exactly those phyla's rows
        assert np.array_equal(strata, ph[ids])
    permuted = [(ids, ytr) for ids, ytr, _ in rec.calls if not np.array_equal(ytr, y[ids])]
    assert permuted, "the null never refit the selector on permuted labels"
    for ids, ytr in permuted:  # permuted WITHIN phylum: per-phylum positive counts are preserved
        for p in set(ph[ids].tolist()):
            m = ph[ids] == p
            assert ytr[m].sum() == y[ids][m].sum()
    for h in res["phyla"]:
        m = res["per_phylum"][h]["models"]["pfam"]
        assert m["K"] in (5, "all") and m["C"] in (0.01, 1.0) and 0 < m["n_selected"] <= m["n_stage1"]
    assert "inner_auc_selected" in res


def test_adding_a_selector_model_leaves_existing_models_bit_identical():
    feats, y, ph, genus = _lopo_data()
    plain = {"plain": {"features": "plain", "grid": [0.01, 0.1, 1.0], "null": True},
             "plain_c1": {"features": "plain", "grid": [1.0], "null": False}}
    base = _run(plain, feats, y, ph, genus)
    both = _run({**plain, "pfam": {"features": "pfam", "grid": [0.01, 1.0], "ks": [5, None], "select": _Recorder(), "null": True}},
                feats, y, ph, genus)
    for h in base["phyla"]:
        for n in plain:
            assert base["per_phylum"][h]["models"][n] == both["per_phylum"][h]["models"][n]
    for n in plain:
        assert base["macro"]["models"][n] == both["macro"]["models"][n]
    assert base["inner_auc"] == both["inner_auc"]


def test_a_selector_model_finds_planted_within_phylum_signal():
    feats, y, ph, genus = _lopo_data()
    res = _run({"pfam": {"features": "pfam", "grid": [0.01, 1.0], "ks": [5, 20, None], "select": _Recorder(), "null": True}},
               feats, y, ph, genus, n_perm=10)
    assert res["macro"]["models"]["pfam"]["auc"] > 0.8
    assert res["macro"]["models"]["pfam"]["null"]["mean"] < 0.65


def test_window_sequences_are_exactly_the_windows_sequence_features_counts(tmp_path):
    import gzip
    import random

    rng = random.Random(11)
    seq = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    contigs = [("c1", seq(30000)), ("c2", seq(20000)), ("c1", seq(9000))]  # a repeated id: the FIRST occurrence wins
    p = tmp_path / "g.fna.gz"
    with gzip.open(p, "wt") as f:
        for cid, s in contigs:
            f.write(f">{cid} some description\n")
            for i in range(0, len(s), 70):
                f.write(s[i:i + 70] + "\n")
    windows = [["c1", 0], ["c2", 5000], ["c1", 21000]]
    wins = ec.window_sequences(p, windows)
    assert [w[0] for w in wins] == ["c1:0", "c2:5000", "c1:21000"]
    assert all(len(s) == ec.WINDOW for _, s in wins)
    assert wins[2][1] == contigs[0][1][21000:21000 + ec.WINDOW]  # from the first c1, not the 9 kb duplicate
    want = ec.sequence_features(p, windows)[1]
    got = np.zeros(260)
    for _, s in wins:
        t, m = ec.count_tetra(s)
        got[:256] += t
        got[256:] += m
    assert np.array_equal(got, want)
