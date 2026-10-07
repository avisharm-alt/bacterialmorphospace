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


# --- the pre-specified confirmatory table -------------------------------------------------------------------
def test_bh_qvalues_known_values_order_and_a_naive_reference():
    q = ec.bh_qvalues([0.01, 0.04, 0.03])
    assert np.allclose(q, [0.03, 0.04, 0.04])  # worked by hand: 0.03, 0.045, 0.04, then the running minimum from the largest
    assert np.allclose(ec.bh_qvalues([0.5]), [0.5])
    assert np.allclose(ec.bh_qvalues([0.9, 0.8, 0.7]), [0.9, 0.9, 0.9])  # capped at 1 and monotone
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(rng.integers(2, 9))
        rank = lambda j: int((p <= p[j]).sum())  # noqa: E731  rank of p_j (ties take the highest rank)
        ref = np.array([min(1.0, min(p[j] * len(p) / rank(j) for j in range(len(p)) if p[j] >= p[i])) for i in range(len(p))])
        assert np.allclose(ec.bh_qvalues(p), ref)  # q_i = min over {j : p_j >= p_i} of p_j m / rank(p_j), capped at 1
    assert np.all(ec.bh_qvalues(p) >= p - 1e-12)


def _doc(p, auc=0.6, n_perm=999, min_prev=0.005):
    ph = ["P1", "P2", "P3", "P4"]
    delta = lambda v: {"delta": v, "ci": [v - 0.05, v + 0.05], "share_boot_positive": 0.5}  # noqa: E731
    model = lambda a, pp: {"auc": a, "ci": [a - 0.03, a + 0.03], "null": {"mean": 0.5, "sd": 0.02, "q95": 0.53, "p": pp}}  # noqa: E731
    return {"phyla": ph, "n_genomes": 2435, "n_perm": n_perm, "pfam": {"min_prev": min_prev},
            "macro": {"models": {"pfam": model(auc, p), "pfam_windows": model(0.52, 0.2), "evo2": model(0.55, 0.1),
                                 "kmer_genome": model(0.52, 0.3), "kmer_windows": model(0.54, 0.2), "gc": model(0.58, 0.05)},
                      "deltas": {k: delta(0.01) for k in ("evo2 - pfam", "evo2 - pfam_windows", "pfam - kmer_genome",
                                                          "pfam_windows - kmer_windows")}},
            "per_phylum": {h: {"models": {"pfam": {"K": 100}, "pfam_windows": {"K": "all"}}} for h in ph}}


def _summary_files(tmp_path, p_by_target, **kw):
    import json

    (tmp_path / "tables").mkdir(exist_ok=True)
    for t, p in p_by_target.items():
        json.dump(_doc(p, **kw), open(tmp_path / "tables" / f"evo2_lopo_{t}_pfam.json", "w"))


def test_pfam_summary_applies_bh_across_the_three_secondary_targets_only(tmp_path, capsys):
    import csv

    m = pytest.importorskip("modal") and __import__("src.evo2_modal", fromlist=["x"])
    _summary_files(tmp_path, {"motility": 0.001, "shape_rod": 0.004, "oxygen_aerobe": 0.02, "oxygen_facultative": 0.30})
    m.pfam_summary.info.raw_f(reports=str(tmp_path))
    rows = {r["target"]: r for r in csv.DictReader(open(tmp_path / "tables" / "pfam_lopo_summary.tsv"), delimiter="\t")}
    assert rows["motility"]["role"] == "primary" and rows["motility"]["q_bh"] == "" and rows["motility"]["positive"] == "yes"
    # BH over {0.004, 0.02, 0.30}: q = 0.012, 0.03, 0.30. Had motility (0.001) been in the family (m = 4) shape_rod's q would be 0.008.
    assert [round(float(rows[t]["q_bh"]), 3) for t in m.PFAM_SECONDARY] == [0.012, 0.03, 0.3]
    assert [rows[t]["positive"] for t in m.PFAM_SECONDARY] == ["yes", "yes", "no"]
    out = capsys.readouterr().out
    assert "primary, tested alone" in out or "is primary, tested alone" in out
    assert "Benjamini-Hochberg corrected across the three" in out and "DESCRIPTIVE" in out and "CHOSEN K PER HELD-OUT PHYLUM" in out


def test_pfam_summary_primary_is_judged_alone_and_a_secondary_with_a_small_raw_p_can_fail_bh(tmp_path):
    import csv

    m = pytest.importorskip("modal") and __import__("src.evo2_modal", fromlist=["x"])
    _summary_files(tmp_path, {"motility": 0.06, "shape_rod": 0.03, "oxygen_aerobe": 0.04, "oxygen_facultative": 0.045})
    m.pfam_summary.info.raw_f(reports=str(tmp_path))
    rows = {r["target"]: r for r in csv.DictReader(open(tmp_path / "tables" / "pfam_lopo_summary.tsv"), delimiter="\t")}
    assert rows["motility"]["positive"] == "no"  # 0.06 > 0.05, no correction applied to the primary
    assert [rows[t]["positive"] for t in m.PFAM_SECONDARY] == ["yes", "yes", "yes"]  # q = 0.045 for all three (<= 0.05)


@pytest.mark.parametrize("kw, needle", [({"n_perm": 50}, "n_perm is 50"), ({"min_prev": 0.01}, "min_prev is 0.01")])
def test_pfam_summary_refuses_runs_that_differ_from_the_pre_specification(tmp_path, kw, needle):
    m = pytest.importorskip("modal") and __import__("src.evo2_modal", fromlist=["x"])
    _summary_files(tmp_path, {"motility": 0.01, "shape_rod": 0.01, "oxygen_aerobe": 0.01, "oxygen_facultative": 0.01}, **kw)
    with pytest.raises(SystemExit, match=needle):
        m.pfam_summary.info.raw_f(reports=str(tmp_path))
    assert not (tmp_path / "tables" / "pfam_lopo_summary.tsv").exists()


def test_pfam_summary_refuses_when_a_target_is_missing(tmp_path):
    m = pytest.importorskip("modal") and __import__("src.evo2_modal", fromlist=["x"])
    _summary_files(tmp_path, {"motility": 0.01, "shape_rod": 0.01, "oxygen_aerobe": 0.01})
    with pytest.raises(SystemExit, match="oxygen_facultative.*is missing"):
        m.pfam_summary.info.raw_f(reports=str(tmp_path))
