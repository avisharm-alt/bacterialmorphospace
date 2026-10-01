import random

import pytest

pytest.importorskip("pyhmmer")
pytest.importorskip("pyrodigal")

from src import annotate_core as ac  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def _hmm_file(tmp_path, seq, acc, ga):
    from pyhmmer import easel, plan7
    abc = easel.Alphabet.amino()
    bg = plan7.Background(abc)
    hmm, _, _ = plan7.Builder(abc).build(easel.TextSequence(name=b"fam", sequence=seq).digitize(abc), bg)
    hmm.accession = acc.encode()
    hmm.cutoffs.gathering = (ga, ga)
    p = tmp_path / "t.hmm"
    with open(p, "wb") as f:
        hmm.write(f)
    return str(p)


def test_search_pfam_hits_only_present_family(tmp_path):
    rng = random.Random(1)
    fam = "".join(rng.choice(AA) for _ in range(150))
    other = "".join(rng.choice(AA) for _ in range(150))
    hmm = _hmm_file(tmp_path, fam, "PF99999.3", 25.0)
    assert ac.search_pfam([("p1", fam), ("p2", other)], hmm, 1) == {"PF99999"}  # versionless accession
    assert ac.search_pfam([("p2", other)], hmm, 1) == set()


def test_gathering_cutoff_is_applied(tmp_path):
    rng = random.Random(2)
    fam = "".join(rng.choice(AA) for _ in range(150))
    hmm = _hmm_file(tmp_path, fam, "PF99998.1", 1e6)  # unreachable cutoff: no family may be reported
    assert ac.search_pfam([("p1", fam)], hmm, 1) == set()


def test_empty_proteins_and_tiny_contig():
    assert ac.search_pfam([], "/nonexistent.hmm", 1) == set()
    assert ac.call_proteins([("c", "ACGT" * 10)]) == []  # metagenomic mode on 40 bp: no genes, no crash


def test_grouped_search_equals_per_group_search_and_attributes_hits_to_the_right_group(tmp_path):
    rng = random.Random(7)
    fam = "".join(rng.choice(AA) for _ in range(150))
    other = "".join(rng.choice(AA) for _ in range(150))
    hmm = _hmm_file(tmp_path, fam, "PF99997.2", 25.0)
    groups = {"gA": [("p1", fam), ("p2", other)],  # same protein id "p1" in both groups, only gA's is the family
              "gB": [("p1", other)],
              "gC": [("p9", fam)],
              "gD": []}
    got = ac.search_pfam_grouped(groups, hmm, 1)
    assert got == {g: ac.search_pfam(ps, hmm, 1) for g, ps in groups.items()}
    assert got == {"gA": {"PF99997"}, "gB": set(), "gC": {"PF99997"}, "gD": set()}
    assert ac.search_pfam_grouped({}, "/nonexistent.hmm", 1) == {}
    assert ac.search_pfam_grouped({"g": []}, "/nonexistent.hmm", 1) == {"g": set()}


def test_annotate_window_groups_returns_builtins_and_handles_windows_without_genes(tmp_path):
    hmm = _hmm_file(tmp_path, "".join(random.Random(8).choice(AA) for _ in range(150)), "PF99996.1", 25.0)
    res = ac.annotate_window_groups({"g1": [("w1", "ACGT" * 10)], "g2": []}, hmm, 1)  # 40 bp and no windows: no genes, no crash
    assert res["groups"]["g1"] == {"families": [], "n_proteins": 0, "bp": 40}
    assert res["groups"]["g2"] == {"families": [], "n_proteins": 0, "bp": 0}
    assert set(res) == {"groups", "gene_call_s", "hmmsearch_s"}
