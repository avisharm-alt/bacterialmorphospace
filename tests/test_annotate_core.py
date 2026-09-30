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
