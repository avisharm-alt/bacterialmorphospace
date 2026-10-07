import pandas as pd
import pytest

from src import pretraining_overlap as po

HEADER = ("Representative genome\tGTDB species\tGTDB taxonomy\tANI circumscription radius\tMean intra-species ANI\t"
          "Min intra-species ANI\tMean intra-species AF\tMin intra-species AF\tNo. clustered genomes\tClustered genomes\n")


def _tax(genus):
    return f"d__Bacteria;p__P;c__C;o__O;f__F;g__{genus}"


def _clusters(tmp_path, name, rows):
    """rows = [(rep, species, genus, [members incl. rep])] written as a GTDB sp_clusters TSV."""
    p = tmp_path / name
    with open(p, "w") as f:
        f.write(HEADER)
        for rep, sp, genus, members in rows:
            f.write(f"{rep}\ts__{sp}\t{_tax(genus)}\t95.0\t97\t96\t0.8\t0.7\t{len(members)}\t{','.join(members)}\n")
    return p


def test_assembly_key_ignores_prefixes_but_keeps_version():
    assert po.assembly_key("RS_GCF_001580535.1") == "001580535.1"
    assert po.assembly_key("GB_GCA_001580535.1") == "001580535.1"
    assert po.assembly_key("GCF_001580535.1") == "001580535.1"
    assert po.assembly_key("GCF_001580535.2") != po.assembly_key("GCF_001580535.1")


def test_read_sp_clusters_parses_members_and_checks_header(tmp_path):
    p = _clusters(tmp_path, "c.tsv", [("RS_GCF_000000001.1", "A a", "A", ["RS_GCF_000000001.1", "GB_GCA_000000002.1"])])
    (rep, sp, tax, members), = po.read_sp_clusters(p)
    assert rep == "000000001.1" and sp == "s__A a" and tax.endswith("g__A") and members == ["000000001.1", "000000002.1"]
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError):
        po.read_sp_clusters(bad)


def test_read_og2_gtdb_keeps_only_gtdb_rows(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("domain,source_dataset,accession_or_taxid\n"
                 "Prokaryote,GTDB_v220,GB_GCA_000000009.1\n"
                 "Prokaryote,GTDB_v220,RS_GCF_000000010.1\n"
                 "Eukaryote,NCBI_Eukaryotic_Genomes,GCA_000000011.1\n")
    assert po.read_og2_gtdb(p) == {"000000009.1", "000000010.1"}


def test_training_genomes_is_r214_reps_plus_og2():
    r214 = [("1", "s", "t", ["1", "2"]), ("3", "s", "t", ["3"])]
    assert po.training_genomes(r214, {"9", "1"}) == {"1", "3", "9"}  # members that are not representatives are not trained on


def test_classify_tiers():
    latest = [("100", "s__X x", _tax("X"), ["100", "101", "102"]),   # 101 is trained on (a 214.1 rep), the cluster is seen
              ("200", "s__Y y", _tax("Y"), ["200", "201"]),          # nothing trained on, but genus Y is trained via the next cluster
              ("300", "s__Y z", _tax("Y"), ["300"]),
              ("400", "s__Z z", _tax("Z"), ["400"])]                  # genus Z has no trained genome
    train = {"101", "300"}
    d = po.classify(["101", "100", "200", "300", "400", "999"], train, latest).set_index("key")
    assert d.loc["101", "tier"] == "genome_seen"
    assert d.loc["100", "tier"] == "species_seen" and d.loc["100", "n_trained_in_species"] == 1
    assert d.loc["200", "tier"] == "unseen" and bool(d.loc["200", "genus_seen"]) is True   # unseen species, seen genus
    assert d.loc["300", "tier"] == "genome_seen"
    assert d.loc["400", "tier"] == "unseen" and bool(d.loc["400", "genus_seen"]) is False
    assert d.loc["999", "tier"] == "not_in_release"  # never silently called unseen


def test_classify_prefix_flip_does_not_change_tier():
    # trained on as RefSeq (GCF), listed in the panel's release as GenBank (GCA): the same assembly
    latest = [(po.assembly_key("GB_GCA_000000100.1"), "s__X x", _tax("X"), [po.assembly_key("GB_GCA_000000100.1")])]
    d = po.classify([po.assembly_key("GB_GCA_000000100.1")], {po.assembly_key("RS_GCF_000000100.1")}, latest)
    assert d.loc[0, "tier"] == "genome_seen"


def test_tier_counts_and_shares():
    df = pd.DataFrame({"ph": ["a", "a", "a", "b"], "tier": ["genome_seen", "unseen", "genome_seen", "species_seen"]})
    c = po.tier_counts(df, "ph").set_index("ph")
    assert c.loc["a", "n"] == 3 and c.loc["a", "genome_seen"] == 2 and c.loc["a", "share_genome_seen"] == pytest.approx(0.6667, abs=1e-4)
    assert c.loc["all", "n"] == 4 and c.loc["all", "species_seen"] == 1


def test_unseen_feasibility_counts_both_classes_and_genera():
    df = pd.DataFrame({"gtdb_phylum": ["a"] * 5, "gtdb_genus": ["g1", "g1", "g2", "g3", "g3"],
                       "tier": ["unseen", "unseen", "unseen", "genome_seen", "species_seen"],
                       "motility": ["yes", "no", "no", "yes", "yes"]})
    f = po.unseen_feasibility(df, {"motility": ("motility", "yes")}, ("unseen",)).iloc[0]
    assert (f.n, f.n_unseen, f.n_unseen_genera) == (5, 3, 2)
    assert (f.motility_pos_unseen, f.motility_neg_unseen) == (1, 2)
    assert f.motility_prev_rest == 1.0
    g = po.unseen_feasibility(df, {"motility": ("motility", "yes")}, ("unseen", "species_seen")).iloc[0]
    assert g.n_unseen == 4
