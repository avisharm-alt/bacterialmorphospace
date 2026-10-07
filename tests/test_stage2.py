from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage2 import (ROOT, audit, benchmark, kmer_features, kmer_predict,
                        load_panel, nearest_rank, read_accession_subset,
                        select_accessions, split_indices, taxonomy_predict)


def toy_panel(n=30):
    return pd.DataFrame({
        "gtdb_species": [f"Species {i}" for i in range(n)],
        "ncbi_assembly_accession": [f"GCF_{i}.1" for i in range(n)],
        "assembly_genbank": [f"GCA_{i}.1" for i in range(n)],
        "gtdb_genus": [f"G{i // 2}" for i in range(n)],
        "gtdb_family": [f"F{i // 4}" for i in range(n)],
        "gtdb_order": [f"O{i // 8}" for i in range(n)],
        "gtdb_class": [f"C{i // 16}" for i in range(n)],
        "gtdb_phylum": ["P" for _ in range(n)],
        "gram": ["positive" if i % 2 else "negative" for i in range(n)],
    })


def complete_toy_panel(n=30):
    df = toy_panel(n)
    df["gtdb_accession"] = [f"RS_GCF_{i}.1" for i in range(n)]
    df["shape"] = ["coccus", "other", "rod"] * (n // 3)
    df["motility"] = ["no", "yes"] * (n // 2)
    df["spore"] = ["no", "yes"] * (n // 2)
    df["oxygen"] = ["aerobe", "anaerobe", "facultative"] * (n // 3)
    df["temperature"] = ["meso", "psychro", "thermo"] * (n // 3)
    df["match_method"] = "accession"
    df["temperature_bin_source"] = "optimum"
    df["oxygen_microaerophile_folded"] = False
    df["in_gtdb_v220_exact"] = True
    df["in_gtdb_v220_stem"] = True
    return df


@pytest.mark.parametrize("rank", ["genus", "family", "order"])
def test_group_splits_do_not_share_taxa_species_or_assemblies(rank):
    df = toy_panel()
    for seed in range(6):
        train, test = split_indices(df, rank, seed)
        assert set(train).isdisjoint(test)
        assert len(train) + len(test) == len(df)
        for field in (f"gtdb_{rank}", "gtdb_species", "ncbi_assembly_accession", "assembly_genbank"):
            assert set(df.iloc[train][field]).isdisjoint(df.iloc[test][field])


def test_duplicate_accession_is_rejected_before_benchmark(tmp_path):
    final = tmp_path / "data/final"
    final.mkdir(parents=True)
    df = toy_panel(4)
    df["gtdb_accession"] = ["RS_GCF_1.1"] * 4
    df.to_csv(final / "core_panel_species.tsv", sep="\t", index=False)
    pd.DataFrame({"gtdb_accession": ["RS_GCF_1.1"], "gtdb_family": ["F"],
                  "in_gtdb_v220_exact": [True], "in_gtdb_v220_stem": [True]}).to_csv(
                      final / "stage2_taxonomy.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate gtdb_accession"):
        load_panel(tmp_path, "core")


def test_two_versions_of_same_assembly_are_rejected(tmp_path):
    final = tmp_path / "data/final"
    final.mkdir(parents=True)
    df = toy_panel(4)
    df["gtdb_accession"] = [f"RS_GCF_{i}.1" for i in range(4)]
    df.loc[1, "assembly_genbank"] = "GCA_0.2"
    df.to_csv(final / "core_panel_species.tsv", sep="\t", index=False)
    pd.DataFrame({"gtdb_accession": df["gtdb_accession"], "gtdb_family": df["gtdb_family"],
                  "in_gtdb_v220_exact": [True] * 4, "in_gtdb_v220_stem": [True] * 4}).to_csv(
                      final / "stage2_taxonomy.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="version-stripped GenBank"):
        load_panel(tmp_path, "core")


def test_taxonomy_fit_uses_train_labels_only_and_falls_back_on_unseen_family():
    train = toy_panel(8)
    train["gram"] = ["negative"] * 4 + ["positive"] * 4
    test = toy_panel(1)
    test["gtdb_genus"] = "unseen"
    test["gtdb_family"] = "unseen"
    test["gtdb_order"] = "O0"
    test["gram"] = "positive"
    labels = ["negative", "positive"]
    first = taxonomy_predict(train, test, "gram", labels)
    test["gram"] = "negative"
    assert np.array_equal(first, taxonomy_predict(train, test, "gram", labels))
    assert nearest_rank(train, test).tolist() == ["same_order"]


def test_kmer_counts_do_not_bridge_contigs_and_centroids_are_train_only(tmp_path):
    fasta = tmp_path / "x.fna"
    fasta.write_text(">a\nAAAA\n>b\nTTTT\n")
    vector = kmer_features(fasta)
    assert vector.sum() == pytest.approx(1)
    assert np.count_nonzero(vector) == 2
    train_x = np.stack([vector, np.eye(256)[85]])
    test_x = np.stack([vector, vector])
    y_train = pd.Series(["A", "B"])
    first = kmer_predict(train_x, y_train, test_x, ["A", "B"])
    assert first.tolist() == ["A", "A"]
    # Test labels are not an argument to the predictor.


def test_cross_panel_spore_missing_is_counted_separately_from_disagreement():
    core = complete_toy_panel(6)
    no_spore = core.copy()
    no_spore.loc[0, "spore"] = None  # same assembly, different BacDive observation
    no_spore.loc[1, "spore"] = None
    no_spore.loc[1, "gtdb_accession"] = "RS_GCF_99.1"  # different selected assembly
    no_spore.loc[2, "spore"] = "yes"  # observed disagreement, not missing
    cross = audit(core, no_spore)["cross_panel"]
    assert cross["core_spore_present_no_spore_missing"] == 2
    assert cross["spore_missing_same_assembly"] == 1
    assert cross["spore_missing_different_assembly"] == 1
    assert cross["spore_nonmissing_disagreement"] == 1


def test_subset_run_uses_same_genomes_and_split_sizes_for_all_models(tmp_path, monkeypatch):
    final = tmp_path / "data/final"
    final.mkdir(parents=True)
    data = complete_toy_panel()
    cols = [col for col in data if col not in ("gtdb_family", "in_gtdb_v220_exact", "in_gtdb_v220_stem")]
    for panel in ("core", "no_spore"):
        rows = data[cols].copy()
        if panel == "no_spore":
            rows.loc[0, "spore"] = None  # spore task must use the core row
        rows.to_csv(final / f"{panel}_panel_species.tsv", sep="\t", index=False)
    data[["gtdb_accession", "gtdb_family", "in_gtdb_v220_exact", "in_gtdb_v220_stem"]].to_csv(
        final / "stage2_taxonomy.tsv", sep="\t", index=False)
    subset = tmp_path / "pilot.txt"
    accessions = data.ncbi_assembly_accession.iloc[:18].tolist()
    subset.write_text("\n".join(accessions) + "\n")
    fasta_dir = tmp_path / "fasta"
    fasta_dir.mkdir()
    for i, accession in enumerate(accessions):
        (fasta_dir / f"{accession}.fna").write_text(f">contig\n{'ACGT'[i % 4] * 20}ACGT\n")
    calls = []
    original = kmer_features
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr("src.stage2.kmer_features", counted)
    benchmark(tmp_path, repeats=1, fasta_dir=fasta_dir, accession_subset=subset)
    assert len(calls) == 18  # one read per assembly across all six trait tasks
    details = pd.read_csv(tmp_path / "reports/stage2_subset/tables/stage2_split_metrics.tsv", sep="\t")
    assert set(details.model) == {"prevalence", "taxonomy", "kmer4"}
    assert set(details.panel_n) == {18}
    for _, group in details.groupby(["trait", "split", "repeat"]):
        assert group[["train_n", "test_n", "train_classes", "test_classes",
                      "train_accessions_sha256", "test_accessions_sha256"]].drop_duplicates().shape[0] == 1
    import json
    manifest = json.loads((tmp_path / "reports/stage2_subset/stage2_run.json").read_text())
    assert manifest["accessions"] == sorted(accessions)
    assert manifest["n_accessions"] == 18
    assert manifest["models"] == ["prevalence", "taxonomy", "kmer4"]
    subset_audit = json.loads((tmp_path / "reports/stage2_subset/stage2_audit.json").read_text())
    assert subset_audit["cross_panel"]["core_spore_present_no_spore_missing"] == 1
    assert not (tmp_path / "reports/tables/stage2_benchmark_summary.tsv").exists()


def test_subset_rejects_duplicates_and_assemblies_missing_from_a_panel(tmp_path):
    path = tmp_path / "subset.txt"
    path.write_text("GCF_1.1\nGCF_1.1\n")
    with pytest.raises(ValueError, match="duplicate accession"):
        read_accession_subset(path)
    path.write_text("GCF_1.1\nGCF_2.1\n")
    core = complete_toy_panel(6)
    no_spore = core[core.ncbi_assembly_accession.ne("GCF_2.1")]
    with pytest.raises(ValueError, match="no_spore panel lacks 1"):
        select_accessions(core, no_spore, read_accession_subset(path))


def test_committed_pilot_accessions_have_same_assembly_and_strain_in_both_panels():
    subset = read_accession_subset(ROOT / "data/final/stage2_fasta_pilot_accessions.txt")
    core, no_spore = select_accessions(load_panel(ROOT, "core"), load_panel(ROOT, "no_spore"), subset)
    assert len(core) == len(no_spore) == len(subset) == 192
    paired = core.merge(no_spore, on="ncbi_assembly_accession", suffixes=("_core", "_no_spore"))
    assert (paired.bacdive_id_core == paired.bacdive_id_no_spore).all()
