from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage2 import (kmer_features, kmer_predict, load_panel, nearest_rank,
                        split_indices, taxonomy_predict)


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
