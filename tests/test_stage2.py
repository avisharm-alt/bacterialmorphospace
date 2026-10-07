import numpy as np
import pandas as pd
import pytest

from src.stage2 import (accession_overlap, assign_folds, centroid_predict, kmer_frequencies,
                        standardize_fit_transform, taxonomy_predict, validate_panel)


def frame(n=25):
    return pd.DataFrame({"gtdb_species": [f"s{i}" for i in range(n)],
                         "ncbi_assembly_accession": [f"GCF_{i:09d}.1" for i in range(n)],
                         "assembly_genbank": [f"GCA_{i:09d}.1" for i in range(n)],
                         "gtdb_genus": [f"g{i // 2}" for i in range(n)],
                         "gtdb_family": [f"f{i // 5}" for i in range(n)],
                         "gtdb_order": [f"o{i // 5}" for i in range(n)],
                         "gtdb_class": ["c"] * n, "gtdb_phylum": ["p"] * n})


@pytest.mark.parametrize("rank", ["within", "genus", "family", "order"])
def test_group_splits_keep_groups_disjoint(rank):
    df = frame()
    fold = assign_folds(df, rank, 42)
    key = df.gtdb_species if rank == "within" else df[f"gtdb_{rank}"]
    for i in range(5):
        assert set(key[fold == i]).isdisjoint(set(key[fold != i]))
        assert (fold == i).any()


def test_duplicate_accessions_rejected_in_both_namespace_forms():
    df = frame()
    df.loc[1, "assembly_genbank"] = df.loc[0, "assembly_genbank"]
    with pytest.raises(ValueError, match="assembly_genbank"):
        validate_panel(df)
    df = frame()
    df.loc[1, "ncbi_assembly_accession"] = df.loc[0, "ncbi_assembly_accession"]
    with pytest.raises(ValueError, match="ncbi_assembly_accession"):
        validate_panel(df)


def test_taxonomy_uses_training_labels_only_and_backs_off():
    train = frame(10)
    train["gram"] = ["positive"] * 6 + ["negative"] * 4
    test = train.iloc[[0, 8]].copy()
    test["gram"] = ["negative", "positive"]  # intentionally contradictory holdout labels
    assert taxonomy_predict(train, test, "gram", min_count=2).tolist() == ["positive", "negative"]
    assert taxonomy_predict(train, test, "gram", min_count=6).tolist() == ["positive", "positive"]


def test_kmer_standardization_is_train_only():
    train = np.array([[0., 1.], [2., 3.]])
    test = np.array([[1000., 2000.]])
    a, b = standardize_fit_transform(train, test)
    np.testing.assert_allclose(a.mean(axis=0), [0, 0])
    np.testing.assert_allclose(b, [[999, 1998]])
    labels = pd.Series(["a", "b"])
    assert centroid_predict(train, labels, test).tolist() == ["b"]


def test_kmers_do_not_cross_contig_or_ambiguous_base(tmp_path):
    fasta = tmp_path / "one.fna"
    fasta.write_text(">a\nAAAA\n>b\nAAANAAAA\n")
    frequencies = kmer_frequencies(fasta)
    assert frequencies[0] == 1 and frequencies.sum() == 1


def test_accession_overlap_uses_genbank_crosswalk_and_versions(tmp_path):
    manifest = tmp_path / "accessions.txt"
    manifest.write_text("GB_GCA_000000000.1\nGCF_000000001.2\n")
    overlap = accession_overlap(frame(2), manifest)
    assert overlap["exact_assembly_accessions"] == 1
    assert overlap["unversioned_accessions"] == 2
