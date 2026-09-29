"""Genome matching on a tiny synthetic GTDB table."""
import json

import pandas as pd
import pytest

from src.config import load_config
from src.join import GtdbIndex, bacdive_taxids, match_stage_table, match_strains, prepare_gtdb
from src.normalise import Normaliser

CFG = load_config()
COLS = ["accession", "ncbi_genbank_assembly_accession", "ncbi_strain_identifiers", "gtdb_taxonomy", "gtdb_representative",
        "gtdb_type_designation_ncbi_taxa", "ncbi_type_material_designation", "ncbi_assembly_level", "ncbi_organism_name",
        "checkm2_completeness", "checkm2_contamination", "genome_size", "ncbi_taxid", "ncbi_species_taxid"]


def tax(genus, species):
    return f"d__Bacteria;p__Pseudomonadota;c__C;o__O;f__F;g__{genus};s__{species}"


def gtdb_raw():
    rows = [
        # accession, genbank, strain ids, taxonomy, rep, type, level, organism, taxid, species taxid
        ("RS_GCF_000000001.1", "GCA_000000001.1", "DSM 20231;ATCC 11775", tax("Escherichia", "Escherichia coli"), "t", "type strain of species", "Complete Genome", "Escherichia coli", 562, 562),
        ("GB_GCA_000000005.1", "GCA_000000005.1", "DSM 20231", tax("Escherichia", "Escherichia coli"), "f", "not type material", "Contig", "Escherichia coli", 562, 562),
        ("GB_GCA_000000002.1", "GCA_000000002.1", "DSM-30083", tax("Bacillus_A", "Bacillus_A subtilis"), "t", "type strain of species", "Scaffold", "Bacillus subtilis", 1423, 1423),
        ("GB_GCA_000000003.1", "GCA_000000003.1", "BS107", tax("Phaeobacter", "Phaeobacter gallaeciensis"), "t", "not type material", "Chromosome", "Phaeobacter gallaeciensis", 1423144, 60890),
        ("GB_GCA_000000004.1", "GCA_000000004.1", "BS 107", tax("Vibrio", "Vibrio cholerae"), "t", "not type material", "Chromosome", "Vibrio cholerae", 666, 666),
        ("GB_GCA_000000006.1", "GCA_000000006.1", "none", tax("Listeria", "Listeria innocua"), "t", "not type material", "Complete Genome", "Listeria innocua", 1642, 1642),
        ("GB_GCA_000000007.1", "GCA_000000007.1", "JCM 1/JCM 2", tax("Aeromonas", "Aeromonas hydrophila"), "t", "not type material", "Complete Genome", "Aeromonas hydrophila", 644, 644),
        # genus renamed between BacDive (Salinibacterium) and GTDB/NCBI (Homoserinimonas); same NCBI species taxid
        ("GB_GCA_000000008.1", "GCA_000000008.1", "SYSU T00001", tax("Homoserinimonas", "Homoserinimonas sedimenticola"), "t", "not type material", "Complete Genome", "Homoserinimonas sedimenticola", 999001, 999000),
    ]
    out = []
    for acc, gb, ids, tx, rep, typ, lvl, org, tid, stid in rows:
        out.append({"accession": acc, "ncbi_genbank_assembly_accession": gb, "ncbi_strain_identifiers": ids, "gtdb_taxonomy": tx,
                    "gtdb_representative": rep, "gtdb_type_designation_ncbi_taxa": typ, "ncbi_type_material_designation": "na",
                    "ncbi_assembly_level": lvl, "ncbi_organism_name": org, "checkm2_completeness": 99 if rep == "t" else 90,
                    "checkm2_contamination": 1 if rep == "t" else 3, "genome_size": 4_000_000, "ncbi_taxid": tid, "ncbi_species_taxid": stid})
    return pd.DataFrame(out)[COLS]


@pytest.fixture(scope="module")
def idx():
    return GtdbIndex(prepare_gtdb(gtdb_raw(), CFG["matching"]["assembly_level_rank"]), Normaliser.from_config(CFG))


def taxids(*ids):
    return json.dumps([{"NCBI tax id": i, "Matching level": "species"} for i in ids])


def strains(*specs):
    rows = []
    for i, s in enumerate(specs, 1):
        rows.append({"bacdive_id": i, "genus": None, "species": None, "culture_collection_no_raw": None,
                     "strain_designation_raw": None, "dsm_number": None, "bacdive_gca": None, "ncbi_tax_ids_raw": None, **s})
    return pd.DataFrame(rows)


def match(idx, *specs):
    m, u = match_strains(strains(*specs), idx, CFG)
    return m.set_index("bacdive_id"), u


def test_prepare_gtdb_derives_accessions_and_ranks():
    g = prepare_gtdb(gtdb_raw(), CFG["matching"]["assembly_level_rank"])
    assert g.loc[0, "assembly_refseq"] == "GCF_000000001.1" and g.loc[0, "assembly_genbank"] == "GCA_000000001.1"
    assert pd.isna(g.loc[2, "assembly_refseq"])
    assert g.loc[2, "genus_gtdb"] == "bacillus" and g.loc[2, "gtdb_phylum"] == "Pseudomonadota"
    assert g.loc[0, "pref_order"] < g.loc[1, "pref_order"]  # representative RefSeq complete beats contig GenBank


def test_missing_gtdb_column_fails_loudly():
    with pytest.raises(KeyError):
        prepare_gtdb(gtdb_raw().drop(columns=["ncbi_strain_identifiers"]), {})


def test_bacdive_taxids_parses_the_record_format():
    assert bacdive_taxids(taxids(60890, 1423144)) == {60890, 1423144}
    assert bacdive_taxids(None) == set() and bacdive_taxids(float("nan")) == set()


def test_direct_accession_ignores_version_and_takes_precedence(idx):
    m, _ = match(idx, {"bacdive_gca": "GCA_000000002", "culture_collection_no_raw": "DSM 20231"})
    r = m.loc[1]
    assert r.match_method == "accession" and r.assembly_genbank == "GCA_000000002.1" and r.bacdive_assembly_in_gtdb


def test_designation_match_across_writings(idx):
    m, _ = match(idx, {"culture_collection_no_raw": "DSMZ 20231"}, {"culture_collection_no_raw": "DSM20231"},
                 {"strain_designation_raw": "dsm-20231T"})
    assert set(m.match_method) == {"designation"} and set(m.assembly_genbank) == {"GCA_000000001.1"}
    assert set(m.match_token_kind) == {"collection"}


def test_best_genome_is_chosen_deterministically_and_ambiguity_is_recorded(idx):
    m, _ = match(idx, {"culture_collection_no_raw": "DSM 20231"})
    r = m.loc[1]
    assert r.gtdb_accession == "RS_GCF_000000001.1" and r.ncbi_assembly_accession == "GCF_000000001.1"
    assert r.n_gtdb_candidates == 2 and not r.match_ambiguous_species


def test_any_designation_matches_not_just_the_first(idx):
    m, _ = match(idx, {"culture_collection_no_raw": "ATCC 99999, JCM 5, DSM 30083"})
    assert m.loc[1].match_method == "designation" and m.loc[1].match_key == "DSM30083"


def test_slash_separated_gtdb_identifiers_are_split_too(idx):
    m, _ = match(idx, {"culture_collection_no_raw": "JCM 2"})
    assert m.loc[1].genome_matched and m.loc[1].assembly_genbank == "GCA_000000007.1"


def test_bare_designation_needs_taxon_agreement(idx):
    m, u = match(idx, {"strain_designation_raw": "BS 107", "genus": "Phaeobacter"},
                 {"strain_designation_raw": "BS 107", "genus": "Xanthomonas"},
                 {"strain_designation_raw": "BS 107", "genus": "Vibrio"})
    assert m.loc[1].assembly_genbank == "GCA_000000003.1" and m.loc[1].match_token_kind == "bare" and m.loc[1].taxon_agree_via == "genus"
    assert not m.loc[2].genome_matched and m.loc[2].designation_bare_hit_rejected
    assert m.loc[3].assembly_genbank == "GCA_000000004.1"
    assert u.set_index("bacdive_id").loc[2, "reason"] == "no_bacdive_assembly;bare_designation_taxon_mismatch"


def test_bare_designation_accepted_on_ncbi_taxid_when_genus_was_renamed(idx):
    m, _ = match(idx, {"strain_designation_raw": "SYSU T00001", "genus": "Salinibacterium", "ncbi_tax_ids_raw": taxids(999000)},
                 {"strain_designation_raw": "SYSU T00001", "genus": "Salinibacterium", "ncbi_tax_ids_raw": taxids(12345)})
    assert m.loc[1].genome_matched and m.loc[1].taxon_agree_via == "ncbi_taxid" and not m.loc[1].genus_agree
    assert not m.loc[2].genome_matched


def test_genus_agreement_uses_gtdb_suffix_stripped_and_ncbi_names(idx):
    m, _ = match(idx, {"culture_collection_no_raw": "DSM 30083", "genus": "Bacillus"})
    assert m.loc[1].genus_agree  # GTDB genus is "Bacillus_A"


def test_placeholders_and_numbers_never_match(idx):
    m, u = match(idx, {"strain_designation_raw": "none"}, {"strain_designation_raw": "870"})
    assert not m.genome_matched.any()
    assert set(u.reason) == {"no_bacdive_assembly;no_usable_designation"}


def test_unmatched_log_reasons(idx):
    m, u = match(idx, {"culture_collection_no_raw": "DSM 999999"},
                 {"bacdive_gca": "GCA_999999999", "culture_collection_no_raw": "DSM 999999"},
                 {"bacdive_gca": "GCA_999999999"})
    reasons = u.set_index("bacdive_id")["reason"].to_dict()
    assert reasons[1] == "no_bacdive_assembly;designations_not_in_gtdb"
    assert reasons[2] == "bacdive_assembly_not_in_gtdb;designations_not_in_gtdb"
    assert reasons[3] == "bacdive_assembly_not_in_gtdb;no_usable_designation"
    assert m.loc[2].bacdive_lists_assembly and not m.loc[2].bacdive_assembly_in_gtdb and not m.loc[2].genome_matched
    assert set(u.columns) >= {"bacdive_id", "keys_tried", "reason", "culture_collection_no_raw"}


def test_unmatched_log_has_columns_even_when_empty(idx):
    _, u = match(idx, {"culture_collection_no_raw": "DSM 20231"})
    assert len(u) == 0 and "reason" in u.columns


def test_designation_route_agreement_with_accession_route(idx):
    m, _ = match(idx, {"bacdive_gca": "GCA_000000001", "culture_collection_no_raw": "ATCC 11775"},
                 {"bacdive_gca": "GCA_000000002", "culture_collection_no_raw": "ATCC 11775"})
    assert m.loc[1].designation_agrees_with_accession == True  # noqa: E712
    assert m.loc[2].designation_agrees_with_accession == False  # noqa: E712


def test_stage_table_reconciles_with_the_final_match_count(idx):
    df = strains({"culture_collection_no_raw": "DSM 20231;ATCC 11775"},     # raw tokens hit
                 {"culture_collection_no_raw": "DSMZ 20231"},               # only after normalisation
                 {"strain_designation_raw": "BS-107", "genus": "Phaeobacter"},  # bare, only after normalisation
                 {"bacdive_gca": "GCA_000000006"},                          # accession only
                 {"strain_designation_raw": "BS 107", "genus": "Xanthomonas"},  # raw hit on the wrong genus: rejected
                 {"culture_collection_no_raw": "DSM 1"})                    # nothing
    m, _ = match_strains(df, idx, CFG)
    st = match_stage_table(m, {"all": pd.Series(True, index=m.index)}).set_index("step")["all n"]
    a, b, c, d, e, info = st.tolist()
    assert (a, b, c) == (1, 1, 3)            # normalisation adds DSMZ and BS-107; the wrong-genus raw hit never counts
    assert d == 1 and e == 4 == int(m.genome_matched.sum())
    assert info == 2                         # without the taxon check the Xanthomonas collision would count


def test_raw_baseline_applies_the_same_taxon_rule(idx):
    m, _ = match(idx, {"strain_designation_raw": "BS 107", "genus": "Xanthomonas"})
    r = m.loc[1]
    assert r.baseline_raw_token_hit_unchecked and not r.baseline_raw_token_hit and not r.genome_matched
