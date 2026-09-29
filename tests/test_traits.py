"""Trait extraction: mapping, conflicts, bin sources, raw preservation."""
import json
import math
from collections import Counter

import pytest

from src.config import load_config
from src.traits import extract_categorical, extract_halophily, extract_identity, extract_numeric, extract_record, parse_numeric

TCFG = load_config()["traits"]


def col():
    return {"unmapped": Counter(), "unparsed": Counter(), "predictions": []}


def rec_morph(**cell):
    return {"Morphology": {"cell morphology": cell}}


def rec_oxygen(*vals):
    return {"Physiology and metabolism": {"oxygen tolerance": [{"@ref": i, "oxygen tolerance": v} for i, v in enumerate(vals)]}}


def rec_temp(*ents):
    return {"Culture and growth conditions": {"culture temp": [
        {"@ref": i, "growth": g, "type": t, "temperature": v} for i, (t, g, v) in enumerate(ents)]}}


def rec_ph(*ents):
    return {"Culture and growth conditions": {"culture pH": [
        {"@ref": i, "ability": a, "type": t, "pH": v} for i, (t, a, v) in enumerate(ents)]}}


# --- categorical ---------------------------------------------------------------
def test_gram_simple_and_raw_kept():
    r = extract_categorical(rec_morph(**{"gram stain": "negative", "@ref": 5}), "gram", TCFG["gram"], col())
    assert r["gram"] == "negative" and r["gram_fine"] == "negative" and not r["gram_conflict"]
    assert json.loads(r["gram_raw"]) == [{"ref": 5, "value": "negative"}]


def test_missing_trait_is_null_not_conflict():
    r = extract_categorical({}, "gram", TCFG["gram"], col())
    assert r["gram"] is None and r["gram_raw"] is None and r["gram_n_obs"] == 0 and not r["gram_conflict"]


def test_disagreeing_observations_null_the_value_and_flag_conflict():
    rec = {"Morphology": {"cell morphology": [{"@ref": 1, "gram stain": "positive"}, {"@ref": 2, "gram stain": "negative"}]}}
    r = extract_categorical(rec, "gram", TCFG["gram"], col())
    assert r["gram"] is None and r["gram_conflict"] and r["gram_fine_conflict"]
    assert len(json.loads(r["gram_raw"])) == 2  # both observations survive in the raw column


def test_agreeing_duplicate_observations_are_not_a_conflict():
    rec = {"Morphology": {"cell morphology": [{"@ref": 1, "motility": "yes"}, {"@ref": 2, "motility": "yes"}]}}
    r = extract_categorical(rec, "motility", TCFG["motility"], col())
    assert r["motility"] == "yes" and not r["motility_conflict"] and r["motility_n_obs"] == 2


@pytest.mark.parametrize("raw,fine,coarse", [
    ("rod-shaped", "rod", "rod"), ("coccus-shaped", "coccus", "coccus"),
    ("spiral-shaped", "spiral", "other"), ("vibrio-shaped", "spiral", "other"),
    ("filament-shaped", "filamentous", "other"), ("oval-shaped", "other", "other"),
    ("ovoid-shaped", "other", "other"), ("star-shaped", "other", "other"),
])
def test_shape_fine_and_coarse(raw, fine, coarse):
    r = extract_categorical(rec_morph(**{"cell shape": raw}), "shape", TCFG["shape"], col())
    assert (r["shape_fine"], r["shape"]) == (fine, coarse)


def test_unmapped_shape_goes_to_other_and_is_logged():
    c = col()
    r = extract_categorical(rec_morph(**{"cell shape": "banana-shaped"}), "shape", TCFG["shape"], c)
    assert r["shape"] == "other" and c["unmapped"][("shape", "banana-shaped")] == 1


def test_unmapped_gram_is_null_and_logged():
    c = col()
    r = extract_categorical(rec_morph(**{"gram stain": "variable"}), "gram", TCFG["gram"], c)
    assert r["gram"] is None and c["unmapped"][("gram", "variable")] == 1


def test_shape_conflict_at_fine_level_can_vanish_at_coarse_level():
    rec = {"Morphology": {"cell morphology": [{"cell shape": "spiral-shaped"}, {"cell shape": "oval-shaped"}]}}
    r = extract_categorical(rec, "shape", TCFG["shape"], col())
    assert r["shape_fine"] is None and r["shape_fine_conflict"]
    assert r["shape"] == "other" and not r["shape_conflict"]


@pytest.mark.parametrize("raw,fine,coarse", [
    ("aerobe", "aerobe", "aerobe"), ("obligate aerobe", "aerobe", "aerobe"),
    ("facultative anaerobe", "facultative", "facultative"), ("microaerophile", "microaerophile", "facultative"),
    ("anaerobe", "anaerobe", "anaerobe"), ("obligate anaerobe", "anaerobe", "anaerobe"),
])
def test_oxygen_bins(raw, fine, coarse):
    r = extract_categorical(rec_oxygen(raw), "oxygen", TCFG["oxygen"], col())
    assert (r["oxygen_fine"], r["oxygen"]) == (fine, coarse)


def test_microaerophile_folding_is_flagged():
    fold = extract_categorical(rec_oxygen("microaerophile"), "oxygen", TCFG["oxygen"], col())
    assert fold["oxygen_microaerophile_folded"] is True
    plain = extract_categorical(rec_oxygen("facultative anaerobe"), "oxygen", TCFG["oxygen"], col())
    assert plain["oxygen_microaerophile_folded"] is False


def test_microaerophile_plus_facultative_conflicts_only_at_fine_level():
    r = extract_categorical(rec_oxygen("microaerophile", "facultative anaerobe"), "oxygen", TCFG["oxygen"], col())
    assert r["oxygen_fine"] is None and r["oxygen_fine_conflict"]
    assert r["oxygen"] == "facultative" and not r["oxygen_conflict"]


def test_aerobe_vs_anaerobe_is_a_real_conflict():
    r = extract_categorical(rec_oxygen("aerobe", "anaerobe"), "oxygen", TCFG["oxygen"], col())
    assert r["oxygen"] is None and r["oxygen_conflict"]


# --- numeric parsing -------------------------------------------------------------
@pytest.mark.parametrize("raw,exp", [
    ("37", (37, 37)), ("22-37", (22, 37)), ("5.0", (5, 5)), ("2.0-30.0", (2, 30)), ("7-41.5", (7, 41.5)),
    (">7", (7, 7)), ("<4.5", (4.5, 4.5)), ("37 °C", (37, 37)), ("pH 7.0", (7, 7)), ("30-22", (22, 30)),
    (37, (37, 37)), ("-2", (-2, -2)), ("-1.8-10", (-1.8, 10)), ("-12-10", (-12, 10)), ("(-2)-(-1)", (-2, -1)),
    ("-2.0-23", (-2, 23)), ("20 to 30", (20, 30)),
])
def test_parse_numeric(raw, exp):
    assert parse_numeric(raw) == exp


@pytest.mark.parametrize("raw", ["", None, "warm", "37 and 40", "-", "25--30"])
def test_parse_numeric_rejects_garbage(raw):
    assert parse_numeric(raw) is None


# --- temperature -------------------------------------------------------------------
def test_optimum_beats_growth_range():
    r = extract_numeric(rec_temp(("optimum", "positive", "60"), ("growth", "positive", "20-70")), "temperature", TCFG["temperature"], col())
    assert r["temperature_bin_source"] == "optimum" and r["temperature"] == "thermo" and r["temperature_fine"] == "thermo"


def test_optimum_given_as_range_uses_midpoint():
    r = extract_numeric(rec_temp(("optimum", "positive", "20-30")), "temperature", TCFG["temperature"], col())
    assert r["temperature_point"] == 25 and r["temperature"] == "meso"


@pytest.mark.parametrize("t,fine,coarse", [(14.9, "psychro", "psychro"), (15, "meso", "meso"), (44.9, "meso", "meso"),
                                          (45, "thermo", "thermo"), (79.9, "thermo", "thermo"), (80, "hyperthermo", "thermo")])
def test_temperature_bin_edges_and_fold(t, fine, coarse):
    r = extract_numeric(rec_temp(("optimum", "positive", str(t))), "temperature", TCFG["temperature"], col())
    assert (r["temperature_fine"], r["temperature"]) == (fine, coarse)


def test_growth_range_uses_midpoint_of_positive_growth_and_is_labelled():
    # the real record 24493: growth 22, 25, 5-30; no growth at 37/41/45
    r = extract_numeric(rec_temp(("growth", "positive", "25"), ("growth", "positive", "22"), ("growth", "positive", "5-30"),
                                 ("growth", "negative", "37"), ("growth", "negative", "41"), ("growth", "negative", "45")),
                        "temperature", TCFG["temperature"], col())
    assert r["temperature_bin_source"] == "growth_range"
    assert (r["temperature_growth_min"], r["temperature_growth_max"], r["temperature_point"]) == (5, 30, 17.5)
    assert r["temperature"] == "meso" and not r["temperature_conflict"]


def test_single_point_growth_is_included_but_flagged():
    r = extract_numeric(rec_temp(("growth", "positive", "37")), "temperature", TCFG["temperature"], col())
    assert r["temperature_bin_source"] == "growth_single_point" and r["temperature"] == "meso"


def test_two_different_single_points_form_a_range():
    r = extract_numeric(rec_temp(("growth", "positive", "28"), ("growth", "positive", "37")), "temperature", TCFG["temperature"], col())
    assert r["temperature_bin_source"] == "growth_range" and r["temperature_point"] == 32.5


def test_negative_growth_inside_positive_span_is_a_conflict():
    r = extract_numeric(rec_temp(("growth", "positive", "20-40"), ("growth", "negative", "30")), "temperature", TCFG["temperature"], col())
    assert r["temperature"] is None and r["temperature_conflict"] and r["temperature_bin_source"] == "growth_range"


def test_negative_growth_at_the_single_positive_point_is_a_conflict():
    r = extract_numeric(rec_temp(("growth", "positive", "37"), ("growth", "negative", "37")), "temperature", TCFG["temperature"], col())
    assert r["temperature"] is None and r["temperature_conflict"]


def test_optima_that_fall_in_different_bins_conflict_and_do_not_fall_back_to_growth():
    r = extract_numeric(rec_temp(("optimum", "positive", "10"), ("optimum", "positive", "30"), ("growth", "positive", "5-35")),
                        "temperature", TCFG["temperature"], col())
    assert r["temperature"] is None and r["temperature_conflict"] and r["temperature_bin_source"] == "optimum"


def test_optima_in_the_same_coarse_bin_do_not_conflict_even_if_fine_bins_differ():
    r = extract_numeric(rec_temp(("optimum", "positive", "60"), ("optimum", "positive", "85")), "temperature", TCFG["temperature"], col())
    assert r["temperature_fine"] is None and r["temperature_fine_conflict"]
    assert r["temperature"] == "thermo" and not r["temperature_conflict"]


def test_min_max_entries_count_as_growth_bounds():
    r = extract_numeric(rec_temp(("minimum", "positive", "10"), ("maximum", "positive", "40")), "temperature", TCFG["temperature"], col())
    assert r["temperature_bin_source"] == "growth_range" and r["temperature_point"] == 25


def test_no_temperature_data():
    r = extract_numeric({}, "temperature", TCFG["temperature"], col())
    assert r["temperature"] is None and r["temperature_bin_source"] == "none" and math.isnan(r["temperature_point"])


def test_unparseable_temperature_is_logged_not_silently_dropped():
    c = col()
    r = extract_numeric(rec_temp(("optimum", "positive", "warm")), "temperature", TCFG["temperature"], c)
    assert r["temperature"] is None and c["unparsed"][("temperature", "warm")] == 1
    assert json.loads(r["temperature_raw"])[0]["value"] == "warm"


# --- pH ------------------------------------------------------------------------------
@pytest.mark.parametrize("v,exp", [("4.0", "acidophile"), ("5.5", "neutrophile"), ("7.0", "neutrophile"), ("8.5", "alkaliphile"), ("10", "alkaliphile")])
def test_ph_bins(v, exp):
    r = extract_numeric(rec_ph(("optimum", "positive", v)), "ph", TCFG["ph"], col())
    assert r["ph"] == exp and r["ph_bin_source"] == "optimum"


def test_ph_growth_range_fallback_and_greater_than():
    r = extract_numeric(rec_ph(("growth", "positive", "4-9")), "ph", TCFG["ph"], col())
    assert r["ph_bin_source"] == "growth_range" and r["ph_point"] == 6.5 and r["ph"] == "neutrophile"
    r2 = extract_numeric(rec_ph(("optimum", "positive", ">7")), "ph", TCFG["ph"], col())
    assert r2["ph_point"] == 7 and r2["ph"] == "neutrophile"


def test_ph_growth_negative_entry_uses_ability_key_not_growth():
    r = extract_numeric(rec_ph(("growth", "positive", "5-9"), ("growth", "no", "7")), "ph", TCFG["ph"], col())
    assert r["ph_conflict"] and r["ph"] is None


# --- halophily -------------------------------------------------------------------------
def test_halophily_keeps_level_and_nacl_tests_separately():
    rec = {"Physiology and metabolism": {"halophily": [
        {"@ref": 1, "halophily level": "Halotolerant", "salt": "NaCl", "growth": "positive", "tested relation": "growth", "concentration": "0-8 %"},
        {"@ref": 2, "salt": "NaCl", "growth": "no", "tested relation": "growth", "concentration": "10 %"}]}}
    r = extract_halophily(rec)
    assert r["halophily_level"] == "halotolerant" and r["nacl_tests_n"] == 2 and r["halophily_level_n_obs"] == 1


def test_halophily_tests_only_means_no_curated_level():
    rec = {"Physiology and metabolism": {"halophily": [{"salt": "NaCl", "growth": "positive", "concentration": "2 %"}]}}
    r = extract_halophily(rec)
    assert r["halophily_level"] is None and r["nacl_tests_n"] == 1


# --- identity / genomes -------------------------------------------------------------------
def test_identity_extracts_accessions_without_version_and_designations():
    rec = {"General": {"DSM-Number": 26640, "doi": "10.13145/bacdive24493.20260601.11"},
           "Name and taxonomic classification": {"species": "Phaeobacter gallaeciensis", "genus": "Phaeobacter", "strain designation": "BS 107",
                                                 "type strain": "yes"},
           "Literature": {"culture collection no.": "DSM 26640, CIP 105210"},
           "Sequence information": {"Genome sequences": [
               {"INSDC accession": "GCA_000511385", "assembly level": "complete", "score": 95.9},
               {"INSDC accession": "GCA_000819625.1", "assembly level": "contig"}, {"BV-BRC accession": "1.1"}]}}
    r = extract_identity(24493, rec, "2026-09-29T00:00:00+00:00")
    assert r["bacdive_gca"] == "GCA_000511385|GCA_000819625" and r["bacdive_gca_n"] == 2
    assert r["type_strain_bacdive"] is True and r["doi_date"] == "20260601"
    assert r["culture_collection_no_raw"] == "DSM 26640, CIP 105210" and r["dsm_number"] == "26640"


def test_predictions_section_is_reported():
    c = col()
    extract_record(1, {"Genome-based predictions": [{"trait": "x"}]}, "t", TCFG, c)
    assert c["predictions"] == [1]


def test_full_record_row_has_every_expected_column():
    row = extract_record(1, {}, "t", TCFG, col())
    for t in ("gram", "shape", "motility", "spore", "oxygen", "temperature", "ph"):
        for suffix in ("", "_raw", "_fine", "_conflict", "_fine_conflict", "_n_obs"):
            assert f"{t}{suffix}" in row, f"{t}{suffix}"
    for c in ("temperature_bin_source", "ph_bin_source", "halophily_level", "nacl_tests_raw", "bacdive_gca", "oxygen_microaerophile_folded"):
        assert c in row


def test_subzero_growth_range_bins_as_psychrophile():
    r = extract_numeric(rec_temp(("growth", "positive", "-1.8-10")), "temperature", TCFG["temperature"], col())
    assert r["temperature_growth_min"] == -1.8 and r["temperature"] == "psychro"


@pytest.mark.parametrize("raw,coarse", [("sphere-shaped", "coccus"), ("diplococcus-shaped", "coccus"),
                                        ("curved-shaped", "other"), ("helical-shaped", "other"), ("pleomorphic-shaped", "other")])
def test_shape_synonyms_are_mapped_explicitly(raw, coarse):
    c = col()
    r = extract_categorical(rec_morph(**{"cell shape": raw}), "shape", TCFG["shape"], c)
    assert r["shape"] == coarse and not c["unmapped"]


@pytest.mark.parametrize("raw", ["aerotolerant", "microaerotolerant"])
def test_aerotolerant_is_deliberately_unmapped(raw):
    c = col()
    r = extract_categorical(rec_oxygen(raw), "oxygen", TCFG["oxygen"], c)
    assert r["oxygen"] is None and c["unmapped"][("oxygen", raw)] == 1
