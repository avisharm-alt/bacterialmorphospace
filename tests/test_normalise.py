"""Unit tests for strain-designation normalisation (the hard part of the genome join)."""
import pytest

from src.normalise import (
    Normaliser,
    canonical_keys,
    normalise_designation,
    parse_designation,
    parse_designations,
    split_designations,
)


# --- one designation -> one canonical key -----------------------------------
@pytest.mark.parametrize(
    "raw",
    ["DSM 20231", "DSM20231", "DSMZ 20231", "dsm 20231", "DSM-20231", "DSM_20231", "DSM 20231T",
     "DSM 20231 T", "DSM 20231^T", "DSM 20231 (T)", "DSM 20231(T)", "DSM 20231 [type strain]",
     "strain DSM 20231", "type strain DSM 20231", "DSMZ-20231", "DSM 020231", "  DSM   20231  ",
     "DSM 20231", "DSM‐20231"],
)
def test_dsm_variants_share_one_key(raw):
    assert normalise_designation(raw) == "DSM20231"


def test_registered_trademark_marks_are_dropped():
    assert normalise_designation("ATCC® 11775™") == "ATCC11775"


def test_alias_ncib_folds_to_ncimb():
    assert normalise_designation("NCIB 8014") == normalise_designation("NCIMB 8014") == "NCIMB8014"


def test_accents_are_folded():
    assert normalise_designation("Schön 12") == "SCHON12"


def test_leading_zeros_only_stripped_after_a_known_collection():
    assert normalise_designation("DSM 000123") == "DSM123"
    # a bare designation keeps its zeros: "TK 0112" is not "TK 112"
    assert normalise_designation("Urakami TK 0112") == "URAKAMITK0112"


def test_trailing_T_is_not_stripped_from_bare_designations():
    assert normalise_designation("K12T") == "K12T"
    assert normalise_designation("ATCC 700T") == "ATCC700"


def test_empty_and_none():
    assert normalise_designation(None) == ""
    assert normalise_designation("") == ""
    assert normalise_designation("   ") == ""


def test_distinct_collections_never_collide():
    assert normalise_designation("DSM 11775") != normalise_designation("ATCC 11775")


# --- multi-designation strings -------------------------------------------------
def test_split_on_slash_semicolon_comma_pipe_equals():
    assert split_designations("ATCC 11775 / DSM 30083") == ["ATCC 11775", "DSM 30083"]
    assert split_designations("ATCC 11775; DSM 30083") == ["ATCC 11775", "DSM 30083"]
    assert split_designations("DSM 46234, ATCC 21370, IMET 10785") == ["DSM 46234", "ATCC 21370", "IMET 10785"]
    assert split_designations("ATCC 11775T = DSM 30083") == ["ATCC 11775T", "DSM 30083"]
    assert split_designations("A 1|B 2") == ["A 1", "B 2"]


def test_split_drops_empty_pieces():
    assert split_designations(";; ,DSM 1, ;") == ["DSM 1"]
    assert split_designations(None) == []
    assert split_designations("") == []


def test_multi_designation_keys_cover_every_designation_not_just_the_first():
    assert canonical_keys("ATCC 11775 / DSM 30083") == ["ATCC11775", "DSM30083"]
    assert canonical_keys("DSM 46234, ATCC 21370, IMET 10785, JCM 2849, Urakami TK 0112") == [
        "DSM46234", "ATCC21370", "IMET10785", "JCM2849", "URAKAMITK0112"]


def test_the_issue_examples_all_reach_the_same_key_as_their_partner():
    # "DSM 20231", "DSM20231", "DSMZ 20231" and "ATCC 11775 / DSM 30083" from the task brief
    a = set(canonical_keys("DSM 20231"))
    b = set(canonical_keys("DSM20231"))
    c = set(canonical_keys("DSMZ 20231"))
    assert a == b == c == {"DSM20231"}
    assert "DSM30083" in canonical_keys("ATCC 11775 / DSM 30083")
    assert "DSM30083" in canonical_keys("DSMZ30083")


def test_duplicate_keys_are_collapsed():
    assert canonical_keys("DSM 20231, DSMZ 20231; DSM20231") == ["DSM20231"]


# --- kinds: what is matchable ---------------------------------------------------
def test_kinds():
    assert parse_designation("DSM 20231").kind == "collection"
    assert parse_designation("DSM 20231").collection == "DSM"
    assert parse_designation("BS 107").kind == "bare"
    assert parse_designation("870").kind == "unusable"        # purely numeric
    assert parse_designation("Marburg").kind == "unusable"    # purely alphabetic
    assert parse_designation("none").kind == "unusable"       # placeholder
    assert parse_designation("A1").kind == "unusable"         # too short
    assert parse_designation("DSM").kind == "unusable"        # code with no number


def test_unusable_designations_never_become_match_keys():
    assert canonical_keys("none; 870; Marburg") == []
    assert canonical_keys("Marburg, DSM 30083") == ["DSM30083"]


def test_parse_designations_keeps_raw_text_for_the_audit_trail():
    d = parse_designations("DSMZ 20231 / ATCC 11775")
    assert [x.raw for x in d] == ["DSMZ 20231", "ATCC 11775"]
    assert [x.canonical for x in d] == ["DSM20231", "ATCC11775"]


# --- config-bound front end -------------------------------------------------------
def test_normaliser_from_config_matches_module_defaults():
    from src.config import load_config

    n = Normaliser.from_config(load_config())
    assert [d.canonical for d in n.parse("DSMZ 20231, ATCC 11775")] == ["DSM20231", "ATCC11775"]
    assert n.parse(None) == ()
    assert n.raw_tokens("A 1; B 2") == ["A 1", "B 2"]


def test_custom_aliases_and_collections_are_respected():
    assert normalise_designation("XYZ 12", aliases={"XYZ": "FOO"}, collections={"FOO"}) == "FOO12"
    assert parse_designation("XYZ 12", aliases={"XYZ": "FOO"}, collections={"FOO"}).kind == "collection"
    assert parse_designation("XYZ 12", collections={"DSM"}).kind == "bare"


def test_normaliser_treats_nan_and_none_as_no_designation():
    n = Normaliser.from_config(__import__("src.config", fromlist=["load_config"]).load_config())
    assert n.parse(float("nan")) == () and n.parse(None) == () and n.parse("") == ()
    assert n.raw_tokens(float("nan")) == []
