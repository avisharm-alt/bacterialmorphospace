import json

import pandas as pd
import pytest

from src import slide_figures as sf

TABLES = "reports/tables"


def test_parse_interval_reads_the_en_dash_form():
    assert sf.parse_interval("110–124") == (110.0, 124.0)
    assert sf.parse_interval("96-109") == (96.0, 109.0)


def test_attrition_rows_are_the_published_counts():
    rows = dict(sf.attrition_rows(f"{TABLES}/attrition.tsv"))
    assert rows["BacDive strains retrieved"] == 102187
    assert rows["... one per GTDB species (analysed)"] == 2580
    assert [v for _, v in sf.attrition_rows(f"{TABLES}/attrition.tsv")] == sorted(
        [v for _, v in sf.attrition_rows(f"{TABLES}/attrition.tsv")], reverse=True)  # a funnel never grows


def test_occupancy_nulls_in_global_to_order_sequence():
    d = sf.occupancy_nulls(f"{TABLES}/occupied_vs_expected.tsv")
    assert list(d["null"]) == ["global", "phylum", "class", "order"]
    assert list(d["observed"]) == [90, 90, 89, 83]
    assert (d["lo"] <= d["expected"]).all() and (d["expected"] <= d["hi"]).all()
    assert d["expected"].is_monotonic_decreasing  # expectation falls as the null respects more lineage


def test_tier_shares_are_computed_from_counts_and_sum_to_one(tmp_path):
    p = tmp_path / "t.tsv"
    pd.DataFrame({"gtdb_phylum": ["a", "all"], "n": [3, 3], "genome_seen": [2, 2], "species_seen": [0, 0], "unseen": [1, 1],
                  "share_genome_seen": [0.67, 0.67], "share_species_seen": [0, 0], "share_unseen": [0.33, 0.33]}).to_csv(p, sep="\t", index=False)
    d = sf.tier_shares(p)
    assert d["share_unseen"].iloc[0] == pytest.approx(1 / 3)  # not the rounded 0.33 stored in the table
    assert all(abs(s - 1.0) < 1e-9 for s in d[["share_genome_seen", "share_species_seen", "share_unseen"]].sum(axis=1))


def _lopo_json():
    models = {k: {"auc": 0.5 + 0.02 * i, "ci": [0.4, 0.62]} for i, (k, _, _) in enumerate(sf.LOPO_MODELS)}
    return {"phyla": ["P1", "P2"], "per_phylum": {h: {"models": models} for h in ("P1", "P2")}, "macro": {"models": models}}


def test_lopo_points_cover_every_phylum_and_the_macro_mean_for_every_model():
    pts = sf.lopo_points(_lopo_json())
    assert len(pts) == (2 + 1) * len(sf.LOPO_MODELS)
    assert {p["group"] for p in pts} == {"P1", "P2", "macro mean"}
    assert all(p["lo"] <= p["auc"] <= p["hi"] for p in pts)


def test_every_figure_renders_to_a_nonempty_png(tmp_path):
    pytest.importorskip("matplotlib")
    for fn, name in ((sf.fig_attrition, "a"), (sf.fig_occupancy, "b"), (sf.fig_pretraining, "c")):
        out = tmp_path / f"{name}.png"
        fn(TABLES, out)
        assert out.stat().st_size > 10_000
    j = tmp_path / "lopo.json"
    j.write_text(json.dumps(_lopo_json()))
    sf.fig_lopo(j, tmp_path / "lopo.png")
    assert (tmp_path / "lopo.png").stat().st_size > 10_000
