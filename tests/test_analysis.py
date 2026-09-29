"""Occupancy statistics: expected counts, permutation null, BH, dereplication."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_set, bh_qvalues, dereplicate, expected_counts, occupancy, permutation_null
from src.config import load_config

CFG = load_config()


def test_expected_counts_are_product_of_marginals():
    codes = np.array([[0, 0, 1, 1], [0, 1, 0, 0]])
    e = expected_counts(codes, (2, 2))
    # p(a)=[.5,.5], p(b)=[.75,.25] -> N*p = 4*[.375,.125,.375,.125]
    assert np.allclose(e, [1.5, 0.5, 1.5, 0.5]) and np.isclose(e.sum(), 4)


def test_permutation_null_preserves_marginals_and_its_mean_matches_expected():
    rng = np.random.default_rng(0)
    codes = np.vstack([rng.integers(0, 2, 300), rng.integers(0, 3, 300)])
    null = permutation_null(codes, (2, 3), 3000, rng)
    assert (null.sum(axis=1) == 300).all()
    assert np.allclose(null.mean(axis=0), expected_counts(codes, (2, 3)), rtol=0.05)


def test_bh_matches_hand_computation():
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, 0.2]))
    # sorted p = .01,.03,.04,.2 -> p*n/rank = .04,.06,.0533,.2 -> running min from the top = .04,.0533,.0533,.2
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def _df(rows):
    base = {"gram": "negative", "shape": "rod", "motility": "yes", "spore": "no", "oxygen": "aerobe", "temperature": "meso",
            "temperature_bin_source": "optimum", "genome_matched": True, "type_strain_bacdive": False,
            "gtdb_is_type_strain_of_species": False, "pref_order": 0, "gtdb_accession": "x"}
    out = []
    for i, r in enumerate(rows):
        d = {**base, **r, "bacdive_id": i}
        for t in ("gram", "shape", "motility", "spore", "oxygen", "temperature"):
            d.setdefault(f"{t}_fine", d[t])
        out.append(d)
    return pd.DataFrame(out)


def test_dereplication_prefers_type_strain_then_assembly():
    df = _df([{"gtdb_species": "s1", "pref_order": 1}, {"gtdb_species": "s1", "pref_order": 0},
              {"gtdb_species": "s1", "pref_order": 5, "type_strain_bacdive": True}, {"gtdb_species": "s2"}])
    d = dereplicate(df)
    assert list(d.bacdive_id) == [2, 3]


def test_analysis_set_requires_genome_and_filters_optimum():
    df = _df([{"gtdb_species": "a"}, {"gtdb_species": "b", "genome_matched": False},
              {"gtdb_species": "c", "temperature_bin_source": "growth_single_point"}, {"gtdb_species": "d", "gram": None}])
    assert list(analysis_set(df, CFG.core, level="strain").bacdive_id) == [0, 2]
    assert list(analysis_set(df, CFG.core, level="strain", temp_source="optimum").bacdive_id) == [0]


def test_forbidden_combination_is_flagged_and_sparse_cells_are_not():
    rng = np.random.default_rng(1)
    n = 2000
    gram = rng.choice(["negative", "positive"], n)
    spore = rng.choice(["no", "yes"], n)
    spore[gram == "negative"] = "no"          # gram-negative spore formers never occur
    df = _df([{"gram": g, "spore": s, "gtdb_species": f"sp{i}"} for i, (g, s) in enumerate(zip(gram, spore))])
    occ = occupancy(df, ["gram", "spore"], CFG, n_perm=500, keep_cells=True)
    t = occ.cell_table.set_index(["gram", "spore"])
    assert t.loc[("negative", "yes"), "observed"] == 0 and t.loc[("negative", "yes"), "emptier_than_chance"]
    # with fixed marginals a structural zero forces its diagonal partner below expectation too,
    # so (positive, no) is depleted-but-occupied; only the empty cell is "empty beyond chance"
    assert t.loc[("positive", "no"), "emptier_than_chance"] and not t.loc[("positive", "no"), "empty_beyond_chance"]
    assert t.loc[("negative", "yes"), "empty_beyond_chance"]
    assert not t.loc[("negative", "no"), "emptier_than_chance"] and not t.loc[("positive", "yes"), "emptier_than_chance"]
    assert occ.summary["empty_beyond_chance_cells"] == 1


def test_independent_traits_flag_nothing():
    rng = np.random.default_rng(2)
    n = 3000
    df = _df([{"gram": g, "spore": s, "motility": m, "gtdb_species": f"sp{i}"} for i, (g, s, m) in
              enumerate(zip(rng.choice(["negative", "positive"], n), rng.choice(["no", "yes"], n), rng.choice(["no", "yes"], n)))])
    occ = occupancy(df, ["gram", "spore", "motility"], CFG, n_perm=500)
    assert occ.summary["flagged_cells"] == 0 and occ.summary["testable_cells"] == 8


def test_cells_with_tiny_expectation_are_untestable():
    df = _df([{"gtdb_species": f"s{i}"} for i in range(5)] + [{"gtdb_species": "z", "gram": "positive"}])
    occ = occupancy(df, ["gram", "spore"], CFG, n_perm=200, keep_cells=True)
    assert occ.summary["testable_cells"] < occ.summary["cells"]
    assert not occ.cell_table["emptier_than_chance"].any()


def test_value_outside_configured_levels_raises():
    df = _df([{"gtdb_species": "a", "shape": "banana"}])
    with pytest.raises(ValueError):
        occupancy(df, ["shape"], CFG, n_perm=10)


def test_genus_level_keeps_one_strain_per_gtdb_genus():
    df = _df([{"gtdb_species": "g1 a", "gtdb_genus": "g1"}, {"gtdb_species": "g1 b", "gtdb_genus": "g1", "type_strain_bacdive": True},
              {"gtdb_species": "g2 a", "gtdb_genus": "g2"}])
    assert list(analysis_set(df, CFG.core, level="genus").bacdive_id) == [1, 2]
    assert len(analysis_set(df, CFG.core, level="species")) == 3
