"""Occupancy statistics: expected counts, permutation null, BH, dereplication."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import (analysis_set, assign_strata, bh_qvalues, dereplicate, detectable_zero, expected_counts, occupancy,
                          permutation_null, phylum_occupancy)
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
            "gtdb_phylum": "P1", "gtdb_class": "C1", "gtdb_order": "O1",
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



# --- stratified null ------------------------------------------------------------------
def test_stratified_expected_is_sum_of_per_stratum_products():
    codes = np.array([[0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0]])
    strata = np.array([0, 0, 0, 0, 1, 1])
    e = expected_counts(codes, (2, 2), strata)
    # stratum 0: perfectly associated but marginals 1/2,1/2 -> 4*[.25]*4 ; stratum 1: 2*[.25]*4
    assert np.allclose(e, [1.5, 1.5, 1.5, 1.5])
    assert np.allclose(expected_counts(codes, (2, 2), None), expected_counts(codes, (2, 2), np.zeros(6, int)))


def test_within_stratum_permutation_preserves_each_strata_marginals():
    rng = np.random.default_rng(3)
    n = 200
    strata = rng.integers(0, 4, n)
    codes = np.vstack([(strata % 2 == 0).astype(int), rng.integers(0, 3, n)])  # trait 0 is fixed by stratum
    null = permutation_null(codes, (2, 3), 500, rng, strata)
    # trait 0 is constant within strata, so its marginal (and the stratum-trait association) is untouched:
    # cells (0, *) and (1, *) keep their row totals in every permutation
    assert (null.reshape(500, 2, 3).sum(axis=2) == np.bincount(codes[0], minlength=2)).all()
    assert np.allclose(null.mean(axis=0), expected_counts(codes, (2, 3), strata), rtol=0.1)


def test_assign_strata_excludes_and_counts_small_strata():
    df = pd.DataFrame({"gtdb_order": ["A"] * 6 + ["B"] * 2 + [None] * 5})
    keep, ids, info = assign_strata(df, "order", min_size=5)
    assert keep.tolist() == [True] * 6 + [False] * 2 + [True] * 5   # the 5 unassigned form their own stratum
    assert info["N_used"] == 11 and info["species_excluded"] == 2 and info["strata_excluded"] == 1 and info["strata_used"] == 2
    assert len(set(ids.tolist())) == 2
    keep, ids, info = assign_strata(df, None, min_size=5)
    assert keep.all() and ids is None and info["null"] == "global"


def _clade_confounded(n_per=400, seed=5):
    """Two clades: clade A all gram-negative and never spore-forming; clade B gram-positive, half spore-forming.
    Gram-negative spore formers are absent only because of which clade carries which trait."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per):
        rows.append({"gram": "negative", "spore": "no", "gtdb_order": "A", "gtdb_species": f"a{i}"})
        rows.append({"gram": "positive", "spore": rng.choice(["no", "yes"]), "gtdb_order": "B", "gtdb_species": f"b{i}"})
    return _df(rows)


def test_positive_control_logic_global_flags_clade_gap_but_order_null_does_not():
    df = _clade_confounded()
    g = occupancy(df, ["gram", "spore"], CFG, n_perm=500, keep_cells=True).cell_table.set_index(["gram", "spore"])
    assert g.loc[("negative", "yes"), "empty_beyond_chance"]
    o = occupancy(df, ["gram", "spore"], CFG, n_perm=500, keep_cells=True, stratify="order").cell_table.set_index(["gram", "spore"])
    assert o.loc[("negative", "yes"), "expected"] == 0 and not o.loc[("negative", "yes"), "testable"]
    assert not o["empty_beyond_chance"].any()


def test_order_null_still_detects_a_gap_that_exists_within_orders():
    rng = np.random.default_rng(6)
    rows = []
    for order in ("A", "B", "C"):
        for i in range(400):
            g = rng.choice(["negative", "positive"])
            s = "no" if g == "negative" else rng.choice(["no", "yes"])   # gap within every order
            rows.append({"gram": g, "spore": s, "gtdb_order": order, "gtdb_species": f"{order}{i}"})
    o = occupancy(_df(rows), ["gram", "spore"], CFG, n_perm=500, keep_cells=True, stratify="order")
    t = o.cell_table.set_index(["gram", "spore"])
    assert t.loc[("negative", "yes"), "empty_beyond_chance"] and o.summary["N_used"] == 1200


def test_phylum_occupancy_counts_exclusive_cells_and_cumulative_share():
    rows = ([{"gram": "negative", "spore": "no", "gtdb_phylum": "Big", "gtdb_species": f"x{i}"} for i in range(10)]
            + [{"gram": "positive", "spore": "yes", "gtdb_phylum": "Big", "gtdb_species": f"y{i}"} for i in range(10)]
            + [{"gram": "positive", "spore": "no", "gtdb_phylum": "Small", "gtdb_species": f"z{i}"} for i in range(3)])
    t = phylum_occupancy(_df(rows), ["gram", "spore"], CFG, n_perm=50).set_index("phylum")
    assert t.loc["Big", "cells occupied"] == 2 and t.loc["Big", "cells only this phylum occupies"] == 2
    assert t.loc["Small", "cells occupied"] == 1 and np.isnan(t.loc["Small", "occupied (within-phylum null mean)"])
    assert t.loc["Small", "cumulative share of all occupied"] == 1.0



def test_exclusion_of_small_strata_never_manufactures_an_empty_cell():
    # the only gram-negative spore formers live in tiny orders; excluding those orders empties the cell
    # among the kept species, but the cell is occupied in the full set and must not be flagged empty
    rng = np.random.default_rng(8)
    rows = []
    for order in ("A", "B", "C"):
        for i in range(400):
            g = rng.choice(["negative", "positive"])
            rows.append({"gram": g, "spore": "no" if g == "negative" else rng.choice(["no", "yes"]),
                         "gtdb_order": order, "gtdb_species": f"{order}{i}"})
    for j in range(6):  # six singleton orders holding the only gram-negative spore formers
        rows.append({"gram": "negative", "spore": "yes", "gtdb_order": f"tiny{j}", "gtdb_species": f"t{j}"})
    o = occupancy(_df(rows), ["gram", "spore"], CFG, n_perm=300, keep_cells=True, stratify="order")
    t = o.cell_table.set_index(["gram", "spore"])
    cell = t.loc[("negative", "yes")]
    assert cell.observed == 0 and cell.observed_all_species == 6 and cell.emptied_by_exclusion
    assert not cell.empty_beyond_chance and o.summary["cells_emptied_by_exclusion"] == 1
    assert o.summary["species_excluded"] == 6



def test_detectable_zero_bounds_and_empirical_threshold():
    rng = np.random.default_rng(9)
    n = 3000
    df = _df([{"gram": g, "spore": s, "motility": m, "gtdb_species": f"sp{i}", "gtdb_order": f"O{i % 6}"} for i, (g, s, m) in
              enumerate(zip(rng.choice(["negative", "positive"], n, p=[0.7, 0.3]), rng.choice(["no", "yes"], n, p=[0.9, 0.1]),
                            rng.choice(["no", "yes"], n)))])
    occ = occupancy(df, ["gram", "spore", "motility"], CFG, n_perm=2000, keep_cells=True, stratify="order")
    d = detectable_zero(occ, CFG)
    m = d["testable cells (m)"]
    assert m == 8
    assert np.isclose(d["E needed, guaranteed (Poisson ln(m/α))"], np.log(m / 0.05))
    assert np.isclose(d["E needed, best case (Poisson ln(1/α))"], np.log(20))
    # every cell here has E >= ~45, far above any threshold: all guaranteed detectable
    assert d["cells where a true zero would be detected on its own (empirical p0 <= α/m)"] == 8
    assert d["cells with E >= minimum detectable constraint"] == 8
    assert d["minimum detectable constraint (E)"] >= max(3, np.log(m / 0.05))
    # the empirical threshold must be one of the cells' expected counts, i.e. every cell above it passes
    ct = occ.cell_table
    assert d["E needed, guaranteed (empirical, this null)"] in set(ct["expected"])


def test_detectable_zero_poisson_matches_permutation_for_a_single_small_cell():
    # a cell with expected ~7 under a global null: p0 ~ exp(-7) ~ 9e-4
    rng = np.random.default_rng(10)
    n = 2000
    df = _df([{"gram": g, "spore": s, "gtdb_species": f"sp{i}"} for i, (g, s) in
              enumerate(zip(rng.choice(["negative", "positive"], n, p=[0.93, 0.07]), rng.choice(["no", "yes"], n, p=[0.95, 0.05])))])
    occ = occupancy(df, ["gram", "spore"], CFG, n_perm=20000, keep_cells=True)
    t = occ.cell_table.set_index(["gram", "spore"])
    e = t.loc[("positive", "yes"), "expected"]
    assert 3 < e < 12
    assert abs(np.log(t.loc[("positive", "yes"), "p_empty_under_null"] + 1e-6) + e) < 1.5  # log p0 ~ -E
