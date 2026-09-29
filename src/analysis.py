"""Panel completeness, dereplication, occupancy and the permutation null.

Terminology
-----------
strain-level   every BacDive strain with a matched GTDB genome (the non-dereplicated table)
species-level  one strain per GTDB species: the PRIMARY analysis. Permuting a table in which a
               genus appears hundreds of times measures genus abundance, not trait constraint.
               Preference within a species: type strain, then best assembly, then lowest BacDive ID.
coarse / fine  the two binnings in config.toml (coarse is primary; fine is a sensitivity)

Occupancy null
--------------
For a set of N strains complete on K traits, each trait column is permuted independently across
strains. That preserves every marginal exactly and destroys all association between traits, so
the permuted tables are draws from "traits independent given the marginals". For every cell:

  expected   N * prod_k p_k(level)            (analytic; equals the permutation mean)
  p_low      P(null count <= observed)        (permutation, one-sided: "emptier than chance")
  q_low      Benjamini-Hochberg over the *testable* cells only

  emptier_than_chance   testable, q_low < alpha, observed < expected
  empty_beyond_chance   ...and observed == 0 (the headline category)

Because the marginals are fixed, depletion in one cell forces compensating depletion elsewhere:
in a 2x2 table with a structural zero, the diagonal partner is also "emptier than chance" while
being well occupied. Only `empty_beyond_chance` cells are candidate forbidden combinations;
occupied-but-depleted cells are reported separately.

A cell is testable only if expected >= `min_expected_for_test` (default 3): with expected 0.3 the
null itself is empty 74% of the time, so an empty cell carries no information. Cells below the
threshold are counted and reported as untestable, never flagged.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .config import Config

ALL_TRAITS = ["gram", "shape", "motility", "spore", "oxygen", "temperature", "ph", "halophily_level"]
TRAIT_LABEL = {"gram": "gram stain", "shape": "cell shape", "motility": "motility", "spore": "spore formation",
               "oxygen": "oxygen tolerance", "temperature": "temperature", "ph": "pH",
               "halophily_level": "halophily (curated level)"}


def col(trait: str, binning: str) -> str:
    """Column holding the trait in the given binning ('coarse' or 'fine')."""
    if binning == "coarse":
        return trait
    if binning == "fine":
        return f"{trait}_fine"
    raise ValueError(binning)


def levels_for(cfg: Config, trait: str, binning: str) -> list[str]:
    t = cfg["traits"][trait]
    return list(t["coarse_levels"] if binning == "coarse" else t["fine_levels"])


# ---------------------------------------------------------------------------
# panel flags
# ---------------------------------------------------------------------------
def _populated(df: pd.DataFrame, trait: str) -> pd.Series:
    return df[f"{trait}_n_obs"] > 0


def add_panel_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    core, morph = cfg.core, cfg.morphology
    for t in ALL_TRAITS:
        df[f"{t}_populated"] = _populated(df, t)
        df[f"{t}_usable"] = df[t].notna()
    df["nacl_tests_populated"] = df["nacl_tests_n"] > 0
    # halophily counts as populated through either the curated level or raw NaCl growth tests
    df["any_trait_populated"] = df[[f"{t}_populated" for t in ALL_TRAITS] + ["nacl_tests_populated"]].any(axis=1)
    df["n_core_usable"] = df[[f"{t}_usable" for t in core]].sum(axis=1)
    df["morph4_complete"] = df[[f"{t}_usable" for t in morph]].all(axis=1)
    df["core6_complete"] = df[[f"{t}_usable" for t in core]].all(axis=1)
    df["morph4_fine_complete"] = df[[col(t, "fine") for t in morph]].notna().all(axis=1)
    df["core6_fine_complete"] = df[[col(t, "fine") for t in core]].notna().all(axis=1)
    df["core6_complete_optimum_temp"] = df["core6_complete"] & (df["temperature_bin_source"] == "optimum")
    df["all8_reference_complete"] = df["core6_complete"] & df["ph_usable"] & df["halophily_level_usable"]
    return df


# ---------------------------------------------------------------------------
# analysis sets and dereplication
# ---------------------------------------------------------------------------
def dereplicate(sub: pd.DataFrame, rank: str = "species") -> pd.DataFrame:
    """One strain per GTDB `rank` (species or genus): type strains first, then best assembly, then lowest BacDive ID.

    At genus level the kept strain is chosen by the same rule, so its trait values are those of one
    representative species - a deliberately crude sensitivity for congener clustering.
    """
    s = sub.copy()
    s["_type"] = (s["type_strain_bacdive"].fillna(False).astype(bool) | s["gtdb_is_type_strain_of_species"].astype(bool))
    s["_key"] = s[f"gtdb_{rank}"].where(s[f"gtdb_{rank}"].notna(), "acc:" + s["gtdb_accession"].astype(str))
    s = s.sort_values(["_type", "pref_order", "bacdive_id"], ascending=[False, True, True], kind="stable")
    s = s.drop_duplicates("_key", keep="first")
    return s.drop(columns=["_type", "_key"]).sort_values("bacdive_id")


def analysis_set(df: pd.DataFrame, traits: list[str], binning: str = "coarse", level: str = "species",
                 temp_source: str = "all", require_genome: bool = True) -> pd.DataFrame:
    """Strains complete on `traits` (in `binning`), with a genome, at strain or species level."""
    m = df[[col(t, binning) for t in traits]].notna().all(axis=1)
    if require_genome:
        m &= df["genome_matched"]
    if temp_source == "optimum" and "temperature" in traits:
        m &= df["temperature_bin_source"] == "optimum"
    elif temp_source not in ("all", "optimum"):
        raise ValueError(temp_source)
    sub = df[m]
    if level in ("species", "genus"):
        if not require_genome:
            raise ValueError("dereplication is defined on GTDB taxonomy, so it needs a genome")
        sub = dereplicate(sub, level)
    elif level != "strain":
        raise ValueError(level)
    return sub


# ---------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------
def _codes(sub: pd.DataFrame, traits: list[str], levels: dict[str, list[str]], binning: str) -> np.ndarray:
    out = []
    for t in traits:
        v = sub[col(t, binning)]
        bad = set(v.unique()) - set(levels[t])
        if bad:
            raise ValueError(f"trait {t}: values outside the configured levels {levels[t]}: {sorted(map(str, bad))[:5]}")
        c = pd.Categorical(v, categories=levels[t]).codes
        out.append(c.astype(np.int64))
    return np.vstack(out)


def expected_counts(codes: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """Expected count per cell under independence (flattened C-order), from the marginals."""
    n = codes.shape[1]
    margins = [np.bincount(codes[k], minlength=sizes[k]) / n for k in range(len(sizes))]
    return (n * reduce(np.multiply.outer, margins)).ravel()


def permutation_null(codes: np.ndarray, sizes: tuple[int, ...], n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """(n_perm, n_cells) cell counts with every trait column permuted independently."""
    ncell = int(np.prod(sizes))
    out = np.empty((n_perm, ncell), dtype=np.uint32)
    for i in range(n_perm):
        perm = rng.permuted(codes, axis=1)
        out[i] = np.bincount(np.ravel_multi_index(tuple(perm), sizes), minlength=ncell)
    return out


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


@dataclass
class Occupancy:
    traits: list[str]
    binning: str
    n: int
    cells: int
    occupied: int
    summary: dict
    cell_table: pd.DataFrame | None


def occupancy(sub: pd.DataFrame, traits: list[str], cfg: Config, binning: str = "coarse", n_perm: int = 2000,
              rng: np.random.Generator | None = None, keep_cells: bool = False) -> Occupancy:
    rng = rng or np.random.default_rng(cfg["analysis"]["seed"])
    lv = {t: levels_for(cfg, t, binning) for t in traits}
    sizes = tuple(len(lv[t]) for t in traits)
    ncell = int(np.prod(sizes))
    n = len(sub)
    a = cfg["analysis"]
    if n == 0:
        return Occupancy(traits, binning, 0, ncell, 0, {"N": 0, "cells": ncell, "occupied": 0, "occupied_frac": 0.0}, None)
    codes = _codes(sub, traits, lv, binning)
    obs = np.bincount(np.ravel_multi_index(tuple(codes), sizes), minlength=ncell)
    exp = expected_counts(codes, sizes)
    null = permutation_null(codes, sizes, n_perm, rng)
    p_low = (1 + (null <= obs).sum(axis=0)) / (n_perm + 1)
    p_high = (1 + (null >= obs).sum(axis=0)) / (n_perm + 1)
    p_empty = (null == 0).mean(axis=0)
    testable = exp >= a["min_expected_for_test"]
    q = np.full(ncell, np.nan)
    q[testable] = bh_qvalues(p_low[testable])
    flagged = testable & (q < a["fdr_alpha"]) & (obs < exp)
    empty_flagged = flagged & (obs == 0)
    occ_null = (null > 0).sum(axis=1)
    occupied = int((obs > 0).sum())
    summary = {
        "N": n, "cells": ncell, "occupied": occupied, "occupied_frac": occupied / ncell, "mean_per_cell": n / ncell,
        "null_occupied_mean": float(occ_null.mean()), "null_occupied_frac": float(occ_null.mean() / ncell),
        "null_occupied_lo": float(np.percentile(occ_null, 2.5)), "null_occupied_hi": float(np.percentile(occ_null, 97.5)),
        "p_fewer_occupied_than_null": float((1 + (occ_null <= occupied).sum()) / (n_perm + 1)),
        "empty_cells": int((obs == 0).sum()), "testable_cells": int(testable.sum()),
        "empty_testable_cells": int(((obs == 0) & testable).sum()),
        "flagged_cells": int(flagged.sum()), "empty_beyond_chance_cells": int(empty_flagged.sum()),
        "depleted_occupied_cells": int((flagged & (obs > 0)).sum()), "n_perm": n_perm,
        "min_p_possible": 1 / (n_perm + 1),
    }
    table = None
    if keep_cells:
        grid = np.array(list(itertools.product(*[lv[t] for t in traits])), dtype=object)
        table = pd.DataFrame(grid, columns=traits)
        table["observed"] = obs
        table["expected"] = exp
        table["null_mean"] = null.mean(axis=0)
        table["p_empty_under_null"] = p_empty
        table["p_low"] = p_low
        table["q_low"] = q
        table["p_high"] = p_high
        table["testable"] = testable
        table["emptier_than_chance"] = flagged
        table["empty_beyond_chance"] = empty_flagged
    return Occupancy(traits, binning, n, ncell, occupied, summary, table)


# ---------------------------------------------------------------------------
# subsets, capacity, co-occurrence, genome availability
# ---------------------------------------------------------------------------
def subset_list(cfg: Config) -> list[tuple[str, list[str]]]:
    core = cfg.core
    out = []
    for k in (4, 5):
        for c in itertools.combinations(core, k):
            out.append((f"{k}-trait", list(c)))
    out.append((f"{len(core)}-trait (full)", list(core)))
    return out


def subset_name(traits: list[str]) -> str:
    return " + ".join(traits)


def occupancy_grid(df: pd.DataFrame, cfg: Config, binning: str, level: str, temp_source: str,
                   n_perm: int | None = None, seed_offset: int = 0, full: Occupancy | None = None) -> pd.DataFrame:
    """Occupancy of every 4-/5-trait subset and the full panel in one configuration.

    Pass `full` (the full-panel result computed elsewhere) so the full map is computed once and
    the grid row agrees exactly with the per-cell table.
    """
    a = cfg["analysis"]
    rows = []
    for j, (kind, traits) in enumerate(subset_list(cfg)):
        is_full = kind.endswith("(full)")
        rng = np.random.default_rng(a["seed"] + seed_offset + j)
        sub = analysis_set(df, traits, binning, level, temp_source)
        strain_level_n = len(analysis_set(df, traits, binning, "strain", temp_source))
        if is_full and full is not None:
            occ = full
        else:
            occ = occupancy(sub, traits, cfg, binning, n_perm or (a["n_perm_full"] if is_full else a["n_perm_subset"]), rng)
        no_genome = len(analysis_set(df, traits, binning, "strain", temp_source, require_genome=False))
        rows.append({"panel": kind, "traits": subset_name(traits), "morphology_map": traits == cfg.morphology,
                     "N_strains_with_genome": strain_level_n, "N_species": len(analysis_set(df, traits, binning, "species", temp_source)),
                     "N_no_genome_requirement": no_genome, **occ.summary})
    return pd.DataFrame(rows)


def capacity_table(df: pd.DataFrame, cfg: Config, binning: str = "coarse", level: str = "species") -> pd.DataFrame:
    """N, cells and expected-count distribution for every trait subset of size 1..6 (no permutation needed)."""
    a = cfg["analysis"]
    rows = []
    for k in range(1, len(cfg.core) + 1):
        for traits in itertools.combinations(cfg.core, k):
            traits = list(traits)
            sub = analysis_set(df, traits, binning, level)
            lv = {t: levels_for(cfg, t, binning) for t in traits}
            sizes = tuple(len(lv[t]) for t in traits)
            cells = int(np.prod(sizes))
            if len(sub):
                exp = expected_counts(_codes(sub, traits, lv, binning), sizes)
                med, mn, mx = float(np.median(exp)), float(exp.min()), float(exp.max())
                testable = float((exp >= a["min_expected_for_test"]).mean())
            else:
                med = mn = mx = testable = float("nan")
            rows.append({"size": k, "traits": subset_name(traits), "N": len(sub), "cells": cells, "N_per_cell": len(sub) / cells,
                         "expected_min": mn, "expected_median": med, "expected_max": mx, "testable_cell_frac": testable})
    return pd.DataFrame(rows)


def coverage_matrices(df: pd.DataFrame, traits: list[str] = ALL_TRAITS) -> dict[str, pd.DataFrame]:
    """Pairwise co-coverage: counts, P(col | row) and lift = P(both) / (P(row) P(col)), on usable values."""
    U = np.column_stack([df[f"{t}_usable"].to_numpy(dtype=float) for t in traits])
    n = len(df)
    counts = U.T @ U
    diag = np.diag(counts)
    cond = counts / diag[:, None]
    lift = counts * n / np.outer(diag, diag)
    np.fill_diagonal(lift, np.nan)  # P(a,a)/P(a)^2 = 1/P(a): not a co-occurrence measure
    lab = [TRAIT_LABEL.get(t, t) for t in traits]
    mk = lambda m: pd.DataFrame(m, index=lab, columns=lab)  # noqa: E731
    return {"counts": mk(counts.astype(int)), "conditional": mk(cond), "lift": mk(lift)}


def independence_check(df: pd.DataFrame, panels: dict[str, list[str]]) -> pd.DataFrame:
    """Observed complete fraction vs the product of the marginal coverages."""
    n = len(df)
    rows = []
    for name, traits in panels.items():
        marg = [df[f"{t}_usable"].mean() for t in traits]
        obs = df[[f"{t}_usable" for t in traits]].all(axis=1).mean()
        prod = float(np.prod(marg))
        rows.append({"panel": name, "traits": len(traits), "observed_complete": int(round(obs * n)), "observed_frac": obs,
                     "product_of_marginals": prod, "expected_complete_if_independent": prod * n,
                     "ratio": obs / prod if prod > 0 else float("nan")})
    return pd.DataFrame(rows)


def _rate(s: pd.Series) -> tuple[int, int, float]:
    return int(s.sum()), int(len(s)), float(s.mean()) if len(s) else float("nan")


def genome_availability(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Does genome availability correlate with trait coverage?"""
    outcomes = {"BacDive lists a GCA": "bacdive_lists_assembly", "GTDB genome matched": "genome_matched"}
    base = {k: _rate(df[c]) for k, c in outcomes.items()}
    sets = {"all BacDive strains": pd.Series(True, index=df.index), "morphology panel complete (4)": df["morph4_complete"],
            "core panel complete (6)": df["core6_complete"], "core panel complete, optimum temperature": df["core6_complete_optimum_temp"]}
    rows = []
    for name, m in sets.items():
        row = {"set": name, "N": int(m.sum())}
        for k, c in outcomes.items():
            k1, n1, r1 = _rate(df.loc[m, c])
            rest = df.loc[~m, c]
            row[f"{k} n"], row[f"{k} %"], row[f"{k} base %"] = k1, 100 * r1, 100 * base[k][2]
            if m.sum() > 0 and (~m).sum() > 0:
                tab = [[k1, n1 - k1], [int(rest.sum()), int((~rest.astype(bool)).sum())]]
                odds, p = fisher_exact(tab)
                row[f"{k} odds ratio vs rest"], row[f"{k} Fisher p"] = float(odds), float(p)
            else:
                row[f"{k} odds ratio vs rest"], row[f"{k} Fisher p"] = float("nan"), float("nan")
        rows.append(row)
    by_n = []
    for n in range(0, len(cfg.core) + 1):
        m = df["n_core_usable"] == n
        row = {"core traits usable": n, "N": int(m.sum())}
        for k, c in outcomes.items():
            row[f"{k} %"] = 100 * float(df.loc[m, c].mean()) if m.sum() else float("nan")
        by_n.append(row)
    per_trait = []
    for t in cfg.core + ["ph", "halophily_level"]:
        m = df[f"{t}_usable"]
        row = {"trait": TRAIT_LABEL.get(t, t), "with trait N": int(m.sum())}
        for k, c in outcomes.items():
            with_t, without = df.loc[m, c], df.loc[~m, c]
            row[f"{k}: with trait %"] = 100 * float(with_t.mean()) if len(with_t) else float("nan")
            row[f"{k}: without %"] = 100 * float(without.mean()) if len(without) else float("nan")
        per_trait.append(row)
    return {"sets": pd.DataFrame(rows), "by_n_traits": pd.DataFrame(by_n), "per_trait": pd.DataFrame(per_trait)}
