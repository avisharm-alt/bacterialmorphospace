"""Evolutionary null for the trait-occupancy map.

The permutation nulls in `analysis.py` shuffle trait values across species (globally, or within a taxon). Trait
combinations are, however, the outcome of evolution along a tree, so this module simulates that process instead:
each trait evolves independently along the actual GTDB tree under a fitted continuous-time Markov (Mk) model, and the
resulting distribution of cell counts is the null "the traits evolved independently". A cell that is emptier than that
null predicts is a candidate for real constraint (correlated evolution), which a shuffle cannot identify.

Guards against a badly specified null (the project has produced one false "forbidden combination" already):
  * rate uncertainty  -- every replicate uses rates drawn from an MCMC posterior, not the point estimate;
  * rate heterogeneity -- the same analysis is repeated with a hidden fast/slow rate class (HRM); a cell is a
    `robust` candidate only if it is flagged under BOTH nulls;
  * calibration       -- `validate` runs the whole pipeline on data simulated with independent traits (and with
    rate-heterogeneous traits) and reports how often it raises a false flag.

The nulls do not fix the observed marginals (a shuffle does), and they say nothing about sampling bias in which
species have phenotypes. Temperature is excluded: with 24 psychrophiles its rates are not identifiable and every cell
involving that state is untestable.

Also here: stochastic character mapping to count independent origins of each trait combination.
"""
from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, phylo
from .config import ROOT, load_config

TRAITS = ["gram", "shape", "motility", "spore", "oxygen"]  # temperature excluded (see module docstring)
KINDS = {"gram": ["ER", "ARD", "HRM"], "motility": ["ER", "ARD", "HRM"], "spore": ["ER", "ARD", "HRM"],
         "shape": ["ER", "SYM", "ARD", "HRM"], "oxygen": ["ER", "SYM", "ARD", "ORD", "HRM"]}
TREE_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120.tree.gz"
TAX_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120_taxonomy.tsv.gz"
VERSION_URL = "https://data.gtdb.ecogenomic.org/releases/latest/VERSION.txt"
ALPHA = 0.05


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _fetch(url: str, dest: Path, min_bytes: int) -> Path:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink()
        raise IOError(f"{url}: download is smaller than {min_bytes} bytes (truncated?)")
    tmp.replace(dest)
    return dest


def load_panel(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(ROOT / cfg["paths"]["final"] / "core_panel_species.tsv", sep="\t")
    if df["gtdb_species"].duplicated().any():
        raise ValueError("core panel must have one row per GTDB species")
    return df


def load_panel_tree(df: pd.DataFrame, cfg=None, check_release: bool = True):
    """The GTDB bac120 tree (same release as the panel) pruned to the panel's species.

    Tree leaves are GTDB species representatives, so each panel species is mapped to its representative leaf through
    the GTDB taxonomy file (a panel genome may be a non-representative genome of its species). Returns (tree, df aligned
    to tree.tips).
    """
    cfg = cfg or load_config()
    raw = ROOT / cfg["paths"]["raw"] / "gtdb"
    if check_release:
        want = json.loads((ROOT / cfg["paths"]["interim"] / "gtdb_info.json").read_text())["release"]
        with urllib.request.urlopen(VERSION_URL, timeout=60) as r:
            have = r.read().decode().split()[0]
        if have != want:
            raise RuntimeError(f"GTDB 'latest' is {have} but the panel was built on {want}; the tree would not match")
    tree_gz = _fetch(TREE_URL, raw / "bac120.tree.gz", 1_000_000)
    tax_gz = _fetch(TAX_URL, raw / "bac120_taxonomy.tsv.gz", 5_000_000)
    with gzip.open(tree_gz, "rt") as f:
        newick = f.read()
    import treeswift as ts
    tree_leaves = {n.label for n in ts.read_tree_newick(newick.strip()).traverse_leaves()}
    leaf_of_species = {}  # species -> the genome that is its tree leaf (the GTDB representative)
    with gzip.open(tax_gz, "rt") as f:
        for line in f:
            acc, t = line.rstrip("\n").split("\t")
            if acc in tree_leaves:
                leaf_of_species[t.split(";")[-1][3:]] = acc
    miss = [s for s in df["gtdb_species"] if s not in leaf_of_species]
    if miss:
        raise KeyError(f"{len(miss)} panel species have no leaf in the tree, e.g. {miss[:3]}")
    leaves = {leaf_of_species[sp]: sp for sp in df["gtdb_species"]}
    tree = phylo.tree_from_newick(newick, keep=set(leaves))
    order = pd.Series(range(len(df)), index=df["gtdb_species"])
    aligned = df.iloc[[order[leaves[lab]] for lab in tree.labels]].reset_index(drop=True)
    aligned["tree_leaf"] = tree.labels
    return tree, aligned


def trait_codes(df: pd.DataFrame, cfg=None) -> tuple[dict, dict]:
    """({trait: int code array aligned to df}, {trait: level names})."""
    cfg = cfg or load_config()
    lv = {t: analysis.levels_for(cfg, t, "coarse") for t in TRAITS}
    codes = {}
    for t in TRAITS:
        bad = set(df[t].unique()) - set(lv[t])
        if bad:
            raise ValueError(f"{t}: values outside the configured levels: {sorted(bad)}")
        codes[t] = pd.Categorical(df[t], categories=lv[t]).codes.astype(int)
    return codes, lv


def sizes_of(lv: dict) -> tuple:
    return tuple(len(lv[t]) for t in TRAITS)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def fit_trait(tree, obs: np.ndarray, k: int, kinds: list, seed: int, n_iter: int = 3000, n_draws: int = 200,
              mcmc_kinds=("ARD", "HRM"), starts: int = 4) -> dict:
    """ML fits of every candidate model (AIC table) plus MCMC posterior draws for the ARD and HRM models."""
    rng = np.random.default_rng(seed)
    out = {"ml": {}, "draws": {}, "mcmc": {}}
    for kind in kinds:
        m = phylo.RateModel(k, kind)
        lik = phylo.Likelihood(tree, m, obs)
        fit = phylo.fit_ml(lik, starts=starts, seed=seed)
        out["ml"][kind] = {"theta": fit["theta"].tolist(), "loglik": fit["loglik"], "aic": fit["aic"], "npar": fit["npar"],
                           "at_bound": fit["at_bound"]}
        if kind in mcmc_kinds:
            it = n_iter if kind == "ARD" else int(n_iter * 4 / 3)
            ch = phylo.run_mcmc(lik, fit["theta"], fit["hessian"], it, rng)
            out["draws"][kind] = phylo.thin(ch["chain"], n_draws).tolist()
            out["mcmc"][kind] = {"iterations": it, "kept": int(len(ch["chain"])), "acceptance": ch["acceptance"],
                                 "min_ess": float(ch["ess"].min()), "median_ess": float(np.median(ch["ess"]))}
    return out


def rate_table(fits: dict, levels: dict) -> pd.DataFrame:
    """Per trait and model: AIC, delta AIC, and the ML / posterior rates for the ARD and HRM models."""
    rows = []
    for t, f in fits.items():
        best = min(v["aic"] for v in f["ml"].values())
        for kind, v in f["ml"].items():
            row = {"trait": t, "model": kind, "npar": v["npar"], "loglik": v["loglik"], "aic": v["aic"], "delta_aic": v["aic"] - best,
                   "ml_rates_or_params": ", ".join(f"{x:.3g}" for x in np.exp(v["theta"])), "at_bound": v["at_bound"]}
            if kind in f["draws"]:
                d = np.exp(np.array(f["draws"][kind]))
                row["posterior_median"] = ", ".join(f"{x:.3g}" for x in np.median(d, axis=0))
                row["posterior_95"] = "; ".join(f"[{a:.2g},{b:.2g}]" for a, b in zip(np.quantile(d, .025, axis=0), np.quantile(d, .975, axis=0)))
                row["mcmc_acceptance"] = f["mcmc"][kind]["acceptance"]
                row["mcmc_min_ess"] = f["mcmc"][kind]["min_ess"]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the null
# ---------------------------------------------------------------------------
def simulate_cell_counts(tree, kind: str, ks: dict, fits: dict, sizes: tuple, reps_per_draw: int, seed: int,
                         traits=TRAITS) -> np.ndarray:
    """(n_replicates, n_cells) counts from simulating every trait independently under its posterior draws."""
    rng = np.random.default_rng(seed)
    per = []
    for t in traits:
        m = phylo.RateModel(ks[t], kind)
        per.append(phylo.simulate_tips(tree, m, np.array(fits[t]["draws"][kind]), reps_per_draw, rng))
    R = per[0].shape[1]
    ncell = int(np.prod(sizes))
    ntip = per[0].shape[0]
    idx = np.ravel_multi_index(tuple(p.astype(np.int64) for p in per), sizes)  # (ntip, R)
    flat = (idx + ncell * np.arange(R)[None, :]).ravel()
    return np.bincount(flat, minlength=ncell * R).reshape(R, ncell).astype(np.uint16)


def cell_statistics(obs: np.ndarray, sim: np.ndarray, alpha: float = ALPHA) -> dict:
    """Per-cell comparison of the observed count with the simulated null.

    p_low = P(sim <= obs) with the +1 correction; testable = the null itself is empty in <= alpha of replicates (so an
    empty observation could reach significance at all); BH over testable cells; `flag` = testable, q < alpha, and
    emptier than expected. `bonferroni_count` is the largest observed count that would still be flagged at alpha/m:
    it turns each cell's detectability into a number (compare with its expected count).
    """
    R = sim.shape[0]
    exp, sd = sim.mean(0), sim.std(0, ddof=1)
    p_low = (1 + (sim <= obs).sum(0)) / (R + 1)
    p_high = (1 + (sim >= obs).sum(0)) / (R + 1)
    p0 = (1 + (sim == 0).sum(0)) / (R + 1)
    testable = p0 <= alpha
    q = np.full(len(obs), np.nan)
    q[testable] = analysis.bh_qvalues(p_low[testable])
    flag = testable & (q < alpha) & (obs < exp)
    m = int(testable.sum())
    thr = alpha / max(m, 1)
    srt = np.sort(sim, axis=0)
    k = int(np.floor(thr * (R + 1))) - 1
    bonf = srt[k] if k >= 0 else np.full(len(obs), -1)
    bonf = np.where(testable, bonf, -1)
    return {"expected": exp, "sd": sd, "p_low": p_low, "p_high": p_high, "p_empty": p0, "testable": testable, "q_low": q,
            "flag": flag, "empty_flag": flag & (obs == 0), "bonferroni_count": bonf, "n_testable": m,
            "occupied_null": (sim > 0).sum(1)}


def run_null(tree, codes: dict, ks: dict, fits: dict, sizes: tuple, kind: str, reps_per_draw: int, seed: int,
             traits=TRAITS) -> dict:
    ncell = int(np.prod(sizes))
    obs = np.bincount(np.ravel_multi_index(tuple(codes[t] for t in traits), sizes), minlength=ncell)
    sim = simulate_cell_counts(tree, kind, ks, fits, sizes, reps_per_draw, seed, traits)
    st = cell_statistics(obs, sim)
    st["observed"] = obs
    return st


def marginal_check(tree, codes: dict, ks: dict, fits: dict, kind: str, reps_per_draw: int, seed: int, traits=TRAITS) -> pd.DataFrame:
    """Observed tip-state counts against the simulated distribution (the evolutionary null does not fix marginals)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in traits:
        m = phylo.RateModel(ks[t], kind)
        sim = phylo.simulate_tips(tree, m, np.array(fits[t]["draws"][kind]), reps_per_draw, rng)
        for s in range(ks[t]):
            c = (sim == s).sum(0)
            obs = int((codes[t] == s).sum())
            rows.append({"trait": t, "state": s, "observed": obs, "sim_mean": c.mean(), "sim_2.5": np.quantile(c, .025),
                         "sim_97.5": np.quantile(c, .975), "obs_percentile": float((c <= obs).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stochastic mapping: independent origins of each trait combination
# ---------------------------------------------------------------------------
def map_origins(tree, ks: dict, fits: dict, codes: dict, sizes: tuple, n_maps: int, seed: int, kind: str = "ARD",
                traits=TRAITS) -> np.ndarray:
    """(n_maps, n_cells): how many times lineages ENTERED each cell in one joint stochastic map.

    Each trait is mapped independently under its own model given its own tip data (a draw from the posterior rates is
    used for every map), and the histories are merged along every branch in time order. An entry into cell c is an event
    that changes the lineage's joint state to c. The root's own cell is not an entry. Events on branches whose
    descendants later lose the state still count: this is the number of independent arrivals in the sampled tree.
    """
    rng = np.random.default_rng(seed)
    N, ncell = tree.n_nodes, int(np.prod(sizes))
    out = np.zeros((n_maps, ncell), dtype=np.int32)
    models = {t: phylo.RateModel(ks[t], kind) for t in traits}
    draws = {t: np.array(fits[t]["draws"][kind]) for t in traits}
    for mp in range(n_maps):
        states = np.zeros((len(traits), N), dtype=int)
        ev = []
        for ti, t in enumerate(traits):
            th = draws[t][rng.integers(len(draws[t]))]
            r = phylo.stochastic_map(tree, models[t], th, codes[t], rng)
            states[ti] = r["states"]
            if len(r["branch"]):
                ev.append(np.column_stack([r["branch"], r["time"], np.full(len(r["branch"]), ti), r["to"]]))
        if not ev:
            continue
        ev = np.concatenate(ev)
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
        cur_branch, vec = -1, None
        for b, _, ti, to in ev:
            b, ti, to = int(b), int(ti), int(to)
            if b != cur_branch:
                cur_branch, vec = b, states[:, tree.parent[b]].copy()
            vec[ti] = to
            out[mp, np.ravel_multi_index(tuple(vec), sizes)] += 1
    return out


def hamming1_neighbours(sizes: tuple) -> list:
    """For every cell (flattened C order), the flattened indices of the cells that differ in exactly one trait."""
    ncell = int(np.prod(sizes))
    nb = []
    for c in range(ncell):
        idx = np.unravel_index(c, sizes)
        cur = []
        for ti, sz in enumerate(sizes):
            for v in range(sz):
                if v != idx[ti]:
                    j = list(idx)
                    j[ti] = v
                    cur.append(int(np.ravel_multi_index(tuple(j), sizes)))
        nb.append(cur)
    return nb


def surprising_absences(observed: np.ndarray, origins: np.ndarray, sizes: tuple, top: int = 10) -> pd.DataFrame:
    """Rank the EMPTY cells by the summed mean origin counts of their Hamming-1 neighbours."""
    nb = hamming1_neighbours(sizes)
    mean_orig = origins.mean(0)
    rows = []
    for c in np.flatnonzero(observed == 0):
        n = nb[c]
        rows.append({"cell": c, "neighbour_origins": float(mean_orig[n].sum()), "n_neighbours": len(n),
                     "occupied_neighbours": int((observed[n] > 0).sum()), "max_neighbour_origins": float(mean_orig[n].max()),
                     "own_visits": float(mean_orig[c])})
    return pd.DataFrame(rows).sort_values("neighbour_origins", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the analysis: tables and comparison with the permutation nulls
# ---------------------------------------------------------------------------
def cell_grid(lv: dict, traits=TRAITS) -> pd.DataFrame:
    import itertools
    g = pd.DataFrame(list(itertools.product(*[lv[t] for t in traits])), columns=list(traits))
    g["cell"] = ["\u00b7".join(map(str, r)) for r in g[list(traits)].to_numpy()]
    return g


def permutation_results(al: pd.DataFrame, cfg, n_perm: int, seed: int, traits=TRAITS) -> tuple[dict, dict]:
    """The repo's own permutation nulls (global, phylum, class, order) on the same cells, via analysis.occupancy.

    Returns ({level: cell table}, {level: null SD per cell}); the SD comes from a separate, smaller run of the same
    shuffle so the two nulls can be compared on how variable a cell's count is.
    """
    tables, sds = {}, {}
    lv = {t: analysis.levels_for(cfg, t, "coarse") for t in traits}
    sizes = tuple(len(lv[t]) for t in traits)
    for level in (None, "phylum", "class", "order"):
        occ = analysis.occupancy(al, list(traits), cfg, "coarse", n_perm, np.random.default_rng(seed), keep_cells=True, stratify=level)
        name = level or "global"
        tables[name] = occ.cell_table
        keep, strata, _ = analysis.assign_strata(al, level, cfg["analysis"]["min_stratum_size"])
        codes = analysis._codes(al[keep], list(traits), lv, "coarse")
        null = analysis.permutation_null(codes, sizes, min(n_perm, 2000), np.random.default_rng(seed + 1), strata)
        sds[name] = null.std(0, ddof=1)
    return tables, sds


def build_comparison(grid: pd.DataFrame, observed: np.ndarray, evo: dict, perm: dict, perm_sd: dict) -> pd.DataFrame:
    """One row per cell: the two evolutionary nulls (ARD / HRM), their robust intersection, and the four permutation nulls."""
    t = grid.copy()
    t["observed"] = observed
    for nm, st in evo.items():
        t[f"evo_{nm}_expected"] = st["expected"]
        t[f"evo_{nm}_sd"] = st["sd"]
        t[f"evo_{nm}_p_low"] = st["p_low"]
        t[f"evo_{nm}_q"] = st["q_low"]
        t[f"evo_{nm}_testable"] = st["testable"]
        t[f"evo_{nm}_emptier"] = st["flag"]
    t["evo_robust_emptier"] = t["evo_ARD_emptier"] & t["evo_HRM_emptier"]
    t["evo_robust_empty"] = t["evo_robust_emptier"] & (observed == 0)
    for lvl, ct in perm.items():
        t[f"perm_{lvl}_expected"] = ct["expected"].to_numpy()
        t[f"perm_{lvl}_sd"] = perm_sd[lvl]
        t[f"perm_{lvl}_q"] = ct["q_low"].to_numpy()
        t[f"perm_{lvl}_testable"] = ct["testable"].to_numpy()
        t[f"perm_{lvl}_emptier"] = ct["emptier_than_chance"].to_numpy()
    lv_names = list(perm)
    t["perm_any_emptier"] = t[[f"perm_{l}_emptier" for l in lv_names]].any(axis=1)
    t["perm_order_emptier"] = t["perm_order_emptier"]
    t["category"] = np.select(
        [t["evo_robust_emptier"] & t["perm_any_emptier"], t["evo_robust_emptier"], t["perm_any_emptier"] & (t["evo_ARD_emptier"] | t["evo_HRM_emptier"]),
         t["perm_any_emptier"]],
        ["flagged by both", "evolutionary only", "permutation + one evolutionary null", "permutation only"], default="neither")
    t["note"] = [_why(r) for _, r in t.iterrows()]
    return t


def _why(r) -> str:
    """One-line reason for a disagreement, from the numbers in the row."""
    if r["category"] in ("neither", "flagged by both"):
        return ""
    evo_e, evo_s = r["evo_ARD_expected"], r["evo_ARD_sd"]
    lv = "order" if r["perm_order_emptier"] else next((l for l in ("phylum", "class", "global") if r[f"perm_{l}_emptier"]), "order")
    pe, ps = r[f"perm_{lv}_expected"], r[f"perm_{lv}_sd"]
    if r["category"] == "permutation only":
        if evo_s > 1.5 * ps:
            return (f"the tree makes this count far more variable (SD {evo_s:.1f} vs {ps:.1f} under the {lv} shuffle), so a shortfall "
                    f"to {int(r['observed'])} from {pe:.1f} is within what independently evolving traits produce")
        return f"expected {evo_e:.1f} under independent evolution vs {pe:.1f} under the {lv} shuffle"
    if evo_e > 1.3 * r["perm_order_expected"]:
        return (f"the order-level shuffle already expects only {r['perm_order_expected']:.1f} (it keeps each order's own trait mix, "
                f"which carries any correlated evolution), while independent evolution expects {evo_e:.1f}")
    return f"independent evolution expects {evo_e:.1f} (SD {evo_s:.1f}); the {lv} shuffle expects {pe:.1f}"


def power_summary(st: dict, alpha: float = ALPHA) -> dict:
    """Which cells the null can test at all and how large a depletion it could detect."""
    m = int(st["n_testable"])
    thr = alpha / max(m, 1)
    p0 = st["p_empty"]
    exp = st["expected"]
    bonf_ok = p0 <= thr
    return {"cells": len(exp), "testable": m, "untestable": int(len(exp) - m),
            "min_expected_testable": float(exp[st["testable"]].min()) if m else np.nan,
            "min_expected_bonferroni_detectable": float(exp[bonf_ok].min()) if bonf_ok.any() else np.nan,
            "n_bonferroni_detectable_if_empty": int(bonf_ok.sum()),
            "poisson_equivalent_E": float(np.log(max(m, 1) / alpha)),
            "median_overdispersion_var_over_mean": float(np.median((st["sd"] ** 2)[exp > 1] / exp[exp > 1])) if (exp > 1).any() else np.nan}


def power_by_expected(st: dict) -> pd.DataFrame:
    """P(the null is empty) by expected-count bin, against the Poisson value exp(-E) a shuffle would give."""
    e, p0 = st["expected"], st["p_empty"]
    bins = [0, 1, 2, 3, 5, 7, 10, 15, 25, 50, 1e9]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (e >= a) & (e < b)
        if m.any():
            rows.append({"expected_count_bin": f"[{a:g}, {b:g})" if b < 1e8 else f">= {a:g}", "cells": int(m.sum()),
                         "median_P_null_empty": float(np.median(p0[m])), "poisson_exp_minus_E": float(np.median(np.exp(-e[m])))})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.to_numpy()) + "\n"


def analyze(n_perm: int = 10_000, reps_per_draw: int = 50, n_maps: int = 200, out: Path | None = None, quick: bool = False) -> dict:
    """Run every analysis from the cached fits and write tables + reports/evolutionary_null_report.md."""
    cfg = load_config()
    out = out or ROOT / cfg["paths"]["reports"]
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    fits = json.loads(FITS_JSON.read_text())
    meta = fits.pop("_meta")
    seed = cfg["analysis"]["seed"]
    df = load_panel(cfg)
    tree, al = load_panel_tree(df, cfg)
    codes, lv = trait_codes(al, cfg)
    ks = {t: len(lv[t]) for t in TRAITS}
    sizes = sizes_of(lv)
    grid = cell_grid(lv)
    obs = np.bincount(np.ravel_multi_index(tuple(codes[t] for t in TRAITS), sizes), minlength=int(np.prod(sizes)))
    n_draws = len(fits["gram"]["draws"]["ARD"])
    reps = max(reps_per_draw // (10 if quick else 1), 2)

    rates = rate_table(fits, lv)
    rates.to_csv(tab / "evolnull_mk_fits.tsv", sep="\t", index=False)
    evo = {k: run_null(tree, codes, ks, fits, sizes, k, reps, seed + 11 * (i + 1)) for i, k in enumerate(("ARD", "HRM"))}
    marg = marginal_check(tree, codes, ks, fits, "ARD", reps, seed + 5)
    marg.to_csv(tab / "evolnull_marginals.tsv", sep="\t", index=False)
    perm, perm_sd = permutation_results(al, cfg, 200 if quick else n_perm, seed)
    comp = build_comparison(grid, obs, evo, perm, perm_sd)
    origins = map_origins(tree, ks, fits, codes, sizes, 3 if quick else n_maps, seed + 3)
    comp["origins_mean"] = origins.mean(0)
    comp["origins_lo"], comp["origins_hi"] = np.quantile(origins, .025, axis=0), np.quantile(origins, .975, axis=0)
    comp["origins_p_zero"] = (origins == 0).mean(0)
    sa = surprising_absences(obs, origins, sizes)
    sa.insert(1, "cell_label", grid["cell"].to_numpy()[sa["cell"].to_numpy()])
    sa.to_csv(tab / "evolnull_surprising_absences.tsv", sep="\t", index=False)
    comp.to_csv(tab / "evolnull_cells.tsv", sep="\t", index=False)
    pw = {k: power_summary(v) for k, v in evo.items()}
    pd.DataFrame(pw).T.to_csv(tab / "evolnull_power.tsv", sep="\t")
    power_by_expected(evo["ARD"]).to_csv(tab / "evolnull_power_by_expected.tsv", sep="\t", index=False)
    res = {"grid": grid, "observed": obs, "evo": evo, "perm": perm, "comparison": comp, "origins": origins, "surprising": sa,
           "rates": rates, "marginals": marg, "power": pw, "meta": meta, "lv": lv, "sizes": sizes, "n_draws": n_draws, "reps": reps}
    (out / "evolutionary_null_report.md").write_text(write_report(res))
    return res


def write_report(r: dict) -> str:
    comp, evo, obs = r["comparison"], r["evo"], r["observed"]
    val = {}
    for name in ("recovery", "coverage", "calibration"):
        p = ROOT / "reports" / "tables" / f"evolnull_validation_{name}.tsv"
        if p.exists():
            val[name] = pd.read_csv(p, sep="\t")
    occ_obs = int((obs > 0).sum())
    L = ["# Evolutionary null for the trait occupancy map", "",
         f"Five traits (temperature excluded: 24 psychrophiles are too few to fit or test), {len(obs)} cells, {int(obs.sum())} species, "
         f"GTDB {r['meta']['gtdb_release']} tree pruned to the panel. {occ_obs} of {len(obs)} cells are occupied.", "",
         "## Method", "",
         "Each trait evolves independently along the tree under a continuous-time Markov model. Rates are drawn from an MCMC posterior "
         f"({r['n_draws']} draws x {r['reps']} replicates each = {r['n_draws'] * r['reps']} replicates), not fixed at the ML estimate, and the "
         "whole analysis is repeated with a hidden fast/slow rate class (HRM). A cell is a **robust candidate** only if it is emptier than "
         "both nulls predict. p = (1 + #{simulated <= observed}) / (1 + replicates), Benjamini-Hochberg over the cells the null can test "
         "(the null is itself empty in <= 5% of replicates). Branch lengths are substitutions per site.", ""]
    if "calibration" in val:
        L += ["## Calibration: does the pipeline cry wolf?", "",
              "Whole pipeline (fit, posterior draws, simulate, BH) on data where the traits evolved independently, so any flag is false. "
              "400-tip subtrees of the panel tree.", "", md_table(val["calibration"]), ""]
    if "recovery" in val:
        L += ["## Parameter recovery on the real tree", "", md_table(val["recovery"].round(3)), ""]
    if "coverage" in val:
        L += ["## Posterior coverage (nominal 90%)", "", md_table(val["coverage"].round(3)), ""]
    L += ["## Model fits (AIC)", "", md_table(r["rates"][["trait", "model", "npar", "loglik", "aic", "delta_aic", "at_bound"]].round(2)), ""]
    hr = r["rates"][r["rates"]["model"] == "HRM"][["trait", "delta_aic"]]
    L += ["ΔAIC is relative to the best model for that trait. Where HRM has a lower AIC than ARD the trait evolves at heterogeneous rates and the "
          "single-rate null is misspecified.", ""]
    mg = r["marginals"]
    L += ["## Do the simulations reproduce the observed trait frequencies?", "",
          "The evolutionary null does not fix marginals (a shuffle does), so this checks the fitted models are not simulating a different world.", "",
          md_table(mg.round(3)), ""]
    for nm in ("ARD", "HRM"):
        st = evo[nm]
        occ_null = st["occupied_null"]
        L += [f"**{nm} null**: {occ_obs} occupied cells observed; null mean {occ_null.mean():.1f} "
              f"(95% range {np.quantile(occ_null, .025):.0f}-{np.quantile(occ_null, .975):.0f}); P(null occupies <= observed) = "
              f"{(1 + (occ_null <= occ_obs).sum()) / (len(occ_null) + 1):.3f}.", ""]
    show = ["cell", "observed", "evo_ARD_expected", "evo_ARD_sd", "evo_ARD_q", "evo_HRM_expected", "evo_HRM_q", "perm_global_q", "perm_phylum_q",
            "perm_class_q", "perm_order_q", "category"]
    flagged = comp[(comp["category"] != "neither")].copy()
    L += ["## Cells emptier than the nulls predict, and how the nulls compare", "",
          f"Evolutionary ARD flags {int(comp['evo_ARD_emptier'].sum())} cells, HRM {int(comp['evo_HRM_emptier'].sum())}, both (robust) "
          f"{int(comp['evo_robust_emptier'].sum())}; permutation nulls flag {int(comp['perm_any_emptier'].sum())} at any level.", ""]
    if len(flagged):
        L += [md_table(flagged[show + ["note"]].round(4)), ""]
    else:
        L += ["No cell is flagged by any null.", ""]
    pc = comp[comp["cell"] == "negative\u00b7coccus\u00b7no\u00b7yes\u00b7aerobe"]
    if len(pc):
        L += ["**Positive control** (Gram-negative, non-motile, endospore-forming aerobic cocci; the permutation analysis's known constraint): ",
              md_table(pc[show].round(4)), ""]
    L += ["## Power: what can and cannot be tested", ""]
    L += [md_table(pd.DataFrame(r["power"]).T.round(3).reset_index().rename(columns={"index": "null"})), ""]
    L += ["An empty cell is detectable only if the null's own chance of being empty is below alpha/m. Because simulated counts are "
          "over-dispersed (clade structure), the expected count needed is higher than a Poisson/shuffle calculation gives:", "",
          md_table(power_by_expected(evo["ARD"]).round(4)), ""]
    untest = comp[~comp["evo_ARD_testable"]]
    L += [f"{len(untest)} of {len(comp)} cells are untestable under the ARD null (expected count too small for an empty observation to be "
          "informative); they are never flagged.", ""]
    L += ["## Independent origins (stochastic character mapping)", "",
          f"Mean number of times lineages entered each occupied cell across {r['origins'].shape[0]} joint maps (each trait mapped independently under a posterior draw of its "
          "rates; root cell not counted).", ""]
    occd = comp[comp["observed"] > 0].sort_values("origins_mean", ascending=False)
    L += ["Most convergent occupied cells:", "", md_table(occd[["cell", "observed", "origins_mean", "origins_lo", "origins_hi"]].head(10).round(1)), "",
          "Occupied cells with the fewest origins:", "", md_table(occd[["cell", "observed", "origins_mean", "origins_lo", "origins_hi"]].tail(10).round(1)), ""]
    L += ["## The ten most surprising absences", "",
          "Empty cells ranked by the summed mean origin counts of their seven Hamming-distance-1 neighbours.", "",
          md_table(r["surprising"].merge(comp[["cell", "evo_ARD_expected", "evo_ARD_q", "evo_HRM_q", "perm_order_q"]].rename(columns={"cell": "cell_label"}),
                                         on="cell_label").round(3)), ""]
    L += ["## Caveats", "",
          "* The null tests **independent evolution of the traits**. A flagged cell means correlated evolution, or a misspecified rate model, or sampling bias in which "
          "species have phenotypes (type strains, complete-panel selection); the calibration study above bounds the second, nothing here bounds the third.",
          "* Traits that changed together on a single branch (Gram stain and sporulation at the origin of the Bacillota) are correlated evolution by construction; "
          "the positive control is expected to be flagged for that reason, not because the combination is forbidden.",
          "* Origin counts come from mapping each trait independently, so they describe the independent-evolution history; and with 2,580 of ~190,000 species sampled "
          "they are lower bounds on origins in the whole tree.",
          "* Branch lengths are substitutions per site, not time. HRM captures rate variation among clades but not every kind of misspecification.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# command line: fit
# ---------------------------------------------------------------------------
FITS_JSON = ROOT / "data" / "interim" / "evolnull_fits.json"
_G: dict = {}


def _fit_task(t):
    from threadpoolctl import threadpool_limits
    g = _G
    with threadpool_limits(limits=1):
        return t, fit_trait(g["tree"], g["codes"][t], g["ks"][t], KINDS[t], g["seed"] + TRAITS.index(t), g["n_iter"], g["n_draws"])


def fit_all(workers: int = 2, n_iter: int = 3000, n_draws: int = 200) -> dict:
    """Fit every trait (ML for all candidate models, MCMC for ARD and HRM) and cache to data/interim/evolnull_fits.json."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    cfg = load_config()
    df = load_panel(cfg)
    tree, al = load_panel_tree(df, cfg)
    codes, lv = trait_codes(al, cfg)
    _G.update(tree=tree, codes=codes, ks={t: len(lv[t]) for t in TRAITS}, seed=cfg["analysis"]["seed"], n_iter=n_iter, n_draws=n_draws)
    order = ["oxygen", "shape", "motility", "spore", "gram"]  # slowest first
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        fits = dict(ex.map(_fit_task, order))
    fits["_meta"] = {"seed": _G["seed"], "n_iter": n_iter, "n_draws": n_draws, "n_tips": len(tree.tips), "levels": lv,
                     "gtdb_release": json.loads((ROOT / cfg["paths"]["interim"] / "gtdb_info.json").read_text())["release"]}
    FITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    FITS_JSON.write_text(json.dumps(fits))
    return fits


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "analyze"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="tiny settings to check the plumbing")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.command == "analyze":
        analyze(quick=a.quick, out=(Path("/tmp/evolnull_quick") if a.quick else None))
        print(f"[analyze finished in {(time.time() - t0) / 60:.1f} min]")
        return
    fits = fit_all(a.workers, a.n_iter)
    print(rate_table({k: v for k, v in fits.items() if k != "_meta"}, fits["_meta"]["levels"]).round(3).to_string(index=False))
    print(f"[fit finished in {(time.time() - t0) / 60:.1f} min] wrote {FITS_JSON}")


if __name__ == "__main__":
    main()
