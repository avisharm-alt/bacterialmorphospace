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


def run_null(tree, codes: dict, ks: dict, fits: dict, sizes: tuple, kind: str, reps_per_draw: int, seed: int) -> dict:
    ncell = int(np.prod(sizes))
    obs = np.bincount(np.ravel_multi_index(tuple(codes[t] for t in TRAITS), sizes), minlength=ncell)
    sim = simulate_cell_counts(tree, kind, ks, fits, sizes, reps_per_draw, seed)
    st = cell_statistics(obs, sim)
    st["observed"] = obs
    return st


def marginal_check(tree, codes: dict, ks: dict, fits: dict, kind: str, reps_per_draw: int, seed: int) -> pd.DataFrame:
    """Observed tip-state counts against the simulated distribution (the evolutionary null does not fix marginals)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in TRAITS:
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
def map_origins(tree, ks: dict, fits: dict, codes: dict, sizes: tuple, n_maps: int, seed: int, kind: str = "ARD") -> np.ndarray:
    """(n_maps, n_cells): how many times lineages ENTERED each cell in one joint stochastic map.

    Each trait is mapped independently under its own model given its own tip data (a draw from the posterior rates is
    used for every map), and the histories are merged along every branch in time order. An entry into cell c is an event
    that changes the lineage's joint state to c. The root's own cell is not an entry. Events on branches whose
    descendants later lose the state still count: this is the number of independent arrivals in the sampled tree.
    """
    rng = np.random.default_rng(seed)
    N, ncell = tree.n_nodes, int(np.prod(sizes))
    out = np.zeros((n_maps, ncell), dtype=np.int32)
    models = {t: phylo.RateModel(ks[t], kind) for t in TRAITS}
    draws = {t: np.array(fits[t]["draws"][kind]) for t in TRAITS}
    for mp in range(n_maps):
        states = np.zeros((len(TRAITS), N), dtype=int)
        ev = []
        for ti, t in enumerate(TRAITS):
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
