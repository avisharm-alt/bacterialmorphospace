"""Validation studies for the evolutionary null: does the machinery recover what it should, and does it cry wolf?

    python -m src.evolnull_validate recovery     repeated simulate -> fit on the real tree: bias, error, bound hits
    python -m src.evolnull_validate coverage     does the MCMC 90% interval contain the true rates ~90% of the time?
    python -m src.evolnull_validate calibration  the whole flagging pipeline on data where traits evolved INDEPENDENTLY:
                                                 how often does it flag a cell? (and with rate-heterogeneous truth)

Every study writes a TSV to reports/tables/ and prints a summary. `calibration` is the one that matters for the claim
"a flagged cell is a candidate for real constraint": under a correct null, BH at 5% should raise a false flag in <= 5%
of independent-trait datasets. It runs on a random 400-tip subtree of the panel tree to keep it affordable; the
behaviour of the null (not its power on the real tree) is what is being checked.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import evolnull as ev
from . import phylo
from .config import ROOT

TREE_NWK = ROOT / "data" / "interim" / "evolnull_panel_tree.nwk"
OUT = ROOT / "reports" / "tables"

# generating rates (per unit branch length) chosen near the ML fits on the real panel
SCENARIOS = {
    "gram-like (k=2)": (2, [0.04, 1.05]),
    "spore-like (k=2)": (2, [0.31, 2.8]),
    "motility-like (k=2)": (2, [2.8, 5.7]),
    "oxygen-like (k=3)": (3, [1.95, 0.1, 4.54, 2.17, 0.03, 0.16]),
}
TRUE_ARD = {"gram": (2, [0.04, 1.05]), "shape": (3, [0.51, 1.27, 2.14, 5.17, 20.65, 4.95]), "motility": (2, [2.8, 5.7]),
            "spore": (2, [0.31, 2.8]), "oxygen": (3, [1.95, 0.1, 4.54, 2.17, 0.03, 0.16])}

_TREE = None


def _init(nwk: str):
    global _TREE
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _TREE = phylo.tree_from_newick(Path(nwk).read_text())


def _simulate_truth(tree, model, theta, rng):
    return phylo.simulate_tips(tree, model, np.atleast_2d(theta), 1, rng)[:, 0]


# ---------------------------------------------------------------------------
# recovery and coverage
# ---------------------------------------------------------------------------
def _recovery_task(args):
    name, k, rates, seed, mcmc_iter = args
    rng = np.random.default_rng(seed)
    m = phylo.RateModel(k, "ARD")
    th = np.log(rates)
    obs = _simulate_truth(_TREE, m, th, rng)
    if len(set(obs.tolist())) < k:
        return None
    lik = phylo.Likelihood(_TREE, m, obs)
    fit = phylo.fit_ml(lik, starts=3, seed=seed, hessian=bool(mcmc_iter))
    row = {"scenario": name, "seed": seed, "theta_true": th.tolist(), "theta_ml": fit["theta"].tolist(), "at_bound": fit["at_bound"]}
    if mcmc_iter:
        ch = phylo.run_mcmc(lik, fit["theta"], fit["hessian"], mcmc_iter, rng)
        d = ch["chain"]
        row.update(lo=np.quantile(d, .05, axis=0).tolist(), hi=np.quantile(d, .95, axis=0).tolist(),
                   post_median=np.median(d, axis=0).tolist(), acceptance=ch["acceptance"], min_ess=float(ch["ess"].min()))
    return row


def run_recovery(workers: int, n_sims: int, mcmc_sims: int, mcmc_iter: int, seed: int = 1):
    tasks = []
    for si, (name, (k, rates)) in enumerate(SCENARIOS.items()):
        n = n_sims if k == 2 else max(n_sims // 2, 10)
        nm = mcmc_sims if k == 2 else max(mcmc_sims // 2, 8)
        tasks += [(name, k, rates, seed * 100_000 + si * 1000 + i, mcmc_iter if i < nm else 0) for i in range(n)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(TREE_NWK),)) as ex:
        res = [r for r in ex.map(_recovery_task, tasks, chunksize=1) if r]
    rows, cov_rows = [], []
    for name in SCENARIOS:
        rs = [r for r in res if r["scenario"] == name]
        true = np.array(rs[0]["theta_true"])
        est = np.array([r["theta_ml"] for r in rs])
        err = est - true
        mc = [r for r in rs if "lo" in r]
        for j in range(len(true)):
            row = {"scenario": name, "param": j, "true_rate": float(np.exp(true[j])), "n_sims": len(rs),
                   "ml_median_rate": float(np.exp(np.median(est[:, j]))), "ml_bias_log": float(np.median(err[:, j])),
                   "ml_rmse_log": float(np.sqrt(np.mean(err[:, j] ** 2))),
                   "share_within_2x": float(np.mean(np.abs(err[:, j]) < np.log(2))), "share_at_bound": float(np.mean([r["at_bound"] for r in rs]))}
            if mc:
                lo, hi = np.array([r["lo"] for r in mc])[:, j], np.array([r["hi"] for r in mc])[:, j]
                pm = np.array([r["post_median"] for r in mc])[:, j]
                row.update(mcmc_sims=len(mc), cri90_coverage=float(np.mean((lo <= true[j]) & (true[j] <= hi))),
                           post_median_bias_log=float(np.median(pm - true[j])), cri90_median_width_log=float(np.median(hi - lo)))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# end-to-end calibration
# ---------------------------------------------------------------------------
HET_SLOW, HET_LOGM, HET_S01, HET_S10 = 0.4, np.log(10.0), 0.6, 1.5  # slow class = 0.4x the base rates, fast class 10x that


def _truth_models(kind):
    out = {}
    for t, (k, rates) in TRUE_ARD.items():
        if kind == "homogeneous":
            out[t] = (phylo.RateModel(k, "ARD"), np.log(rates))
        else:
            m = phylo.RateModel(k, "HRM")
            out[t] = (m, np.concatenate([np.log(np.array(rates) * HET_SLOW), [HET_LOGM, np.log(HET_S01), np.log(HET_S10)]]))
    return out


def _calibration_task(args):
    truth, seed, n_tips, n_iter, n_draws, reps = args
    rng = np.random.default_rng(seed)
    labels = list(_TREE.labels)
    keep = set(rng.choice(labels, n_tips, replace=False))
    tree = phylo.tree_from_newick(_TREE.newick, keep=keep)
    models = _truth_models(truth)
    for _ in range(20):  # resample until every state of every trait is present a few times
        obs = {t: _simulate_truth(tree, m, th, rng) for t, (m, th) in models.items()}
        if all(np.bincount(obs[t], minlength=TRUE_ARD[t][0]).min() >= 4 for t in obs):
            break
    else:
        return None
    ks = {t: TRUE_ARD[t][0] for t in ev.TRAITS}
    sizes = tuple(ks[t] for t in ev.TRAITS)
    fits = {t: ev.fit_trait(tree, obs[t], ks[t], ["ARD", "HRM"], seed + i, n_iter=n_iter, n_draws=n_draws, starts=2)
            for i, t in enumerate(ev.TRAITS)}
    res = {"truth": truth, "seed": seed}
    flags = {}
    for kind in ("ARD", "HRM"):
        st = ev.run_null(tree, obs, ks, fits, sizes, kind, reps, seed + 7)
        flags[kind] = st["flag"]
        pl = st["p_low"][st["testable"]]
        res.update({f"{kind}_testable": int(st["testable"].sum()), f"{kind}_flagged": int(st["flag"].sum()),
                    f"{kind}_empty_flagged": int(st["empty_flag"].sum()), f"{kind}_share_p_below_05": float(np.mean(pl < 0.05)),
                    f"{kind}_min_q": float(np.nanmin(st["q_low"])) if st["testable"].any() else np.nan})
    res["robust_flagged"] = int((flags["ARD"] & flags["HRM"]).sum())
    return res


def run_calibration(workers: int, n_datasets: int, n_tips: int, n_iter: int, n_draws: int, reps: int, seed: int = 3):
    tasks = [(truth, seed * 10_000 + i + (0 if truth == "homogeneous" else 5000), n_tips, n_iter, n_draws, reps)
             for truth in ("homogeneous", "heterogeneous") for i in range(n_datasets)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(TREE_NWK),)) as ex:
        res = [r for r in ex.map(_calibration_task, tasks, chunksize=1) if r]
    return pd.DataFrame(res)


def summarise_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for truth in ("homogeneous", "heterogeneous"):
        d = df[df["truth"] == truth]
        for null, col in (("homogeneous ARD null", "ARD"), ("heterogeneous HRM null", "HRM"), ("robust (flagged under both)", "robust")):
            f = d["robust_flagged"] if col == "robust" else d[f"{col}_flagged"]
            rows.append({"data generated by": f"independent traits, {truth} rates", "null used": null, "datasets": len(d),
                         "datasets with >=1 false flag": int((f > 0).sum()), "false-flag rate": float((f > 0).mean()),
                         "mean flagged cells": float(f.mean()),
                         "share of testable p<0.05": float(d[f"{'ARD' if col == 'robust' else col}_share_p_below_05"].mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["recovery", "coverage", "calibration", "tree"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0, help="datasets/simulations (study-specific default)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.study == "tree" or not TREE_NWK.exists():
        df = ev.load_panel()
        tree, _ = ev.load_panel_tree(df)
        TREE_NWK.parent.mkdir(parents=True, exist_ok=True)
        TREE_NWK.write_text(tree.newick)
        print(f"wrote {TREE_NWK} ({len(tree.tips)} tips)")
        if a.study == "tree":
            return
    t0 = time.time()
    if a.study == "recovery":
        r = run_recovery(a.workers, a.n or 60, 0, 0)
        r.to_csv(OUT / "evolnull_validation_recovery.tsv", sep="\t", index=False)
    elif a.study == "coverage":
        r = run_recovery(a.workers, a.n or 40, a.n or 40, 1500, seed=2)
        r.to_csv(OUT / "evolnull_validation_coverage.tsv", sep="\t", index=False)
    else:
        raw = run_calibration(a.workers, a.n or 30, 400, 600, 60, 50)
        raw.to_csv(OUT / "evolnull_validation_calibration_raw.tsv", sep="\t", index=False)
        r = summarise_calibration(raw)
        r.to_csv(OUT / "evolnull_validation_calibration.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(r.round(3).to_string(index=False))
    print(f"[{a.study} finished in {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    sys.exit(main())
