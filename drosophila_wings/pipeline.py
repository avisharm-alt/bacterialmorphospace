"""Command-line entry point for the DGRP wing-shape baseline (steps 1 and 2 of the wing plan).

    python -m drosophila_wings.pipeline prepare    --wings <table>          Procrustes-align wings -> line means
    python -m drosophila_wings.pipeline genotypes  --genotypes <bed|vcf>    filter variants -> cached npz + GRM
    python -m drosophila_wings.pipeline baseline   [--covariates ...]       nested group CV -> report
    python -m drosophila_wings.pipeline all        all three
    python -m drosophila_wings.pipeline simulate   write a synthetic dataset in the same raw formats (for testing)

Outputs go under --data-dir (default /mnt/project-files/data/drosophila): derived/ for intermediates, results/ for the
report. See drosophila_wings/README.md for where to get the raw files.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, model, procrustes, relatedness, shape

DEFAULT_DATA = Path("/mnt/project-files/data/drosophila")
log = logging.getLogger("drosophila_wings")


def _dirs(args) -> tuple[Path, Path]:
    d = Path(args.data_dir)
    (d / "derived").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d / "derived", d / "results"


# ---------------------------------------------------------------- step 1a: shapes

def cmd_prepare(args) -> None:
    derived, _ = _dirs(args)
    w = io.load_wings(args.wings, line_col=args.line_col, sex_col=args.sex_col, keep_cols=tuple(args.cell_cols))
    if args.sex:
        keep = (w.meta.get("sex") == args.sex[0].upper()).to_numpy()
        w = io.Wings(w.coords[keep], w.meta.loc[keep].reset_index(drop=True))
    aligned, cs, consensus = procrustes.gpa(w.coords)
    V = procrustes.tangent(aligned, consensus)
    cells = ["sex"] + list(args.cell_cols)
    if args.remove_allometry:
        V = shape.remove_allometry(V, np.log(cs), w.meta, cells)
    V = shape.center_within(V, w.meta, cells)
    lm, n = shape.line_means(V, w.meta, cells, args.min_wings)
    rel = shape.split_half_reliability(V, w.meta, cells, np.random.default_rng(args.seed), list(lm.index))
    np.savez_compressed(derived / "wings_tangent.npz", V=V, centroid_size=cs, consensus=consensus,
                        line=w.meta["line"].to_numpy(), sex=w.meta.get("sex", pd.Series([""] * len(V))).to_numpy())
    k = consensus.shape[0]
    lm.columns = [f"{a}{i + 1}" for i in range(k) for a in ("x", "y")]
    lm.index.name = "line"
    lm.to_csv(derived / "line_means.csv")
    n.rename("n_wings").to_csv(derived / "line_counts.csv")
    summary = {"wings_file": str(args.wings), "n_wings": int(len(V)), "n_lines_total": int(w.meta["line"].nunique()),
               "n_lines_kept": int(len(lm)), "min_wings": args.min_wings, "landmarks": int(k), "cells": cells,
               "sex_filter": args.sex, "remove_allometry": bool(args.remove_allometry),
               "line_mean_reliability": rel, "wings_per_line_median": float(n.median())}
    (derived / "shape_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("prepare: %s", summary)


# ---------------------------------------------------------------- step 1b: genotypes

def cmd_genotypes(args) -> None:
    derived, _ = _dirs(args)
    g = io.read_genotypes(args.genotypes, maf=args.maf, max_missing=args.max_missing, thin=args.thin)
    lm = pd.read_csv(derived / "line_means.csv", index_col="line", dtype={"line": str})
    common = [l for l in lm.index if l in set(g.lines)]
    missing = sorted(set(lm.index) - set(g.lines), key=int)
    if missing:
        log.warning("%d phenotyped lines have no genotypes: %s", len(missing), missing[:20])
    g = g.subset(common)
    # re-filter on the phenotyped lines only, so MAF is that of the analysed panel
    ok = io._filter_block(g.G, args.maf, args.max_missing)
    g = io.Genotypes(g.lines, np.ascontiguousarray(g.G[:, ok]), g.variants.loc[ok].reset_index(drop=True))
    io.save_genotypes(g, derived / "genotypes_filtered.npz")
    K = relatedness.grm(g.G)
    np.savez_compressed(derived / "grm.npz", K=K, lines=np.array(g.lines))
    log.info("genotypes: %d lines x %d variants; GRM written", *g.G.shape)


# ---------------------------------------------------------------- step 2: baseline

def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def cmd_baseline(args) -> None:
    derived, results = _dirs(args)
    lm = pd.read_csv(derived / "line_means.csv", index_col="line", dtype={"line": str})
    z = np.load(derived / "grm.npz")
    glines = [str(x) for x in z["lines"]]
    lines = [l for l in glines if l in lm.index]
    pos = [glines.index(l) for l in lines]
    K = z["K"][np.ix_(pos, pos)]
    Y = lm.loc[lines].to_numpy()
    summary = json.loads((derived / "shape_summary.json").read_text())

    F = None
    cov_cols: list[str] = []
    if args.covariates:
        cov = io.load_covariates(args.covariates)
        X = io.design_matrix(cov, lines)
        cov_cols = list(X.columns)
        F = X.to_numpy(float)
        log.info("covariates: %s", cov_cols)

    groups = relatedness.related_groups(K, args.related_threshold)
    singletons = np.arange(len(lines))
    pairs = relatedness.top_pairs(K, lines, 15)
    rng = np.random.default_rng(args.seed)

    rows, per_line, lam_log = [], [], {}
    for scheme, grp in (("related-grouped", groups), ("random (leaky)", singletons)):
        reps = []
        for r in range(args.repeats):
            res = model.cross_validate(Y, K, grp, rng, F, args.outer_k, args.inner_k)
            lam_log.setdefault(scheme, []).extend(res.lambdas)
            for m, P in res.pred.items():
                reps.append({"scheme": scheme, "repeat": r, "model": m, **model.metrics(Y, P, res.pred["mean"])})
                if scheme == "related-grouped" and r == 0:
                    d = np.sqrt(((P - Y) ** 2).sum(axis=1))
                    per_line.append(pd.DataFrame({"line": lines, "model": m, "fold": res.folds, "procrustes_dist": d}))
            if scheme == "related-grouped" and r == 0:
                first_folds = res.folds
                first_r2 = next(x["r2_vs_mean"] for x in reps if x["model"] == "snp" and x["repeat"] == 0)
        rows.extend(reps)
    df = pd.DataFrame(rows)
    null = model.permutation_null(Y, K, groups, first_folds, rng, args.n_perm, args.inner_k) if args.n_perm else np.array([])
    perm_p = float((1 + (null >= first_r2).sum()) / (1 + len(null))) if len(null) else None

    pcs = df["pc_r"].apply(pd.Series).add_prefix("pc_r")
    df = pd.concat([df.drop(columns="pc_r"), pcs], axis=1)
    df.to_csv(results / "cv_repeats.csv", index=False)
    pd.concat(per_line).to_csv(results / "per_line_errors.csv", index=False)
    agg = df.groupby(["scheme", "model"], sort=False).agg(["mean", "std"])

    out = {"n_lines": len(lines), "n_variants_in_grm": None, "related_threshold": args.related_threshold,
           "n_related_groups": int(groups.max() + 1), "largest_group": int(np.bincount(groups).max()),
           "covariates": cov_cols, "line_mean_reliability": summary.get("line_mean_reliability"),
           "permutation": {"observed_r2_repeat0": first_r2, "n_perm": int(len(null)), "p": perm_p,
                           "null_mean": float(null.mean()) if len(null) else None,
                           "null_q95": float(np.quantile(null, 0.95)) if len(null) else None},
           "chosen_lambda_multipliers": {k: sorted(set(v)) for k, v in lam_log.items()},
           "summary": {f"{s} | {m}": {c: float(agg.loc[(s, m), (c, "mean")]) for c in
                                      ("r2_vs_mean", "procrustes_dist_mean", "share_lines_improved")}
                       for s, m in agg.index}}
    gz = derived / "genotypes_filtered.npz"
    if gz.exists():
        out["n_variants_in_grm"] = int(np.load(gz)["G"].shape[1])
    (results / "baseline_metrics.json").write_text(json.dumps(out, indent=2))

    # ---- markdown report
    L = [f"# DGRP wing shape: genomic baseline ({'SIMULATED DATA' if args.simulated else 'real data'})", ""]
    if args.simulated:
        L += ["> **These numbers come from simulated wings and genotypes. They test the code, not biology.**", ""]
    L += [f"- Wings: {summary['n_wings']:,} from `{Path(summary['wings_file']).name}`; {summary['landmarks']} landmarks; "
          f"cells averaged within line: {', '.join(summary['cells'])}; sex filter: {summary['sex_filter'] or 'none'}; "
          f"allometry removed: {summary['remove_allometry']}.",
          f"- Lines analysed: {len(lines)} (phenotyped with at least {summary['min_wings']} wings and genotyped); "
          f"median wings per line {summary['wings_per_line_median']:.0f}.",
          f"- Variants in the GRM: {out['n_variants_in_grm'] or 'n/a'}.",
          f"- Related-line groups (normalised GRM > {args.related_threshold}): {out['n_related_groups']} groups, "
          f"largest {out['largest_group']} lines.",
          f"- Covariates: {', '.join(cov_cols) if cov_cols else 'none supplied'}.",
          f"- **Noise ceiling.** Split-half reliability of line means = {_fmt(out['line_mean_reliability'])}. "
          "That is the R² a perfect genotype-to-mean predictor could reach against these noisy means.", "",
          f"## Held-out lines ({args.outer_k}-fold outer CV, {args.repeats} repeats, penalty by inner {args.inner_k}-fold group CV)",
          "", "R² vs mean is 1 − SSE(model)/SSE(training-mean shape), pooled over held-out lines; >0 means the model beats the "
          "mean shape. Distances are Procrustes (tangent-space Euclidean) distances to the observed line mean.", "",
          "| split | model | R² vs mean (± sd) | mean distance | lines improved | r on PC1 | r on PC2 | r on PC3 |",
          "|---|---|---|---|---|---|---|---|"]
    for s, m in agg.index:
        a = agg.loc[(s, m)]
        L.append(f"| {s} | {m} | {_fmt(a[('r2_vs_mean', 'mean')])} ± {_fmt(a[('r2_vs_mean', 'std')])} | "
                 f"{_fmt(a[('procrustes_dist_mean', 'mean')], 5)} | {_fmt(a[('share_lines_improved', 'mean')], 2)} | "
                 + " | ".join(_fmt(a.get((f"pc_r{i}", "mean"), np.nan), 2) for i in range(3)) + " |")
    L += ["", "## Is the SNP model's gain real?", ""]
    if len(null):
        L += [f"Shuffling genotypes across lines ({len(null)} permutations, same outer folds): null R² mean "
              f"{_fmt(out['permutation']['null_mean'])}, 95th percentile {_fmt(out['permutation']['null_q95'])}; "
              f"observed {_fmt(first_r2)}; permutation p = {_fmt(perm_p, 4)}."]
    L += ["", "The `random (leaky)` rows ignore relatedness and are reported only to show how much leakage would "
          "inflate the score. The `related-grouped` rows are the honest estimate.", "",
          "## Most related line pairs (normalised GRM)", "", "| line a | line b | relationship |", "|---|---|---|"]
    L += [f"| {r.line_a} | {r.line_b} | {r.relationship:.3f} |" for r in pairs.itertuples()]
    (results / "baseline_report.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", results / "baseline_report.md")


# ---------------------------------------------------------------- simulate

def cmd_simulate(args) -> None:
    from .simulate import write_dataset
    paths = write_dataset(Path(args.out), np.random.default_rng(args.seed), n_lines=args.n_lines, h2=args.h2)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m drosophila_wings.pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def prep_args(s):
        s.add_argument("--wings", required=True, help="wing table: one row per wing, x1..xK / y1..yK columns")
        s.add_argument("--line-col")
        s.add_argument("--sex-col")
        s.add_argument("--cell-cols", nargs="*", default=[], help="extra columns to centre within and average over (e.g. lab)")
        s.add_argument("--sex", choices=["F", "M"], help="analyse one sex only (default: both, centred within sex)")
        s.add_argument("--min-wings", type=int, default=10)
        s.add_argument("--remove-allometry", action="store_true")

    def geno_args(s):
        s.add_argument("--genotypes", required=True, help="plink prefix, .vcf(.gz), .tgeno or cached .npz")
        s.add_argument("--maf", type=float, default=0.05)
        s.add_argument("--max-missing", type=float, default=0.2)
        s.add_argument("--thin", type=int, default=1, help="keep every n-th variant (speed)")

    def base_args(s):
        s.add_argument("--covariates", nargs="*", default=[], help="tables keyed by line (e.g. wolbachia.xlsx inversion.xlsx)")
        s.add_argument("--related-threshold", type=float, default=0.05)
        s.add_argument("--outer-k", type=int, default=5)
        s.add_argument("--inner-k", type=int, default=5)
        s.add_argument("--repeats", type=int, default=5)
        s.add_argument("--n-perm", type=int, default=200)
        s.add_argument("--simulated", action="store_true", help="label the report as simulated")

    prep_args(sub.add_parser("prepare"))
    geno_args(sub.add_parser("genotypes"))
    base_args(sub.add_parser("baseline"))
    a = sub.add_parser("all")
    prep_args(a), geno_args(a), base_args(a)
    s = sub.add_parser("simulate")
    s.add_argument("--out", required=True)
    s.add_argument("--n-lines", type=int, default=150)
    s.add_argument("--h2", type=float, default=0.5)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    if args.cmd == "all":
        cmd_prepare(args), cmd_genotypes(args), cmd_baseline(args)
    else:
        {"prepare": cmd_prepare, "genotypes": cmd_genotypes, "baseline": cmd_baseline, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
