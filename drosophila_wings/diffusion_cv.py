"""Step 3 entry point: cross-validated genotype-conditioned diffusion model of wing shape, plus a possibility-space sketch.

    python -m drosophila_wings.diffusion_cv run --backend modal            full run on Modal T4 GPUs (one container per fold)
    python -m drosophila_wings.diffusion_cv run --backend local --quick    small CPU run to test the code

Inputs come from `derived/` (written by `python -m drosophila_wings.pipeline prepare/genotypes`), and the outer folds are
copied from the baseline's `results/per_line_errors.csv`, so step 3 is scored on exactly the step 2 held-out lines.
Outputs go to `results/diffusion/` (or `--out`).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import generative as gm
from .diffusion import DiffusionConfig

DEFAULT_DATA = Path("/mnt/project-files/data/drosophila")
log = logging.getLogger("drosophila_wings")


def _run_jobs(backend: str, payload, folds, cfg, args) -> tuple[list[dict], dict]:
    jobs = [(int(f), v) for v in gm.VARIANTS for f in folds]
    if backend == "modal":
        import modal

        from .modal_app import app, fold_remote, full_remote
        with modal.enable_output(), app.run():
            full_call = full_remote.spawn(payload, cfg, args.geno_pcs, args.seed) if args.space else None
            res = list(fold_remote.starmap([(payload, f, cfg, args.n_gen, args.geno_pcs, s, args.seed) for f, s in jobs]))
            full = full_call.get() if full_call is not None else None
        return res, full
    res = []
    for f, s in jobs:
        t0 = time.time()
        res.append(gm.run_fold(payload, f, cfg, n_gen=args.n_gen, n_geno_pc=args.geno_pcs, variant=s, seed=args.seed,
                               score_train=not args.quick))
        log.info("fold %d %s done in %.0fs", f, s, time.time() - t0)
    full = gm.run_full(payload, cfg, n_geno_pc=args.geno_pcs, seed=args.seed,
                       n_synth=40 if args.quick else 400, n_per_synth=64 if args.quick else 128,
                       n_per_line=64 if args.quick else 256) if args.space else None
    return res, full


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def space_summary(full: dict, d: gm.WingData) -> tuple[pd.DataFrame, np.ndarray]:
    """Compare the spread of generated line means with the observed line means, in the observed line-mean PC space."""
    Y = d.Y
    mu = Y.mean(axis=0)
    _, s, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    k = 5
    obs = (Y - mu) @ Vt[:k].T
    cov = np.cov(obs.T)
    icov = np.linalg.inv(cov)
    chi95 = float(np.quantile(np.einsum("ij,jk,ik->i", obs, icov, obs), 0.95))
    rows = [{"set": "observed lines", "n_lines": len(Y), "between_line_var": float(((Y - mu) ** 2).sum(1).mean()),
             "within_line_var_median": np.nan, "share_inside_observed_95pct": 0.95, "pc1_range": float(np.ptp(obs[:, 0]))}]
    for name, M in full["line_means"].items():
        P = (M - mu) @ Vt[:k].T
        mah = np.einsum("ij,jk,ik->i", P, icov, P)
        rows.append({"set": f"generated: {name}", "n_lines": len(M), "between_line_var": float(((M - M.mean(0)) ** 2).sum(1).mean()),
                     "within_line_var_median": float(np.median(full["line_vars"][name])),
                     "share_inside_observed_95pct": float((mah <= chi95).mean()), "pc1_range": float(np.ptp(P[:, 0]))})
    return pd.DataFrame(rows), Vt[:k]


def plots(out: Path, d: gm.WingData, rec: pd.DataFrame, full: dict | None, Vt: np.ndarray | None, samples: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    t = rec[rec["split"] == "test"].pivot_table(index="line", columns="method", values="energy")
    if {"mean+resid", "diffusion-geno"} <= set(t.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        for m, col in (("diffusion-geno", "#2a6fdb"), ("diffusion-shuffled", "#d9822b"), ("diffusion-nogeno", "#888888")):
            if m in t:
                ax.scatter(t["mean+resid"] * 1e3, t[m] * 1e3, s=12, alpha=0.7, label=m, color=col)
        lim = [0, float(np.nanmax(t.to_numpy()) * 1e3 * 1.05)]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("energy distance, mean + residuals baseline (×10⁻³)")
        ax.set_ylabel("energy distance, diffusion (×10⁻³)")
        ax.set_title("Held-out lines: below the line = closer than baseline")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "energy_per_line.png", dpi=150)
        plt.close(fig)
        made.append("energy_per_line.png")
    if full is not None and Vt is not None:
        mu = d.Y.mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        obs = (d.Y - mu) @ Vt[:2].T
        colors = {"random genotypes": "#9bbcf0", "crosses": "#8cc084", "extrapolated (2x)": "#e7a36b", "real lines": "#2a6fdb"}
        for name in ("extrapolated (2x)", "random genotypes", "crosses", "real lines"):
            P = (full["line_means"][name] - mu) @ Vt[:2].T
            ax.scatter(P[:, 0] * 1e3, P[:, 1] * 1e3, s=8, alpha=0.6, color=colors[name], label=f"generated: {name}")
        ax.scatter(obs[:, 0] * 1e3, obs[:, 1] * 1e3, s=14, facecolor="none", edgecolor="k", lw=0.7, label="observed lines")
        ax.set_xlabel("line-mean PC1 (×10⁻³)")
        ax.set_ylabel("line-mean PC2 (×10⁻³)")
        ax.set_title("Wing line means: observed vs generated")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "possibility_space.png", dpi=150)
        plt.close(fig)
        made.append("possibility_space.png")
    if samples:
        line, s = next(iter(samples.items()))
        names = [m for m in ("mean+resid", "diffusion-geno", "diffusion-shuffled") if m in s]
        Xo = d.V[d.wing_line == line]
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.2 * (len(names) + 1), 3), sharex=True, sharey=True)
        for ax, (title, X) in zip(axes, [("observed", Xo)] + [(n, s[n]) for n in names]):
            pts = X[:60].reshape(len(X[:60]), -1, 2)
            ax.scatter(pts[..., 0].ravel(), pts[..., 1].ravel(), s=0.3, alpha=0.3, color="#2a6fdb")
            ax.set_title(f"line {line}: {title}", fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / "example_line_wings.png", dpi=150)
        plt.close(fig)
        made.append("example_line_wings.png")
    return made


def cmd_run(args) -> None:
    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "results" / "diffusion"
    out.mkdir(parents=True, exist_ok=True)
    d = gm.load(data_dir / "derived", data_dir / "results")
    cfg = DiffusionConfig(steps=args.steps, width=args.width, depth=args.depth, cond_noise=args.cond_noise,
                          sample_steps=args.sample_steps).to_dict()
    folds = sorted(set(d.folds.tolist()))
    if args.quick:
        # small CPU test: fewer wings per line and fewer lines, same code path
        rng = np.random.default_rng(args.seed)
        keep_lines = set(np.array(d.lines)[rng.permutation(len(d.lines))[:60]])
        li = [i for i, l in enumerate(d.lines) if l in keep_lines]
        m = np.isin(d.wing_line, list(keep_lines))
        sub = np.flatnonzero(m)
        sub = sub[rng.random(len(sub)) < 0.4]
        d = gm.WingData(d.V[sub], d.wing_line[sub], d.sex[sub], [d.lines[i] for i in li], d.K[np.ix_(li, li)],
                        d.Y[li], d.folds[li], d.consensus)
        folds = folds[:2]
    log.info("lines %d, wings %d, folds %s, cfg %s", len(d.lines), len(d.V), folds, cfg)
    t0 = time.time()
    results, full = _run_jobs(args.backend, d.to_payload(), folds, cfg, args)
    log.info("all jobs done in %.0fs", time.time() - t0)

    rec = pd.DataFrame([r for res in results for r in res["records"]])
    means = np.stack(rec.pop("gen_mean").to_numpy())
    rec.to_csv(out / "per_line_scores.csv", index=False)
    np.savez_compressed(out / "generated_line_means.npz", means=means, line=rec["line"].to_numpy(),
                        method=rec["method"].to_numpy(), split=rec["split"].to_numpy())
    rec["gen_mean"] = list(means)
    train_mean = {f: d.Y[d.folds != f].mean(axis=0) for f in folds}
    summ = gm.summarise(rec, d.Y, d.lines, train_mean)
    summ.to_csv(out / "summary_test_lines.csv", index=False)
    tr = rec[rec["split"] == "train"].groupby("method")[["energy", "mean_dist"]].mean()

    samples = {}
    for res in results:
        for line, s in res["samples"].items():
            samples.setdefault(line, {}).update(s)
    space, Vt = (space_summary(full, d) if full is not None else (None, None))
    if space is not None:
        space.to_csv(out / "possibility_space.csv", index=False)
        np.savez_compressed(out / "possibility_space_samples.npz", genotype_free=full["genotype_free"],
                            real_line_samples=full["real_line_samples"], lines=np.array(full["lines"]),
                            **{f"means_{k.split()[0]}": v for k, v in full["line_means"].items()})
    figs = plots(out, d, rec, full, Vt, samples)

    code_norms = {"train": float(np.mean([r["code_norm_train"] for r in results])),
                  "test": float(np.mean([r["code_norm_test"] for r in results]))}
    spread_rel = gm.spread_reliability(d, np.random.default_rng(args.seed))
    metrics = {"quick_test": bool(args.quick), "genotype_code_norm": code_norms, "spread_reliability": spread_rel, "backend": args.backend, "n_lines": len(d.lines), "n_wings": int(len(d.V)),
               "folds": folds, "config": cfg, "genotype_pcs": args.geno_pcs, "n_gen_per_line": args.n_gen,
               "n_shape_pcs": results[0]["n_shape_pcs"], "seconds": round(time.time() - t0),
               "summary_test_lines": summ.to_dict(orient="records"),
               "train_lines_mean": tr.reset_index().to_dict(orient="records"),
               "loss_curves": {f"fold{r['fold']}-{r['variant']}": r["loss_curve"] for r in results},
               "possibility_space": None if space is None else space.to_dict(orient="records")}
    (out / "diffusion_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    report(out, args, d, summ, tr, space, figs, metrics)
    log.info("wrote %s", out / "diffusion_report.md")


def report(out, args, d, summ, tr, space, figs, metrics) -> None:
    S = summ.set_index("method")
    L = [f"# DGRP wing shape: genotype-conditioned diffusion model{' (QUICK CPU TEST, not a result)' if args.quick else ''}", ""]
    L += [f"- Lines: {len(d.lines)}; wings: {len(d.V):,}; held-out folds: the step 2 related-grouped folds "
          f"({len(metrics['folds'])} folds, repeat 0).",
          f"- Model: MLP denoiser (width {args.width}, {args.depth} residual blocks) over {metrics['n_shape_pcs']} whitened "
          f"shape PCs (99.9% of variance), {args.steps:,} training steps per fold, cosine schedule, "
          f"{args.sample_steps} ancestral sampling steps.",
          f"- Genotype code: top {args.geno_pcs} genomic PCs of the training lines' GRM; held-out lines are projected onto "
          "them. The code is dropped for 15% of training wings, so the same network also gives a genotype-free model.",
          f"- {args.n_gen} wings generated per held-out line, sex-matched to its real wings.",
          f"- **Genotype codes of held-out lines are short.** Mean code length is {metrics['genotype_code_norm']['train']:.2f} "
          f"for training lines but {metrics['genotype_code_norm']['test']:.2f} for held-out lines: an unrelated line projects "
          "close to the centre of the training lines' genomic PCs, so the model sees it as 'a typical line'.",
          f"- **Spread ceiling.** Split-half reliability of each line's log within-line variance = "
          f"{metrics['spread_reliability']:.3f}: lines differ reliably in how variable their wings are.", "",
          "## Held-out lines", "",
          "Energy distance compares the whole generated distribution with the line's real wings (lower is better, 0 = "
          "indistinguishable). R² vs mean is the step 2 metric on the generated line mean. Spread is the total within-line "
          "variance; log ratio > 0 means too spread out. Rows are compared line by line with `mean+resid`; the "
          "`oracle-mean+` rows are compared with `oracle-mean+resid`, so they test spread alone.", "",
          "| method | energy ×10⁻³ (mean) | lines better than reference | Wilcoxon p | mean-shape R² vs mean | "
          "spread log ratio | spread r across lines | spread R² vs reference | W1 of distances ×10⁻³ |",
          "|---|---|---|---|---|---|---|---|---|"]
    order = ["mean+resid", "pooled", "diffusion-nogeno", "diffusion-geno", "diffusion-shuffled", "diffusion-resid-nogeno",
             "diffusion-resid-geno", "oracle-mean+resid", "oracle-mean+diffusion-resid-nogeno",
             "oracle-mean+diffusion-resid-geno"]
    for m in [o for o in order if o in S.index]:
        r = S.loc[m]
        L.append(f"| {m} | {_fmt(r.energy_mean * 1e3)} | {_fmt(r.lines_better_than_reference, 2)} | "
                 f"{_fmt(r.energy_wilcoxon_p, 4)} | {_fmt(r.mean_r2_vs_train_mean)} | {_fmt(r.spread_log_ratio)} | "
                 f"{_fmt(r.spread_r_across_lines, 2)} | {_fmt(r.spread_r2_vs_reference)} | {_fmt(r.w1_dist_mean * 1e3)} |")
    L += ["", "`mean+resid` is the baseline to beat (training mean shape plus real within-line scatter). "
          "`oracle-mean+resid` uses the held-out line's own observed mean and is not achievable; it shows how much a "
          "perfect mean prediction would be worth.", ""]
    if len(tr):
        L += ["## Training lines (in-sample check)", "",
              "The same scores on lines the model was trained on. A large gap between these and the held-out scores means "
              "the model can reproduce lines it has seen but the genotype code does not carry over to new lines.", "",
              "| method | energy ×10⁻³ | mean distance ×10⁻³ |", "|---|---|---|"]
        L += [f"| {m} | {_fmt(r.energy * 1e3)} | {_fmt(r.mean_dist * 1e3)} |" for m, r in tr.iterrows()]
        L.append("")
    if space is not None:
        L += ["## Possibility space sketch (model trained on all lines)", "",
              "Generated line means for real genotypes, random genotype codes (new lines from the same population), crosses "
              "(midpoints of two real lines' codes) and extrapolated codes (2× the population spread), against the observed "
              "line means. Spread is in the observed line-mean PC space (top 5 PCs).", "",
              "| set | lines | between-line variance ×10⁻⁶ | median within-line variance ×10⁻⁶ | inside observed 95% region | PC1 range ×10⁻³ |",
              "|---|---|---|---|---|---|"]
        for r in space.itertuples():
            L.append(f"| {r.set} | {r.n_lines} | {_fmt(r.between_line_var * 1e6, 2)} | {_fmt(r.within_line_var_median * 1e6, 2)} | "
                     f"{_fmt(r.share_inside_observed_95pct, 2)} | {_fmt(r.pc1_range * 1e3, 2)} |")
        L.append("")
    if figs:
        L += ["## Figures", ""] + [f"- `{f}`" for f in figs]
    (out / "diffusion_report.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m drosophila_wings.diffusion_cv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["local", "modal"], default="local")
    r.add_argument("--out", help="output folder (default <data-dir>/results/diffusion)")
    r.add_argument("--quick", action="store_true", help="60 lines, 40%% of wings, 2 folds: a CPU smoke test")
    r.add_argument("--steps", type=int, default=30_000)
    r.add_argument("--width", type=int, default=256)
    r.add_argument("--depth", type=int, default=4)
    r.add_argument("--cond-noise", type=float, default=0.3)
    r.add_argument("--sample-steps", type=int, default=250)
    r.add_argument("--geno-pcs", type=int, default=10)
    r.add_argument("--n-gen", type=int, default=512)
    r.add_argument("--no-space", dest="space", action="store_false", help="skip the all-lines possibility-space model")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    cmd_run(args)


if __name__ == "__main__":
    main()
