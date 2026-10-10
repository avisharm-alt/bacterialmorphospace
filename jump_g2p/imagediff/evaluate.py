"""Score generated Cell Painting crops against real crops of held-out genes.

    python -m jump_g2p.imagediff.evaluate <sel_dir> <profile_results_dir>

For each held-out gene: energy distance between its generated and real crops, and the distance between generated and
real mean features, in two feature spaces (DINOv2 per-channel, 50 whitened PCs fitted on real training crops; and
42 CellProfiler-like features, z-scored on real training crops). Lower is better. Variants from one network:
cond (w = 1), free (gene code dropped), shuffled (another held-out gene's code), guided2 (w = 2).
Also: across held-out genes, the correlation between each gene's real and generated shift from the fold mean
(does the model move the right features in the right direction?). Two real references: half of a gene's wells
against the other half (the floor), and real crops of the other held-out genes (a perfectly realistic generator
that ignores the gene).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

from .features import cp_names

VARIANTS = ("cond", "free", "shuffled", "guided2")
N_REAL = 512


def energy_distance(a, b):
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def spaces(z):
    ref_d = z["real_train_ref_dino"].astype(np.float32)
    pca = PCA(50, whiten=True, random_state=0).fit(ref_d)
    ref_c = z["real_train_ref_cp"]
    mu, sd = ref_c.mean(0), ref_c.std(0) + 1e-6
    out = {}
    for name in ("real_test",) + VARIANTS:
        out[("dino", name)] = pca.transform(z[f"{name}_dino"].astype(np.float32))
        out[("cp", name)] = (z[f"{name}_cp"] - mu) / sd
    return out


def score_fold(z, genes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    S = spaces(z)
    rng = np.random.default_rng(0)
    tg, n_gen = z["test_genes"], int(z["n_gen"])
    rg, rw = z["real_test_gene"], z["real_test_well"]
    rows, shifts = [], {sp: {v: [] for v in ("real",) + VARIANTS} for sp in ("dino", "cp")}
    for j, g in enumerate(tg):
        idx = np.where(rg == g)[0]
        wells = np.unique(rw[idx])
        row = {"symbol": genes.at[g, "symbol"], "gene_index": int(g), "reliability": genes.at[g, "reliability"],
               "n_real_crops": len(idx), "n_wells": len(wells), "fold": int(z["fold"])}
        sub = rng.choice(idx, min(N_REAL, len(idx)), replace=False)
        half = set(rng.permutation(wells)[: len(wells) // 2])
        ia = np.array([i for i in idx if rw[i] in half]); ib = np.array([i for i in idx if rw[i] not in half])
        for sp in ("dino", "cp"):
            real = S[(sp, "real_test")]
            r = real[sub]
            shifts[sp]["real"].append(real[idx].mean(0))
            for v in VARIANTS:
                s = S[(sp, v)][j * n_gen:(j + 1) * n_gen]
                row[f"{sp}_ed_{v}"] = energy_distance(s, r)
                row[f"{sp}_mean_err_{v}"] = float(np.linalg.norm(s.mean(0) - r.mean(0)))
                shifts[sp][v].append(s.mean(0))
            # gene-blind real reference: real crops of the other held-out genes in this fold
            other = rng.choice(np.where(rg != g)[0], n_gen, replace=False)
            row[f"{sp}_ed_real_other_genes"] = energy_distance(real[other], r)
            if len(ia) and len(ib):
                a = real[rng.choice(ia, min(n_gen, len(ia)), replace=False)]
                row[f"{sp}_ed_real_half"] = energy_distance(a, real[rng.choice(ib, min(N_REAL, len(ib)), replace=False)])
        rows.append(row)
    return pd.DataFrame(rows), {sp: {v: np.stack(x) for v, x in d.items()} for sp, d in shifts.items()}


def shift_correlation(sh: dict) -> dict:
    """Per feature, Pearson r across genes between the real and generated shift from the fold mean."""
    real = sh["real"] - sh["real"].mean(0)
    out = {}
    for v in VARIANTS:
        gen = sh[v] - sh[v].mean(0)
        r = (real * gen).sum(0) / (np.linalg.norm(real, axis=0) * np.linalg.norm(gen, axis=0) + 1e-9)
        cos = (real * gen).sum(1) / (np.linalg.norm(real, axis=1) * np.linalg.norm(gen, axis=1) + 1e-9)
        out[v] = {"per_feature_r": r, "per_gene_cosine": cos}
    return out


def main(sel: str, prof: str) -> None:
    sel_p = Path(sel)
    genes = pd.read_csv(sel_p / "genes.csv")
    dfs, sh_all = [], {"dino": {}, "cp": {}}
    for f in sorted((sel_p / "folds").glob("fold*.npz")):
        z = np.load(f, allow_pickle=True)
        df, sh = score_fold(z, genes)
        dfs.append(df)
        for sp in sh:
            for v, x in sh[sp].items():   # centre within fold, then pool
                sh_all[sp].setdefault(v, []).append(x - x.mean(0))
    df = pd.concat(dfs, ignore_index=True)
    sh_all = {sp: {v: np.concatenate(x) for v, x in d.items()} for sp, d in sh_all.items()}
    out = sel_p / "results"
    out.mkdir(exist_ok=True)
    summ = {"n_genes": int(len(df)), "folds": sorted(df["fold"].unique().tolist()), "spaces": {}}
    for sp in ("dino", "cp"):
        s = {}
        for m in ("ed", "mean_err"):
            for v in VARIANTS:
                s[f"{m}_{v}"] = float(df[f"{sp}_{m}_{v}"].mean())
            for a, b in [("cond", "free"), ("cond", "shuffled"), ("guided2", "free"), ("guided2", "cond")]:
                d = df[f"{sp}_{m}_{a}"] - df[f"{sp}_{m}_{b}"]
                s[f"{m}_{a}_better_than_{b}_frac"] = float((d < 0).mean())
                s[f"{m}_p_{a}_vs_{b}"] = float(wilcoxon(d, alternative="less").pvalue)
        s["ed_real_half"] = float(df[f"{sp}_ed_real_half"].mean())
        s["ed_real_other_genes"] = float(df[f"{sp}_ed_real_other_genes"].mean())
        sc = shift_correlation(sh_all[sp])
        for v in VARIANTS:
            s[f"shift_cosine_{v}"] = float(np.mean(sc[v]["per_gene_cosine"]))
            s[f"shift_feature_r_median_{v}"] = float(np.median(sc[v]["per_feature_r"]))
        cmp = sc["cond"]["per_gene_cosine"] - sc["shuffled"]["per_gene_cosine"]
        s["shift_cosine_p_cond_vs_shuffled"] = float(wilcoxon(cmp, alternative="greater").pvalue)
        if sp == "cp":
            s["shift_feature_r"] = {v: dict(zip(cp_names(), np.round(sc[v]["per_feature_r"], 3).tolist()))
                                    for v in ("cond", "guided2", "shuffled")}
        summ["spaces"][sp] = s
    # the profile-level model on the same genes, for reference
    pp = Path(prof) / "diffusion_per_gene.csv"
    if pp.exists():
        p = pd.read_csv(pp)
        p = p[p["symbol"].isin(df["symbol"])]
        summ["profile_model_same_genes"] = {
            "n": int(len(p)), "ed_cond": float(p["ed_cond"].mean()), "ed_free": float(p["ed_free"].mean()),
            "ed_shuffled": float(p["ed_shuffled"].mean()),
            "p_cond_vs_free": float(wilcoxon(p["ed_cond"] - p["ed_free"], alternative="less").pvalue),
            "cond_better_frac": float((p["ed_cond"] < p["ed_free"]).mean())}
    df.to_csv(out / "per_gene.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
