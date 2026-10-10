"""Step 3: score a genotype-conditioned diffusion model of wing shape on held-out DGRP lines, and sample it (step 4 sketch).

For each outer fold (the same relatedness-grouped folds as the step 2 baseline) a diffusion model is trained on the wings
of the training lines and asked to generate wings for every held-out line from that line's genotype code alone. The
generated wings are compared with the line's real wings.

Methods compared on each held-out line (all sex-matched to the line's real wings):
  mean+resid          training mean shape + within-line residuals of random training wings (the baseline to beat: right
                      average spread, no genotype)
  pooled              random training wings (the population marginal: between-line spread counted as within-line)
  diffusion-nogeno    the diffusion model with the genotype code dropped (guidance w = 0)
  diffusion-geno      the diffusion model given the line's genotype code (w = 1)
  diffusion-shuffled  a second diffusion model trained with genotype codes shuffled across training lines (control)
  diffusion-resid-*   a third model trained on within-line residuals only, added to the training mean shape: tests whether
                      genotype predicts the scatter around a line's mean, with the mean held at the baseline's
  oracle-mean+...     the same residual generators (pooled residuals, or the residual model) added to the line's own
                      observed mean: not achievable, but compares the spread alone with the mean fixed at the truth

Scores per held-out line (tangent-space units, Euclidean = Procrustes distance):
  energy distance     unbiased energy distance between generated and real wings (0 = same distribution)
  mean                distance of the generated line mean to the real line mean; pooled into R² vs the training mean
  spread              total within-line variance (summed over coordinates, within sex) of generated vs real wings
  W1 of distances     1-Wasserstein distance between the distributions of each wing's distance to its line-sex mean
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- data

@dataclass
class WingData:
    V: np.ndarray          # wings x 2k tangent coordinates (sex- and lab-centred, from `prepare`)
    wing_line: np.ndarray  # line id per wing (str)
    sex: np.ndarray        # 1 = female, 0 = male, per wing
    lines: list[str]       # analysed lines, in GRM order
    K: np.ndarray          # GRM over `lines`
    Y: np.ndarray          # observed line means (lines x 2k), as used by the baseline
    folds: np.ndarray      # outer fold per line
    consensus: np.ndarray  # k x 2 consensus shape (for drawing)

    def to_payload(self) -> dict:
        return {"V": self.V.astype(np.float32), "wing_line": self.wing_line.astype(str), "sex": self.sex.astype(np.int8),
                "lines": np.array(self.lines), "K": self.K, "Y": self.Y, "folds": self.folds, "consensus": self.consensus}

    @classmethod
    def from_payload(cls, p: dict) -> "WingData":
        return cls(np.asarray(p["V"], dtype=np.float64), np.asarray(p["wing_line"]).astype(str), np.asarray(p["sex"]),
                   [str(x) for x in p["lines"]], p["K"], p["Y"], np.asarray(p["folds"]), p["consensus"])


def load(derived, results, folds_from: str | None = "per_line_errors.csv", outer_k: int = 5, related_threshold=0.1,
         seed: int = 20261010) -> WingData:
    """Wings and GRM from `derived/`; outer folds copied from the baseline's per-line file when it exists."""
    from pathlib import Path

    from . import relatedness
    derived, results = Path(derived), Path(results)
    z = np.load(derived / "wings_tangent.npz", allow_pickle=True)
    g = np.load(derived / "grm.npz")
    lm = pd.read_csv(derived / "line_means.csv", index_col="line", dtype={"line": str})
    glines = [str(x) for x in g["lines"]]
    lines = [l for l in glines if l in lm.index]
    pos = [glines.index(l) for l in lines]
    K = g["K"][np.ix_(pos, pos)]
    wl = np.array([str(x) for x in z["line"]])
    keep = np.isin(wl, lines)
    sex = (np.array([str(s) for s in z["sex"]])[keep] == "F").astype(np.int8)
    folds = None
    if folds_from and (results / folds_from).exists():
        pl = pd.read_csv(results / folds_from, dtype={"line": str})
        f = pl[pl["model"] == "mean"].set_index("line")["fold"]
        if set(lines) <= set(f.index):
            folds = f.loc[lines].to_numpy()
    if folds is None:
        groups = relatedness.related_groups(K, related_threshold)
        folds = relatedness.group_kfold(groups, outer_k, np.random.default_rng(seed))
    return WingData(z["V"][keep], wl[keep], sex, lines, K, lm.loc[lines].to_numpy(), folds, z["consensus"])


# ---------------------------------------------------------------- representations

class ShapeSpace:
    """PCA of training wings, whitened per PC; keeps enough PCs for `var_kept` of the variance."""

    def __init__(self, V: np.ndarray, var_kept: float = 0.999):
        self.mu = V.mean(axis=0)
        _, s, Vt = np.linalg.svd(V - self.mu, full_matrices=False)
        ev = s ** 2 / (len(V) - 1)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), var_kept)) + 1
        self.C, self.sd = Vt[:k], np.sqrt(ev[:k])

    def encode(self, V):
        return (V - self.mu) @ self.C.T / self.sd

    def decode(self, Z):
        return (Z * self.sd) @ self.C + self.mu


def genotype_codes(K: np.ndarray, train: np.ndarray, n_pc: int = 10) -> np.ndarray:
    """Genomic PCs of the training lines, projected out-of-sample for the rest (Nyström), standardised on train.

    For training lines this is U·sqrt(s) from the eigendecomposition of K_train; any other line gets K[line, train]·U/sqrt(s),
    which is the same map applied to its genotype.
    """
    s, U = np.linalg.eigh(K[np.ix_(train, train)])
    s, U = s[::-1][:n_pc], U[:, ::-1][:, :n_pc]
    codes = K[:, train] @ U / np.sqrt(np.clip(s, 1e-12, None))
    m, sd = codes[train].mean(axis=0), codes[train].std(axis=0)
    return (codes - m) / np.where(sd > 0, sd, 1)


# ---------------------------------------------------------------- per-line scores

def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Unbiased (U-statistic) energy distance 2E|a-b| - E|a-a'| - E|b-b'|; expectation 0 when A, B share a law."""
    def mean_pdist(X):
        n = len(X)
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.clip(sq[:, None] + sq[None] - 2 * X @ X.T, 0, None))
        return D.sum() / (n * (n - 1))
    sa, sb = (A ** 2).sum(1), (B ** 2).sum(1)
    cross = np.sqrt(np.clip(sa[:, None] + sb[None] - 2 * A @ B.T, 0, None)).mean()
    return float(2 * cross - mean_pdist(A) - mean_pdist(B))


def cell_summary(X: np.ndarray, sex: np.ndarray, sexes) -> tuple[np.ndarray, float, np.ndarray]:
    """(line mean weighting each sex equally, total within-sex variance, each wing's distance to its sex-cell mean)."""
    means, resid = [], np.empty_like(X)
    for s in sexes:
        m = sex == s
        mu = X[m].mean(axis=0)
        means.append(mu)
        resid[m] = X[m] - mu
    present = np.isin(sex, list(sexes))
    R = resid[present]
    return np.mean(means, axis=0), float((R ** 2).sum() / max(len(R) - len(sexes), 1)), np.sqrt((R ** 2).sum(axis=1))


def sex_matched(sex_obs: np.ndarray, n_gen: int) -> np.ndarray:
    """Sex labels for n_gen generated wings in the observed line's proportions (at least 8 per sex present)."""
    sexes, counts = np.unique(sex_obs, return_counts=True)
    n = np.maximum(np.round(n_gen * counts / counts.sum()).astype(int), 8)
    return np.repeat(sexes, n)


def score_line(Xg, sg, Xo, so) -> dict:
    sexes = np.unique(so)
    mg, vg, dg = cell_summary(Xg, sg, sexes)
    mo, vo, do = cell_summary(Xo, so, sexes)
    return {"energy": energy_distance(Xg, Xo), "mean_dist": float(np.linalg.norm(mg - mo)), "var_gen": vg, "var_obs": vo,
            "w1_dist": float(stats.wasserstein_distance(dg, do)), "gen_mean": mg}


def spread_reliability(d: WingData, rng: np.random.Generator, n_splits: int = 10) -> float:
    """Spearman-Brown split-half reliability of each line's log within-line variance: the ceiling for predicting spread."""
    li = {l: i for i, l in enumerate(d.lines)}
    wl = np.array([li[l] for l in d.wing_line])
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        for i in range(len(d.lines)):
            idx = rng.permutation(np.flatnonzero(wl == i))
            sexes = np.unique(d.sex[idx])
            h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
            if not all(np.isin(sexes, d.sex[h]).all() and len(h) > 2 * len(sexes) for h in (h1, h2)):
                continue
            a.append(np.log(cell_summary(d.V[h1], d.sex[h1], np.unique(d.sex[h1]))[1]))
            b.append(np.log(cell_summary(d.V[h2], d.sex[h2], np.unique(d.sex[h2]))[1]))
        r = np.corrcoef(a, b)[0, 1]
        rs.append(2 * r / (1 + r))
    return float(np.mean(rs))


# ---------------------------------------------------------------- one fold

def _residual_pool(d: WingData, idx_lines: np.ndarray):
    """Training wings and their residuals around their own line-sex cell mean."""
    m = np.isin(d.wing_line, [d.lines[i] for i in idx_lines])
    V, wl, sx = d.V[m], d.wing_line[m], d.sex[m]
    R = V.copy()
    for (_, _), idx in pd.DataFrame({"l": wl, "s": sx}).groupby(["l", "s"]).indices.items():
        R[idx] -= V[idx].mean(axis=0)
    return V, R, sx


def _draw(pool_sex: np.ndarray, sg: np.ndarray, rng) -> np.ndarray:
    out = np.empty(len(sg), dtype=int)
    for s in np.unique(sg):
        cand = np.flatnonzero(pool_sex == s)
        out[sg == s] = rng.choice(cand, size=(sg == s).sum(), replace=True)
    return out


VARIANTS = ("geno", "shuffled", "residual")


def run_fold(payload: dict, fold: int, cfg: dict, n_gen: int = 512, n_geno_pc: int = 10, variant: str = "geno",
             score_train: bool = True, device: str | None = None, seed: int = 0) -> dict:
    """Train on the lines outside `fold`, generate for every line, score. Returns plain dicts and arrays.

    variant  "geno": the model on whole wings (also scores the baselines and the genotype-free mode)
             "shuffled": the same, trained on genotype codes permuted across training lines (the control)
             "residual": the model on within-line residuals (wing minus its line-sex mean), so it can only learn how
                         genotype shapes the scatter around a line's mean. Generated residuals are added to the training
                         mean shape, and to the line's own observed mean (oracle) to compare spread alone.
    """
    from .diffusion import ConditionalDiffusion, DiffusionConfig
    assert variant in VARIANTS, variant
    d = WingData.from_payload(payload)
    rng = np.random.default_rng(seed + 1000 * fold + VARIANTS.index(variant))
    train = np.flatnonzero(d.folds != fold)
    test = np.flatnonzero(d.folds == fold)
    is_test = set(test.tolist())
    line_index = {l: i for i, l in enumerate(d.lines)}
    wing_li = np.array([line_index[l] for l in d.wing_line])
    tr_w = np.isin(wing_li, train)
    Vtr, Rtr, Str = _residual_pool(d, train)  # rows in the same order as d.V[tr_w]
    ymean = d.Y[train].mean(axis=0)

    Xtrain = Rtr if variant == "residual" else d.V[tr_w]
    space = ShapeSpace(Xtrain)
    codes = genotype_codes(d.K, train, n_geno_pc)
    train_codes = codes.copy()
    if variant == "shuffled":
        train_codes[train] = codes[train][rng.permutation(len(train))]
    c = DiffusionConfig(**cfg)
    c.seed = seed + fold
    model = ConditionalDiffusion(space.C.shape[0], n_geno_pc, c, device=device)
    model.fit(space.encode(Xtrain), train_codes[wing_li[tr_w]], d.sex[tr_w])

    score_lines = np.concatenate([test, train]) if score_train else test
    sg_by_line = {i: sex_matched(d.sex[wing_li == i], n_gen) for i in score_lines}
    G_all = np.concatenate([np.repeat(codes[i][None], len(sg_by_line[i]), axis=0) for i in score_lines])
    S_all = np.concatenate([sg_by_line[i] for i in score_lines])
    bounds = np.cumsum([0] + [len(sg_by_line[i]) for i in score_lines])
    gens = {"geno": {"diffusion-nogeno": 0.0, "diffusion-geno": 1.0}, "shuffled": {"diffusion-shuffled": 1.0},
            "residual": {"diffusion-resid-nogeno": 0.0, "diffusion-resid-geno": 1.0}}[variant]
    samples = {name: space.decode(model.sample(G_all, S_all, w=w, seed=seed + 7 * fold + int(10 * w)))
               for name, w in gens.items()}

    records, keep_samples = [], {}
    for j, i in enumerate(score_lines):
        Xo, so = d.V[wing_li == i], d.sex[wing_li == i]
        sg = sg_by_line[i]
        methods = {k: v[bounds[j]:bounds[j + 1]] for k, v in samples.items()}
        if variant == "residual":
            for k in list(methods):
                R = methods[k]
                methods[k] = ymean + R if i in is_test else d.Y[i] + R
                if i in is_test:
                    methods["oracle-mean+" + k] = d.Y[i] + R
        if variant == "geno" and i in is_test:
            pick = _draw(Str, sg, rng)
            methods["mean+resid"] = ymean + Rtr[pick]
            methods["oracle-mean+resid"] = d.Y[i] + Rtr[pick]
            methods["pooled"] = Vtr[_draw(Str, sg, rng)]
        for name, Xg in methods.items():
            r = score_line(Xg, sg, Xo, so)
            records.append({"line": d.lines[i], "fold": int(fold), "split": "test" if i in is_test else "train",
                            "method": name, **{k: v for k, v in r.items() if k != "gen_mean"},
                            "gen_mean": r["gen_mean"].astype(np.float32)})
        if i in is_test and j < 6:
            keep_samples[d.lines[i]] = {k: v[:128].astype(np.float32) for k, v in methods.items()}
    norms = np.linalg.norm(codes, axis=1)
    return {"fold": int(fold), "variant": variant, "records": records, "loss_curve": model.losses,
            "n_shape_pcs": int(space.C.shape[0]), "samples": keep_samples,
            "code_norm_train": float(norms[train].mean()), "code_norm_test": float(norms[test].mean())}


# ---------------------------------------------------------------- possibility space

def run_full(payload: dict, cfg: dict, n_geno_pc: int = 10, n_per_line: int = 256, n_synth: int = 400,
             n_per_synth: int = 128, device: str | None = None, seed: int = 0) -> dict:
    """Train on every line, then generate: each real line, synthetic lines with random genotype codes, crosses
    (midpoints of two real lines' codes), extrapolated codes (2x), and genotype-free wings."""
    from .diffusion import ConditionalDiffusion, DiffusionConfig
    d = WingData.from_payload(payload)
    rng = np.random.default_rng(seed)
    allx = np.arange(len(d.lines))
    line_index = {l: i for i, l in enumerate(d.lines)}
    wing_li = np.array([line_index[l] for l in d.wing_line])
    space = ShapeSpace(d.V)
    codes = genotype_codes(d.K, allx, n_geno_pc)
    c = DiffusionConfig(**cfg)
    c.seed = seed
    model = ConditionalDiffusion(space.C.shape[0], n_geno_pc, c, device=device)
    model.fit(space.encode(d.V), codes[wing_li], d.sex)

    pairs = np.array([rng.choice(len(d.lines), 2, replace=False) for _ in range(n_synth)])
    sets = {
        "real lines": (codes, n_per_line),
        "random genotypes": (rng.normal(size=(n_synth, n_geno_pc)), n_per_synth),
        "crosses": ((codes[pairs[:, 0]] + codes[pairs[:, 1]]) / 2, n_per_synth),
        "extrapolated (2x)": (2 * rng.normal(size=(n_synth, n_geno_pc)), n_per_synth),
    }
    out = {"codes": {}, "line_means": {}, "line_vars": {}, "loss_curve": model.losses, "lines": d.lines}
    for name, (C, n) in sets.items():
        sex = np.tile(np.repeat([0, 1], n // 2), len(C))
        G = np.repeat(C, n, axis=0)
        X = space.decode(model.sample(G, sex, w=1.0, seed=seed + len(name)))
        X = X.reshape(len(C), n, -1)
        s = sex.reshape(len(C), n)
        summ = [cell_summary(X[k], s[k], (0, 1)) for k in range(len(C))]
        out["codes"][name] = C
        out["line_means"][name] = np.stack([m for m, _, _ in summ]).astype(np.float32)
        out["line_vars"][name] = np.array([v for _, v, _ in summ])
        if name == "real lines":
            out["real_line_samples"] = X[:, :32].astype(np.float32)
    sex = np.repeat([0, 1], 2048)
    out["genotype_free"] = space.decode(model.sample(np.zeros((len(sex), n_geno_pc)), sex, w=0.0, seed=seed + 99)
                                        ).astype(np.float32)
    return out


# ---------------------------------------------------------------- summaries

def summarise(records: pd.DataFrame, Y: np.ndarray, lines: list[str], train_mean_by_fold: dict) -> pd.DataFrame:
    """Held-out-line summary per method. R² vs mean pools every test line, as in the baseline.

    Energy and spread are compared line by line with `mean+resid`, or with `oracle-mean+resid` for the oracle-mean rows
    (so those rows compare spread alone).
    """
    from .model import metrics
    t = records[records["split"] == "test"]
    li = {l: i for i, l in enumerate(lines)}
    rows = []
    for m, g in t.groupby("method", sort=False):
        ref = "oracle-mean+resid" if m.startswith("oracle-mean+") else "mean+resid"
        base = t[t["method"] == ref].set_index("line")
        g = g.set_index("line").loc[base.index]
        Yt = Y[[li[l] for l in g.index]]
        P = np.stack(g["gen_mean"].to_numpy())
        P0 = np.stack([train_mean_by_fold[f] for f in g["fold"]])
        mm = metrics(Yt, P, P0)
        lv_o, lv_g = np.log(g["var_obs"]), np.log(g["var_gen"])
        lv_0 = np.log(base["var_gen"].to_numpy())  # the reference's spread: average within-line spread of training lines
        e, e0 = g["energy"].to_numpy(), base["energy"].to_numpy()
        same = m == ref
        rows.append({
            "method": m, "reference": ref, "n_lines": len(g),
            "energy_mean": float(e.mean()), "energy_median": float(np.median(e)),
            "lines_better_than_reference": np.nan if same else float((e < e0).mean()),
            "energy_wilcoxon_p": np.nan if same else float(stats.wilcoxon(e, e0, alternative="less").pvalue),
            "mean_r2_vs_train_mean": mm["r2_vs_mean"], "mean_dist": mm["procrustes_dist_mean"],
            "spread_log_ratio": float((lv_g - lv_o).mean()), "spread_abs_log_ratio": float(np.abs(lv_g - lv_o).mean()),
            "spread_r_across_lines": float(np.corrcoef(lv_g, lv_o)[0, 1]) if lv_g.std() > 1e-9 else np.nan,
            "spread_r2_vs_reference": float(1 - ((lv_g - lv_o) ** 2).sum() / ((lv_0 - lv_o) ** 2).sum()),
            "w1_dist_mean": float(g["w1_dist"].mean()),
        })
    return pd.DataFrame(rows)
