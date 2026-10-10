"""Gene-informed genotype codes for the step 3 diffusion model (replacing whole-genome PCs).

A code is a line's *predicted* wing phenotype from a chosen set of SNPs: kernel ridge from those SNPs to 11 targets (the
top 10 PCs of training line-mean shapes, and the log within-line variance). The model is fitted on training lines only.
Training lines get out-of-fold (inner group CV) predictions, so the diffusion model is not trained on overfit codes;
held-out lines get predictions from the fit on all training lines. Codes are standardised on the training lines.

SNP sets:
  wing-genes    SNPs within ±5 kb of the 91 wing-development genes knocked down in Pitchers et al. 2019 (File S5).
                Chosen from developmental biology, not from these data, so not circular.
  fold-gwas     the top N SNPs by multivariate association with the targets, re-selected inside every training set
                (outer and inner), so held-out lines never inform the selection.
  pitchers-hits the 2,396 MANOVA-significant SNPs of Pitchers et al. 2019 (File S3). They were found using all DGRP lines,
                held-out ones included, so this code is LEAKY: a reference for how much circular selection inflates scores.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from . import model, relatedness

CODE_SETS = ("wing-genes", "fold-gwas", "pitchers-hits")


# ---------------------------------------------------------------- SNP sets

def wing_gene_names(file_s5: Path) -> list[str]:
    x = pd.read_excel(file_s5, header=None)
    names = []
    for v in x.iloc[4:, 0].dropna().astype(str):
        if v.startswith("Key to"):
            break
        names.append(v.strip())
    return names


def gene_spans(annotation_dir: Path, names: list[str]) -> pd.DataFrame:
    """chrom/start/end (dm3 = FlyBase r5) spanning all transcripts of each named gene."""
    with gzip.open(annotation_dir / "flyBase2004Xref.txt.gz", "rt") as f:
        xref = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1], names=["tx", "symbol"], quoting=3)
    with gzip.open(annotation_dir / "flyBaseGene.txt.gz", "rt") as f:
        genes = pd.read_csv(f, sep="\t", header=None, usecols=[1, 2, 4, 5], names=["tx", "chrom", "start", "end"])
    m = genes.merge(xref, on="tx")
    m = m[m["symbol"].isin(names)]
    m["chrom"] = m["chrom"].str.replace("chr", "", regex=False)
    return m.groupby(["symbol", "chrom"], as_index=False).agg(start=("start", "min"), end=("end", "max"))


def snps_near(chrom: np.ndarray, pos: np.ndarray, spans: pd.DataFrame, flank: int = 5000) -> np.ndarray:
    keep = np.zeros(len(pos), dtype=bool)
    for r in spans.itertuples():
        keep |= (chrom == r.chrom) & (pos >= r.start - flank) & (pos <= r.end + flank)
    return np.flatnonzero(keep)


def pitchers_hits(file_s3: Path, chrom: np.ndarray, pos: np.ndarray) -> np.ndarray:
    y = pd.read_excel(file_s3, sheet_name=0, header=3)
    y = y.dropna(subset=["FB5"])
    arm = y["Chrom. Segment"].astype(str).str.extract(r"^(X|2L|2R|3L|3R|4)")[0]
    key = set(zip(arm, y["FB5"].astype(int)))
    return np.flatnonzero([(c, p) in key for c, p in zip(chrom, pos)])


# ---------------------------------------------------------------- targets and codes

def targets(Y: np.ndarray, logvar: np.ndarray, train: np.ndarray, n_pc: int = 10) -> np.ndarray:
    """Training-fitted top PCs of line means plus standardised log within-line variance, for every line."""
    mu = Y[train].mean(axis=0)
    _, _, Vt = np.linalg.svd(Y[train] - mu, full_matrices=False)
    S = (Y - mu) @ Vt[:n_pc].T
    S /= S[train].std(axis=0)
    v = (logvar - logvar[train].mean()) / logvar[train].std()
    return np.column_stack([S, v])


def assoc_scores(G: np.ndarray, T: np.ndarray, idx: np.ndarray, block: int = 200_000) -> np.ndarray:
    """Per-SNP sum over targets of squared correlation, on lines `idx`."""
    Tc = T[idx] - T[idx].mean(axis=0)
    Tc /= np.linalg.norm(Tc, axis=0)
    out = np.empty(G.shape[1], dtype=np.float32)
    for s in range(0, G.shape[1], block):
        Z = relatedness.standardise(G[idx, s:s + block]).astype(np.float32)
        Z -= Z.mean(axis=0)
        n = np.linalg.norm(Z, axis=0)
        n[n == 0] = 1
        out[s:s + block] = (((Z / n).T @ Tc) ** 2).sum(axis=1)
    return out


def _kernel(G: np.ndarray, snps: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Linear kernel on the chosen SNPs, standardised with allele frequencies of `ref` lines."""
    X = G[:, snps].astype(float)
    X[X < 0] = np.nan
    p = np.nanmean(X[ref], axis=0) / 2
    sd = np.sqrt(2 * p * (1 - p))
    ok = sd > 0
    Z = (X[:, ok] - 2 * p[ok]) / sd[ok]
    Z[np.isnan(Z)] = 0
    return Z @ Z.T / max(ok.sum(), 1)


def _fit_predict(G, snps_fn, T, groups, train, test, inner_k, rng) -> np.ndarray:
    """Kernel ridge on the SNPs that `snps_fn(train)` picks; penalty by inner group CV. Predictions for `test`."""
    snps = snps_fn(train)
    K = _kernel(G, snps, train)
    lam = model.choose_lambda(T, train, K, None, groups, inner_k, rng)
    return model.fit_predict(T, train, test, K, None, np.array([lam]))[0]


def fold_codes(G: np.ndarray, T: np.ndarray, groups: np.ndarray, train: np.ndarray, snp_set: str, fixed_snps=None,
               n_top: int = 2000, inner_k: int = 5, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Codes for every line under one outer fold: inner out-of-fold predictions for train, full-train fit elsewhere.

    Returns (standardised codes, raw predictions on the target scale)."""
    rng = rng or np.random.default_rng(0)
    if snp_set == "fold-gwas":
        def snps_fn(idx):
            return np.argsort(assoc_scores(G, T, idx))[::-1][:n_top]
    else:
        def snps_fn(idx):
            return fixed_snps
    n = len(T)
    codes = np.full(T.shape, np.nan)
    rest = np.setdiff1d(np.arange(n), train)
    codes[rest] = _fit_predict(G, snps_fn, T, groups, train, rest, inner_k, rng)
    inner = relatedness.group_kfold(groups[train], inner_k, rng)
    for f in np.unique(inner):
        tr, te = train[inner != f], train[inner == f]
        codes[te] = _fit_predict(G, snps_fn, T, groups, tr, te, inner_k, rng)
    m, sd = codes[train].mean(axis=0), codes[train].std(axis=0)
    return (codes - m) / np.where(sd > 0, sd, 1), codes


def code_check(T: np.ndarray, codes_raw: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """How well the (unstandardised) code predicts held-out targets: R² vs training mean on shape PCs and on spread."""
    def r2(a, p, base):
        return float(1 - ((a - p) ** 2).sum() / ((a - base) ** 2).sum())
    base = T[train].mean(axis=0)
    return {"shape_pc_r2": r2(T[test, :-1], codes_raw[test, :-1], base[:-1]),
            "spread_r2": r2(T[test, -1], codes_raw[test, -1], base[-1]),
            "spread_r": float(np.corrcoef(T[test, -1], codes_raw[test, -1])[0, 1])}
