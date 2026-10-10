"""Gene-network inputs for the held-out-gene test: STRING links and RB-TnSeq fitness profiles.

    python -m keio_g2p.network <raw_dir> <esm.npz> <out_dir> <homolog_pairs.tsv>

raw_dir must also hold string/511145.protein.links.detailed.v12.0.txt.gz (STRING v12, E. coli K-12) and
fitness/keio_gene_fitness.tsv.gz + fitness/keio_genes.tsv (Fitness Browser, organism "Keio" = BW25113 RB-TnSeq).

Two ways to use a network, both on the same folds as evaluate.py:
- features: a spectral embedding of the STRING graph (unsupervised, built once from links only) or the gene's fitness
  profile across conditions, fed to the same ridge/logistic models;
- guilt by association: a held-out gene's shape is predicted as the link-weighted average of its *training*
  neighbours' shapes (test genes' own labels are never used).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import data
from .evaluate import N_PERM, SEED, heldout_splits, oof, score

CHANNELS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database", "textmining"]


def string_matrix(raw: Path, bnums: list[str], channels: list[str], min_score: int = 400) -> sparse.csr_matrix:
    """Gene x gene weights (0..1) recombined from the chosen STRING channels with STRING's own noisy-OR rule
    (each channel's prior of 0.041 removed first), keeping links scoring at least `min_score`/1000."""
    d = pd.read_csv(raw / "string/511145.protein.links.detailed.v12.0.txt.gz", sep=" ")
    idx = {b: i for i, b in enumerate(bnums)}
    a = d["protein1"].str.split(".").str[1].map(idx)
    b = d["protein2"].str.split(".").str[1].map(idx)
    keep = a.notna() & b.notna()
    prior = 0.041
    p_none = np.ones(keep.sum())
    for ch in channels:
        s = d.loc[keep, ch].to_numpy() / 1000
        s = np.clip((s - prior) / (1 - prior), 0, 1)
        p_none *= 1 - s
    w = (1 - p_none) * (1 - prior) + prior
    w[w < min_score / 1000] = 0
    n = len(bnums)
    M = sparse.coo_matrix((w, (a[keep].astype(int), b[keep].astype(int))), shape=(n, n)).tocsr()
    M = M.maximum(M.T)
    M.eliminate_zeros()
    return M


def spectral(M: sparse.csr_matrix, k: int = 64) -> np.ndarray:
    """Top eigenvectors of the symmetric-normalised adjacency (isolated genes get zeros)."""
    deg = np.asarray(M.sum(1)).ravel()
    dinv = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1e-12)), 0)
    A = sparse.diags(dinv) @ M @ sparse.diags(dinv)
    vals, vecs = eigsh(A, k=k, which="LA")
    return vecs[:, np.argsort(-vals)].astype(np.float32)


def fitness_matrix(raw: Path, bnums: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Gene x experiment fitness (genes without data get zeros) and a has-data mask."""
    f = pd.read_csv(raw / "fitness/keio_gene_fitness.tsv.gz", sep="\t")
    g = pd.read_csv(raw / "fitness/keio_genes.tsv", sep="\t")
    loc2b = dict(zip(g["locusId"].astype(str), g["sysName"].astype(str)))
    f["bnum"] = f["locusId"].astype(str).map(loc2b)
    W = f.pivot_table(index="bnum", columns="expName", values="fit")
    W = W.reindex(bnums)
    has = W.notna().any(axis=1).to_numpy()
    return W.fillna(0).to_numpy(dtype=np.float32), has


def neighbour_oof(M: sparse.csr_matrix, Y: np.ndarray, hit: np.ndarray, splits) -> tuple[np.ndarray, np.ndarray]:
    P = np.zeros_like(Y)
    p = np.zeros(len(Y))
    for tr, te in splits:
        mask = np.zeros(len(Y)); mask[tr] = 1
        Mt = M[te] @ sparse.diags(mask)
        w = np.asarray(Mt.sum(1)).ravel()
        num = Mt @ Y
        nh = Mt @ hit
        ok = w > 0
        P[te] = Y[tr].mean(0); p[te] = hit[tr].mean()
        P[te[ok]] = num[ok] / w[ok, None]
        # shrink toward the base rate when the neighbourhood is thin
        p[te[ok]] = (nh[ok] + hit[tr].mean()) / (w[ok] + 1)
    return P, p


def main(raw: str, esm: str, out: str, homologs: str) -> None:
    raw_p, out_p = Path(raw), Path(out)
    k = data.load(raw_p)
    Y, hit = k.Y, k.genes["hit"].to_numpy().astype(int)
    bnums = k.genes["bnum"].tolist()
    splits = heldout_splits(k.genes["accession"].tolist(), pd.read_csv(homologs, sep="\t", header=None))
    E = data.load_esm(Path(esm), k.genes["accession"].tolist())
    G_seq, _ = data.go_matrix(raw_p, k.genes, drop_phenotype=True)
    E64 = PCA(64, random_state=SEED).fit_transform(StandardScaler().fit_transform(E))

    nets = {
        "STRING all channels": string_matrix(raw_p, bnums, CHANNELS),
        "STRING without text mining": string_matrix(raw_p, bnums, [c for c in CHANNELS if c != "textmining"]),
        "STRING without text mining or databases": string_matrix(
            raw_p, bnums, [c for c in CHANNELS if c not in ("textmining", "database")]),
    }
    F, has_fit = fitness_matrix(raw_p, bnums)
    Fz = StandardScaler().fit_transform(F)
    C = np.corrcoef(Fz[has_fit]) if has_fit.sum() else None
    cof = np.zeros((len(Y), len(Y)), dtype=np.float32)
    if C is not None:
        hi = np.flatnonzero(has_fit)
        C = np.where(C > 0.6, C, 0); np.fill_diagonal(C, 0)
        cof[np.ix_(hi, hi)] = C
    nets["cofitness (r > 0.6)"] = sparse.csr_matrix(cof)

    rows = []
    rng = np.random.default_rng(SEED)
    split_name = "held-out gene blocks"

    def add(name, P, p):
        r = {"model": name, "split": split_name, **score(Y, P, hit, p)}
        rows.append(r); print(r, flush=True)

    for name, M in nets.items():
        add(f"{name}: neighbour average", *neighbour_oof(M, Y, hit, splits))
        q = rng.permutation(len(Y))
        add(f"{name}: neighbour average, shuffled genes", *neighbour_oof(M[q][:, q], Y, hit, splits))
    S = {name: spectral(M) for name, M in nets.items() if name.startswith("STRING")}
    S_clean = S["STRING without text mining or databases"]
    add("STRING (no text/db) spectral ridge", *oof(S_clean, Y, hit, splits, "ridge"))
    add("fitness profile ridge", *oof(F, Y, hit, splits, "ridge_pc"))
    combo = np.hstack([G_seq, E64, S_clean, PCA(64, random_state=SEED).fit_transform(Fz)])
    add("GO + ESM-2 + STRING (no text/db) + fitness ridge", *oof(combo, Y, hit, splits, "ridge"))

    # Permutation null for the best network input on AUROC and R², shuffling gene <-> network.
    best = max((r for r in rows if "shuffled" not in r["model"]), key=lambda r: r["auroc_mutant"])
    print("best:", best["model"], flush=True)
    null = {"r2_all": [], "auroc_mutant": []}
    bname = best["model"]
    for _ in range(N_PERM):
        q = rng.permutation(len(Y))
        if bname.endswith("neighbour average"):
            M = nets[bname.split(":")[0]]
            Pn, pn = neighbour_oof(M[q][:, q], Y, hit, splits)
        elif bname.startswith("GO + ESM-2"):
            Pn, pn = oof(combo[q], Y, hit, splits, "ridge")
        elif bname.startswith("fitness"):
            Pn, pn = oof(F[q], Y, hit, splits, "ridge_pc")
        else:
            Pn, pn = oof(S_clean[q], Y, hit, splits, "ridge")
        s = score(Y, Pn, hit, pn)
        null["r2_all"].append(s["r2_all"]); null["auroc_mutant"].append(s["auroc_mutant"])
    perm = {"model": bname, "n": N_PERM,
            "p": {m: float((1 + np.sum(np.array(v) >= best[m])) / (1 + N_PERM)) for m, v in null.items()},
            "null_95": {m: float(np.quantile(v, 0.95)) for m, v in null.items()}}
    res = pd.DataFrame(rows)
    res.to_csv(out_p / "network_scores.csv", index=False)
    meta = {"genes_with_fitness": int(has_fit.sum()), "fitness_experiments": int(F.shape[1]),
            "string_links": {n: int(M.nnz // 2) for n, M in nets.items()},
            "genes_with_links": {n: int((np.asarray(M.sum(1)).ravel() > 0).sum()) for n, M in nets.items()},
            "permutation": perm}
    (out_p / "network_meta.json").write_text(json.dumps(meta, indent=2))
    print(res.to_string(index=False)); print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
