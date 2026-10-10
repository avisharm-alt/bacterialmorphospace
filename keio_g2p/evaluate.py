"""Held-out-gene test: does a gene's sequence (or annotation) predict the cell shape of its knockout?

    python -m keio_g2p.evaluate <raw_dir> <esm.npz> <out_dir> <homolog_pairs.tsv>

Each held-out fold removes whole chromosomal blocks of genes (so operon partners, which share polar effects and
function, never straddle train and test) together with all their homologs (MMseqs2 all-vs-all, E < 1e-5, both
sequences at least 50% covered). Scores are pooled over out-of-fold predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from . import data

BLOCK = 30          # consecutive genes (by b-number) per chromosomal block
N_FOLDS = 10
N_PERM = 100
SEED = 0


def heldout_splits(accessions: list[str], pairs: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds of whole chromosomal blocks (genes are sorted by b-number); every homolog of a test gene is then purged
    from that fold's training set. (Merging homologs into the groups instead chains the whole genome into one group,
    because large families such as ABC transporters are spread around the chromosome.)"""
    n = len(accessions)
    idx = {a: i for i, a in enumerate(accessions)}
    hom: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(pairs[0], pairs[1]):
        if a in idx and b in idx:
            hom[idx[a]].add(idx[b]); hom[idx[b]].add(idx[a])
    block = np.arange(n) // BLOCK
    out = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(np.zeros(n), groups=block):
        bad = set().union(*(hom[i] for i in te))
        out.append((np.array([i for i in tr if i not in bad]), te))
    return out


def _ridge(Xtr, Ytr, Xte, n_pc=None):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if n_pc:
        p = PCA(n_components=min(n_pc, Xtr.shape[1]), random_state=SEED).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Xtr, Ytr)
    return m.predict(Xte)


def _knn(Xtr, Ytr, Xte, k=10):
    sc = StandardScaler().fit(Xtr)
    return KNeighborsRegressor(n_neighbors=k, metric="cosine").fit(sc.transform(Xtr), Ytr).predict(sc.transform(Xte))


def _logit(Xtr, ytr, Xte, n_pc=None):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if n_pc:
        p = PCA(n_components=min(n_pc, Xtr.shape[1]), random_state=SEED).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    m = LogisticRegression(C=0.01, max_iter=2000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def oof(X, Y, hit, splits, kind):
    """Out-of-fold shape predictions and shape-mutant probabilities."""
    P = np.zeros_like(Y)
    p = np.zeros(len(Y))
    for tr, te in splits:
        if kind == "mean":
            P[te] = Y[tr].mean(0)
            p[te] = hit[tr].mean()
            continue
        if kind == "knn":
            P[te] = _knn(X[tr], Y[tr], X[te])
        else:
            P[te] = _ridge(X[tr], Y[tr], X[te], n_pc=64 if kind == "ridge_pc" else None)
        p[te] = _logit(X[tr], hit[tr], X[te], n_pc=64 if kind == "ridge_pc" else None)
    return P, p


def score(Y, P, hit, p):
    r2 = float(np.mean([r2_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))
    # Shape mutants only: does the predicted change point the right way? (cosine of predicted vs true score vectors)
    H = hit.astype(bool)
    cos = np.sum(Y[H] * P[H], 1) / (np.linalg.norm(Y[H], axis=1) * np.linalg.norm(P[H], axis=1) + 1e-12)
    r2_hit = float(np.mean([r2_score(Y[H, j], P[H, j]) for j in range(Y.shape[1])]))
    auc = float(roc_auc_score(hit, p)) if p.std() > 0 else 0.5
    return {"r2_all": r2, "r2_mutants": r2_hit, "cos_mutants": float(np.mean(cos)), "auroc_mutant": auc}


def main(raw: str, esm: str, out: str, homologs: str) -> None:
    raw_p, out_p = Path(raw), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    k = data.load(raw_p)
    Y, hit = k.Y, k.genes["hit"].to_numpy().astype(int)
    E = data.load_esm(Path(esm), k.genes["accession"].tolist())
    G_all, _ = data.go_matrix(raw_p, k.genes, drop_phenotype=False)
    G_seq, _ = data.go_matrix(raw_p, k.genes, drop_phenotype=True)
    grouped = heldout_splits(k.genes["accession"].tolist(), pd.read_csv(homologs, sep="\t", header=None))
    random = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(E))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Y))

    # Genomic neighbour: predict a knockout from the two genes either side of it. Only meaningful (and only possible)
    # under the random split; it is the operon-leakage the grouped split is built to remove.
    pos_rank = np.arange(len(Y))
    nb = np.stack([np.roll(pos_rank, s) for s in (-2, -1, 1, 2)], 1)

    rows = []

    def run(name, X, kind, splits, split_name):
        P, p = oof(X, Y, hit, splits, kind)
        r = {"model": name, "split": split_name, **score(Y, P, hit, p)}
        rows.append(r)
        print(r, flush=True)
        return P, p

    for split_name, splits in [("held-out gene blocks", grouped), ("random genes (leaky)", random)]:
        run("training mean", E, "mean", splits, split_name)
        run("ESM-2 ridge", E, "ridge_pc", splits, split_name)
        run("ESM-2 nearest genes", E, "knn", splits, split_name)
        run("ESM-2 shuffled", E[perm], "ridge_pc", splits, split_name)
        run("GO ridge", G_all, "ridge", splits, split_name)
        run("GO without phenotype evidence", G_seq, "ridge", splits, split_name)
        run("GO + ESM-2", np.hstack([G_seq, PCA(64, random_state=SEED).fit_transform(StandardScaler().fit_transform(E))]),
            "ridge", splits, split_name)
    # neighbour baseline under the random split
    P = np.zeros_like(Y); p = np.zeros(len(Y))
    for tr, te in random:
        trs = set(tr)
        for i in te:
            js = [j for j in nb[i] if j in trs]
            P[i] = Y[js].mean(0) if js else Y[tr].mean(0)
            p[i] = hit[js].mean() if js else hit[tr].mean()
    rows.append({"model": "genomic neighbours", "split": "random genes (leaky)", **score(Y, P, hit, p)})
    print(rows[-1])

    # Permutation null for the main contrast (GO + ESM-2 ridge, grouped): shuffle which gene has which embedding.
    obs = next(r for r in rows if r["model"] == "GO + ESM-2" and r["split"] == "held-out gene blocks")
    null = {"r2_all": [], "auroc_mutant": []}
    Xc = np.hstack([G_seq, PCA(64, random_state=SEED).fit_transform(StandardScaler().fit_transform(E))])
    for b in range(N_PERM):
        q = rng.permutation(len(Y))
        Pn, pn = oof(Xc[q], Y, hit, grouped, "ridge")
        s = score(Y, Pn, hit, pn)
        null["r2_all"].append(s["r2_all"]); null["auroc_mutant"].append(s["auroc_mutant"])
    perm_p = {m: float((1 + np.sum(np.array(v) >= obs[m])) / (1 + N_PERM)) for m, v in null.items()}

    res = pd.DataFrame(rows)
    res.to_csv(out_p / "heldout_gene_scores.csv", index=False)
    meta = {"n_genes": int(len(Y)), "n_mutants": int(hit.sum()),
            "homologs_purged_per_fold": [int(len(Y) - len(te) - len(tr)) for tr, te in grouped],
            "features": data.SHAPE, "block": BLOCK, "homologs": "MMseqs2 E<1e-5, coverage>=0.5", "folds": N_FOLDS,
            "go_terms_all": int(G_all.shape[1]), "go_terms_no_phenotype": int(G_seq.shape[1]),
            "permutation": {"model": "GO + ESM-2", "n": N_PERM, "p": perm_p,
                            "null_mean": {m: float(np.mean(v)) for m, v in null.items()},
                            "null_95": {m: float(np.quantile(v, 0.95)) for m, v in null.items()}}}
    (out_p / "heldout_gene_meta.json").write_text(json.dumps(meta, indent=2))
    print(res.to_string(index=False))
    print(json.dumps(meta["permutation"], indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
