"""Can a held-out gene's knockout morphology be predicted from what is known about the gene?

    python -m jump_g2p.predictability <raw_dir> <homolog_pairs.tsv> <out_dir>

Folds hold out whole chromosome arms (CRISPR cuts cause arm-level 'proximity bias', so genes on one arm look alike
for technical reasons), and every paralog of a test gene is purged from training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import data

N_FOLDS = 10
SHRINK = 10.0


def splits_for(genes: pd.DataFrame, homologs: pd.DataFrame):
    groups = genes["arm"].replace("", "unknown").to_numpy()
    idx = {g: i for i, g in enumerate(genes["gene_id"])}
    hom = [set() for _ in range(len(genes))]
    for a, b in zip(homologs[0].astype(str), homologs[1].astype(str)):
        if a in idx and b in idx:
            hom[idx[a]].add(idx[b]); hom[idx[b]].add(idx[a])
    out = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(genes)), groups=groups):
        bad = set().union(*(hom[i] for i in te))
        out.append((np.array([i for i in tr if i not in bad]), te))
    return out


def ridge(Xtr, Ytr, Xte, n_pc=64):
    sc = StandardScaler().fit(Xtr)
    p = PCA(min(n_pc, Xtr.shape[1]), random_state=0).fit(sc.transform(Xtr))
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(p.transform(sc.transform(Xtr)), Ytr)
    return m.predict(p.transform(sc.transform(Xte)))


def neighbours(M, Y, tr, te, shrink=SHRINK):
    mask = np.zeros(len(Y)); mask[tr] = 1
    Mt = M[te] @ sparse.diags(mask)
    w = np.asarray(Mt.sum(1)).ravel()
    return (Mt @ Y + shrink * Y[tr].mean(0)) / (w + shrink)[:, None]


def metrics(Y, P, rel):
    r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    cos = (Y * P).sum(1) / (np.linalg.norm(Y, axis=1) * np.linalg.norm(P, axis=1) + 1e-9)
    act = rel > np.quantile(rel, 0.75)
    return {"r2_mean_over_features": float(r2.mean()), "r2_top50_features": float(np.sort(r2)[-50:].mean()),
            "cosine_all": float(cos.mean()), "cosine_reproducible_quartile": float(cos[act].mean())}


def main(raw: str, homologs: str, out: str) -> None:
    raw_p, out_p = Path(raw), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    jd = data.load(raw_p)
    Y, g = jd.Y, jd.genes
    print(f"{len(g)} genes; median reliability {g['reliability'].median():.2f}", flush=True)
    splits = splits_for(g, pd.read_csv(homologs, sep="\t", header=None))
    D, has = data.depmap(raw_p, g["gene_id"].tolist())
    M = data.string_matrix(raw_p, g["symbol"].tolist())
    print(f"DepMap covers {has.sum()} genes; STRING links {M.nnz // 2}", flush=True)
    rel = g["reliability"].to_numpy()
    res = {}
    preds = {k: np.zeros_like(Y) for k in ("mean", "depmap", "string", "both")}
    rng = np.random.default_rng(0)
    q = rng.permutation(len(Y))
    preds["depmap_shuffled"] = np.zeros_like(Y)
    for tr, te in splits:
        preds["mean"][te] = Y[tr].mean(0)
        preds["depmap"][te] = ridge(D[tr], Y[tr], D[te])
        preds["depmap_shuffled"][te] = ridge(D[q][tr], Y[tr], D[q][te])
        preds["string"][te] = neighbours(M, Y, tr, te)
        preds["both"][te] = (preds["depmap"][te] + preds["string"][te]) / 2
    for k, P in preds.items():
        res[k] = metrics(Y, P, rel)
        print(k, res[k], flush=True)
    np.savez_compressed(out_p / "predicted_profiles.npz", **preds, Y=Y)
    g.to_csv(out_p / "genes.csv", index=False)
    (out_p / "predictability.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
