"""Permutation test and per-feature R² for the fitness-profile ridge (the one input with positive shape R²).

    python -m keio_g2p.perm_fitness <raw_dir> <homolog_pairs.tsv> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from . import data, network
from .evaluate import N_PERM, heldout_splits, oof, score


def main(raw: str, homologs: str, out: str) -> None:
    raw_p = Path(raw)
    k = data.load(raw_p)
    Y, hit = k.Y, k.genes["hit"].to_numpy().astype(int)
    splits = heldout_splits(k.genes["accession"].tolist(), pd.read_csv(homologs, sep="\t", header=None))
    F, _ = network.fitness_matrix(raw_p, k.genes["bnum"].tolist())
    P, p = oof(F, Y, hit, splits, "ridge_pc")
    obs = score(Y, P, hit, p)
    rng = np.random.default_rng(1)
    null = []
    for _ in range(N_PERM):
        q = rng.permutation(len(Y))
        Pn, pn = oof(F[q], Y, hit, splits, "ridge_pc")
        null.append(score(Y, Pn, hit, pn))
    res = {"observed": obs,
           "per_feature_r2": {f: float(r2_score(Y[:, j], P[:, j])) for j, f in enumerate(data.SHAPE)}}
    for m in ("r2_all", "cos_mutants", "auroc_mutant"):
        v = np.array([s[m] for s in null])
        res[f"p_{m}"] = float((1 + (v >= obs[m]).sum()) / (1 + N_PERM))
        res[f"null95_{m}"] = float(np.quantile(v, 0.95))
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
