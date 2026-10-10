"""Genomic relationship matrix and relatedness-aware cross-validation folds.

Related lines must never sit on both sides of a train/test split: in the bacterial work, phylogenetic leakage inflated
every score. Lines whose normalised genomic relationship exceeds a threshold are joined (single linkage) into one group,
and whole groups are assigned to folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardise(G: np.ndarray) -> np.ndarray:
    """VanRaden-style standardisation of allele counts (lines x variants); missing calls become 0 after centring."""
    G = G.astype(float)
    G[G < 0] = np.nan
    p = np.nanmean(G, axis=0) / 2
    sd = np.sqrt(2 * p * (1 - p))
    Z = (G - 2 * p) / np.where(sd > 0, sd, 1)
    Z[np.isnan(Z)] = 0.0
    Z[:, sd == 0] = 0.0
    return Z


def grm(G: np.ndarray, block: int = 100_000) -> np.ndarray:
    """Genomic relationship matrix ZZ'/m, accumulated in variant blocks to bound memory."""
    n, m = G.shape
    K = np.zeros((n, n))
    for s in range(0, m, block):
        Z = standardise(G[:, s:s + block])
        K += Z @ Z.T
    return K / max(m, 1)


def normalised(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def related_groups(K: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph joining lines with normalised relationship > threshold. Returns group labels."""
    R = normalised(K)
    n = len(R)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(R, 1) > threshold)):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def top_pairs(K: np.ndarray, lines: list[str], n: int = 20) -> pd.DataFrame:
    R = normalised(K)
    iu = np.triu_indices(len(R), 1)
    order = np.argsort(R[iu])[::-1][:n]
    return pd.DataFrame({"line_a": [lines[i] for i in iu[0][order]], "line_b": [lines[j] for j in iu[1][order]],
                         "relationship": R[iu][order]})


def group_kfold(groups: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fold index per item; whole groups go to one fold, largest groups first into the currently smallest fold."""
    labels, sizes = np.unique(groups, return_counts=True)
    k = min(k, len(labels))
    order = rng.permutation(len(labels))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of_group = np.empty(len(labels), dtype=int)
    load = np.zeros(k, dtype=int)
    for g in order:
        f = int(np.flatnonzero(load == load.min())[rng.integers((load == load.min()).sum())])
        fold_of_group[g] = f
        load[f] += sizes[g]
    lookup = dict(zip(labels, fold_of_group))
    return np.array([lookup[g] for g in groups])
