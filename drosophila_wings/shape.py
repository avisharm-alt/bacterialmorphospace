"""From aligned wings to line-mean shapes, plus the split-half reliability that caps any line-mean predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def center_within(V: np.ndarray, meta: pd.DataFrame, by: list[str]) -> np.ndarray:
    """Subtract each `by` group's mean shape (e.g. sex, lab), then add back the grand mean."""
    by = [c for c in by if c in meta.columns]
    if not by:
        return V
    out = V.copy()
    for _, idx in meta.groupby(by, sort=False).indices.items():
        out[idx] -= V[idx].mean(axis=0)
    return out + V.mean(axis=0)


def remove_allometry(V: np.ndarray, log_cs: np.ndarray, meta: pd.DataFrame, by: list[str]) -> np.ndarray:
    """Remove the pooled within-group regression of shape on log centroid size (common allometric slope)."""
    by = [c for c in by if c in meta.columns]
    x = log_cs.astype(float).copy()
    Vc = V - V.mean(axis=0)
    if by:
        Vc = center_within(V, meta, by) - V.mean(axis=0)
        for _, idx in meta.groupby(by, sort=False).indices.items():
            x[idx] -= x[idx].mean()
    else:
        x -= x.mean()
    slope = (x @ Vc) / (x @ x)
    return V - np.outer(x, slope)


def line_means(V: np.ndarray, meta: pd.DataFrame, cells: list[str], min_wings: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Line means that weight each cell (e.g. each sex) equally. Lines with fewer than `min_wings` wings are dropped.

    Returns (line x coordinate frame, wings per line).
    """
    cells = [c for c in cells if c in meta.columns]
    keys = ["line"] + cells
    df = pd.DataFrame(V, index=pd.MultiIndex.from_frame(meta[keys]) if cells else meta["line"].to_numpy())
    cell_means = df.groupby(level=list(range(len(keys)))).mean()
    lm = cell_means.groupby(level=0).mean() if cells else cell_means
    n = meta.groupby("line").size()
    keep = n[n >= min_wings].index
    lm = lm.loc[lm.index.isin(keep)].sort_index()
    return lm, n.loc[lm.index]


def split_half_reliability(V: np.ndarray, meta: pd.DataFrame, cells: list[str], rng: np.random.Generator,
                           lines: list[str], n_splits: int = 20) -> float:
    """Spearman-Brown corrected multivariate split-half reliability of line means.

    This is the share of between-line variance in observed line means that is not sampling noise, i.e. the R^2 a perfect
    genotype-to-mean predictor would reach on these data.
    """
    sub = meta["line"].isin(lines).to_numpy()
    V, meta = V[sub], meta.loc[sub].reset_index(drop=True)
    cells = [c for c in cells if c in meta.columns]
    rs = []
    for _ in range(n_splits):
        half = np.zeros(len(meta), dtype=bool)
        for _, idx in meta.groupby(["line"] + cells, sort=False).indices.items():
            pick = rng.permutation(idx)[: len(idx) // 2]
            half[pick] = True
        a, _ = line_means(V[half], meta.loc[half].reset_index(drop=True), cells, 1)
        b, _ = line_means(V[~half], meta.loc[~half].reset_index(drop=True), cells, 1)
        common = a.index.intersection(b.index)
        A = a.loc[common].to_numpy() - a.loc[common].to_numpy().mean(axis=0)
        B = b.loc[common].to_numpy() - b.loc[common].to_numpy().mean(axis=0)
        rs.append((A * B).sum() / np.sqrt((A ** 2).sum() * (B ** 2).sum()))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)
