"""Does a genome predict the 3D shape of an ant from a genus the model has never seen?

Every model is kernel ridge regression from a species code to the species' mean worker shape descriptors, so the
only thing that differs between them is the code:

    mean           no information (training mean)
    subfamily      taxonomy only (training mean of the same subfamily)
    tree           Brownian-motion covariance from the GAGA dated phylogeny (the family tree alone, no DNA)
    genome         gene-family copy-number profile (GAGA orthogroups), linear kernel
    genome+tree    the two kernels mixed
    genome-shuffled  the genome kernel with species labels permuted (null; repeated)

Folds hold out whole genera (leave-one-genus-out). Ridge strength and mixing weight are picked by an inner
leave-one-genus-out loop on the training species only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# shape descriptor blocks; each block gets equal total weight in the pooled score
BLOCKS = {
    "size": ("log_size", "core_log_size"),
    "proportions": ("axis_ratio", "core_axis_ratio", "sphericity"),
    "distance_distribution": ("d2", "radial"),
    "angle_distribution": ("a3",),
    "long_axis_profile": ("long_profile",),
    "body_profile": ("core_profile", "core_width"),
    "thickness": ("thickness", "granulometry", "thick_quantiles"),
}
LAMBDAS = np.logspace(-3, 3, 13)
MIX = np.linspace(0.0, 1.0, 6)


def normalise_kernel(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / d[:, None] / d[None, :]


def linear_kernel(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    Xc = Xc / np.clip(Xc.std(0), 1e-9, None)
    return normalise_kernel(Xc @ Xc.T / X.shape[1] + 1e-9 * np.eye(len(X)))


def krr_predict(K: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, lam: float) -> np.ndarray:
    mu = Y[tr].mean(0)
    Ktr = K[np.ix_(tr, tr)]
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(tr)), Y[tr] - mu)
    return mu + K[np.ix_(te, tr)] @ alpha


def _sse(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float((((pred - y) ** 2) * w).sum())


def inner_select(Ks: list[np.ndarray], Y, w, tr, groups) -> tuple[int, float]:
    """Pick (kernel index, lambda) by leave-one-genus-out within the training species.

    One eigendecomposition per inner fold serves every lambda.
    """
    g = groups[tr]
    ug = np.unique(g)
    sse = np.zeros((len(Ks), len(LAMBDAS)))
    for ki, K in enumerate(Ks):
        for gg in ug:
            te_i, tr_i = tr[g == gg], tr[g != gg]
            mu = Y[tr_i].mean(0)
            evals, V = np.linalg.eigh(K[np.ix_(tr_i, tr_i)])
            B = K[np.ix_(te_i, tr_i)] @ V
            R = V.T @ (Y[tr_i] - mu)
            for li, lam in enumerate(LAMBDAS):
                pred = mu + B @ (R / (evals + lam)[:, None])
                sse[ki, li] += _sse(pred, Y[te_i], w)
    ki, li = np.unravel_index(np.argmin(sse), sse.shape)
    return int(ki), float(LAMBDAS[li])


def cv_predict(Ks: list[np.ndarray], Y, w, groups) -> tuple[np.ndarray, list]:
    """Outer leave-one-genus-out predictions; `Ks` is the menu of kernels the inner loop may choose from."""
    pred = np.zeros_like(Y)
    picks = []
    for gg in np.unique(groups):
        te = np.flatnonzero(groups == gg)
        tr = np.flatnonzero(groups != gg)
        ki, lam = inner_select(Ks, Y, w, tr, groups)
        pred[te] = krr_predict(Ks[ki], Y, tr, te, lam)
        picks.append((gg, ki, lam))
    return pred, picks


def mean_predict(Y, groups) -> np.ndarray:
    pred = np.zeros_like(Y)
    for gg in np.unique(groups):
        te, tr = groups == gg, groups != gg
        pred[te] = Y[tr].mean(0)
    return pred


def group_mean_predict(Y, groups, labels) -> np.ndarray:
    """Training mean of species sharing `labels` (e.g. subfamily); falls back to the overall training mean."""
    pred = np.zeros_like(Y)
    for gg in np.unique(groups):
        te, tr = np.flatnonzero(groups == gg), np.flatnonzero(groups != gg)
        mu = Y[tr].mean(0)
        for i in te:
            same = tr[labels[tr] == labels[i]]
            pred[i] = Y[same].mean(0) if len(same) else mu
    return pred


def r2(pred, Y, w, base) -> float:
    """1 - SSE(model) / SSE(training-mean model), pooled with block weights."""
    return 1 - _sse(pred, Y, w) / _sse(base, Y, w)


def block_r2(pred, Y, base, cols: list[str]) -> dict[str, float]:
    out = {}
    for b, keys in BLOCKS.items():
        m = np.array([any(c.startswith(k + "_") for k in keys) and not
                      any(c.startswith(k2 + "_") for k2 in _longer(keys)) for c in cols])
        if m.any():
            out[b] = 1 - float(((pred[:, m] - Y[:, m]) ** 2).sum() / ((base[:, m] - Y[:, m]) ** 2).sum())
    return out


def _longer(keys):
    # "core_profile" must not be swallowed by "profile"-like prefixes; block keys are already exact, so none
    return ()


def block_weights(cols: list[str]) -> np.ndarray:
    """Per-column weight so that every block contributes equally (columns are standardised beforehand)."""
    w = np.zeros(len(cols))
    for keys in BLOCKS.values():
        m = np.array([any(c.startswith(k + "_") for k in keys) for c in cols])
        if m.any():
            w[m] = 1.0 / m.sum()
    return w / w.sum()


def per_species_errors(pred, Y, w) -> np.ndarray:
    return (((pred - Y) ** 2) * w).sum(1)
