"""Genomic ridge (GBLUP-style kernel ridge) prediction of line-mean shape, with nested group cross-validation.

Models, all predicting every tangent coordinate of the line-mean shape at once with one shared penalty:
  mean        the training-fold mean shape (the baseline to beat)
  covariates  OLS on known covariates (e.g. Wolbachia, inversion karyotypes)
  snp         intercept + kernel ridge on the genomic relationship matrix (equivalent to ridge on all SNPs)
  snp+cov     covariates as fixed effects, kernel ridge on the residual

Fixed effects are fitted by OLS first and the ridge is fitted to their residuals (a two-stage approximation to GBLUP's
joint mixed-model solution). The ridge penalty is picked by inner group cross-validation inside each training fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .relatedness import group_kfold

LAMBDA_GRID = np.logspace(-3, 3, 25)  # multiples of mean(diag(K_train))


def _fixed(F_tr: np.ndarray, Y_tr: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(F_tr, Y_tr, rcond=None)
    return beta


def _ridge_path(K_tr: np.ndarray, R_tr: np.ndarray, K_te_tr: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Predictions (n_lambda, n_test, p) of kernel ridge for every penalty, from one eigendecomposition."""
    s, U = np.linalg.eigh(K_tr)
    s = np.clip(s, 0, None)
    UtR = U.T @ R_tr
    A = K_te_tr @ U
    return np.stack([A @ (UtR / (s + lam)[:, None]) for lam in lambdas])


def fit_predict(Y: np.ndarray, train: np.ndarray, test: np.ndarray, K: np.ndarray | None, F: np.ndarray | None,
                lambdas: np.ndarray) -> np.ndarray:
    """Predictions for `test` under each penalty (n_lambda, n_test, p); a single slice when there is no kernel."""
    F = np.ones((len(Y), 1)) if F is None else np.column_stack([np.ones(len(Y)), F])
    beta = _fixed(F[train], Y[train])
    base = F[test] @ beta
    if K is None:
        return base[None]
    R = Y[train] - F[train] @ beta
    scale = np.mean(np.diag(K[np.ix_(train, train)]))
    return base[None] + _ridge_path(K[np.ix_(train, train)], R, K[np.ix_(test, train)], lambdas * scale)


def choose_lambda(Y, train, K, F, groups, inner_k, rng, lambdas=LAMBDA_GRID) -> float:
    folds = group_kfold(groups[train], inner_k, rng)
    sse = np.zeros(len(lambdas))
    for f in np.unique(folds):
        tr, te = train[folds != f], train[folds == f]
        P = fit_predict(Y, tr, te, K, F, lambdas)
        sse += ((P - Y[te][None]) ** 2).sum(axis=(1, 2))
    return float(lambdas[int(np.argmin(sse))])


@dataclass
class CVResult:
    pred: dict[str, np.ndarray]  # model -> (n_lines, p) out-of-fold predictions
    lambdas: list[float] = field(default_factory=list)
    folds: np.ndarray | None = None


def cross_validate(Y: np.ndarray, K: np.ndarray, groups: np.ndarray, rng: np.random.Generator, F: np.ndarray | None = None,
                   outer_k: int = 5, inner_k: int = 5, models: tuple[str, ...] = ("mean", "covariates", "snp", "snp+cov"),
                   folds: np.ndarray | None = None) -> CVResult:
    """Outer group k-fold; whole relatedness groups are held out together. Inner group CV picks the ridge penalty."""
    n = len(Y)
    if F is None or F.shape[1] == 0:
        F = None
        models = tuple(m for m in models if m in ("mean", "snp"))
    folds = group_kfold(groups, outer_k, rng) if folds is None else folds
    pred = {m: np.full(Y.shape, np.nan) for m in models}
    lams = []
    for f in np.unique(folds):
        train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        if "mean" in pred:
            pred["mean"][test] = fit_predict(Y, train, test, None, None, LAMBDA_GRID)[0]
        if "covariates" in pred:
            pred["covariates"][test] = fit_predict(Y, train, test, None, F, LAMBDA_GRID)[0]
        for m, Fm in (("snp", None), ("snp+cov", F)):
            if m in pred:
                lam = choose_lambda(Y, train, K, Fm, groups, inner_k, rng)
                lams.append(lam)
                pred[m][test] = fit_predict(Y, train, test, K, Fm, np.array([lam]))[0]
    return CVResult(pred, lams, folds)


def metrics(Y: np.ndarray, P: np.ndarray, P_mean: np.ndarray, n_pcs: int = 5) -> dict:
    """Out-of-sample accuracy of P against the mean-shape baseline P_mean, on line-mean tangent coordinates."""
    d = np.sqrt(((P - Y) ** 2).sum(axis=1))
    d0 = np.sqrt(((P_mean - Y) ** 2).sum(axis=1))
    out = {
        "r2_vs_mean": float(1 - (d ** 2).sum() / (d0 ** 2).sum()),
        "procrustes_dist_mean": float(d.mean()),
        "baseline_procrustes_dist_mean": float(d0.mean()),
        "share_lines_improved": float((d < d0).mean()),
    }
    if not np.allclose(d, d0):
        out["wilcoxon_p"] = float(stats.wilcoxon(d, d0, alternative="less").pvalue)
    Yc = Y - Y.mean(axis=0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    ys, ps = Yc @ Vt[:n_pcs].T, (P - Y.mean(axis=0)) @ Vt[:n_pcs].T
    out["pc_r"] = [float(np.corrcoef(ys[:, i], ps[:, i])[0, 1]) if ps[:, i].std() > 0 else float("nan")
                   for i in range(min(n_pcs, Vt.shape[0]))]
    return out


def permutation_null(Y: np.ndarray, K: np.ndarray, groups: np.ndarray, folds: np.ndarray, rng: np.random.Generator,
                     n_perm: int, inner_k: int = 5) -> np.ndarray:
    """R^2 vs mean of the snp model after shuffling which genotype belongs to which line (same outer folds)."""
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(Y))
        Kp = K[np.ix_(perm, perm)]
        res = cross_validate(Y, Kp, groups[perm], rng, None, folds=folds, inner_k=inner_k, models=("mean", "snp"))
        null[i] = metrics(Y, res.pred["snp"], res.pred["mean"])["r2_vs_mean"]
    return null
