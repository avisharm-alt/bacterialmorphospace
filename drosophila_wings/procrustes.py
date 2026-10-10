"""Generalised Procrustes analysis for 2-D landmark configurations.

Configurations are arrays of shape (n, k, 2). Reflections are never allowed. Semilandmarks are treated as fixed
landmarks (no sliding).
"""
from __future__ import annotations

import numpy as np


def centroid_size(X: np.ndarray) -> np.ndarray:
    """Square root of the summed squared distances of the landmarks from their centroid, per configuration."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.sqrt((Xc ** 2).sum(axis=(1, 2)))


def _rotate_onto(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate each centred configuration in X onto ref (k, 2) by orthogonal Procrustes, without reflection."""
    M = np.einsum("nki,kj->nij", X, ref)  # X_i^T ref, (n, 2, 2)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(X), 2))
    D[:, 1] = d
    R = (U * D[:, None, :]) @ Vt
    return X @ R


def gpa(X: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align configurations (n, k, 2). Returns (aligned unit-size shapes, centroid sizes, consensus of unit size)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[2] != 2:
        raise ValueError(f"expected (n, k, 2) landmarks, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("landmarks contain NaN or inf; drop incomplete wings first")
    cs = centroid_size(X)
    if (cs <= 0).any():
        raise ValueError("degenerate configuration with zero centroid size")
    Y = (X - X.mean(axis=1, keepdims=True)) / cs[:, None, None]
    ref = Y[0]
    for _ in range(max_iter):
        Y = _rotate_onto(Y, ref)
        new = Y.mean(axis=0)
        new /= np.sqrt((new ** 2).sum())
        if ((new - ref) ** 2).sum() < tol:
            ref = new
            break
        ref = new
    Y = _rotate_onto(Y, ref)
    return Y, cs, ref


def tangent(Y: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    """Orthogonal projection of aligned unit-size shapes onto the tangent space at the consensus; flat (n, 2k)."""
    u = consensus.reshape(-1)
    u = u / np.linalg.norm(u)
    V = Y.reshape(len(Y), -1)
    return V - np.outer(V @ u, u) + u
