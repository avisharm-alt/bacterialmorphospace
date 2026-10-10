"""Train the outline diffusion model for one fold and sample the held-out genes (runs on Modal or locally)."""
from __future__ import annotations

import numpy as np


def aux_matrix(plate: np.ndarray, time: np.ndarray, n_plates: int) -> np.ndarray:
    """Imaging nuisance covariates: plate one-hot and elapsed imaging time (z). A plate of -1 means 'average plate'."""
    A = np.zeros((len(plate), n_plates + 1), dtype=np.float32)
    known = plate >= 0
    A[np.flatnonzero(known), plate[known]] = 1.0
    A[~known, :n_plates] = 1.0 / n_plates
    A[:, -1] = time
    return A


def run_fold(p: dict, cfg: dict) -> dict:
    from .diffusion import ConditionalDiffusion, DiffusionConfig
    c = DiffusionConfig(**{"width": 256, "depth": 4, "steps": 20_000, "batch": 2048, **cfg})
    A = aux_matrix(p["plate"], p["time"], p["n_plates"])
    m = ConditionalDiffusion(p["X"].shape[1], p["code"].shape[1], c, a_dim=A.shape[1]).fit(p["X"], p["code"], A)
    n = p["n_gen"]
    At = np.repeat(aux_matrix(p["test_plate"], p["test_time"], p["n_plates"]), n, axis=0)
    G = np.repeat(p["test_code"], n, axis=0)
    Gs = np.repeat(p["test_code_shuffled"], n, axis=0)
    return {"free": m.sample(G, At, w=0.0, seed=1), "cond": m.sample(G, At, w=1.0, seed=2),
            "shuffled": m.sample(Gs, At, w=1.0, seed=3)}
