"""Train the profile diffusion model for one fold and sample the held-out genes (Modal or local)."""
from __future__ import annotations

import numpy as np


def run_fold(p: dict, cfg: dict) -> dict:
    from keio_g2p.diffusion import ConditionalDiffusion, DiffusionConfig
    c = DiffusionConfig(**{"width": 512, "depth": 4, "steps": 15_000, "batch": 1024, "cond_noise": 0.05, **cfg})
    A = np.zeros((len(p["X"]), 1), dtype=np.float32)
    m = ConditionalDiffusion(p["X"].shape[1], p["code"].shape[1], c, a_dim=1).fit(p["X"], p["code"], A)
    n = p["n_gen"]
    G = np.repeat(p["test_code"], n, axis=0)
    Gs = np.repeat(p["test_code_shuffled"], n, axis=0)
    At = np.zeros((len(G), 1), dtype=np.float32)
    return {"free": m.sample(G, At, w=0.0, seed=1), "cond": m.sample(G, At, w=1.0, seed=2),
            "shuffled": m.sample(Gs, At, w=1.0, seed=3)}
