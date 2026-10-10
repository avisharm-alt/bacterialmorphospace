"""Modal GPU functions for jump_g2p. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET."""
from __future__ import annotations

import modal

app = modal.App("jump-profile-diffusion")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "numpy>=1.26", "scipy>=1.11", "pandas>=2.0", "scikit-learn>=1.4")
         .add_local_python_source("keio_g2p", "jump_g2p"))


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def fold_remote(payload: dict, cfg: dict) -> dict:
    from jump_g2p.model import run_fold
    return run_fold(payload, cfg)


@app.function(image=image, gpu="T4", timeout=7200)
def train_full_remote(payload: dict, cfg: dict) -> bytes:
    """Train on every gene; return the EMA network and config for jump_g2p.predict."""
    import io

    import numpy as np
    import torch

    from keio_g2p.diffusion import ConditionalDiffusion, DiffusionConfig
    c = DiffusionConfig(**{"width": 512, "depth": 4, "steps": 15_000, "batch": 1024, "cond_noise": 0.05, **cfg})
    m = ConditionalDiffusion(payload["X"].shape[1], payload["code"].shape[1], c, a_dim=1)
    m.fit(payload["X"], payload["code"], np.zeros((len(payload["X"]), 1), dtype=np.float32))
    buf = io.BytesIO()
    torch.save({"ema": m.ema.state_dict(), "cfg": c.to_dict(), "x_dim": payload["X"].shape[1],
                "g_dim": payload["code"].shape[1]}, buf)
    return buf.getvalue()
