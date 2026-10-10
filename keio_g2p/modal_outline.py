"""Modal GPU function for the outline diffusion folds. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET."""
from __future__ import annotations

import modal

app = modal.App("keio-outline-diffusion")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "numpy>=1.26", "scipy>=1.11", "pandas>=2.0", "scikit-learn>=1.4")
         .add_local_python_source("keio_g2p"))


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def fold_remote(payload: dict, cfg: dict) -> dict:
    from keio_g2p.outline_model import run_fold
    return run_fold(payload, cfg)


@app.function(image=image, gpu="T4", timeout=7200)
def train_full_remote(payload: dict, cfg: dict) -> bytes:
    """Train on every gene and return the model state (for the predict tool)."""
    import io

    import torch

    from keio_g2p.diffusion import ConditionalDiffusion, DiffusionConfig
    from keio_g2p.outline_model import aux_matrix
    c = DiffusionConfig(**{"width": 256, "depth": 4, "steps": 20_000, "batch": 2048, **cfg})
    A = aux_matrix(payload["plate"], payload["time"], payload["n_plates"])
    m = ConditionalDiffusion(payload["X"].shape[1], payload["code"].shape[1], c, a_dim=A.shape[1])
    m.fit(payload["X"], payload["code"], A)
    buf = io.BytesIO()
    torch.save({"ema": m.ema.state_dict(), "cfg": c.to_dict(), "x_dim": payload["X"].shape[1],
                "g_dim": payload["code"].shape[1], "a_dim": A.shape[1]}, buf)
    return buf.getvalue()
