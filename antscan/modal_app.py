"""Modal GPU entry point for the 3D diffusion step. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.

    python -m antscan.diffusion3d --config gpu --backend modal

The payload (about 200 volumes of 32^3 plus codes, ~30 MB) travels as a function argument; no volume is needed.
"""
from __future__ import annotations

import modal

app = modal.App("antscan-genome-diffusion")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "numpy>=1.26", "scipy>=1.11", "pandas>=2.0")
         .add_local_python_source("antscan"))


@app.function(image=image, gpu="T4", timeout=3 * 3600, retries=1)
def fold_remote(payload: dict, fold: int, cfg: dict, variant: str) -> dict:
    from antscan.diffusion3d import run_fold
    return run_fold(payload, fold, cfg, variant)
