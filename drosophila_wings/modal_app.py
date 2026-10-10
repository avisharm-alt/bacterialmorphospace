"""Modal GPU functions for step 3. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.

Used by `python -m drosophila_wings.diffusion_cv run --backend modal`; the data travel as function arguments (about 10 MB),
so no Modal volume is needed.
"""
from __future__ import annotations

import modal

app = modal.App("dgrp-wing-diffusion")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "numpy>=1.26", "scipy>=1.11", "pandas>=2.0")
         .add_local_python_source("drosophila_wings"))


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def fold_remote(payload: dict, fold: int, cfg: dict, n_gen: int, n_geno_pc: int, variant: str, seed: int) -> dict:
    from drosophila_wings.generative import run_fold
    return run_fold(payload, fold, cfg, n_gen=n_gen, n_geno_pc=n_geno_pc, variant=variant, seed=seed)


@app.function(image=image, gpu="T4", timeout=3600, retries=1)
def full_remote(payload: dict, cfg: dict, n_geno_pc: int, seed: int) -> dict:
    from drosophila_wings.generative import run_full
    return run_full(payload, cfg, n_geno_pc=n_geno_pc, seed=seed)
