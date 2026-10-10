"""Driver for the image diffusion experiment.

    python -m jump_g2p.imagediff.run prep  <raw_dir> <sel_dir> [--plates=N]
    python -m jump_g2p.imagediff.run train <sel_dir> [--folds=0,1,2,3,4] [--steps=N]
    python -m jump_g2p.imagediff.run collect <sel_dir>

<sel_dir> is the output of jump_g2p.imagediff.select (manifest.csv, codes.npz). `train` spawns one detached Modal
call per fold; `collect` downloads each fold<k>.npz of real and generated crop features (plus example crops) into
<sel_dir>/folds/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def prep(raw: str, sel: str, n_plates: int | None = None) -> None:
    import modal

    from .modal_img import app, prep_plate
    man = pd.read_csv(Path(sel) / "manifest.csv")
    pl = pd.read_csv(Path(raw) / "plate.csv.gz")
    batch = dict(zip(pl["Metadata_Plate"], pl["Metadata_Batch"]))
    args = []
    for p, d in man.groupby("plate"):
        args.append((p, batch[p], d.loc[d["gene_index"] >= 0, "well"].tolist(), d.loc[d["gene_index"] < 0, "well"].tolist()))
    args = args[:n_plates] if n_plates else args
    with modal.enable_output(), app.run():
        for r in prep_plate.starmap(args, return_exceptions=True):
            print(r, flush=True)


def train(sel: str, folds: list[int], cfg: dict) -> None:
    import modal
    sel_p = Path(sel)
    man = pd.read_csv(sel_p / "manifest.csv")
    rows = man[man["gene_index"] >= 0][["plate", "well", "gene_index"]].to_dict("records")
    z = np.load(sel_p / "codes.npz")
    chosen = z["chosen"]
    out = sel_p / "folds"
    out.mkdir(exist_ok=True)
    calls = []
    for k in folds:
        trg = chosen[z["train"][k][chosen]]
        teg = chosen[z["test"][k][chosen]]
        calls.append((k, rows, z["codes"][k], trg, teg, cfg))
    # the app must be deployed first (modal deploy jump_g2p/imagediff/modal_img.py); calls run detached and
    # write <volume>/results/fold<k>.npz, which `collect` downloads
    f = modal.Function.from_name("jump-image-diffusion", "train_fold")
    for c in calls:
        print("spawned fold", c[0], f.spawn(*c).object_id, flush=True)


def collect(sel: str) -> None:
    import subprocess
    out = Path(sel) / "folds"
    out.mkdir(exist_ok=True)
    for k in range(5):
        subprocess.run(["modal", "volume", "get", "--force", "jump-image-crops", f"results/fold{k}.npz", str(out / f"fold{k}.npz")])


if __name__ == "__main__":
    cmd, a = sys.argv[1], [x for x in sys.argv[2:] if not x.startswith("--")]
    opt = dict(x[2:].split("=", 1) for x in sys.argv[2:] if x.startswith("--"))
    if cmd == "prep":
        prep(a[0], a[1], int(opt["plates"]) if "plates" in opt else None)
    elif cmd == "train":
        cfg = {"steps": int(opt["steps"])} if "steps" in opt else {}
        train(a[0], [int(x) for x in opt.get("folds", "0,1,2,3,4").split(",")], cfg)
    elif cmd == "collect":
        collect(a[0])
