"""Modal functions for the image diffusion model: fetch and crop JUMP images, train per fold, sample, featurise.

Crops live in the Modal volume "jump-image-crops" (one .npy per plate), so the ~150 GB of raw TIFFs are read from
the public cellpainting-gallery bucket once, inside Modal, and never pass through the sandbox.
"""
from __future__ import annotations

import modal

app = modal.App("jump-image-diffusion")
vol = modal.Volume.from_name("jump-image-crops", create_if_missing=True)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy>=1.26", "tifffile>=2024.1"))
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.5.1", "numpy>=1.26", "timm==1.0.11", "huggingface_hub==0.26.2")
             .run_commands("python -c \"import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)\"")
             .add_local_python_source("jump_g2p"))

BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"
FIELDS = (1, 3, 7, 9)   # 4 of the 9 fields per well
TILE, DOWN = 128, 2     # 128 x 128 crops after 2x average pooling (256 x 256 native pixels, ~15-25 U2OS cells)


@app.function(image=cpu_image, volumes={"/vol": vol}, timeout=3600, retries=2, cpu=4, memory=8192)
def prep_plate(plate: str, batch: str, wells: list[str], ctrl_wells: list[str]) -> dict:
    """Download fields, normalise each channel with plate-level control percentiles, cut 128 px crops."""
    import io
    import re
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import numpy as np
    import tifffile

    out = Path(f"/vol/tiles/{plate}.npz")
    if out.exists():
        return {"plate": plate, "skipped": True}
    pre = f"cpg0016-jump/source_13/images/{batch}/images/{plate}/{plate}_"
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(pre)}" + (
            f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        x = urllib.request.urlopen(url, timeout=120).read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            break
        token = m.group(1)
    bykey = {}
    for k in keys:
        m = re.search(r"_([A-P]\d\d)_T0001F(\d{3})L01A\d\dZ01C(\d\d)\.tif$", k)
        if m:
            bykey[(m.group(1), int(m.group(2)), int(m.group(3)))] = k

    def fetch(wf):
        well, f = wf
        chans = []
        for c in range(1, 6):
            k = bykey.get((well, f, c))
            if k is None:
                return wf, None
            for attempt in range(4):
                try:
                    b = urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(k)}", timeout=120).read()
                    break
                except Exception:
                    if attempt == 3:
                        raise
            a = tifffile.imread(io.BytesIO(b)).astype(np.float32)
            chans.append(a)
        return wf, np.stack(chans)

    todo = [(w, f) for w in list(wells) + list(ctrl_wells) for f in FIELDS]
    with ThreadPoolExecutor(16) as ex:
        raw = dict(ex.map(fetch, todo))
    ctrl = [raw[(w, f)] for w in ctrl_wells for f in FIELDS if raw.get((w, f)) is not None]
    ref = ctrl if ctrl else [v for v in raw.values() if v is not None]
    lo = np.median([np.percentile(v.reshape(5, -1), 0.5, axis=1) for v in ref], axis=0)
    hi = np.median([np.percentile(v.reshape(5, -1), 99.8, axis=1) for v in ref], axis=0)
    tiles, twell, tfield = [], [], []
    for (w, f), v in raw.items():
        if v is None:
            continue
        v = (v - lo[:, None, None]) / (hi - lo)[:, None, None]
        C, H, W = v.shape
        v = v[:, : H // DOWN * DOWN, : W // DOWN * DOWN].reshape(C, H // DOWN, DOWN, W // DOWN, DOWN).mean((2, 4))
        v = (np.clip(v, 0, 1) * 255).round().astype(np.uint8)
        for i in range(v.shape[1] // TILE):
            for j in range(v.shape[2] // TILE):
                tiles.append(v[:, i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]); twell.append(w); tfield.append(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tiles=np.stack(tiles), well=np.array(twell), field=np.array(tfield), lo=lo, hi=hi)
    vol.commit()
    return {"plate": plate, "n_tiles": len(tiles), "missing": sum(v is None for v in raw.values())}


def _load_tiles(manifest_rows: list[dict]):
    """All crops of gene wells in the manifest: tiles (N,5,128,128) uint8, gene index, well key."""
    from pathlib import Path

    import numpy as np
    want = {(r["plate"], r["well"]): r["gene_index"] for r in manifest_rows}
    T, G, Wk = [], [], []
    for p in sorted(Path("/vol/tiles").glob("*.npz")):
        z = np.load(p)
        g = np.array([want.get((p.stem, w), -2) for w in z["well"]])
        keep = g >= 0
        if keep.any():
            T.append(z["tiles"][keep]); G.append(g[keep]); Wk += [f"{p.stem}:{w}" for w in z["well"][keep]]
    return np.concatenate(T), np.concatenate(G), np.array(Wk)


@app.function(image=gpu_image, gpu="H100", volumes={"/vol": vol}, timeout=6 * 3600, memory=65536, retries=3)
def train_fold(k: int, manifest_rows: list[dict], codes, train_genes, test_genes, cfg: dict, n_gen: int = 128,
               n_ref: int = 6000, run: str = "") -> dict:
    """Train on crops of training genes; sample held-out genes four ways; return features of real and generated crops."""
    import io
    import time

    import numpy as np
    import torch

    from jump_g2p.imagediff import features as fx
    from jump_g2p.imagediff.unet import ImageDiffusion, ImageDiffusionConfig

    t0 = time.time()
    tiles, gidx, wkey = _load_tiles(manifest_rows)
    print(f"fold {k}: {len(tiles)} crops loaded in {time.time() - t0:.0f}s", flush=True)
    codes = np.asarray(codes, dtype=np.float32)
    tr = np.isin(gidx, train_genes)
    dev = torch.device("cuda")
    c = ImageDiffusionConfig(**cfg)
    m = ImageDiffusion(tiles.shape[1], codes.shape[1], c)
    X = torch.as_tensor(tiles[tr], device=dev)
    C = torch.as_tensor(codes[gidx[tr]], device=dev)
    print(f"training on {int(tr.sum())} crops of {len(np.unique(gidx[tr]))} genes; "
          f"{sum(p.numel() for p in m.net.parameters()) / 1e6:.1f}M params", flush=True)
    import os
    os.makedirs("/vol/ckpt", exist_ok=True)
    tag = "_".join(f"{a}{b}" for a, b in sorted(cfg.items()))
    pre = f"{run}/" if run else ""   # separate checkpoints, models and results per experiment (e.g. "chain")
    os.makedirs(f"/vol/ckpt/{pre}", exist_ok=True)
    path = f"/vol/ckpt/{pre}fold{k}{'_' + tag if tag else ''}.pt"

    def load():
        vol.reload()
        return torch.load(path, map_location="cuda", weights_only=False) if os.path.exists(path) else None

    def save(state):
        torch.save(state, path + ".tmp"); os.replace(path + ".tmp", path); vol.commit()

    m.fit(X, C, ckpt=(load, save))
    del X, C
    torch.cuda.empty_cache()
    buf = io.BytesIO()
    torch.save({"ema": m.ema.state_dict(), "cfg": c.to_dict(), "c_dim": codes.shape[1]}, buf)
    os.makedirs(f"/vol/models/{pre}", exist_ok=True)
    with open(f"/vol/models/{pre}fold{k}.pt", "wb") as f:
        f.write(buf.getvalue())
    vol.commit()

    test_genes = np.array([g for g in test_genes if (gidx == g).any()])
    rng = np.random.default_rng(k)
    perm = rng.permutation(len(test_genes))
    while len(test_genes) > 1 and (perm == np.arange(len(test_genes))).any():
        perm = rng.permutation(len(test_genes))
    G = np.repeat(codes[test_genes], n_gen, 0)
    Gs = np.repeat(codes[test_genes][perm], n_gen, 0)
    t1 = time.time()
    gen = {"cond": m.sample(G, w=1.0, seed=1), "free": m.sample(G, w=0.0, seed=2),
           "shuffled": m.sample(Gs, w=1.0, seed=3), "guided2": m.sample(G, w=2.0, seed=4)}
    print(f"sampled {sum(len(v) for v in gen.values())} crops in {time.time() - t1:.0f}s", flush=True)

    te = np.isin(gidx, test_genes)
    ref = rng.choice(np.where(tr)[0], min(n_ref, int(tr.sum())), replace=False)
    dino = fx.load_dino(dev)
    feats = {}
    for name, arr in [("real_test", tiles[te]), ("real_train_ref", tiles[ref])] + list(gen.items()):
        feats[f"{name}_dino"] = fx.dino_features(dino, arr, dev).astype(np.float16)
        feats[f"{name}_cp"] = fx.cp_features(arr, dev).astype(np.float32)
    keep = np.concatenate([np.arange(j * n_gen, j * n_gen + 8) for j in range(len(test_genes))])
    examples = {f"gen_{v}": a[keep] for v, a in gen.items()}
    ex_real = np.concatenate([np.where(gidx == g)[0][:8] for g in test_genes])
    res = {"fold": k, "test_genes": test_genes, "shuffled_perm": perm, "n_gen": n_gen,
            "real_test_gene": gidx[te], "real_test_well": wkey[te], "losses": m.losses, **feats, **examples,
            "examples_real": tiles[ex_real], "examples_real_gene": gidx[ex_real],
            "minutes": (time.time() - t0) / 60}
    os.makedirs(f"/vol/results/{pre}", exist_ok=True)
    np.savez_compressed(f"/vol/results/{pre}fold{k}.npz", **{a: np.asarray(b) for a, b in res.items()})
    vol.commit()
    return {"fold": k, "minutes": res["minutes"], "losses": m.losses}
