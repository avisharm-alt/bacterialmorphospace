"""Genome-conditioned 3D diffusion on Antscan body volumes, tested on held-out genera.

Each scan is a 32^3 occupancy volume in its own principal-axis frame (`solid32`: whole body, or `core32`: legs and
antennae stripped). A small 3D U-Net DDPM learns p(volume | code), where the code for a species is one of

    none       zeros (unconditional: "a typical ant")
    tree       Nystrom projection of the GAGA dated-tree covariance (family tree only, no DNA)
    genome     PCA of gene-family copy numbers (GAGA orthogroups), fitted on training species only
    shuffled   genome codes permuted across species (null)

Folds hold out whole genera (grouped K-fold). For each test species we sample volumes and score them against
that species' real scans (soft IoU and voxel MSE of the sample mean). The question is the same as in
run_baselines: does `genome` beat `tree`, and do both beat `none` and `shuffled`?

Runs on CPU for a smoke test (`--tiny`); the real config wants a GPU (see modal_app.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

DATA = "/mnt/project-files/data/antscan"
VARIANTS = ("none", "tree", "genome", "shuffled")


# --- data -----------------------------------------------------------------------------------------------------

def load_payload(shapes: str, selection_csv: str, gaga: str, volume: str = "solid32",
                 pretrain_csv: str | None = None) -> dict:
    """Everything a fold needs, as plain numpy (small enough to ship to a remote GPU as an argument)."""
    from . import evaluate as ev
    from . import genome as gn

    sel = pd.read_csv(selection_csv)
    sel = sel[sel.caste == "worker"]
    vols, sids = [], []
    for r in sel.itertuples():
        p = f"{shapes}/{r.specimen_code}.npz"
        if os.path.exists(p):
            vols.append(np.load(p)[volume].astype(np.float32))
            sids.append(r.gaga_id)
    sids = np.array(sids)
    ids = sorted(set(sids))
    meta = sel.drop_duplicates("gaga_id").set_index("gaga_id").loc[ids]
    tree = gn.phylo_covariance(open(f"{gaga}/GAGA_dated_phylogeny_onlydates_newick.tre").read(), ids).to_numpy()
    G = gn.genome_features(gn.read_genecounts(f"{gaga}/Orthogroup_table_N0_genecount.tsv.gz", ids)).to_numpy()
    payload = {"vols": np.stack(vols), "scan_species": sids, "species": np.array(ids),
               "genus": meta.Genus.to_numpy(), "K_tree": ev.normalise_kernel(tree + 1e-6 * np.eye(len(ids))),
               "G": G.astype(np.float32)}
    if pretrain_csv and os.path.exists(pretrain_csv):
        # every other Antscan scan (any taxon, any caste): unconditional shape prior. run_fold drops the
        # held-out genera's scans from it, so the test genera are never seen in any form.
        pre = pd.read_csv(pretrain_csv)
        pre = pre[~pre.specimen_code.isin(sel.specimen_code)]
        pv, pg = [], []
        for r in pre.itertuples():
            p = f"{shapes}/{r.specimen_code}.npz"
            if os.path.exists(p):
                pv.append(np.load(p)[volume].astype(np.float32))
                pg.append(r.Genus)
        if pv:
            payload["pre_vols"], payload["pre_genus"] = np.stack(pv), np.array(pg)
    return payload


def genus_folds(genus: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Fold index per species; whole genera go to one fold, folds balanced by species count."""
    rng = np.random.default_rng(seed)
    ug, cnt = np.unique(genus, return_counts=True)
    order = rng.permutation(len(ug))
    order = order[np.argsort(-cnt[order], kind="stable")]
    load = np.zeros(k)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[ug[i]] = f
        load[f] += cnt[i]
    return np.array([fold_of[g] for g in genus])


def species_codes(payload: dict, train: np.ndarray, variant: str, dim: int, seed: int) -> np.ndarray:
    """(n_species, dim) codes fitted on training species only; test species are projected, never fitted."""
    n = len(payload["species"])
    if variant == "none":
        return np.zeros((n, dim), np.float32)
    if variant == "tree":
        K = payload["K_tree"]
        evals, evecs = np.linalg.eigh(K[np.ix_(train, train)])
        top = np.argsort(evals)[::-1][:dim]
        lam, V = evals[top], evecs[:, top]
        Z = K[:, train] @ V / np.sqrt(np.clip(lam, 1e-9, None))  # Nystrom: train rows = V * sqrt(lam)
    else:
        G = payload["G"].astype(np.float64)
        mu, sd = G[train].mean(0), np.clip(G[train].std(0), 1e-9, None)
        Gs = (G - mu) / sd
        _, _, Vt = np.linalg.svd(Gs[train], full_matrices=False)
        Z = Gs @ Vt[:dim].T
        if variant == "shuffled":
            Z = Z[np.random.default_rng(seed + 991).permutation(n)]
    Z = Z / np.clip(Z[train].std(0), 1e-9, None)
    if Z.shape[1] < dim:  # fewer training species than code dimensions
        Z = np.pad(Z, ((0, 0), (0, dim - Z.shape[1])))
    return Z.astype(np.float32)


# --- model ----------------------------------------------------------------------------------------------------

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_model(ch: int, cond_dim: int):
    torch, nn, F = _torch()

    def tembed(t, dim):
        half = dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return torch.cat([a.sin(), a.cos()], 1)

    class Block(nn.Module):
        def __init__(self, cin, cout, edim):
            super().__init__()
            self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv3d(cin, cout, 3, padding=1)
            self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1)
            self.film = nn.Linear(edim, 2 * cout)
            self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, e):
            h = self.c1(F.silu(self.n1(x)))
            s, b = self.film(e).chunk(2, 1)
            h = self.n2(h) * (1 + s[..., None, None, None]) + b[..., None, None, None]
            return self.skip(x) + self.c2(F.silu(h))

    class UNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            edim = 4 * ch
            self.edim = edim
            self.t_mlp = nn.Sequential(nn.Linear(ch, edim), nn.SiLU(), nn.Linear(edim, edim))
            self.c_mlp = nn.Sequential(nn.Linear(cond_dim, edim), nn.SiLU(), nn.Linear(edim, edim))
            self.null = nn.Parameter(torch.zeros(edim))
            self.inp = nn.Conv3d(1, ch, 3, padding=1)
            self.d1, self.d2, self.d3 = Block(ch, ch, edim), Block(ch, 2 * ch, edim), Block(2 * ch, 4 * ch, edim)
            self.mid = Block(4 * ch, 4 * ch, edim)
            self.u3, self.u2, self.u1 = Block(8 * ch, 2 * ch, edim), Block(4 * ch, ch, edim), Block(2 * ch, ch, edim)
            self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, 1, 3, padding=1))

        def forward(self, x, t, c, drop=None):
            e = self.t_mlp(tembed(t, self.t_mlp[0].in_features))
            ce = self.c_mlp(c)
            if drop is not None:
                ce = torch.where(drop[:, None], self.null.expand_as(ce), ce)
            e = e + ce
            h0 = self.inp(x)
            h1 = self.d1(h0, e)                       # 32
            h2 = self.d2(F.avg_pool3d(h1, 2), e)      # 16
            h3 = self.d3(F.avg_pool3d(h2, 2), e)      # 8
            m = self.mid(F.avg_pool3d(h3, 2), e)      # 4
            u = self.u3(torch.cat([F.interpolate(m, scale_factor=2), h3], 1), e)
            u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), e)
            u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
            return self.out(u)

    return UNet3D()


def _schedule(T: int):
    torch, _, _ = _torch()
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999)
    return betas, torch.cumprod(1 - betas, 0)


def run_fold(payload: dict, fold: int, cfg: dict, variant: str, seed: int = 0) -> dict:
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folds = genus_folds(payload["genus"], cfg["k_folds"], seed=0)
    species = payload["species"]
    train = np.flatnonzero(folds != fold)
    test = np.flatnonzero(folds == fold)
    Z = species_codes(payload, train, variant, cfg["code_dim"], seed)
    sp_index = {s: i for i, s in enumerate(species)}
    scan_sp = np.array([sp_index[s] for s in payload["scan_species"]])
    tr_scans = np.flatnonzero(np.isin(scan_sp, train))

    vols = payload["vols"]
    is_pre = np.zeros(len(vols), bool)
    if cfg.get("pretrain", True) and "pre_vols" in payload:
        keep = ~np.isin(payload["pre_genus"], payload["genus"][test])
        vols = np.concatenate([vols, payload["pre_vols"][keep]])
        is_pre = np.concatenate([is_pre, np.ones(int(keep.sum()), bool)])
        scan_sp = np.concatenate([scan_sp, np.zeros(int(keep.sum()), int)])  # code ignored: always dropped
        tr_scans = np.concatenate([tr_scans, np.flatnonzero(is_pre)])
    out_pre = int(is_pre.sum())
    X = torch.tensor(vols * 2 - 1)[:, None]  # occupancy in [0,1] -> [-1,1]
    C = torch.tensor(Z)
    is_pre_t = torch.tensor(is_pre)
    model = build_model(cfg["ch"], cfg["code_dim"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    betas, abar = _schedule(cfg["T"])
    abar = abar.to(dev)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(cfg["steps"]):
        b = rng.choice(tr_scans, cfg["batch"])
        x0 = X[b].to(dev)
        if cfg.get("augment", True):  # the PCA frame's sign is a convention; mirror the left-right axis freely
            flip = torch.rand(len(b), device=dev) < 0.5
            x0 = torch.where(flip[:, None, None, None, None], x0.flip(-1), x0)
        c = C[scan_sp[b]].to(dev)
        t = torch.randint(0, cfg["T"], (len(b),), device=dev)
        eps = torch.randn_like(x0)
        a = abar[t][:, None, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        drop = (torch.rand(len(b), device=dev) < cfg["p_uncond"]) | is_pre_t[b].to(dev)
        loss = F.mse_loss(model(xt, t, c, drop), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    # sample: DDIM, classifier-free guidance
    model.eval()
    out = {"fold": fold, "variant": variant, "test_species": species[test].tolist(), "loss_tail": float(np.mean(losses[-100:])),
           "n_pretrain_scans": out_pre}
    scores = []
    ts = np.linspace(cfg["T"] - 1, 0, cfg["sample_steps"]).round().astype(int)
    with torch.no_grad():
        for i in test:
            n = cfg["n_samples"]
            c = C[i : i + 1].repeat(n, 1).to(dev)
            x = torch.randn(n, 1, 32, 32, 32, device=dev)
            for j, t in enumerate(ts):
                tt = torch.full((n,), int(t), device=dev)
                e_c = model(x, tt, c, torch.zeros(n, dtype=torch.bool, device=dev))
                e_u = model(x, tt, c, torch.ones(n, dtype=torch.bool, device=dev))
                e = e_u + cfg["guidance"] * (e_c - e_u)
                a = abar[t]
                x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
                a_next = abar[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=dev)
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e
            gen = ((x0 + 1) / 2).cpu().numpy()[:, 0]
            real = payload["vols"][(scan_sp == i)[: len(payload["vols"])] & ~is_pre[: len(payload["vols"])]]
            scores.append(score_volumes(gen, real))
    out["scores"] = scores
    return out


def score_volumes(gen: np.ndarray, real: np.ndarray) -> dict:
    """Generated vs real scans of one species; mirror-invariant (the left-right sign is arbitrary)."""
    g, r = gen.mean(0), real.mean(0)

    def siou(a, b):
        return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9))

    iou = max(siou(g, r), siou(g[..., ::-1], r))
    mse = min(float(((g - r) ** 2).mean()), float(((g[..., ::-1] - r) ** 2).mean()))
    return {"soft_iou": iou, "mse": mse}


CONFIGS = {
    "tiny": dict(k_folds=5, ch=8, code_dim=8, lr=2e-4, T=200, steps=30, batch=4, p_uncond=0.15,
                 sample_steps=10, n_samples=2, guidance=2.0),
    "gpu": dict(k_folds=5, ch=32, code_dim=16, lr=2e-4, T=1000, steps=20000, batch=32, p_uncond=0.15,
                sample_steps=50, n_samples=8, guidance=2.0),
}


def summarise(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for sp, s in zip(r["test_species"], r["scores"]):
            rows.append({"variant": r["variant"], "fold": r["fold"], "species": sp, **s})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="tiny")
    ap.add_argument("--backend", choices=["local", "modal"], default="local")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--volume", default="solid32", choices=["solid32", "core32"])
    ap.add_argument("--no-pretrain", action="store_true", help="train on the genome species' scans only")
    ap.add_argument("--shapes", default=f"{DATA}/shapes")
    ap.add_argument("--gaga", default=f"{DATA}/gaga")
    ap.add_argument("--out", default=f"{DATA}/results/diffusion")
    a = ap.parse_args(argv)
    cfg = CONFIGS[a.config]
    payload = load_payload(a.shapes, f"{a.shapes}/selection_worker.csv", a.gaga, a.volume,
                           None if a.no_pretrain else f"{a.shapes}/selection_all_taxa.csv")
    folds = a.folds if a.folds is not None else list(range(cfg["k_folds"]))
    jobs = [(f, v) for v in a.variants for f in folds]
    if a.backend == "modal":
        from .modal_app import app, fold_remote
        with app.run():
            results = list(fold_remote.starmap([(payload, f, cfg, v) for f, v in jobs]))
    else:
        results = [run_fold(payload, f, cfg, v) for f, v in jobs]
    os.makedirs(a.out, exist_ok=True)
    df = summarise(results)
    df.to_csv(f"{a.out}/scores_{a.config}_{a.volume}.csv", index=False)
    summary = df.groupby("variant")[["soft_iou", "mse"]].mean()
    print(summary)
    with open(f"{a.out}/summary_{a.config}_{a.volume}.json", "w") as f:
        json.dump({"config": cfg, "volume": a.volume, "summary": summary.to_dict(),
                   "loss_tail": {f"{r['variant']}/{r['fold']}": r["loss_tail"] for r in results}}, f, indent=2)


if __name__ == "__main__":
    main()
