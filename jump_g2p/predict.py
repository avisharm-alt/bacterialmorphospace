"""Gene in, morphology out: generate Cell Painting morphology for a CRISPR knockout of any human gene.

    python -m jump_g2p.predict <GENE_SYMBOL> [--model data/jump/model] [--raw data/jump/raw] [--out out_dir]
                               [--n 64] [--no-images]

Writes to out_dir/<GENE>/:
- samples.npz      n generated well profiles (K whitened PCs and the 599 JUMP features)
- summary.json     predicted effect strength, closest imaged knockouts, whether the gene was in training
- <GENE>.png       generated profiles vs controls, plus real Cell Painting fields of the imaged wells closest to the
                   prediction and of a non-targeting control (real images retrieved for illustration, not generated
                   pixels)

For a gene that was imaged in JUMP, its own wells and every paralog are excluded when building its code, but the
diffusion network did see its wells in training, so the honest test of the tool is the held-out evaluation
(diffusion_cv.py) or a gene that was never imaged.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from keio_g2p.diffusion import ConditionalDiffusion, DiffusionConfig

from .bundle import code_for


def load_bundle(md: Path):
    a = np.load(md / "arrays.npz")
    st = torch.load(md / "model.pt", map_location="cpu")
    cfg = DiffusionConfig(**st["cfg"])
    m = ConditionalDiffusion(st["x_dim"], st["g_dim"], cfg, device="cpu", a_dim=1)
    m.ema.load_state_dict(st["ema"])
    with open(md / "pca.pkl", "rb") as f:
        pca = pickle.load(f)
    inp = {"depmap": a["D"].astype(np.float32), "string": sparse.load_npz(md / "string.npz"),
           "coess": sparse.load_npz(md / "coess.npz")}
    genes = pd.read_csv(md / "genes.csv", dtype={"gene_id": str})
    hom = pd.read_csv(md / "homologs.tsv", sep="\t", header=None, dtype=str)
    return m, pca, inp, genes, a, hom


def predict(symbol: str, md: Path, n: int = 64, seed: int = 0) -> dict:
    m, pca, inp, genes, a, hom = load_bundle(md)
    hit = genes.index[genes["symbol"].str.upper() == symbol.upper()]
    if len(hit) == 0:
        raise SystemExit(f"{symbol}: not in DepMap 24Q2 or JUMP, so there is no gene information to condition on")
    q = int(hit[0])
    jidx = a["jidx"]
    gid = genes.at[q, "gene_id"]
    paralogs = set(hom.loc[hom[0] == gid, 1]) | set(hom.loc[hom[1] == gid, 0])
    para_idx = set(genes.index[genes["gene_id"].isin(paralogs)])
    tr = np.array([j for j in jidx if j != q and j not in para_idx])
    code = code_for(inp, a["T"], tr, np.array([q]))
    G = np.repeat(code, n, axis=0)
    A = np.zeros((n, 1), dtype=np.float32)
    S = m.sample(G, A, w=1.0, seed=seed)
    S0 = m.sample(G, A, w=0.0, seed=seed + 1)
    # closest imaged knockouts by mean profile (excluding the query itself)
    T = a["T"][jidx]
    d = np.linalg.norm(T - S.mean(0), axis=1)
    order = [i for i in np.argsort(d) if jidx[i] != q][:5]
    # closest real wells to the generated mean, and a typical control
    wd = np.linalg.norm(a["X"] - S.mean(0), axis=1)
    wd[a["well_gene"] == q] = np.inf
    near_wells = np.argsort(wd)[:3]
    ctrl_mean = a["ctrl_X"].mean(0)
    ci = int(np.argmin(np.linalg.norm(a["ctrl_X"] - ctrl_mean, axis=1)))
    code_norms = np.linalg.norm(code_for(inp, a["T"], jidx[: len(jidx) // 2], jidx[len(jidx) // 2:]), axis=1)
    return {
        "symbol": genes.at[q, "symbol"], "gene_id": gid, "imaged_in_jump": bool(q in set(jidx)),
        "has_depmap": bool(a["has"][q]), "string_links_to_imaged": int((inp["string"][q][:, jidx] > 0).sum()),
        "predicted_shift": float(np.linalg.norm(code)),
        "predicted_shift_percentile": float((code_norms < np.linalg.norm(code)).mean() * 100),
        "generated_vs_gene_free_mean_distance": float(np.linalg.norm(S.mean(0) - S0.mean(0))),
        "distance_from_controls": float(np.linalg.norm(S.mean(0) - ctrl_mean)),
        "closest_imaged_knockouts": [{"symbol": genes.at[jidx[i], "symbol"], "distance": float(d[i])} for i in order],
        "nearest_real_wells": [{"symbol": genes.at[a["well_gene"][i], "symbol"], "plate": str(a["well_plate"][i]),
                                "well": str(a["well_well"][i])} for i in near_wells],
        "control_well": {"plate": str(a["ctrl_plate"][ci]), "well": str(a["ctrl_well"][ci])},
        "_samples": S, "_free": S0, "_features": pca.inverse_transform(S), "_ctrl": a["ctrl_X"],
    }


def figure(res: dict, raw: Path, path: Path, images: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 5 if images else 1
    fig, ax = plt.subplots(1, ncol, figsize=(4 * ncol, 4.2))
    ax = np.atleast_1d(ax)
    c, S, F = res["_ctrl"], res["_samples"], res["_free"]
    # x: along the predicted change (generated mean minus control mean); y: largest remaining direction of spread
    u = S.mean(0) - c.mean(0)
    u /= np.linalg.norm(u) + 1e-9
    allp = np.vstack([c, F, S])
    r = allp - np.outer(allp @ u, u)
    v = np.linalg.svd(r - r.mean(0), full_matrices=False)[2][0]
    for pts, col, lab, sz in [(c, "#bbbbbb", "non-targeting controls (real)", 4),
                              (F, "#8da0cb", "gene-free model", 6),
                              (S, "#d95f02", f"generated: {res['symbol']} KO", 8)]:
        ax[0].scatter(pts @ u, pts @ v, s=sz, c=col, alpha=0.7, label=lab)
    lo, hi = np.percentile(allp @ u, [0.5, 99.5]); lo2, hi2 = np.percentile(allp @ v, [0.5, 99.5])
    ax[0].set_xlim(lo - 0.5, hi + 0.5); ax[0].set_ylim(lo2 - 0.5, hi2 + 0.5)
    ax[0].set_xlabel("along predicted morphology change"); ax[0].set_ylabel("other variation")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].set_title(f"{res['symbol']} knockout: generated wells", fontsize=10)
    if images:
        from .images import composite, field
        panels = [(w, f"closest real well: {w['symbol']} KO") for w in res["nearest_real_wells"]]
        panels.append((res["control_well"], "non-targeting control"))
        for k, (w, title) in enumerate(panels, start=1):
            try:
                img = composite(field(str(raw), w["plate"], w["well"]))
                ax[k].imshow(img[200:712, 300:812])
            except Exception as e:  # noqa: BLE001 - a missing image should not lose the prediction
                ax[k].text(0.5, 0.5, f"image unavailable\n{e}", ha="center", fontsize=7)
            ax[k].set_title(title, fontsize=9); ax[k].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gene")
    ap.add_argument("--model", default="/mnt/project-files/data/jump/model")
    ap.add_argument("--raw", default="/mnt/project-files/data/jump/raw")
    ap.add_argument("--out", default="/mnt/project-files/data/jump/predictions")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    res = predict(a.gene, Path(a.model), a.n)
    od = Path(a.out) / res["symbol"]
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "samples.npz", pcs=res["_samples"], features=res["_features"])
    summary = {k: v for k, v in res.items() if not k.startswith("_")}
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    figure(res, Path(a.raw), od / f"{res['symbol']}.png", images=not a.no_images)
    print(json.dumps(summary, indent=2))
    print("wrote", od)


if __name__ == "__main__":
    main()
