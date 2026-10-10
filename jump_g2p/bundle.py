"""Build and train the final "gene in, morphology out" model on every JUMP CRISPR gene.

    python -m jump_g2p.bundle <raw_dir> <homolog_pairs.tsv> <model_dir>

The gene universe is every gene in DepMap 24Q2 or JUMP (about 18.5k), so the tool can be asked about genes that were
never imaged. Inputs per gene: DepMap dependency profile, STRING links (no text mining or curated databases) and
co-essentiality neighbours. Training genes get cross-fitted codes exactly as in diffusion_cv.py.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from . import data
from .diffusion_cv import K
from .predictability import neighbours, ridge

SEED = 0


def universe(raw: Path, jd) -> pd.DataFrame:
    cols = pd.read_csv(raw / "CRISPRGeneEffect_24Q2.csv", nrows=0, index_col=0).columns
    dm = pd.DataFrame([re.match(r"(.+) \((\d+)\)", c).groups() for c in cols], columns=["symbol", "gene_id"])
    u = pd.concat([jd.genes[["symbol", "gene_id"]], dm]).drop_duplicates("gene_id").drop_duplicates("symbol")
    gt = data.gene_table(raw).drop_duplicates("gene_id").set_index("gene_id")
    u["arm"] = u["gene_id"].map(gt["arm"]).fillna("unknown").replace("", "unknown")
    return u.reset_index(drop=True)


def coess_sparse(D: np.ndarray, has: np.ndarray, k: int = 50, thr: float = 0.2, chunk: int = 2000):
    Z = D - D.mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    Z[~has] = 0
    rows, cols, vals = [], [], []
    for s in range(0, len(Z), chunk):
        C = Z[s:s + chunk] @ Z.T
        for r in range(len(C)):
            C[r, s + r] = 0
        idx = np.argpartition(-C, k, axis=1)[:, :k]
        v = np.take_along_axis(C, idx, 1)
        keep = v > thr
        rr = np.repeat(np.arange(s, s + len(C)), k).reshape(len(C), k)
        rows.append(rr[keep]); cols.append(idx[keep]); vals.append(v[keep])
    n = len(Z)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def code_for(inp: dict, T: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    a = ridge(inp["depmap"][tr], T[tr], inp["depmap"][te])
    b = neighbours(inp["string"], T, tr, te, shrink=10.0)
    c = neighbours(inp["coess"], T, tr, te, shrink=1.0)
    return ((a + b + c) / 3).astype(np.float32)


def main(raw: str, homologs: str, model_dir: str) -> None:
    raw_p, md = Path(raw), Path(model_dir)
    md.mkdir(parents=True, exist_ok=True)
    jd = data.load(raw_p)
    u = universe(raw_p, jd)
    pos = {g: i for i, g in enumerate(u["gene_id"])}
    jidx = np.array([pos[g] for g in jd.genes["gene_id"]])
    D, has = data.depmap(raw_p, u["gene_id"].tolist())
    inp = {"depmap": D, "string": data.string_matrix(raw_p, u["symbol"].tolist()), "coess": coess_sparse(D, has)}
    print(f"universe {len(u)} genes, {len(jidx)} imaged; STRING links {inp['string'].nnz // 2}", flush=True)

    sym2j = {s: i for i, s in enumerate(jd.genes["symbol"])}
    w = jd.wells[jd.wells["Metadata_Symbol"].isin(sym2j)]
    W = w[jd.feats].to_numpy(dtype=np.float32)
    wj = w["Metadata_Symbol"].map(sym2j).to_numpy()
    pca = PCA(K, whiten=True, random_state=SEED).fit(W)
    X = pca.transform(W).astype(np.float32)
    T = np.zeros((len(u), K), dtype=np.float32)
    Tj = np.zeros((len(jidx), K), dtype=np.float32)
    np.add.at(Tj, wj, X)
    Tj /= np.bincount(wj, minlength=len(jidx))[:, None]
    T[jidx] = Tj

    code = np.zeros_like(T)
    arms = u["arm"].to_numpy()
    for itr, ite in GroupKFold(n_splits=5).split(jidx, groups=arms[jidx]):
        code[jidx[ite]] = code_for(inp, T, jidx[itr], jidx[ite])

    import modal

    from .modal_app import app, train_full_remote
    with modal.enable_output(), app.run():
        state = train_full_remote.remote({"X": X, "code": code[jidx[wj]]}, {})
    (md / "model.pt").write_bytes(state)

    p, feats = data.load_profiles(raw_p)
    c = p[p["Metadata_Symbol"] == "non-targeting"]
    np.savez_compressed(md / "arrays.npz", D=D.astype(np.float16), has=has, T=T, jidx=jidx, X=X,
                        well_gene=jidx[wj], well_plate=w["Metadata_Plate"].to_numpy().astype(str),
                        well_well=w["Metadata_Well"].to_numpy().astype(str),
                        ctrl_X=pca.transform(c[feats].to_numpy(dtype=np.float32)).astype(np.float32),
                        ctrl_plate=c["Metadata_Plate"].to_numpy().astype(str),
                        ctrl_well=c["Metadata_Well"].to_numpy().astype(str))
    sparse.save_npz(md / "string.npz", inp["string"].tocsr())
    sparse.save_npz(md / "coess.npz", inp["coess"].tocsr())
    u.to_csv(md / "genes.csv", index=False)
    with open(md / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    hom = pd.read_csv(homologs, sep="\t", header=None)
    hom = hom[hom[0].astype(str).isin(pos) & hom[1].astype(str).isin(pos)]
    hom[[0, 1]].to_csv(md / "homologs.tsv", sep="\t", header=False, index=False)
    (md / "info.json").write_text(json.dumps({"K": K, "n_universe": len(u), "n_imaged": int(len(jidx)),
                                              "n_wells": int(len(X)), "features": jd.feats}, indent=1))
    print("saved model bundle to", md)


if __name__ == "__main__":
    main(*sys.argv[1:4])
