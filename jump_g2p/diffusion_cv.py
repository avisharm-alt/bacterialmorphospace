"""Held-out-gene test of a gene-conditioned diffusion model over JUMP Cell Painting knockout profiles.

    python -m jump_g2p.diffusion_cv <raw_dir> <homolog_pairs.tsv> <out_dir> [--local] [--codes=network|esm|evo2|esm+evo2]

--codes=network (default) builds gene codes from DepMap, STRING and co-essentiality; esm / evo2 build them from the
gene's protein (ESM-2 650M) or coding DNA (Evo 2 7B) embedding only, with no experimental data about the gene.

Each sample is one well's morphology profile (599 Cell Painting features, reduced to K whitened PCs fitted on
training wells). Per fold (whole chromosome arms held out, paralogs purged):
1. Gene code = predicted mean profile PCs of the gene, averaging three cross-fitted predictors trained on other genes:
   ridge on its DepMap dependency profile, the STRING-weighted mean of its training neighbours, and the
   co-essentiality-weighted mean of its training neighbours. Training genes get codes from inner folds that never saw
   them, so the model learns how far to trust codes of the quality it will meet on new genes.
2. A conditional diffusion model (keio_g2p.diffusion) on (well PCs | code), trained on Modal. w = 1 is gene-conditioned,
   w = 0 gene-free (same network); "shuffled" gives the held-out genes each other's codes.
3. Per held-out gene: energy distance between generated and real wells, and the distance between the generated mean
   and the real mean profile. Lower is better.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from . import data
from .predictability import neighbours, ridge, splits_for

K = 32
N_GEN = 64
SEED = 0


def coessentiality(D: np.ndarray, has: np.ndarray, k: int = 50, thr: float = 0.2) -> sparse.csr_matrix:
    Z = D - D.mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    C = Z @ Z.T
    np.fill_diagonal(C, 0)
    C[~has] = 0
    C[:, ~has] = 0
    W = np.where(C > thr, C, 0)
    np.put_along_axis(W, np.argsort(-W, axis=1)[:, k:], 0, axis=1)
    return sparse.csr_matrix(W)


def knn_graph(E: np.ndarray, k: int = 20) -> sparse.csr_matrix:
    """Cosine k-nearest-neighbour graph of sequence embeddings (centred), weights = cosine similarity."""
    Z = E - E.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    rows, cols, vals = [], [], []
    for s0 in range(0, len(Z), 2000):
        C = Z[s0:s0 + 2000] @ Z.T
        for r in range(len(C)):
            C[r, s0 + r] = -1
        idx = np.argpartition(-C, k, axis=1)[:, :k]
        v = np.clip(np.take_along_axis(C, idx, 1), 0, None)
        rows.append(np.repeat(np.arange(s0, s0 + len(C)), k)); cols.append(idx.ravel()); vals.append(v.ravel())
    n = len(Z)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def predict_means(inp: dict, T: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    if "seq" in inp:  # sequence-only codes: ridge on the embedding plus its nearest training genes
        a = ridge(inp["seq"][tr], T[tr], inp["seq"][te])
        b = neighbours(inp["seq_knn"], T, tr, te, shrink=1.0)
        return ((a + b) / 2).astype(np.float32)
    a = ridge(inp["depmap"][tr], T[tr], inp["depmap"][te])
    b = neighbours(inp["string"], T, tr, te, shrink=10.0)
    c = neighbours(inp["coess"], T, tr, te, shrink=1.0)
    return ((a + b + c) / 3).astype(np.float32)


def cross_fitted(inp, T, tr, te, arms, n_inner=5):
    code = np.zeros_like(T, dtype=np.float32)
    for itr, ite in GroupKFold(n_splits=n_inner).split(tr, groups=arms[tr]):
        code[tr[ite]] = predict_means(inp, T, tr[itr], tr[ite])
    code[te] = predict_means(inp, T, tr, te)
    return code


def energy_distance(a, b):
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def sequence_inputs(genes: pd.DataFrame, codes: str, derived: Path) -> dict:
    """Embeddings aligned to `genes`; genes without a sequence get the mean embedding (so their code is ~the mean)."""
    parts = []
    if "esm" in codes:
        z = np.load(derived / "esm2_650m_human.npz", allow_pickle=True)
        prot = pd.read_csv(derived / "jump_proteins.tsv", sep="\t", dtype={"gene_id": str})
        acc2g = dict(zip(prot["Entry"], prot["gene_id"]))
        m = {acc2g[a]: e for a, e in zip(z["accession"], z["emb"])}
        parts.append(np.stack([m.get(g) if g in m else np.full(z["emb"].shape[1], np.nan) for g in genes["gene_id"]]))
    if "evo2" in codes:
        z = np.load(derived / "evo2_cds.npz", allow_pickle=True)
        m = dict(zip(z["symbol"], z["emb"]))
        parts.append(np.stack([m.get(s) if s in m else np.full(z["emb"].shape[1], np.nan) for s in genes["symbol"]]))
    E = np.hstack(parts).astype(np.float32)
    miss = np.isnan(E).any(1)
    E[miss] = np.nanmean(E[~miss], axis=0)
    print(f"sequence codes ({codes}): {int((~miss).sum())} of {len(E)} genes have sequences", flush=True)
    return {"seq": E, "seq_knn": knn_graph(E)}


def main(raw: str, homologs: str, out: str, local: bool = False, codes: str = "network") -> None:
    raw_p, out_p = Path(raw), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    jd = data.load(raw_p)
    g = jd.genes
    sym2i = {s: i for i, s in enumerate(g["symbol"])}
    w = jd.wells[jd.wells["Metadata_Symbol"].isin(sym2i)]
    W = w[jd.feats].to_numpy(dtype=np.float32)
    wg = w["Metadata_Symbol"].map(sym2i).to_numpy()
    if codes == "network":
        D, has = data.depmap(raw_p, g["gene_id"].tolist())
        inp = {"depmap": D, "string": data.string_matrix(raw_p, g["symbol"].tolist()), "coess": coessentiality(D, has)}
    else:
        inp = sequence_inputs(g, codes, Path(homologs).parent)
    arms = g["arm"].replace("", "unknown").to_numpy()
    splits = splits_for(g, pd.read_csv(homologs, sep="\t", header=None))
    rng = np.random.default_rng(SEED)
    folds = []
    for tr, te in splits:
        trw = np.isin(wg, tr)
        pca = PCA(K, whiten=True, random_state=SEED).fit(W[trw])
        X = pca.transform(W).astype(np.float32)
        T = np.zeros((len(g), K), dtype=np.float32)
        np.add.at(T, wg, X)
        T /= np.bincount(wg, minlength=len(g))[:, None]
        code = cross_fitted(inp, T, tr, te, arms)
        perm = rng.permutation(len(te))
        folds.append({"X": X[trw], "code": code[wg[trw]], "test_code": code[te], "test_code_shuffled": code[te][perm],
                      "n_gen": N_GEN, "_X": X, "_te": te, "_T": T, "_code": code})
    Tt = np.concatenate([f["_T"][f["_te"]] for f in folds]); Ct = np.concatenate([f["_code"][f["_te"]] for f in folds])
    code_r2 = 1 - ((Tt - Ct) ** 2).sum(0) / ((Tt - Tt.mean(0)) ** 2).sum(0)
    print("held-out R² of codes per PC:", np.round(code_r2, 3), flush=True)
    payloads = [{k: v for k, v in f.items() if not k.startswith("_")} for f in folds]
    if local:
        from .model import run_fold
        results = [run_fold(payloads[0], {"steps": 300})]
        folds = folds[:1]
    else:
        import modal
        from .modal_app import app, fold_remote
        with modal.enable_output(), app.run():
            results = list(fold_remote.map(payloads, kwargs={"cfg": {}}))
    rows = []
    for f, r in zip(folds, results):
        for j, i in enumerate(f["_te"]):
            real = f["_X"][wg == i]
            row = {"symbol": g.at[i, "symbol"], "reliability": g.at[i, "reliability"], "n_wells": len(real)}
            for v, S in r.items():
                s = S[j * N_GEN:(j + 1) * N_GEN]
                row[f"ed_{v}"] = energy_distance(s, real)
                row[f"mean_err_{v}"] = float(np.linalg.norm(s.mean(0) - real.mean(0)))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_p / "diffusion_per_gene.csv", index=False)
    summ = {"codes": codes, "n_genes": int(len(df)), "n_wells": int(len(W)), "K": K, "code_r2_per_pc": code_r2.tolist(), "results": {}}
    for name, d in [("all", df), ("reproducible quartile", df[df["reliability"] > df["reliability"].quantile(0.75)])]:
        s = {"n": int(len(d))}
        for m in ("ed", "mean_err"):
            for v in ("free", "cond", "shuffled"):
                s[f"{m}_{v}"] = float(d[f"{m}_{v}"].mean())
            for o in ("free", "shuffled"):
                diff = d[f"{m}_cond"] - d[f"{m}_{o}"]
                s[f"{m}_cond_better_than_{o}_frac"] = float((diff < 0).mean())
                s[f"{m}_p_vs_{o}"] = float(wilcoxon(diff, alternative="less").pvalue)
        summ["results"][name] = s
    (out_p / "diffusion_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    codes = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--codes=")), "network")
    main(*a[:3], local="--local" in sys.argv, codes=codes)
