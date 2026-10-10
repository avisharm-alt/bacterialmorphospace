"""Held-out-gene test of a gene-conditioned diffusion model over JUMP Cell Painting knockout profiles.

    python -m jump_g2p.diffusion_cv <raw_dir> <homolog_pairs.tsv> <out_dir> [--local] [--codes=network|esm|evo2|esm+evo2|chain|depmap] [--seed=N]

--codes=network (default) builds gene codes from DepMap, STRING and co-essentiality; esm / evo2 build them from the
gene's protein (ESM-2 650M) or coding DNA (Evo 2 7B) embedding only, with no experimental data about the gene.
--codes=chain predicts each held-out gene's DepMap profile from its protein embedding (a ridge trained on the ~18k
screened genes off the held-out arms, paralogs removed), then builds the code from that predicted profile
(DepMap ridge plus co-essentiality neighbours); --codes=depmap builds the same code from the measured profile.

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


def chain_depmap(c: dict, te: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Predicted DepMap profiles of JUMP genes `te` from their protein sequence alone. The sequence -> DepMap ridge
    trains on every screened human gene except those on the arms of `te` or `exclude` and their paralogs."""
    held = c["jmap"][np.concatenate([te, exclude])]
    bad = np.isin(c["arm"], np.unique(c["arm"][held])) | c["isseq_missing"]
    for i in held:
        bad[list(c["hom"][i])] = True
    bad[held] = True
    trn = np.flatnonzero(c["has"] & ~bad)
    p = PCA(64, random_state=SEED).fit(c["D"][trn])
    Z = ridge(c["E"][trn], p.transform(c["D"][trn]), c["E"][c["jmap"][te]], n_pc=256)
    return p.inverse_transform(Z).astype(np.float32)


def predict_means(inp: dict, T: np.ndarray, tr: np.ndarray, te: np.ndarray, exclude=None) -> np.ndarray:
    if "chain" in inp:  # sequence -> predicted DepMap -> shape: DepMap ridge and co-essentiality, on predicted profiles
        c = inp["chain"]
        if c.get("true_depmap"):  # reference: the same code built from the gene's measured DepMap profile
            Dte = c["D"][c["jmap"][te]]
        else:
            Dte = chain_depmap(c, te, np.array([], int) if exclude is None else exclude)
        Dtr = c["D"][c["jmap"][tr]]
        ok = c["has"][c["jmap"][tr]]
        a = ridge(Dtr[ok], T[tr][ok], Dte)
        Zt = Dte - Dte.mean(1, keepdims=True); Zt /= np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-9
        Zr = Dtr - Dtr.mean(1, keepdims=True); Zr /= np.linalg.norm(Zr, axis=1, keepdims=True) + 1e-9
        C = Zt @ Zr[ok].T
        np.put_along_axis(C, np.argsort(-C, axis=1)[:, 50:], 0, axis=1)
        C = np.where(C > 0.2, C, 0)
        b = (C @ T[tr][ok] + T[tr].mean(0)) / (C.sum(1) + 1.0)[:, None]
        return ((a + b) / 2).astype(np.float32)
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
        code[tr[ite]] = predict_means(inp, T, tr[itr], tr[ite], exclude=te)
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


def chain_inputs(g: pd.DataFrame, raw: Path, derived: Path, homologs: str) -> dict:
    """Everything chain_depmap needs, over the universe of genes screened in DepMap or imaged in JUMP."""
    z = [np.load(derived / f, allow_pickle=True) for f in ("esm2_650m_human.npz", "esm2_650m_depmap_extra.npz")]
    prot = pd.concat([pd.read_csv(derived / f, sep="\t", dtype={"gene_id": str})
                      for f in ("jump_proteins.tsv", "depmap_extra_proteins.tsv")])
    acc2g = dict(zip(prot["Entry"], prot["gene_id"]))
    emb = {acc2g[a]: e for zz in z for a, e in zip(zz["accession"], zz["emb"])}
    cols = pd.read_csv(raw / "CRISPRGeneEffect_24Q2.csv", nrows=0, index_col=0).columns
    u = list(dict.fromkeys(g["gene_id"].tolist() + [c.rsplit("(", 1)[1].rstrip(")") for c in cols]))
    pos = {x: i for i, x in enumerate(u)}
    D, has = data.depmap(raw, u)
    gt = data.gene_table(raw).drop_duplicates("gene_id").set_index("gene_id")
    arm = pd.Series(u).map(gt["arm"]).fillna("unknown").replace("", "unknown").to_numpy()
    miss = np.array([x not in emb for x in u])
    E = np.stack([emb.get(x, np.zeros(1280, np.float32)) for x in u]).astype(np.float32)
    hom = [set() for _ in u]
    h = pd.read_csv(homologs, sep="\t", header=None, dtype=str)
    for a, b in zip(h[0], h[1]):
        if a in pos and b in pos:
            hom[pos[a]].add(pos[b]); hom[pos[b]].add(pos[a])
    jmap = np.array([pos[x] for x in g["gene_id"]])
    print(f"chain: {len(u)} genes, {int((has & ~miss).sum())} with DepMap and a protein sequence", flush=True)
    return {"chain": {"E": E, "D": D, "has": has, "arm": arm, "hom": hom, "jmap": jmap, "isseq_missing": miss}}


def main(raw: str, homologs: str, out: str, local: bool = False, codes: str = "network", seed: int = 0) -> None:
    raw_p, out_p = Path(raw), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    jd = data.load(raw_p)
    g = jd.genes
    sym2i = {s: i for i, s in enumerate(g["symbol"])}
    w = jd.wells[jd.wells["Metadata_Symbol"].isin(sym2i)]
    W = w[jd.feats].to_numpy(dtype=np.float32)
    wg = w["Metadata_Symbol"].map(sym2i).to_numpy()
    if codes in ("chain", "depmap"):
        inp = chain_inputs(g, raw_p, Path(homologs).parent, homologs)
        inp["chain"]["true_depmap"] = codes == "depmap"
    elif codes == "network":
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
            results = list(fold_remote.map(payloads, kwargs={"cfg": {"seed": seed}}))
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
    summ = {"codes": codes, "train_seed": seed, "n_genes": int(len(df)), "n_wells": int(len(W)), "K": K, "code_r2_per_pc": code_r2.tolist(), "results": {}}
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
    seed = int(next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--seed=")), "0"))
    main(*a[:3], local="--local" in sys.argv, codes=codes, seed=seed)
