"""Held-out-gene test of a gene-conditioned diffusion model over single-cell outlines.

    python -m keio_g2p.outline_cv <raw_dir> <derived_dir> <homolog_pairs.tsv> <out_dir> [--local]

Per fold (the same chromosomal-block folds with homolog purging as evaluate.py):
1. PCA (whitened, K components) of flattened canonical outlines, fitted on training cells only.
   Outlines are then corrected for imaging batch effects: each gene's cells are shifted by the median well-mean of
   the *other* strains on its plate (most strains look wild type, so this is the plate offset), and a linear trend in
   imaging time fitted on training genes is removed. Without this, block folds partly align with plates and plate
   offsets swamp the gene signal.
2. A gene code: the predicted mean outline PCs of the gene, averaging (a) ridge regression from its RB-TnSeq fitness
   profile and (b) the STRING-weighted mean of its training neighbours, shrunk toward the training mean.
   Training genes get *cross-fitted* codes (predicted by models that never saw them), so the diffusion model learns
   how far to trust a code of the quality it will meet on new genes, instead of treating it as a gene ID.
3. A conditional diffusion model on (outline PCs | gene code, plate, imaging time), trained on Modal. One network
   gives the gene-conditioned model (w = 1) and the gene-free model (w = 0); a third variant feeds the held-out genes
   each other's codes (shuffled).
4. Per held-out gene: energy distance between generated and real outlines (whitened PC space). Lower is better.
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

from .evaluate import BLOCK, _ridge, heldout_splits

K = 12                 # outline PCs modelled
CELLS_PER_GENE = 150   # training cells sampled per gene
N_GEN = 150            # samples generated per held-out gene and variant
N_EVAL = 150           # real cells per held-out gene used for scoring
SEED = 0


NB_SHRINK = 10.0       # pseudo-weight of the training mean in the STRING neighbour average


def predict_means(od, T: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Predicted mean outline PCs for genes `te` from genes `tr`: average of fitness ridge and shrunk STRING
    neighbour mean."""
    ridge = _ridge(od.fitness[tr], T[tr], od.fitness[te], n_pc=64)
    mask = np.zeros(len(T)); mask[tr] = 1
    Mt = od.string[te] @ sparse.diags(mask)
    w = np.asarray(Mt.sum(1)).ravel()
    nb = (Mt @ T + NB_SHRINK * T[tr].mean(0)) / (w + NB_SHRINK)[:, None]
    return ((ridge + nb) / 2).astype(np.float32)


def cross_fitted_codes(od, T: np.ndarray, tr: np.ndarray, te: np.ndarray, n_inner: int = 5):
    """Codes for training genes from inner block folds; codes for test genes from all training genes."""
    code = np.zeros_like(T, dtype=np.float32)
    block = tr // BLOCK
    for itr, ite in GroupKFold(n_splits=n_inner).split(tr, groups=block):
        code[tr[ite]] = predict_means(od, T, tr[itr], tr[ite])
    code[te] = predict_means(od, T, tr, te)
    return code


def batch_correct(Xg: list[np.ndarray], plate: np.ndarray, time: np.ndarray, tr: np.ndarray) -> tuple[list, dict]:
    """Shift each gene's cells by its plate offset (median well-mean of the other strains on the plate) and remove a
    linear imaging-time trend fitted on training genes."""
    means = np.stack([x.mean(0) for x in Xg])
    off = np.zeros_like(means)
    for p in np.unique(plate):
        idx = np.flatnonzero(plate == p)
        for i in idx:
            others = idx[idx != i]
            off[i] = np.median(means[others], axis=0) if len(others) else 0
    resid = means - off
    A = np.c_[np.ones(len(tr)), time[tr]]
    beta = np.linalg.lstsq(A, resid[tr], rcond=None)[0][1]
    shift = off + time[:, None] * beta[None]
    return [x - shift[i] for i, x in enumerate(Xg)], {"time_beta": beta}


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def build_fold(od, tr: np.ndarray, te: np.ndarray, rng: np.random.Generator) -> dict:
    sub = [c[rng.choice(len(c), min(len(c), CELLS_PER_GENE), replace=False)] for c in od.cells]
    pca = PCA(K, whiten=True, random_state=SEED).fit(np.concatenate([sub[i] for i in tr]))
    plate = od.genes["plate_idx"].to_numpy()
    time = od.genes["time"].to_numpy().astype(np.float32)
    Xg, corr = batch_correct([pca.transform(s).astype(np.float32) for s in sub], plate, time, tr)
    T = np.stack([x.mean(0) for x in Xg])
    code = cross_fitted_codes(od, T, tr, te)
    gi = np.concatenate([np.full(len(Xg[i]), i) for i in tr])
    X = np.concatenate([Xg[i] for i in tr])
    perm = rng.permutation(te)
    return {"X": X, "code": code[gi], "plate": plate[gi], "time": time[gi], "n_plates": od.n_plates,
            "test_code": code[te], "test_code_shuffled": code[perm], "test_plate": plate[te], "test_time": time[te],
            "n_gen": N_GEN, "_pca": pca, "_Xg": Xg, "_corr": corr, "_te": te, "_T": T, "_code": code}


def score_fold(fold: dict, samples: dict, od) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for j, i in enumerate(fold["_te"]):
        real = fold["_Xg"][i]  # batch-corrected PCs of the cells sampled for this gene
        real = real[rng.choice(len(real), min(len(real), N_EVAL), replace=False)]
        r = {"gene": od.genes.at[i, "gene"], "hit": bool(od.genes.at[i, "hit"]),
             "code_norm": float(np.linalg.norm(fold["_code"][i])), "true_shift": float(np.linalg.norm(fold["_T"][i]))}
        for v, S in samples.items():
            r[f"ed_{v}"] = energy_distance(S[j * N_GEN:(j + 1) * N_GEN], real)
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for name, d in [("all", df), ("shape mutants", df[df["hit"]]),
                    ("top 10% predicted shift", df[df["code_norm"] >= df["code_norm"].quantile(0.9)])]:
        s = {"n": int(len(d))}
        for v in ("free", "cond", "shuffled"):
            s[f"ed_{v}"] = float(d[f"ed_{v}"].mean())
        for other in ("free", "shuffled"):
            diff = d["ed_cond"] - d[f"ed_{other}"]
            s[f"cond_better_than_{other}_frac"] = float((diff < 0).mean())
            s[f"p_wilcoxon_vs_{other}"] = float(wilcoxon(diff, alternative="less").pvalue) if len(d) > 5 else None
        out[name] = s
    return out


def main(raw: str, derived: str, homologs: str, out: str, local: bool = False) -> None:
    from . import outlines
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    od = outlines.load(Path(raw), Path(derived))
    print(f"{len(od.genes)} genes with >= {outlines.MIN_CELLS} cells, {int(od.genes['hit'].sum())} shape mutants, "
          f"{sum(len(c) for c in od.cells)} cells", flush=True)
    splits = heldout_splits(od.genes["accession"].tolist(), pd.read_csv(homologs, sep="\t", header=None))
    rng = np.random.default_rng(SEED)
    folds = [build_fold(od, tr, te, rng) for tr, te in splits]
    Tt = np.concatenate([f["_T"][f["_te"]] for f in folds]); Ct = np.concatenate([f["_code"][f["_te"]] for f in folds])
    code_r2 = [float(1 - ((Tt[:, j] - Ct[:, j]) ** 2).sum() / ((Tt[:, j] - Tt[:, j].mean()) ** 2).sum()) for j in range(K)]
    print("held-out R² of predicted mean outline PCs:", np.round(code_r2, 3), flush=True)
    payloads = [{k: v for k, v in f.items() if not k.startswith("_")} for f in folds]
    if local:
        from .outline_model import run_fold
        results = [run_fold(p, {"steps": 300}) for p in payloads[:1]]
        folds = folds[:1]
    else:
        import modal
        from .modal_outline import app, fold_remote
        with modal.enable_output(), app.run():
            results = list(fold_remote.map(payloads, kwargs={"cfg": {}}))
    df = pd.concat([score_fold(f, r, od) for f, r in zip(folds, results)], ignore_index=True)
    df.to_csv(out_p / "outline_diffusion_per_gene.csv", index=False)
    summary = {"n_genes": int(len(od.genes)), "n_cells": int(sum(len(c) for c in od.cells)), "K": K,
               "code_r2_per_pc": code_r2, "results": summarise(df)}
    (out_p / "outline_diffusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*a[:4], local="--local" in sys.argv)
