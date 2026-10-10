"""JUMP Cell Painting CRISPR knockouts (cpg0016, source_13, U2OS) joined to gene inputs.

Raw files (data/jump/raw in the project folder):
- profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet  JUMP-assembled CRISPR well profiles (599 features)
- crispr.csv.gz                    JCP2022 id -> NCBI gene id and symbol (jump-cellpainting/datasets)
- CRISPRGeneEffect_24Q2.csv        DepMap 24Q2 Chronos gene effects (cell lines x genes)
- 9606.protein.links.detailed.v12.0.txt.gz, 9606.protein.info.v12.0.txt.gz   STRING v12 human
- Homo_sapiens.gene_info.gz        NCBI gene table (chromosome and band, for proximity-safe folds)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROFILES = "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
CONTROLS = {"no-guide", "non-targeting"}


@dataclass
class JumpData:
    genes: pd.DataFrame        # symbol, gene_id, chrom, arm, n_wells, reliability
    Y: np.ndarray              # gene-level mean profile (genes x 599)
    wells: pd.DataFrame        # well-level profiles with symbol
    feats: list[str]


def load_profiles(raw: Path) -> tuple[pd.DataFrame, list[str]]:
    p = pd.read_parquet(raw / PROFILES)
    c = pd.read_csv(raw / "crispr.csv.gz")
    p = p.merge(c, on="Metadata_JCP2022", how="left")
    feats = [x for x in p.columns if not x.startswith("Metadata")]
    return p, feats


def gene_table(raw: Path) -> pd.DataFrame:
    g = pd.read_csv(raw / "Homo_sapiens.gene_info.gz", sep="\t", usecols=["GeneID", "Symbol", "chromosome", "map_location"],
                    dtype=str)
    arm = g["map_location"].fillna("").str.extract(r"^(\w+?)([pq])")
    g["arm"] = np.where(arm[1].notna(), g["chromosome"] + arm[1].fillna(""), g["chromosome"])
    return g.rename(columns={"GeneID": "gene_id", "Symbol": "symbol", "chromosome": "chrom"})


def load(raw: Path, min_wells: int = 3, seed: int = 0) -> JumpData:
    p, feats = load_profiles(raw)
    p = p[~p["Metadata_Symbol"].isin(CONTROLS) & p["Metadata_Symbol"].notna()]
    rng = np.random.default_rng(seed)
    rows, Y = [], []
    gt = gene_table(raw).drop_duplicates("gene_id").set_index("gene_id")
    for (sym, gid), d in p.groupby(["Metadata_Symbol", "Metadata_NCBI_Gene_ID"]):
        X = d[feats].to_numpy(dtype=np.float32)
        if len(X) < min_wells:
            continue
        # split-half reliability of the gene's profile (how reproducible its morphology is)
        idx = rng.permutation(len(X))
        a, b = X[idx[: len(X) // 2]].mean(0), X[idx[len(X) // 2:]].mean(0)
        rel = float(np.corrcoef(a, b)[0, 1])
        gid = str(int(float(gid)))
        rows.append({"symbol": sym, "gene_id": gid, "n_wells": len(X), "reliability": rel,
                     "chrom": gt["chrom"].get(gid, ""), "arm": gt["arm"].get(gid, "")})
        Y.append(X.mean(0))
    genes = pd.DataFrame(rows)
    return JumpData(genes=genes, Y=np.stack(Y), wells=p, feats=feats)


def depmap(raw: Path, gene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Gene x cell-line Chronos effects (NaN -> column mean; genes not screened -> zeros) and a has-data mask."""
    d = pd.read_csv(raw / "CRISPRGeneEffect_24Q2.csv", index_col=0)
    ids = [re.search(r"\((\d+)\)", c).group(1) for c in d.columns]
    d.columns = ids
    d = d.loc[:, ~d.columns.duplicated()]
    d = d.fillna(d.mean())
    M = d.T.reindex(gene_ids)
    has = M.notna().all(1).to_numpy()
    return M.fillna(0).to_numpy(dtype=np.float32), has


def string_matrix(raw: Path, symbols: list[str], drop: tuple[str, ...] = ("textmining", "database"),
                  min_score: int = 400) -> sparse.csr_matrix:
    info = pd.read_csv(raw / "9606.protein.info.v12.0.txt.gz", sep="\t", usecols=[0, 1])
    info.columns = ["pid", "symbol"]
    idx = {s: i for i, s in enumerate(symbols)}
    p2i = {p: idx[s] for p, s in zip(info["pid"], info["symbol"]) if s in idx}
    chans = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database", "textmining"]
    use = [c for c in chans if c not in drop]
    ii, jj, ww = [], [], []
    prior = 0.041
    for chunk in pd.read_csv(raw / "9606.protein.links.detailed.v12.0.txt.gz", sep=" ", chunksize=2_000_000):
        a = chunk["protein1"].map(p2i)
        b = chunk["protein2"].map(p2i)
        k = a.notna() & b.notna()
        if not k.any():
            continue
        p_none = np.ones(k.sum())
        for ch in use:
            s = np.clip((chunk.loc[k, ch].to_numpy() / 1000 - prior) / (1 - prior), 0, 1)
            p_none *= 1 - s
        w = (1 - p_none) * (1 - prior) + prior
        keep = w >= min_score / 1000
        ii.append(a[k].to_numpy()[keep].astype(int)); jj.append(b[k].to_numpy()[keep].astype(int)); ww.append(w[keep])
    n = len(symbols)
    M = sparse.coo_matrix((np.concatenate(ww), (np.concatenate(ii), np.concatenate(jj))), shape=(n, n)).tocsr()
    return M.maximum(M.T)


