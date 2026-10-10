"""Load the Keio cell-shape scores (Campos et al. 2018, Dataset EV2) and join them to gene features.

Inputs (all under one raw directory):
- MSB-14-e7573-s004.xlsx  Dataset EV2, sheet "Scores": one row per imaged strain, z-like scores vs wild type.
- uniprot_k12.tsv         UniProt proteome UP000000625 (E. coli K-12 MG1655): accession, gene names, b-number, sequence.
- ecocyc.gaf.gz           GO annotations for E. coli K-12 from current.geneontology.org.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Cell-shape features only: the mean and cell-to-cell CV of length, width, area, volume, surface area, perimeter,
# surface-to-volume, circularity and aspect ratio. Growth, nucleoid and cell-cycle columns are left out.
SHAPE = ["<L>", "CV_L", "<W>", "CV_W", "<A>", "CV_A", "<V>", "CV_V", "<SA>", "CV_SA", "<P>", "CV_P",
         "<SA/V>", "CV_SA/V", "<C>", "CV_C", "<Ar>", "CV_Ar"]

# GO evidence codes that come from a mutant phenotype or expression experiment; dropped in the "no phenotype" variant
# so that a gene annotated *because* its knockout looks odd cannot feed its own answer back in.
PHENOTYPE_EVIDENCE = {"IMP", "IGI", "IEP", "HMP", "HGI", "HEP"}


@dataclass
class KeioData:
    genes: pd.DataFrame       # one row per knocked-out gene: gene, accession, bnum, pos (chromosome order), hit
    Y: np.ndarray             # genes x len(SHAPE) shape scores
    sequences: list[str]


def load_scores(raw: Path) -> pd.DataFrame:
    d = pd.read_excel(raw / "MSB-14-e7573-s004.xlsx", sheet_name="Scores")
    d = d.rename(columns={"Gene deletion": "gene"})
    d = d[~d["gene"].astype(str).str.match(r"^WT\d+")].copy()
    d["island"] = d["MorphoIsland"].astype(str)
    # The paper's shape mutants: strains placed in a morphological island other than wild type ('0').
    # '-' = not a mutant, '-1' = mutant but not clustered (kept as a mutant).
    d["hit"] = ~d["island"].isin(["-", "0"])
    # A handful of genes appear in several wells; average their scores and call them a hit if any well was.
    agg = d.groupby("gene").agg({**{c: "mean" for c in SHAPE}, "hit": "max"}).reset_index()
    return agg


def load_uniprot(raw: Path) -> pd.DataFrame:
    u = pd.read_csv(raw / "uniprot_k12.tsv", sep="\t")
    u["bnum"] = u["Gene Names (ordered locus)"].fillna("").str.extract(r"\b(b\d{4})\b")[0]
    return u.dropna(subset=["bnum"])


def match_genes(scores: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Map Keio gene names to UniProt by primary name, then by synonym (case-insensitive, unique matches only)."""
    prim = {str(g).lower(): i for i, g in zip(uni.index, uni["Gene Names (primary)"]) if isinstance(g, str)}
    syn: dict[str, list[int]] = {}
    for i, s in zip(uni.index, uni["Gene Names (synonym)"]):
        for name in str(s).split() if isinstance(s, str) else []:
            syn.setdefault(name.lower(), []).append(i)
    jw: dict[str, int] = {}
    for i, o in zip(uni.index, uni["Gene Names (ordered locus)"]):
        for name in str(o).split() if isinstance(o, str) else []:
            if name.startswith("JW"):
                jw[name.split(".")[0]] = i
    idx = []
    for g in scores["gene"].astype(str):
        k = g.lower()
        if g.startswith("JW"):  # Keio strains named by JW locus when the gene had no name in 2006
            idx.append(jw.get(g.split(".")[0], -1))
        elif k in prim:
            idx.append(prim[k])
        elif len(syn.get(k, [])) == 1:
            idx.append(syn[k][0])
        else:
            idx.append(-1)
    s = scores.copy()
    s["uidx"] = idx
    s = s[s["uidx"] >= 0]
    s = s.drop_duplicates("uidx", keep=False)  # two Keio names on one protein: ambiguous, drop both
    return s


def load(raw: Path) -> KeioData:
    scores = load_scores(raw)
    uni = load_uniprot(raw)
    s = match_genes(scores, uni)
    u = uni.loc[s["uidx"]]
    genes = pd.DataFrame({"gene": s["gene"].to_numpy(), "accession": u["Entry"].to_numpy(),
                          "bnum": u["bnum"].to_numpy(), "hit": s["hit"].to_numpy()})
    genes["pos"] = genes["bnum"].str[1:].astype(int)
    order = np.argsort(genes["pos"].to_numpy(), kind="stable")
    genes = genes.iloc[order].reset_index(drop=True)
    Y = s[SHAPE].to_numpy(dtype=float)[order]
    return KeioData(genes=genes, Y=Y, sequences=u["Sequence"].to_numpy()[order].tolist())


def go_matrix(raw: Path, genes: pd.DataFrame, drop_phenotype: bool, min_genes: int = 5) -> tuple[np.ndarray, list[str]]:
    """Binary gene x GO-term matrix (direct annotations only), terms annotated to at least `min_genes` genes.

    UniProtKB rows are matched by accession, EcoCyc and ComplexPortal rows by gene symbol (column 3, case-insensitive).
    """
    acc = {a: i for i, a in enumerate(genes["accession"])}
    sym = {str(g).lower(): i for i, g in enumerate(genes["gene"])}
    pairs = set()
    with gzip.open(raw / "ecocyc.gaf.gz", "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 7 or "NOT" in p[3]:
                continue
            if drop_phenotype and p[6] in PHENOTYPE_EVIDENCE:
                continue
            i = acc.get(p[1], sym.get(p[2].lower()))
            if i is not None:
                pairs.add((i, p[4]))
    terms = pd.Series([t for _, t in pairs]).value_counts()
    keep = sorted(terms[terms >= min_genes].index)
    col = {t: j for j, t in enumerate(keep)}
    X = np.zeros((len(genes), len(keep)), dtype=np.float32)
    for i, t in pairs:
        if t in col:
            X[i, col[t]] = 1.0
    return X, keep


def load_esm(path: Path, accessions: list[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    pos = {a: i for i, a in enumerate(z["accession"])}
    return z["emb"][[pos[a] for a in accessions]]


