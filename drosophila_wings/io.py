"""Readers for wing landmark tables, DGRP genotypes (plink .bed, VCF, .tgeno) and line covariates.

Every reader returns line IDs normalised by `line_id` so that "RAL-21", "line_21", "DGRP_021" and 21 all match.
Genotypes come back as an int8 matrix (lines x variants) of alternate-allele counts 0/1/2, with -1 for missing.
"""
from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LINE_COLS = ("line", "dgrp", "dgrp_line", "line_id", "ral", "genotype", "strain")
SEX_COLS = ("sex",)
_XY_PATTERNS = (
    re.compile(r"^(?:lm|landmark|p|pt)?[._]?(?P<axis>[xy])[._]?(?P<idx>\d+)$", re.I),  # x1, X01, LM_x1
    re.compile(r"^(?:lm|landmark|p|pt)?[._]?(?P<idx>\d+)[._]?(?P<axis>[xy])$", re.I),  # 1x, LM1.X, P1_y
)


def line_id(value) -> str:
    """Canonical DGRP line ID: the line number without prefix or leading zeros ("RAL-021" -> "21")."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    m = re.search(r"(\d+)\s*$", s)
    if not m:
        raise ValueError(f"cannot read a DGRP line number from {value!r}")
    return str(int(m.group(1)))


def _find_col(df: pd.DataFrame, wanted: str | None, candidates: tuple[str, ...], what: str) -> str | None:
    if wanted:
        if wanted not in df.columns:
            raise KeyError(f"{what} column {wanted!r} not in {list(df.columns)[:20]}...")
        return wanted
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def landmark_columns(columns) -> tuple[list[str], list[str]]:
    """Pair x/y landmark columns by index. Returns (x columns, y columns) in landmark order."""
    xs, ys = {}, {}
    for c in columns:
        for pat in _XY_PATTERNS:
            m = pat.match(str(c))
            if m:
                (xs if m.group("axis").lower() == "x" else ys)[int(m.group("idx"))] = c
                break
    idx = sorted(set(xs) & set(ys))
    if set(xs) ^ set(ys):
        log.warning("unpaired landmark columns ignored: %s", sorted(set(xs) ^ set(ys))[:10])
    return [xs[i] for i in idx], [ys[i] for i in idx]


def sex_code(value) -> str:
    """F or M from labels such as "F", "female", "M", "male" and Pitchers' "probablyM"; anything else is kept upper-cased."""
    s = str(value).strip().upper()
    if s in ("F", "FEMALE"):
        return "F"
    if s in ("M", "MALE") or s.endswith("M"):
        return "M"
    return s


@dataclass
class Wings:
    """Wing-level landmark data: coords (n, k, 2) and a metadata frame with at least `line`."""
    coords: np.ndarray
    meta: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    sep = "\t" if suf.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def load_wings(path: str | Path, line_col: str | None = None, sex_col: str | None = None,
               keep_cols: tuple[str, ...] = ()) -> Wings:
    """Read a wing table (one row per wing, x/y columns per landmark). Rows with any missing coordinate are dropped."""
    df = read_table(path)
    lc = _find_col(df, line_col, LINE_COLS, "line")
    if lc is None:
        raise KeyError(f"no line column found in {path}; pass --line-col (columns: {list(df.columns)[:30]})")
    sc = _find_col(df, sex_col, SEX_COLS, "sex")
    xc, yc = landmark_columns(df.columns)
    if len(xc) < 3:
        raise KeyError(f"found {len(xc)} landmark x/y pairs in {path}; expected columns like x1..xK, y1..yK")
    coords = np.stack([df[xc].to_numpy(float), df[yc].to_numpy(float)], axis=2)
    ok = np.isfinite(coords).all(axis=(1, 2)) & df[lc].notna().to_numpy()
    if (~ok).any():
        log.warning("dropping %d of %d wings with missing landmarks or line", (~ok).sum(), len(ok))
    meta = pd.DataFrame({"line": df.loc[ok, lc].map(line_id).to_numpy()})
    if sc is not None:
        meta["sex"] = df.loc[ok, sc].map(sex_code).to_numpy()
    for c in keep_cols:
        meta[c] = df.loc[ok, c].to_numpy()
    log.info("%s: %d wings, %d lines, %d landmarks", Path(path).name, ok.sum(), meta["line"].nunique(), len(xc))
    return Wings(coords[ok], meta.reset_index(drop=True))


# ---------------------------------------------------------------- genotypes

@dataclass
class Genotypes:
    lines: list[str]
    G: np.ndarray  # int8 (lines x variants), alt-allele count, -1 missing
    variants: pd.DataFrame  # chrom, pos, id

    def subset(self, lines: list[str]) -> "Genotypes":
        pos = {l: i for i, l in enumerate(self.lines)}
        return Genotypes(list(lines), self.G[[pos[l] for l in lines]], self.variants)


def _filter_block(G: np.ndarray, maf: float, max_missing: float) -> np.ndarray:
    """Boolean mask over columns of an int8 block: passes MAF and missingness and is polymorphic."""
    miss = G < 0
    n_obs = (~miss).sum(axis=0)
    alt = np.where(miss, 0, G).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = alt / (2 * n_obs)
    q = np.minimum(p, 1 - p)
    return (miss.mean(axis=0) <= max_missing) & (q >= maf) & (n_obs > 0)


def read_plink(prefix: str | Path, maf: float = 0.05, max_missing: float = 0.2, thin: int = 1,
               chunk: int = 200_000) -> Genotypes:
    """Read a SNP-major plink .bed/.bim/.fam. Counts are of the .bim allele-2 (A2) column, treated as ALT."""
    prefix = str(prefix).removesuffix(".bed")
    fam = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None, usecols=[1], names=["iid"], dtype=str)
    bim = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None, usecols=[0, 1, 3], names=["chrom", "id", "pos"],
                      dtype={"chrom": str, "id": str, "pos": np.int64})
    n, m = len(fam), len(bim)
    bpv = (n + 3) // 4
    raw = np.memmap(prefix + ".bed", dtype=np.uint8, mode="r")
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise ValueError(f"{prefix}.bed is not a SNP-major plink bed file")
    if len(raw) != 3 + bpv * m:
        raise ValueError(f"{prefix}.bed has {len(raw) - 3} bytes, expected {bpv * m} for {n} samples x {m} variants")
    # plink 2-bit codes: 00 hom A1, 01 missing, 10 het, 11 hom A2  ->  A2 count 0, -1, 1, 2
    lut = np.array([0, -1, 1, 2], dtype=np.int8)
    keep_blocks, keep_idx = [], []
    for start in range(0, m, chunk):
        stop = min(m, start + chunk)
        idx = np.arange(start, stop)[::thin] if thin > 1 else np.arange(start, stop)
        block = np.asarray(raw[3 + start * bpv: 3 + stop * bpv]).reshape(stop - start, bpv)
        if thin > 1:
            block = block[idx - start]
        codes = np.stack([(block >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(len(idx), -1)[:, :n]
        g = lut[codes].T
        ok = _filter_block(g, maf, max_missing)
        keep_blocks.append(g[:, ok])
        keep_idx.append(idx[ok])
    G = np.ascontiguousarray(np.concatenate(keep_blocks, axis=1))
    sel = np.concatenate(keep_idx)
    log.info("%s.bed: %d lines, %d of %d variants kept (maf>=%.2f, missing<=%.2f, thin=%d)",
             prefix, n, G.shape[1], m, maf, max_missing, thin)
    return Genotypes([line_id(i) for i in fam["iid"]], G, bim.iloc[sel].reset_index(drop=True))


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_vcf(path: str | Path, maf: float = 0.05, max_missing: float = 0.2, thin: int = 1,
             snps_only: bool = True, block: int = 50_000) -> Genotypes:
    """Stream a (gzipped) VCF, keeping biallelic SNPs. Haploid or diploid GT calls; anything unparseable is missing."""
    path = Path(path)
    gt_map: dict[str, int] = {}

    def code(gt: str) -> int:
        v = gt_map.get(gt)
        if v is None:
            alleles = re.split(r"[/|]", gt)
            if any(a not in ("0", "1") for a in alleles):
                v = -1
            else:
                v = sum(int(a) for a in alleles) * (2 // len(alleles))  # haploid "1" -> 2 (inbred line)
            gt_map[gt] = v
        return v

    samples: list[str] = []
    rows: list[np.ndarray] = []
    meta: list[tuple[str, int, str]] = []
    kept_blocks, kept_meta = [], []
    seen = 0

    def flush():
        if not rows:
            return
        g = np.stack(rows, axis=1)
        ok = _filter_block(g, maf, max_missing)
        kept_blocks.append(g[:, ok])
        kept_meta.extend(m for m, k in zip(meta, ok) if k)
        rows.clear()
        meta.clear()

    with _open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            f = line.rstrip("\n").split("\t")
            ref, alt = f[3], f[4]
            if "," in alt or (snps_only and (len(ref) != 1 or len(alt) != 1)):
                continue
            seen += 1
            if thin > 1 and seen % thin:
                continue
            gt_i = f[8].split(":").index("GT")
            rows.append(np.fromiter((code(s.split(":")[gt_i]) for s in f[9:]), dtype=np.int8, count=len(samples)))
            meta.append((f[0], int(f[1]), f[2]))
            if len(rows) >= block:
                flush()
    flush()
    G = np.ascontiguousarray(np.concatenate(kept_blocks, axis=1)) if kept_blocks else np.zeros((len(samples), 0), np.int8)
    log.info("%s: %d lines, %d of %d biallelic variants kept", path.name, len(samples), G.shape[1], seen)
    return Genotypes([line_id(s) for s in samples], G, pd.DataFrame(kept_meta, columns=["chrom", "pos", "id"]))


def read_tgeno(path: str | Path, maf: float = 0.05, max_missing: float = 0.2, thin: int = 1,
               block: int = 50_000) -> Genotypes:
    """DGRP2 `dgrp2.tgeno`: columns chr pos id ref alt refc altc qual cov line_...; calls 0=ref, 2=alt, 1=het, -=missing."""
    path = Path(path)
    lut = {"0": 0, "1": 1, "2": 2}
    kept_blocks, kept_meta, rows, meta = [], [], [], []
    seen = 0

    def flush():
        if rows:
            g = np.stack(rows, axis=1)
            ok = _filter_block(g, maf, max_missing)
            kept_blocks.append(g[:, ok])
            kept_meta.extend(m for m, k in zip(meta, ok) if k)
            rows.clear()
            meta.clear()

    with _open(path) as fh:
        header = fh.readline().split()
        first = next(i for i, h in enumerate(header) if h.lower().startswith(("line", "ral", "dgrp")))
        samples = header[first:]
        for line in fh:
            f = line.split()
            if len(f[3]) != 1 or len(f[4]) != 1:
                continue
            seen += 1
            if thin > 1 and seen % thin:
                continue
            rows.append(np.fromiter((lut.get(c, -1) for c in f[first:]), dtype=np.int8, count=len(samples)))
            meta.append((f[0], int(f[1]), f[2]))
            if len(rows) >= block:
                flush()
    flush()
    G = np.ascontiguousarray(np.concatenate(kept_blocks, axis=1))
    log.info("%s: %d lines, %d of %d SNPs kept", path.name, len(samples), G.shape[1], seen)
    return Genotypes([line_id(s) for s in samples], G, pd.DataFrame(kept_meta, columns=["chrom", "pos", "id"]))


def read_genotypes(path: str | Path, **kw) -> Genotypes:
    p = str(path)
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=False)
        return Genotypes(list(z["lines"]), z["G"], pd.DataFrame({"chrom": z["chrom"], "pos": z["pos"], "id": z["id"]}))
    if p.endswith((".vcf", ".vcf.gz")):
        return read_vcf(p, **kw)
    if p.endswith((".tgeno", ".tgeno.gz")):
        return read_tgeno(p, **kw)
    return read_plink(p, **kw)


def save_genotypes(g: Genotypes, path: str | Path) -> None:
    np.savez_compressed(path, lines=np.array(g.lines), G=g.G, chrom=g.variants["chrom"].astype(str).to_numpy(),
                        pos=g.variants["pos"].to_numpy(), id=g.variants["id"].astype(str).to_numpy())


# ---------------------------------------------------------------- covariates

def load_covariates(paths: list[str | Path], line_col: str | None = None) -> pd.DataFrame:
    """Merge covariate tables on line (e.g. DGRP2 wolbachia.xlsx and inversion.xlsx); index = canonical line ID.

    Columns are kept as read; `design_matrix` turns them into numbers.
    """
    out = None
    for p in paths:
        df = read_table(p)
        lc = _find_col(df, line_col, LINE_COLS, "line") or df.columns[0]
        df = df.dropna(subset=[lc]).copy()
        df.index = df.pop(lc).map(line_id)
        df.index.name = "line"
        df = df[~df.index.duplicated()]
        out = df if out is None else out.join(df, how="outer", rsuffix=f"_{Path(p).stem}")
    return out if out is not None else pd.DataFrame()


def design_matrix(cov: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """Numeric design for the given lines: numeric columns as is (NaN -> column mean), others one-hot (drop first)."""
    cov = cov.reindex(lines)
    parts = []
    for c in cov.columns:
        s = cov[c]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.fillna(s.mean()).rename(str(c)).to_frame())
        else:
            d = pd.get_dummies(s.astype("string").str.strip().str.upper(), prefix=str(c), drop_first=True, dtype=float)
            parts.append(d)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=lines)
    return X.loc[:, X.std(axis=0) > 0]
