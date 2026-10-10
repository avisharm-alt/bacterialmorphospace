"""Assemble per-gene cell outlines and the gene inputs used to condition the outline diffusion model.

Outlines come from keio_g2p.modal_segment (derived/outlines/plate<P>_row<R>.npz, one (n, 64, 2) array per well).
Wells are mapped to deleted genes with Campos et al. 2018 Dataset EV1 (raw/campos_raw_data.csv: plate, well, gene,
elapsed imaging time).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import data, network

MIN_CELLS = 30


@dataclass
class OutlineData:
    genes: pd.DataFrame            # gene, accession, bnum, hit, plate_idx, time (one row per gene, b-number order)
    cells: list[np.ndarray]        # per gene: (n, 128) float32 flattened canonical outlines (pixels)
    fitness: np.ndarray            # per gene: RB-TnSeq fitness profile (183 conditions; zeros when missing)
    string: "sparse.csr_matrix"    # gene x gene STRING weights (no text mining or curated databases)
    n_plates: int


def load_wells(derived: Path) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for f in sorted((derived / "outlines").glob("plate*_row*.npz")):
        plate = re.match(r"plate(\d+)_row", f.name).group(1)
        z = np.load(f)
        for w in z.files:
            out[(plate, w)] = z[w].astype(np.float32).reshape(len(z[w]), -1)
    return out


STRING_CHANNELS = [c for c in network.CHANNELS if c not in ("textmining", "database")]


def load(raw: Path, derived: Path) -> OutlineData:
    k = data.load(raw)
    wells = load_wells(derived)
    meta = pd.read_csv(raw / "campos_raw_data.csv")
    meta["Plate nb"] = meta["Plate nb"].astype(str)
    meta = meta[meta["Plate nb"].str.isdigit()]
    plates = sorted(meta["Plate nb"].unique(), key=int)
    pidx = {p: i for i, p in enumerate(plates)}
    t = meta["Elapsed time"].astype(float)
    meta["time_z"] = (t - t.mean()) / t.std()
    by_gene: dict[str, list] = {}
    for g, p, w, tz in zip(meta["Gene deletion"].astype(str), meta["Plate nb"], meta["Well nb"], meta["time_z"]):
        c = wells.get((p, w))
        if c is not None and len(c):
            by_gene.setdefault(g, []).append((c, pidx[p], tz))
    rows, cells = [], []
    for i, r in k.genes.iterrows():
        recs = by_gene.get(r["gene"])
        if not recs:
            continue
        c = np.concatenate([x[0] for x in recs])
        if len(c) < MIN_CELLS:
            continue
        # a gene imaged in several wells keeps the plate/time of its largest well
        best = max(recs, key=lambda x: len(x[0]))
        rows.append({**r.to_dict(), "plate_idx": best[1], "time": best[2], "n_cells": len(c)})
        cells.append(c)
    genes = pd.DataFrame(rows).reset_index(drop=True)
    b = genes["bnum"].tolist()
    F, _ = network.fitness_matrix(raw, b)
    return OutlineData(genes=genes, cells=cells, fitness=F, string=network.string_matrix(raw, b, STRING_CHANNELS),
                       n_plates=len(plates))
