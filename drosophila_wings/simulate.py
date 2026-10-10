"""Synthetic DGRP-like dataset in the raw formats the pipeline reads, for tests and dry runs only.

Inbred lines (allele counts 0/2) with a few closely related pairs, a block of SNPs in an inversion, polygenic effects on
wing shape, sexual dimorphism, per-wing noise, and a random rotation, scale and translation per wing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_plink(prefix: Path, iids: list[str], G: np.ndarray, chrom: list[str], pos: np.ndarray) -> None:
    """SNP-major plink bed; G is lines x variants of A2 counts (-1 missing)."""
    n, m = G.shape
    code = np.select([G == 0, G == 1, G == 2], [0, 2, 3], default=1).astype(np.uint8).T  # (m, n)
    pad = (-n) % 4
    code = np.pad(code, ((0, 0), (0, pad)))
    q = code.reshape(m, -1, 4)
    packed = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).astype(np.uint8)
    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        fh.write(packed.tobytes())
    pd.DataFrame({"fid": iids, "iid": iids, "p": 0, "m": 0, "sex": 0, "ph": -9}).to_csv(
        f"{prefix}.fam", sep=" ", header=False, index=False)
    pd.DataFrame({"chrom": chrom, "id": [f"{c}_{p}_SNP" for c, p in zip(chrom, pos)], "cm": 0, "pos": pos,
                  "a1": "A", "a2": "G"}).to_csv(f"{prefix}.bim", sep="\t", header=False, index=False)


def simulate(rng: np.random.Generator, n_lines: int = 150, n_snps: int = 10000, k: int = 48, n_causal: int = 60,
             h2: float = 0.5, related_share: float = 0.1, wings_per_line: tuple[int, int] = (40, 80)) -> dict:
    p = np.clip(rng.beta(0.6, 0.6, n_snps), 0.05, 0.95)
    G = (rng.random((n_lines, n_snps)) < p).astype(np.int8) * 2
    n_rel = int(related_share * n_lines)
    for i in range(n_rel):  # line i becomes a close relative of line n_rel + i
        src = n_rel + i
        copy = rng.random(n_snps) < 0.85
        G[i, copy] = G[src, copy]
    inv = rng.random(n_lines) < 0.2
    block = slice(n_snps // 2, n_snps // 2 + 150)
    G[inv, block] = np.where(rng.random((inv.sum(), 150)) < 0.9, 2, G[inv, block])
    G[rng.random(G.shape) < 0.01] = -1
    chrom = np.repeat(["2L", "2R", "3L", "3R", "X"], int(np.ceil(n_snps / 5)))[:n_snps].tolist()
    pos = np.arange(n_snps) * 1000 + 1

    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    mean = np.column_stack([1.0 * np.cos(t), 0.45 * np.sin(t) + 0.08 * np.sin(2 * t)])
    Gf = np.where(G < 0, 0, G).astype(float)
    Z = (Gf - Gf.mean(0)) / np.where(Gf.std(0) > 0, Gf.std(0), 1)
    causal = rng.choice(n_snps, n_causal, replace=False)
    effects = rng.normal(size=(n_causal, k * 2))
    g = Z[:, causal] @ effects
    g = (g - g.mean(0)) / g.std() * np.sqrt(h2) * 0.01  # genetic sd 0.01 * sqrt(h2) per coordinate
    noise_sd = 0.01 * np.sqrt(1 - h2)
    sex_shift = rng.normal(scale=0.01, size=k * 2)

    ids = rng.choice(np.arange(1, 1000), n_lines, replace=False)
    rows = []
    for i in range(n_lines):
        for _ in range(rng.integers(*wings_per_line)):
            sex = "F" if rng.random() < 0.5 else "M"
            s = mean.reshape(-1) + g[i] + (sex_shift if sex == "M" else 0) + rng.normal(scale=noise_sd, size=k * 2)
            s = s.reshape(k, 2)
            a = rng.uniform(-0.5, 0.5)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            s = (s @ R) * rng.uniform(400, 600) + rng.uniform(-50, 50, 2)
            rows.append([f"RAL-{ids[i]}", sex, *s[:, 0], *s[:, 1]])
    wings = pd.DataFrame(rows, columns=["Line", "Sex"] + [f"x{j + 1}" for j in range(k)] + [f"y{j + 1}" for j in range(k)])
    cov = pd.DataFrame({"line": [f"line_{x}" for x in ids], "wolbachia": rng.choice(["y", "n"], n_lines),
                        "In_3R_P": np.where(inv, "INV/INV", "ST/ST")})
    return {"G": G, "chrom": chrom, "pos": pos, "ids": ids, "wings": wings, "cov": cov, "genetic": g,
            "related_pairs": [(ids[i], ids[n_rel + i]) for i in range(n_rel)]}


def write_dataset(out: Path, rng: np.random.Generator, **kw) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    sim = simulate(rng, **kw)
    paths = {"wings": out / "sim_wings.csv", "genotypes": out / "sim_dgrp", "covariates": out / "sim_covariates.csv"}
    sim["wings"].to_csv(paths["wings"], index=False)
    write_plink(paths["genotypes"], [f"line_{x}" for x in sim["ids"]], sim["G"], sim["chrom"], sim["pos"])
    sim["cov"].to_csv(paths["covariates"], index=False)
    return paths
