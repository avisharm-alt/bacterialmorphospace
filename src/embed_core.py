"""Pure logic for the Evo 2 embedding run: no Modal, no torch, importable anywhere.

Everything the Modal app (`src/evo2_modal.py`) needs that can be tested without a GPU lives here:
NCBI path handling, download verification, window placement, stratified sampling, cost
estimation and the phylum-grouped classifier evaluation.

Window design. For a run at depth `n` the windows are evenly spaced over the longest contig.
The sweep embeds ONE pool of `max(depths)` evenly spaced windows per genome and derives every
smaller depth as an evenly spaced subset of that pool (`subset_indices`). That costs
`max(depths)` forward passes per genome instead of `sum(depths)`, and it makes the depth
comparison paired. A subset window sits within half a pool step of where an independent
linspace(n) window would be (0.5% of the contig length for a 100-window pool).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants fixed by the working recipe
# ---------------------------------------------------------------------------
MODEL_NAME = "evo2_7b_base"  # the only config with use_fp8_input_projections: False
LAYER = "blocks.28.mlp.l3"
EMB_DIM = 4096
WINDOW = 8192
DEFAULT_DEPTHS = (10, 25, 50, 100)

# Measured on A100-40GB (Colab), used only for pre-flight estimates.
S_PER_WINDOW = 0.77
S_OVERHEAD_PER_GENOME = 2.0  # contig parse, tokenisation, volume commit
S_LOAD_PER_CONTAINER = 120.0  # weights from the volume + scale-down idle
USD_PER_GPU_HOUR = 2.10  # Modal A100-40GB list price (0.000583 USD/s); the dashboard is authoritative

N_PANEL = 2580
MIN_GENOME_BYTES = 20_000  # no bacterial assembly is smaller; catches empty / stub files
MIN_PHYLUM_SPECIES = 10
FTP_HOST = "https://ftp.ncbi.nlm.nih.gov"


# ---------------------------------------------------------------------------
# NCBI paths and download verification
# ---------------------------------------------------------------------------
_ACC_RE = re.compile(r"(GC[AF])_(\d{3})(\d{3})(\d{3})\.(\d+)")


def ncbi_parent_url(acc: str) -> str:
    """Directory that contains the per-assembly folder, e.g. .../genomes/all/GCF/000/005/845/."""
    m = _ACC_RE.fullmatch(acc)
    if not m:
        raise ValueError(f"not a versioned NCBI assembly accession: {acc!r}")
    db, a, b, c, _ = m.groups()
    return f"{FTP_HOST}/genomes/all/{db}/{a}/{b}/{c}/"


def find_assembly_dir(listing_html: str, acc: str) -> str | None:
    """Pick `<acc>_<asm name>` out of an NCBI directory listing. The name is not derivable from the accession."""
    hits = re.findall(rf'href="({re.escape(acc)}_[^"/]+)/?"', listing_html)
    return sorted(set(hits))[0] if hits else None


def parse_md5_file(text: str) -> dict[str, str]:
    """`md5checksums.txt` lines look like `<md5>  ./<file>`; returns {filename: md5}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].removeprefix("./")] = parts[0].lower()
    return out


def md5_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_download(path: str | Path, expected_bytes: int | None = None,
                    expected_md5: str | None = None) -> None:
    """Raise unless `path` is a plausible, complete genome file.

    curl exits 0 on empty and truncated transfers, so success is never inferred from an exit
    status: the file must exist, exceed MIN_GENOME_BYTES, match Content-Length when the server
    sent one, match NCBI's md5 when available, and decompress to FASTA.
    """
    p = Path(path)
    if not p.exists():
        raise IOError(f"{p} does not exist")
    size = p.stat().st_size
    if size < MIN_GENOME_BYTES:
        raise IOError(f"{p.name}: {size} bytes is below the {MIN_GENOME_BYTES}-byte floor (empty or stub download)")
    if expected_bytes is not None and size != expected_bytes:
        raise IOError(f"{p.name}: {size} bytes on disk, server announced {expected_bytes} (truncated)")
    if expected_md5 is not None and md5_of(p) != expected_md5:
        raise IOError(f"{p.name}: md5 mismatch against NCBI md5checksums.txt")
    try:
        with gzip.open(p, "rb") as f:
            first = f.read(1)
            while f.read(1 << 20):  # decompress to the end: proves the gzip stream is complete
                pass
    except (OSError, EOFError) as e:
        raise IOError(f"{p.name}: corrupt or truncated gzip ({e})") from e
    if first != b">":
        raise IOError(f"{p.name}: does not decompress to FASTA")


def longest_contig(fasta_gz: str | Path) -> tuple[str, str]:
    """(header id, uppercase sequence) of the longest record in a gzipped FASTA."""
    best_id, best_len, best_parts = "", -1, []
    cur_id, cur_parts, cur_len = "", [], 0

    def close():
        nonlocal best_id, best_len, best_parts
        if cur_len > best_len:
            best_id, best_len, best_parts = cur_id, cur_len, cur_parts

    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id:
                    close()
                cur_id, cur_parts, cur_len = line[1:].split()[0], [], 0
            else:
                s = line.strip()
                cur_parts.append(s)
                cur_len += len(s)
        if cur_id:
            close()
    if best_len < 0:
        raise ValueError(f"no FASTA records in {fasta_gz}")
    return best_id, "".join(best_parts).upper()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def window_starts(contig_len: int, n: int, window: int = WINDOW) -> list[int]:
    """n evenly spaced window starts covering the whole contig, first at 0 and last flush with the end."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if contig_len < window:
        raise ValueError(f"longest contig is {contig_len} bp, shorter than one {window} bp window")
    span = contig_len - window
    if n == 1:
        return [span // 2]
    return [int(i * span / (n - 1) + 0.5) for i in range(n)]


def subset_indices(pool_n: int, n: int) -> list[int]:
    """Evenly spaced, distinct indices into a pool of `pool_n` windows (first and last always included)."""
    if not 1 <= n <= pool_n:
        raise ValueError(f"depth {n} is not within a pool of {pool_n}")
    if n == pool_n:
        return list(range(pool_n))
    if n == 1:
        return [pool_n // 2]
    return [int(i * (pool_n - 1) / (n - 1) + 0.5) for i in range(n)]


def depth_seconds(window_times: list[float], pool_n: int, n: int) -> float:
    """GPU seconds the forward passes of depth-n's windows took, from per-window timings of the pool."""
    return sum(window_times[i] for i in subset_indices(pool_n, n))


# ---------------------------------------------------------------------------
# Sampling and grouping
# ---------------------------------------------------------------------------
def pool_groups(phyla: list[str], min_species: int = MIN_PHYLUM_SPECIES, other: str = "other") -> list[str]:
    """Pool phyla with fewer than `min_species` species (in the whole panel) into `other`."""
    counts: dict[str, int] = {}
    for p in phyla:
        counts[p] = counts.get(p, 0) + 1
    return [p if counts[p] >= min_species else other for p in phyla]


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Cap-and-fill allocation: every group gets min(size, cap) with the largest cap that fits `total`,
    then leftovers go one at a time to the groups with the most unsampled species.

    Proportional sampling would spend 80% of a 200-genome budget on four phyla; this spreads it.
    """
    total = min(total, sum(sizes.values()))
    cap = max(c for c in range(max(sizes.values()) + 1) if sum(min(s, c) for s in sizes.values()) <= total)
    alloc = {g: min(s, cap) for g, s in sizes.items()}
    rem = total - sum(alloc.values())
    while rem > 0:
        for g in sorted(sizes, key=lambda g: (-(sizes[g] - alloc[g]), g)):
            if rem == 0:
                break
            if alloc[g] < sizes[g]:
                alloc[g] += 1
                rem -= 1
    return alloc


def stratified_sample(rows: list[dict], total: int, seed: int, group_key: str = "group") -> list[dict]:
    """Deterministic sample of `total` rows stratified by `group_key`; returned sorted by accession."""
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r[group_key], []).append(r)
    alloc = allocate({g: len(v) for g, v in by_group.items()}, total)
    rng = random.Random(seed)
    picked = []
    for g in sorted(by_group):
        pool = sorted(by_group[g], key=lambda r: r["ncbi_assembly_accession"])
        picked += rng.sample(pool, alloc[g])
    return sorted(picked, key=lambda r: r["ncbi_assembly_accession"])


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def estimate_gpu(n_genomes: int, windows_per_genome: int, n_containers: int = 1,
                 s_per_window: float = S_PER_WINDOW) -> dict:
    seconds = (n_genomes * (windows_per_genome * s_per_window + S_OVERHEAD_PER_GENOME)
               + n_containers * S_LOAD_PER_CONTAINER)
    hours = seconds / 3600
    return {"gpu_seconds": seconds, "gpu_hours": hours, "usd": hours * USD_PER_GPU_HOUR}


def usd(gpu_seconds: float) -> float:
    return gpu_seconds / 3600 * USD_PER_GPU_HOUR


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_depth(X, y, groups, n_splits: int = 5) -> dict:
    """roc_auc of scaled logistic regression under GroupKFold (groups = pooled phylum).

    Folds are computed per fold rather than through cross_val_score so that a test fold with a
    single class (possible when a phylum is uniformly motile or not) is skipped and counted,
    not silently scored as NaN. GroupKFold is deterministic, so every depth sees identical folds.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    k = len(aucs)
    return {
        "auc_mean": float(np.mean(aucs)) if k else float("nan"),
        "auc_std": float(np.std(aucs, ddof=1)) if k > 1 else float("nan"),
        "folds_scored": k,
        "folds_total": n_splits,
    }


def select_depth(results: list[dict]) -> dict:
    """One-standard-error rule: the smallest depth whose mean AUC is within one SE of the best depth's."""
    scored = [r for r in results if not math.isnan(r["auc_mean"])]
    best = max(scored, key=lambda r: r["auc_mean"])
    se = best["auc_std"] / math.sqrt(best["folds_scored"]) if best["folds_scored"] > 1 else 0.0
    ok = [r for r in scored if r["auc_mean"] >= best["auc_mean"] - se]
    chosen = min(ok, key=lambda r: r["depth"])
    return {"selected_depth": chosen["depth"], "best_depth": best["depth"], "one_se": se,
            "rule": "smallest depth with mean AUC within one SE of the best depth"}


def project_hours(s_per_genome: float, n: int = N_PANEL) -> float:
    return s_per_genome * n / 3600


def dump_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
