"""Pure logic for the Evo 2 embedding run: no Modal, no torch, importable anywhere.

Everything the Modal app (`src/evo2_modal.py`) needs that can be tested without a GPU lives here:
NCBI path handling, download verification, window placement, stratified sampling, cost
estimation and the phylum-grouped classifier evaluation.

Window design. Windows are drawn from ALL contigs of at least one window (8,192 bp), not from
the longest contig alone: a draft assembly's longest contig is one fragment, and how fragmented
an assembly is tracks how well studied the organism is, which can correlate with the traits.
The usable contigs are laid end to end on one axis of length T (`usable_len`) and `n` points are
spaced evenly along it (systematic sampling), so each contig receives windows in proportion to its
length. Each window is placed on the contig that owns its point, clamped inside it, so no window
spans a junction. Contigs shorter than one window are dropped, and only so much sequence is
"usable": `frac_overlapping` compares n*8192 against T.

The sweep embeds ONE pool of `max(depths)` windows per genome and derives every smaller depth as
an evenly spaced subset of that pool (`subset_indices`). That costs `max(depths)` forward passes
per genome instead of `sum(depths)`, and it makes the depth comparison paired. A subset window
sits within half a pool step of where an independent n-window draw would be (0.5% of T for a
100-window pool).
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
# Recorded in every meta file; the evaluation refuses to mix checkpoints from different sampling schemes.
SAMPLING = "all_contigs_systematic_v1"

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


def read_contigs(fasta_gz: str | Path, min_len: int = WINDOW) -> tuple[list[tuple[str, str]], int, int]:
    """(usable contigs, total bp, number of contigs) from a gzipped FASTA.

    Usable contigs are those of at least `min_len` bp, as (header id, uppercase sequence) in file order.
    """
    usable, total, n_contigs = [], 0, 0
    cur_id, parts, cur_len = None, [], 0

    def close():
        nonlocal total, n_contigs
        total += cur_len
        n_contigs += 1
        if cur_len >= min_len:
            usable.append((cur_id, "".join(parts).upper()))

    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    close()
                cur_id, parts, cur_len = (line[1:].split() or [""])[0], [], 0
            else:
                s = line.strip()
                parts.append(s)
                cur_len += len(s)
        if cur_id is not None:
            close()
    if n_contigs == 0:
        raise ValueError(f"no FASTA records in {fasta_gz}")
    return usable, total, n_contigs


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def pool_windows(contig_lens: list[int], n: int, window: int = WINDOW) -> list[tuple[int, int]]:
    """n windows as (contig index, start) drawn across ALL contigs of at least `window` bp.

    Usable contigs are laid end to end (length T) and n points are spaced evenly from 0 to T-1
    (systematic sampling), so a contig gets windows in proportion to its length. The window for a
    point is centred on it and clamped inside its own contig, so no window crosses a junction.
    The first and last points are at the two ends of the axis, which is what makes an evenly
    spaced subset of a larger pool line up with an independent smaller draw (`subset_indices`).
    Contig indices refer to the input list; contigs shorter than `window` are skipped.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    usable = [(i, L) for i, L in enumerate(contig_lens) if L >= window]
    if not usable:
        raise ValueError(f"no contig of at least {window} bp (longest is {max(contig_lens, default=0)} bp)")
    total = sum(L for _, L in usable)
    out, k, offset = [], 0, 0  # k indexes `usable`; offset is where usable[k] starts on the axis
    for j in range(n):
        p = total / 2 if n == 1 else j * (total - 1) / (n - 1)
        while k < len(usable) - 1 and p >= offset + usable[k][1]:
            offset += usable[k][1]
            k += 1
        idx, L = usable[k]
        out.append((idx, int(min(max(p - offset - window / 2, 0), L - window))))
    return out


def usable_len(contig_lens: list[int], window: int = WINDOW) -> int:
    """Total bp on contigs that can host a window: the sequence a depth is actually drawn from."""
    return sum(L for L in contig_lens if L >= window)


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


# ---------------------------------------------------------------------------
# Diagnostics for a suspicious result (all CPU-side, read the checkpoints only)
# ---------------------------------------------------------------------------
def _pipeline(C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def manual_auc(y, score) -> float:
    """AUC as the Mann-Whitney rank statistic for the POSITIVE = True class, independent of sklearn."""
    import numpy as np
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((rankdata(score)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fold_report(X, y, groups, n_splits: int = 5, C: float = 1.0) -> dict:
    """Per-fold structure of the grouped CV, with the class sklearn treats as positive made explicit.

    For each GroupKFold fold: the held-out groups, train/test sizes, motility prevalence in train and
    test, the held-out AUC (sklearn and an independent rank-statistic implementation), and the mean
    predicted P(positive) on the test fold. `classes` is the fitted classifier's class order, so
    `predict_proba[:, 1]` is the probability of `classes[1]`.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X, y, groups = np.asarray(X), np.asarray(y).astype(bool), np.asarray(groups)
    folds, classes = [], None
    for i, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        model = _pipeline(C).fit(X[tr], y[tr])
        classes = [bool(c) for c in model.classes_]
        row = {"fold": i, "test_groups": sorted(set(groups[te])), "n_train": len(tr), "n_test": len(te),
               "prev_train": float(y[tr].mean()), "prev_test": float(y[te].mean())}
        if len(set(y[te])) == 2:
            p = model.predict_proba(X[te])[:, 1]
            row.update(auc=float(roc_auc_score(y[te], p)), auc_manual=manual_auc(y[te], p),
                       mean_p_motile=float(p.mean()), mean_p_motile_actual_yes=float(p[y[te]].mean()),
                       mean_p_motile_actual_no=float(p[~y[te]].mean()))
        else:
            row.update(auc=float("nan"), auc_manual=float("nan"), mean_p_motile=float("nan"),
                       mean_p_motile_actual_yes=float("nan"), mean_p_motile_actual_no=float("nan"))
        folds.append(row)
    aucs = [f["auc"] for f in folds if f["auc"] == f["auc"]]
    return {"folds": folds, "classes": classes, "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "prev_test_range": [min(f["prev_test"] for f in folds), max(f["prev_test"] for f in folds)]}


def permutation_null(X, y, groups, n_splits: int = 5, n_perm: int = 50, seed: int = 0,
                     within_groups: bool = False, n_jobs: int = 1) -> dict:
    """Distribution of the grouped-CV mean AUC when labels carry no information about the embeddings.

    within_groups=False shuffles labels over all genomes (also destroys the phylum-prevalence
    structure). within_groups=True shuffles only inside each group, keeping every phylum's prevalence,
    which is the right null for "is there signal beyond the phylum prior?". Both use the identical folds.
    """
    import numpy as np
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    X, y, groups = np.asarray(X), np.asarray(y).astype(bool), np.asarray(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    perms = []
    for _ in range(n_perm):
        yp = y.copy()
        if within_groups:
            for idx in idx_by_group:
                yp[idx] = y[rng.permutation(idx)]
        else:
            yp = y[rng.permutation(len(y))]
        perms.append(yp)
    got = Parallel(n_jobs=n_jobs)(delayed(evaluate_depth)(X, yp, groups, n_splits) for yp in perms)
    a = np.asarray([g["auc_mean"] for g in got if g["auc_mean"] == g["auc_mean"]])
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "q05": float(np.quantile(a, .05)),
            "q95": float(np.quantile(a, .95)), "n": int(len(a)), "values": [float(v) for v in a]}


def ungrouped_auc(X, y, n_splits: int = 5, repeats: int = 3, seed: int = 0, C: float = 1.0) -> dict:
    """Same model, ordinary stratified CV (phyla shared between train and test), for contrast with GroupKFold."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y = np.asarray(X), np.asarray(y).astype(bool)
    means = []
    for r in range(repeats):
        aucs = []
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y):
            aucs.append(roc_auc_score(y[te], _pipeline(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
        means.append(float(np.mean(aucs)))
    return {"auc_mean": float(np.mean(means)), "auc_sd_over_repeats": float(np.std(means, ddof=1)) if repeats > 1 else float("nan")}


def gc_content(fasta_gz: str | Path) -> float:
    """G+C fraction over all unambiguous bases of a gzipped FASTA (whole genome, every contig)."""
    gc = at = 0
    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            if line[0] != ">":
                u = line.upper()
                g, c, a, t = u.count("G"), u.count("C"), u.count("A"), u.count("T")
                gc += g + c
                at += a + t
    if gc + at == 0:
        raise ValueError(f"no ACGT bases in {fasta_gz}")
    return gc / (gc + at)


def pairing_control(X, gc, groups, n_splits: int = 5, alpha: float = 1000.0, seed: int = 0) -> dict:
    """Do the embeddings belong to their genomes? Predict each genome's GC from its embedding.

    Independent of the motility labels: genome GC is read from the FASTA, embeddings from the volume,
    both keyed by accession. Evo 2 embeddings carry GC strongly, so correctly paired rows give a high
    out-of-clade Spearman r; rows paired to the wrong genome give ~0. `shuffled_pairing_r` re-runs the
    same model with the GC values permuted across genomes as the null.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, gc, groups = np.asarray(X), np.asarray(gc, dtype=float), np.asarray(groups)

    def r_of(target):
        pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), X, target,
                                 groups=groups, cv=GroupKFold(n_splits=n_splits))
        return float(spearmanr(pred, target).statistic)

    rng = np.random.default_rng(seed)
    return {"spearman_r": r_of(gc), "shuffled_pairing_r": r_of(gc[rng.permutation(len(gc))])}


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


def plain(obj):
    """Reduce `obj` to builtin str/int/float/bool/None/dict/list, or raise TypeError.

    Modal pickles every remote return value back to the local process, which has no torch or numpy.
    A str subclass such as torch.__version__ (TorchVersion), a numpy scalar, or a torch dtype pickles
    by reference to its defining module and fails to deserialize locally. Everything that leaves a
    container goes through here, so a stray non-primitive fails inside the container with its path,
    not as an opaque DeserializationError on the far side.
    """
    def walk(x, path):
        if x is None or type(x) in (bool, int, float, str):
            return x
        if isinstance(x, bool):
            return bool(x)
        if isinstance(x, str):
            return str(x)
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return float(x)
        if isinstance(x, dict):
            return {walk(k, path + ".<key>"): walk(v, f"{path}[{k!r}]") for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(x)]
        if hasattr(x, "item") and type(x).__module__.split(".")[0] == "numpy":
            return walk(x.item(), path)  # numpy scalar -> builtin
        raise TypeError(f"non-primitive {type(x).__module__}.{type(x).__qualname__} at {path}: convert with str()/float() before returning")

    return walk(obj, "return")


def dump_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
