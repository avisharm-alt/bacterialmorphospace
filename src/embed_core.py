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
S_PER_WINDOW = 0.82  # sweep-measured: 0.84 s/window at depth 10, 0.79 at 100 (Colab said 0.77)
S_OVERHEAD_PER_GENOME = 1.0  # contig parse, tokenisation, volume commit (sweep spend implies < 1 s)
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


# ---------------------------------------------------------------------------
# Trait choice: how much of each trait's variance is within phylum?
# ---------------------------------------------------------------------------
def _gini(counts) -> float:
    n = sum(counts)
    return 1 - sum((c / n) ** 2 for c in counts) if n else 0.0


def variance_decomposition(labels: list, groups: list) -> dict:
    """Split a categorical trait's variance (Gini impurity) into between- and within-group parts.

    `within_share` is the fraction of the trait's variance that group identity does NOT explain.
    `off_majority` counts genomes that disagree with their own group's majority class: the amount of
    within-group information a classifier could learn from. High share on a very imbalanced trait
    (86% rods, 93% mesophiles) is still little absolute variance, so both are reported.
    """
    from collections import Counter

    n = len(labels)
    total = Counter(labels)
    by: dict = {}
    for lab, g in zip(labels, groups):
        by.setdefault(g, Counter())[lab] += 1
    g_total = _gini(list(total.values()))
    g_within = sum(sum(c.values()) / n * _gini(list(c.values())) for c in by.values())
    off = sum(sum(c.values()) - max(c.values()) for c in by.values())
    return {"n": n, "gini_total": g_total, "gini_within": g_within,
            "within_share": g_within / g_total if g_total else float("nan"),
            "off_majority": off, "off_majority_frac": off / n, "majority_baseline": max(total.values()) / n}


def auc_se(n_pos: int, n_neg: int, auc: float = 0.7) -> float:
    """Hanley-McNeil standard error of an AUC estimated on n_pos positives and n_neg negatives."""
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var)


def trait_targets(rows: list[dict], traits: list[str], groups: list[str], min_class: int = 100,
                  min_group: int = 30, min_each_class: int = 15, auc_for_se: float = 0.7) -> list[dict]:
    """One row per binary target (a two-class trait, or each class of a multi-class trait vs the rest).

    A group is `testable` when holding it out leaves a within-group ranking that can be scored:
    at least `min_group` genomes with at least `min_each_class` of each class. `se_at_full_panel` is
    the Hanley-McNeil SE of that held-out AUC if every panel genome of the group is used.
    """
    from collections import Counter

    out = []
    for t in traits:
        vals = [r[t] for r in rows]
        cls = Counter(vals)
        if len(cls) == 2:
            targets = [(t, min(cls, key=cls.get), f"{t}: {min(cls, key=cls.get)} vs {max(cls, key=cls.get)}")]
        else:
            targets = [(t, c, f"{t}: {c} vs rest") for c, k in cls.most_common() if k >= min_class]
        for _, pos, name in targets:
            y = [v == pos for v in vals]
            dec = variance_decomposition(y, groups)
            per_group = {}
            for g in sorted(set(groups)):
                k = sum(1 for yy, gg in zip(y, groups) if gg == g)
                p_ = sum(1 for yy, gg in zip(y, groups) if gg == g and yy)
                per_group[g] = (k, p_)
            testable = {g: (k, p_) for g, (k, p_) in per_group.items()
                        if k >= min_group and p_ >= min_each_class and k - p_ >= min_each_class}
            out.append({"target": name, "prevalence": sum(y) / len(y), "n_positive": sum(y), **dec,
                        "testable_groups": len(testable),
                        "testable_genomes": sum(k for k, _ in testable.values()),
                        "per_group": {g: {"n": k, "n_positive": p_, "prevalence": p_ / k,
                                          "testable": g in testable,
                                          "se_at_full_panel": auc_se(p_, k - p_, auc_for_se)}
                                      for g, (k, p_) in per_group.items()}})
    return out


# ---------------------------------------------------------------------------
# Sampling-depth reliability: how much does a genome's vector move with WHICH windows are drawn?
# ---------------------------------------------------------------------------
def cosine_report(mu) -> dict:
    """Mean cosine between different genomes' vectors: raw, mean-centred, and centred + per-dimension scaled.

    Raw vectors share a huge common component (cosine ~1 between unrelated genomes), so any raw
    cosine-based reliability is flat for trivial reasons. Centring across genomes removes that shared
    direction; scaling then gives every dimension equal weight, as StandardScaler does for the classifier.
    """
    import numpy as np

    mu = np.asarray(mu, dtype=np.float64)

    def mean_offdiag_cos(M):
        Z = M / np.linalg.norm(M, axis=1, keepdims=True)
        C = Z @ Z.T
        return float((C.sum() - np.trace(C)) / (len(M) * (len(M) - 1)))

    centred = mu - mu.mean(0)
    s = centred.std(0, ddof=1)
    s[s < 1e-8] = 1.0
    return {"raw": mean_offdiag_cos(mu), "centred": mean_offdiag_cos(centred), "centred_scaled": mean_offdiag_cos(centred / s)}


def _group_idx(groups, g):
    import numpy as np

    if groups is None:
        return [np.arange(g)]
    groups = np.asarray(groups)
    return [np.flatnonzero(groups == u) for u in sorted(set(groups))]


def _center(x, gi, axis):
    """Subtract each group's mean along the genome axis. One group = centre across all genomes."""
    import numpy as np

    out = np.array(x, copy=True)
    for idx in gi:
        sl = [slice(None)] * x.ndim
        sl[axis] = idx
        out[tuple(sl)] = x[tuple(sl)] - x[tuple(sl)].mean(axis=axis, keepdims=True)
    return out


def _scale_from(mu, gi=None):
    import numpy as np

    mu = np.asarray(mu, dtype=np.float64)
    gi = gi or [np.arange(len(mu))]
    c = _center(mu, gi, 0)
    s = np.sqrt((c ** 2).sum(0) / (len(mu) - len(gi)))
    s[s < 1e-8] = 1.0
    return s


def _wb(a, b, s, groups=None, rows=None):
    """Within- and between-genome variance (summed over standardised dimensions) from two disjoint draws.

    a, b: (reps, genomes, dims) means of two disjoint n-window subsets of the same genome.
    within  = E ||a - b||^2 / 2                        (variance of one n-window vector about its genome's mean)
    between = Cov_genomes(a, b)                          (the shared, genome-specific part; noise cancels)
    With `groups`, genomes are centred within their group first, so `between` is the variance
    among genomes of the same phylum (dof = genomes - groups).
    """
    import numpy as np

    if rows is not None:
        a, b = a[:, rows], b[:, rows]
        groups = None if groups is None else np.asarray(groups)[rows]
    a, b = a / s, b / s
    g = a.shape[1]
    gi = _group_idx(groups, g)
    w = 0.5 * ((a - b) ** 2).mean(axis=(0, 1), dtype=np.float64).sum()
    ac, bc = _center(a, gi, 1), _center(b, gi, 1)
    bt = ((ac * bc).sum(1, dtype=np.float64) / (g - len(gi))).mean(0).sum()
    return float(w), float(bt)


def _summarise(a, b, s, rng, n_boot, groups=None):
    import numpy as np

    g = a.shape[1]
    w, bt = _wb(a, b, s, groups)
    gi = _group_idx(groups, g)  # resample genomes WITHIN each group so group sizes (and the dof) stay fixed
    draw = lambda: np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in gi])  # noqa: E731
    boots = np.asarray([[wb_ / bb_, bb_ / (bb_ + wb_)] for wb_, bb_ in
                        (_wb(a, b, s, groups, rows=draw()) for _ in range(n_boot)) if bb_ > 0])
    # Resampling with replacement shrinks variance components by ~(m-1)/m for groups of m genomes, which
    # would bias the interval low; shift the bootstrap distribution so it is centred on the point estimate.
    point = np.array([w / bt, bt / (bt + w)])
    shift = point - boots.mean(0)
    lo, hi = np.quantile(boots, .025, axis=0) + shift, np.quantile(boots, .975, axis=0) + shift
    return {"within": w, "between": bt, "ratio": float(point[0]), "reliability": float(point[1]),
            "ratio_ci": [float(lo[0]), float(hi[0])], "reliability_ci": [float(lo[1]), float(hi[1])]}


def reliability_curve(pools, depths_random, depths_systematic, depths_model, reps: int = 10,
                      n_boot: int = 100, seed: int = 0, groups=None) -> dict:
    """Within/between-genome variance ratio and split-half reliability as a function of windows per genome.

    pools: (genomes, pool_windows, dims), windows in genome order (they are evenly spaced along it).
    Everything is computed after centring across genomes and scaling each dimension by its
    between-genome SD at full depth, so the shared component does not swamp the ratio.
    With `groups` (e.g. phylum), centring and scaling are done WITHIN each group, so `between` is the
    variation among genomes of the same phylum: the differences a within-phylum comparison must resolve.

    Three estimates, all giving within = E||a-b||^2/2 and between = Cov_genomes(a, b) for two
    disjoint n-window vectors a and b of the same genome:
      * random      two disjoint RANDOM n-window subsets (conservative: ignores even spacing).
      * systematic  two interleaved EVENLY SPACED n-window subsets (offset by half a stride), the design
                    the sweep and embed_all actually use. Needs n | pool and 2n <= pool.
      * model       within(n) = mean per-window variance / n, valid up to n = pool where no disjoint
                    halves exist. It matches `random` (windows drawn at random) by construction.
    `systematic_gain` = within_random / within_systematic: >1 means even spacing beats random draws.
    Reliability = between / (between + within) is the split-half ICC of an n-window vector.
    """
    import numpy as np

    pools = np.asarray(pools, dtype=np.float32)
    g, pool_n, d = pools.shape
    rng = np.random.default_rng(seed)
    gi = _group_idx(groups, g)
    dof = g - len(gi)
    mu = pools.mean(1)
    s = _scale_from(mu, gi)
    v_bar = float((pools.var(1, ddof=1, dtype=np.float64) / s ** 2).sum(1).mean())
    dims_eff = float(((_center(mu.astype(np.float64), gi, 0) ** 2).sum(0) / dof / s ** 2).sum())
    b_full = dims_eff - v_bar / pool_n  # between-genome variance net of the noise left in the pool mean

    random_, systematic = {}, {}
    for n in depths_random:
        if 2 * n > pool_n:
            continue
        a = np.empty((reps, g, d), dtype=np.float32)
        b = np.empty_like(a)
        for r in range(reps):
            perm = np.argsort(rng.random((g, pool_n)), axis=1)
            a[r] = np.take_along_axis(pools, perm[:, :n, None], axis=1).mean(1)
            b[r] = np.take_along_axis(pools, perm[:, n:2 * n, None], axis=1).mean(1)
        random_[n] = _summarise(a, b, s, rng, n_boot, groups)
    for n in depths_systematic:
        if pool_n % n or 2 * n > pool_n:
            continue
        stride = pool_n // n
        half = max(stride // 2, 1)
        offsets = sorted({int(o) for o in np.linspace(0, max(stride - half - 1, 0), min(reps, max(stride - half, 1)))})
        a = np.stack([pools[:, np.arange(n) * stride + o].mean(1) for o in offsets])
        b = np.stack([pools[:, np.arange(n) * stride + o + half].mean(1) for o in offsets])
        systematic[n] = {**_summarise(a, b, s, rng, n_boot, groups), "offsets": len(offsets)}
        if n in random_:
            systematic[n]["systematic_gain"] = random_[n]["within"] / systematic[n]["within"]
    model = {n: {"within": v_bar / n, "between": b_full, "ratio": v_bar / n / b_full,
                 "reliability": b_full / (b_full + v_bar / n), "extrapolated": n not in random_}
             for n in depths_model}
    return {"random": random_, "systematic": systematic, "model": model, "per_window_variance": v_bar,
            "between_full": b_full, "dims_effective": dims_eff, "cosine": cosine_report(mu),
            "within_groups": groups is not None, "n_groups": len(gi)}


def flattens_at(xs, ys, higher_is_better: bool = True, frac: float = 0.95, noise: float = 0.0) -> dict:
    """Smallest x reaching `frac` of the total gain from the first point to the best point.

    If the total gain is within 2*noise the curve has no detectable depth effect and none is reported.
    """
    sign = 1.0 if higher_is_better else -1.0
    v = [sign * y for y in ys]
    gain = max(v) - v[0]
    if gain <= 2 * noise:
        return {"flat_within_noise": True, "n": None, "gain": float(sign * gain)}
    thr = v[0] + frac * gain
    return {"flat_within_noise": False, "n": next(x for x, val in zip(xs, v) if val >= thr), "gain": float(sign * gain)}


def gc_quality(X, gc, groups, n_splits: int = 5, alpha: float = 1000.0) -> dict:
    """Out-of-phylum prediction of genome GC from an embedding: Spearman r and R^2 of cross-validated predictions."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, gc = np.asarray(X), np.asarray(gc, dtype=float)
    pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), X, gc, groups=np.asarray(groups),
                             cv=GroupKFold(n_splits=n_splits))
    return {"spearman": float(spearmanr(pred, gc).statistic),
            "r2": float(1 - ((gc - pred) ** 2).sum() / ((gc - gc.mean()) ** 2).sum())}


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


# ---------------------------------------------------------------------------
# Sequence-composition baselines: tetranucleotide frequencies and GC
# ---------------------------------------------------------------------------
_LUT = None
_CANON = None


def _lut():
    import numpy as np

    global _LUT
    if _LUT is None:
        _LUT = np.full(256, 4, dtype=np.uint8)
        for i, ch in enumerate(b"ACGT"):
            _LUT[ch] = i
    return _LUT


def canonical_tetra_map():
    """(256 -> 136 class index). A 4-mer and its reverse complement share a class; palindromes stand alone."""
    import numpy as np

    global _CANON
    if _CANON is None:
        comp = [3, 2, 1, 0]  # A0 C1 G2 T3
        rc = np.empty(256, dtype=int)
        for k in range(256):
            b = [(k >> 6) & 3, (k >> 4) & 3, (k >> 2) & 3, k & 3]
            r = [comp[x] for x in reversed(b)]
            rc[k] = r[0] * 64 + r[1] * 16 + r[2] * 4 + r[3]
        canon = np.minimum(np.arange(256), rc)
        ids = {c: i for i, c in enumerate(sorted(set(canon.tolist())))}
        _CANON = np.array([ids[c] for c in canon])
    return _CANON


def count_tetra(seq: str):
    """(256 tetranucleotide counts, 4 mononucleotide counts) of one sequence; k-mers containing non-ACGT are skipped."""
    import numpy as np

    a = _lut()[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    valid = a < 4
    mono = np.bincount(a[valid], minlength=4).astype(np.float64)
    if len(a) < 4:
        return np.zeros(256), mono
    a32 = a.astype(np.int32)
    idx = a32[:-3] * 64 + a32[1:-2] * 16 + a32[2:-1] * 4 + a32[3:]
    ok = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
    return np.bincount(idx[ok], minlength=256).astype(np.float64), mono


def iter_contigs(fasta_gz: str | Path):
    """Yield (id, uppercase sequence) for every record of a gzipped FASTA."""
    cid, parts = None, []
    with gzip.open(fasta_gz, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if cid is not None:
                    yield cid, "".join(parts).upper()
                cid, parts = (line[1:].split() or [""])[0], []
            else:
                parts.append(line.strip())
    if cid is not None:
        yield cid, "".join(parts).upper()


def sequence_features(fasta_gz: str | Path, windows: list | None = None):
    """(2, 260) array: row 0 = whole genome, row 1 = the given windows only; each row = 256 tetra counts + 4 mono counts.

    `windows` is a list of [contig id, start] (as saved in the embedding meta); each is WINDOW bp. Row 1 lets a
    composition baseline see exactly the sequence Evo 2 saw, which is a small part of the genome at low depth.
    """
    import numpy as np

    genome, seqs = np.zeros(260), {}
    for cid, seq in iter_contigs(fasta_gz):
        t, m = count_tetra(seq)
        genome[:256] += t
        genome[256:] += m
        if windows is not None:
            seqs.setdefault(cid, seq)
    win = np.zeros(260)
    for cid, start in windows or []:
        t, m = count_tetra(seqs[cid][start:start + WINDOW])
        win[:256] += t
        win[256:] += m
    return np.stack([genome, win])


def tetra_frequencies(row):
    """136 canonical tetranucleotide frequencies from a 260-vector of counts."""
    import numpy as np

    counts = np.bincount(canonical_tetra_map(), weights=np.asarray(row[:256], dtype=float), minlength=136)
    return counts / counts.sum()


def gc_from_counts(row) -> float:
    m = row[256:]
    return float((m[1] + m[2]) / m.sum())


# ---------------------------------------------------------------------------
# Leave-one-phylum-out evaluation with per-phylum scoring
# ---------------------------------------------------------------------------
LOPO_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _lr_scores(X_tr, y_tr, X_te, C):
    return _pipeline(C).fit(X_tr, y_tr).decision_function(X_te)


def _auc_or_nan(y, s) -> float:
    y = np_asarray_bool(y)
    return manual_auc(y, s) if 0 < y.sum() < len(y) else float("nan")


def np_asarray_bool(y):
    import numpy as np

    return np.asarray(y).astype(bool)


def cluster_bootstrap(scores: dict, y, clusters, n_boot: int, rng) -> dict:
    """Resample whole clusters (genera) with replacement; AUC of every model on each SAME resample.

    Genomes of one genus are near-copies of each other in both embedding and label, so resampling genomes
    would understate the uncertainty. Returns {model: array(n_boot)}; resamples lacking a class are dropped
    for all models together, so paired differences stay aligned.
    """
    import numpy as np

    y = np.asarray(y).astype(bool)
    clusters = np.asarray(clusters)
    members = [np.flatnonzero(clusters == c) for c in sorted(set(clusters))]
    out = {m: [] for m in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if not 0 < y[idx].sum() < len(idx):
            continue
        for m, sc in scores.items():
            out[m].append(manual_auc(y[idx], sc[idx]))
    return {m: np.asarray(v) for m, v in out.items()}


def lopo_evaluate(features: dict, y, phyla, genera, models: dict, compare: list, n_perm: int = 50, n_boot: int = 1000,
                  seed: int = 0, n_jobs: int = 1) -> dict:
    """Hold out one phylum at a time, train on the rest, score AUC WITHIN the held-out phylum.

    features: {feature set name: (N, d) array}. models: {model name: {"features": set, "grid": [C...], "null": bool}}.
    For each model and held-out phylum, C is chosen by an inner leave-one-phylum-out over the training phyla
    (metric: mean within-phylum AUC), so a 4096-d model and a 136-d model are each regularised for their
    own size. A one-value grid means "fixed C, no tuning". Uncertainty is a genus-cluster bootstrap; the
    null refits on labels shuffled WITHIN each phylum (phylum prevalence kept, within-phylum signal destroyed).
    A constant score ties every pair, so the "majority class" floor is exactly 0.5 for a within-phylum AUC.
    """
    import numpy as np
    from joblib import Parallel, delayed

    y = np.asarray(y).astype(bool)
    phyla, genera = np.asarray(phyla), np.asarray(genera)
    order = sorted(set(phyla))
    idx = {h: np.flatnonzero(phyla == h) for h in order}
    rows_of = lambda hs: np.concatenate([idx[h] for h in hs])  # noqa: E731

    # 1. inner tuning: for outer h, inner t, C -> within-phylum AUC of a model trained on (others \ t)
    tasks = []
    for name, spec in models.items():
        for h in order:
            rest = [g for g in order if g != h]
            if len(spec["grid"]) > 1:
                for t in rest:
                    tr = rows_of([g for g in rest if g != t])
                    for C in spec["grid"]:
                        tasks.append((name, h, t, C, tr, idx[t]))
    def inner(name, h, t, C, tr, te):
        X = features[models[name]["features"]]
        return name, h, C, _auc_or_nan(y[te], _lr_scores(X[tr], y[tr], X[te], C))
    got = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(inner)(*t) for t in tasks)
    inner_auc: dict = {}
    for name, h, C, a in got:
        inner_auc.setdefault((name, h, C), []).append(a)
    chosen = {}
    for name, spec in models.items():
        for h in order:
            if len(spec["grid"]) == 1:
                chosen[(name, h)] = spec["grid"][0]
            else:  # highest mean inner AUC; ties -> the stronger regulariser (smaller C)
                chosen[(name, h)] = max(sorted(spec["grid"]), key=lambda C: (np.nanmean(inner_auc[(name, h, C)]), -C))

    # 2. outer fit + within-phylum scores
    def outer(name, h):
        X = features[models[name]["features"]]
        tr = rows_of([g for g in order if g != h])
        return (name, h), _lr_scores(X[tr], y[tr], X[idx[h]], chosen[(name, h)])
    scores = dict(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(outer)(n, h) for n in models for h in order))

    # 3. cluster bootstrap, paired across models
    rng = np.random.default_rng(seed)
    boots, per = {}, {}
    for h in order:
        te = idx[h]
        b = cluster_bootstrap({n: scores[(n, h)] for n in models}, y[te], genera[te], n_boot, rng)
        boots[h] = b
        per[h] = {"n": int(len(te)), "n_positive": int(y[te].sum()), "prevalence": float(y[te].mean()),
                  "n_genera": int(len(set(genera[te]))), "majority_accuracy": float(max(y[te].mean(), 1 - y[te].mean())),
                  "floor_auc": 0.5, "models": {}, "deltas": {}}
        for n in models:
            auc = _auc_or_nan(y[te], scores[(n, h)])
            lo, hi = (float(np.quantile(b[n], .025)), float(np.quantile(b[n], .975))) if len(b[n]) else (float("nan"),) * 2
            per[h]["models"][n] = {"auc": auc, "ci": [lo, hi], "C": float(chosen[(n, h)])}
        for a_, b_ in compare:
            k = min(len(boots[h][a_]), len(boots[h][b_]))
            d = boots[h][a_][:k] - boots[h][b_][:k]
            per[h]["deltas"][f"{a_} - {b_}"] = {"delta": per[h]["models"][a_]["auc"] - per[h]["models"][b_]["auc"],
                                                 "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                                                 "share_boot_positive": float((d > 0).mean())}
    macro = {"models": {}, "deltas": {}}
    kmin = min(len(boots[h][n]) for h in order for n in models)
    for n in models:
        m = np.mean([boots[h][n][:kmin] for h in order], axis=0)
        macro["models"][n] = {"auc": float(np.mean([per[h]["models"][n]["auc"] for h in order])),
                              "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))]}
    for a_, b_ in compare:
        d = np.mean([boots[h][a_][:kmin] - boots[h][b_][:kmin] for h in order], axis=0)
        macro["deltas"][f"{a_} - {b_}"] = {"delta": macro["models"][a_]["auc"] - macro["models"][b_]["auc"],
                                            "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                                            "share_boot_positive": float((d > 0).mean())}

    # 4. shuffle null: labels permuted within each phylum, refit with the chosen C, scored in the held-out phylum
    null_models = [n for n, spec in models.items() if spec.get("null", True)]
    if n_perm and null_models:
        prng = np.random.default_rng(seed + 1)
        perms = []
        for _ in range(n_perm):
            yp = y.copy()
            for h in order:
                yp[idx[h]] = y[prng.permutation(idx[h])]
            perms.append(yp)

        def null_task(k, name, h):
            X = features[models[name]["features"]]
            tr = rows_of([g for g in order if g != h])
            yp = perms[k]
            return k, name, h, _auc_or_nan(yp[idx[h]], _lr_scores(X[tr], yp[tr], X[idx[h]], chosen[(name, h)]))
        got = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(null_task)(k, n, h) for k in range(n_perm) for n in null_models for h in order)
        nul: dict = {}
        for k, n, h, a in got:
            nul.setdefault((n, h), {})[k] = a
        for h in order:
            for n in null_models:
                v = np.asarray([nul[(n, h)][k] for k in range(n_perm)])
                obs = per[h]["models"][n]["auc"]
                per[h]["models"][n]["null"] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                                                "q95": float(np.nanquantile(v, .95)),
                                                "p": float((1 + np.nansum(v >= obs)) / (1 + np.sum(~np.isnan(v))))}
        for n in null_models:
            m = np.mean([[nul[(n, h)][k] for h in order] for k in range(n_perm)], axis=1)
            obs = macro["models"][n]["auc"]
            macro["models"][n]["null"] = {"mean": float(m.mean()), "sd": float(m.std(ddof=1)), "q95": float(np.quantile(m, .95)),
                                          "p": float((1 + np.sum(m >= obs)) / (1 + len(m)))}
    return {"phyla": order, "models": list(models), "per_phylum": per, "macro": macro, "n_perm": n_perm, "n_boot": n_boot,
            "inner_auc": {f"{n}|{h}|{C}": float(np.nanmean(v)) for (n, h, C), v in inner_auc.items()}}
