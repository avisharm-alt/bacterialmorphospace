"""Leakage-aware, species-level trait benchmark on the committed Stage 1 panels.

No test labels enter fitting. Genomes are optional because sequence files are not
part of the Stage 1 cache. A GTDB v232 metadata file may supply family for older
panel TSVs; newly generated panel TSVs include family directly.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config

RANKS = ("genus", "family", "order", "class", "phylum")
SEED = 20261007
FOLDS = 5
REPEATS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_panel(df: pd.DataFrame) -> None:
    required = {"gtdb_species", "ncbi_assembly_accession", "assembly_genbank", "gtdb_genus",
                "gtdb_order", "gtdb_class", "gtdb_phylum"}
    missing = required - set(df)
    if missing:
        raise ValueError(f"missing panel columns: {sorted(missing)}")
    for col in ("gtdb_species", "ncbi_assembly_accession", "assembly_genbank"):
        if df[col].isna().any() or df[col].duplicated().any():
            raise ValueError(f"missing or duplicate {col}; resolve before splitting")


def add_family(df: pd.DataFrame, metadata: Path) -> pd.DataFrame:
    """Match GTDB accession exactly; no genus-level inference of family."""
    meta = pd.read_csv(metadata, sep="\t", compression="infer", usecols=["accession", "gtdb_taxonomy"])
    if meta.accession.duplicated().any():
        raise ValueError("duplicate GTDB metadata accessions")
    families = meta.set_index("accession").gtdb_taxonomy.str.extract(r"(?:^|;)f__([^;]*)", expand=False)
    out = df.copy()
    out["gtdb_family"] = out.gtdb_accession.map(families).replace("", pd.NA)
    if out.gtdb_family.isna().any():
        raise ValueError(f"family unresolved for {out.gtdb_family.isna().sum()} panel accessions")
    return out


def assign_folds(df: pd.DataFrame, level: str, seed: int, n_folds: int = FOLDS) -> np.ndarray:
    """A group occurs in exactly one test fold. Balance folds by species count."""
    if level == "within":
        group = df.gtdb_species.astype(str)
    else:
        col = f"gtdb_{level}"
        if col not in df or df[col].isna().any():
            raise ValueError(f"{col} is required for {level} splits")
        group = df[col].astype(str)
    sizes = group.value_counts()
    if len(sizes) < n_folds:
        raise ValueError(f"{level}: fewer than {n_folds} groups")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(sizes))
    ordered = sorted(shuffled, key=lambda i: -int(sizes.iloc[i]))
    loads = np.zeros(n_folds, dtype=int)
    mapping = {}
    for i in ordered:
        fold = int(np.argmin(loads))
        mapping[sizes.index[i]] = fold
        loads[fold] += int(sizes.iloc[i])
    return group.map(mapping).to_numpy(dtype=int)


def prevalence(y_train: pd.Series, n: int) -> np.ndarray:
    counts = y_train.value_counts()
    return np.repeat(sorted(counts[counts == counts.max()].index)[0], n)


def taxonomy_predict(train: pd.DataFrame, test: pd.DataFrame, trait: str, min_count: int = 5) -> np.ndarray:
    """Training labels alone define taxon-majority lookups; rare taxa back off."""
    fallback = prevalence(train[trait], 1)[0]
    prediction = np.repeat(fallback, len(test)).astype(object)
    unresolved = np.ones(len(test), dtype=bool)
    for rank in RANKS:
        col = f"gtdb_{rank}"
        if col not in train or col not in test:
            continue
        counts = train.groupby(col)[trait].value_counts().rename("n").reset_index()
        totals = counts.groupby(col).n.transform("sum")
        counts = counts.loc[totals >= min_count].sort_values([col, "n", trait], ascending=[True, False, True])
        lookup = counts.drop_duplicates(col).set_index(col)[trait]
        values = test[col].map(lookup)
        found = unresolved & values.notna().to_numpy()
        prediction[found] = values.to_numpy()[found]
        unresolved[found] = False
    return prediction


def standardize_fit_transform(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean and scale on train only; use same transform on held-out rows."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    return (train - mean) / scale, (test - mean) / scale


def centroid_predict(x_train: np.ndarray, y_train: pd.Series, x_test: np.ndarray) -> np.ndarray:
    a, b = standardize_fit_transform(x_train, x_test)
    classes = np.array(sorted(y_train.unique()))
    centers = np.stack([a[y_train.to_numpy() == label].mean(axis=0) for label in classes])
    d2 = ((b[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return classes[d2.argmin(axis=1)]


def kmer_frequencies(path: Path, k: int = 4) -> np.ndarray:
    """Streaming fixed 4-mer frequencies; discard ambiguous windows and contig boundaries."""
    counts = np.zeros(4 ** k, dtype=np.float64)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        state = 0
        valid = 0
        mask = 4 ** k - 1
        for line in fh:
            if line.startswith(">"):
                state = valid = 0
                continue
            for base in line.strip().upper():
                digit = "ACGT".find(base)
                if digit < 0:
                    state = valid = 0
                else:
                    state = ((state << 2) | digit) & mask
                    valid += 1
                    if valid >= k:
                        counts[state] += 1
    if not counts.sum():
        raise ValueError(f"no valid {k}-mers in {path}")
    return counts / counts.sum()


def load_genomes(df: pd.DataFrame, directory: Path) -> np.ndarray:
    paths = []
    for acc in df.ncbi_assembly_accession:
        choices = [directory / f"{acc}{suffix}" for suffix in (".fna.gz", ".fna", ".fa.gz", ".fa")]
        hits = [p for p in choices if p.is_file()]
        if len(hits) != 1:
            raise ValueError(f"expected one FASTA for {acc}; found {len(hits)} in {directory}")
        paths.append(hits[0])
    return np.stack([kmer_frequencies(p) for p in paths])


def accession_overlap(df: pd.DataFrame, manifest: Path) -> dict:
    """Screen an externally supplied corpus accession manifest; keep its scope explicit."""
    pattern = re.compile(r"(?:RS_|GB_)?GC[AF]_\d{9}(?:\.\d+)?")
    text = manifest.read_text()
    corpus = {m.group().removeprefix("RS_").removeprefix("GB_") for m in pattern.finditer(text)}
    if not corpus:
        raise ValueError("no assembly accessions found in pretraining manifest")
    panel = set(df.ncbi_assembly_accession.dropna()) | set(df.assembly_genbank.dropna())
    exact = panel & corpus
    stem = {a.split(".")[0] for a in panel} & {a.split(".")[0] for a in corpus}
    return {"manifest_sha256": sha256(manifest), "corpus_accessions_parsed": len(corpus),
            "exact_assembly_accessions": len(exact), "unversioned_accessions": len(stem),
            "scope": "accession screen only; requires a complete, provenance-verified OpenGenome2 manifest"}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Scores over test-present labels; one-class test folds are unscorable."""
    labels = sorted(set(y_true))
    if len(labels) < 2:
        return {"balanced_accuracy": np.nan, "macro_f1": np.nan}
    recall, f1 = [], []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        fp = np.sum((y_true != label) & (y_pred == label))
        recall.append(tp / (tp + fn))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"balanced_accuracy": float(np.mean(recall)), "macro_f1": float(np.mean(f1))}


def nearest_rank(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Closest shared GTDB rank, an ordinal proxy rather than a branch distance."""
    result = np.repeat("none", len(test)).astype(object)
    for rank in reversed(RANKS):
        col = f"gtdb_{rank}"
        if col in train and col in test:
            shared = test[col].isin(set(train[col].dropna()))
            result[shared.to_numpy()] = rank
    return result


def benchmark(df: pd.DataFrame, traits: list[str], genomes: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, distance_rows = [], []
    levels = ["within", "genus", "order"]
    if "gtdb_family" in df and df.gtdb_family.notna().all():
        levels.insert(2, "family")
    for trait in traits:
        sub = df[df[trait].notna()].copy()
        indices = np.flatnonzero(df[trait].notna().to_numpy())
        x = genomes[indices] if genomes is not None else None
        for level in levels:
            for repeat in range(REPEATS):
                fold_ids = assign_folds(sub, level, SEED + repeat)
                for fold in range(FOLDS):
                    test_mask = fold_ids == fold
                    train, test = sub.iloc[~test_mask], sub.iloc[test_mask]
                    truth = test[trait].to_numpy()
                    predictions = {"prevalence": prevalence(train[trait], len(test)),
                                   "taxonomy": taxonomy_predict(train, test, trait)}
                    if x is not None:
                        predictions["kmer4_centroid"] = centroid_predict(x[~test_mask], train[trait], x[test_mask])
                    dist = nearest_rank(train, test)
                    for model, pred in predictions.items():
                        rows.append({"trait": trait, "split": level, "repeat": repeat, "fold": fold,
                                     "model": model, "n_train": len(train), "n_test": len(test),
                                     "test_classes": len(set(truth)), **metrics(truth, pred)})
                        if repeat == 0:
                            distance_rows.extend({"trait": trait, "split": level, "model": model,
                                                  "distance": d, "true": y, "pred": p}
                                                 for d, y, p in zip(dist, truth, pred))
    return pd.DataFrame(rows), pd.DataFrame(distance_rows)


def summarize(folds: pd.DataFrame, distance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Across-fold range reflects split variability, not a population confidence interval."""
    summary = folds.groupby(["trait", "split", "model"], as_index=False).agg(
        n_train_min=("n_train", "min"), n_test_total=("n_test", lambda x: int(x.sum() / REPEATS)),
        scorable_folds=("balanced_accuracy", "count"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        ba_p05=("balanced_accuracy", lambda x: x.quantile(.05)),
        ba_p95=("balanced_accuracy", lambda x: x.quantile(.95)),
        macro_f1=("macro_f1", "mean"), f1_p05=("macro_f1", lambda x: x.quantile(.05)),
        f1_p95=("macro_f1", lambda x: x.quantile(.95)))
    out = []
    for (trait, split, model, rank), part in distance.groupby(["trait", "split", "model", "distance"]):
        out.append({"trait": trait, "split": split, "model": model, "nearest_training_rank": rank,
                    "n": len(part), **metrics(part.true.to_numpy(), part.pred.to_numpy())})
    return summary, pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtdb-metadata", type=Path, help="GTDB v232 bac120_metadata.tsv.gz, to add family to old panel TSVs")
    ap.add_argument("--genomes", type=Path, help="directory of accession-named .fna[.gz] files")
    ap.add_argument("--accessions", type=Path, help="optional exact accession list to benchmark a genome subset")
    ap.add_argument("--pretraining-accessions", type=Path, help="optional OpenGenome2 accession manifest for overlap screen")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stage2")
    args = ap.parse_args(argv)
    cfg = load_config()
    panel_paths = [cfg.path("final") / f"{p}_panel_species.tsv" for p in ("core", "no_spore")]
    core, no_spore = [pd.read_csv(p, sep="\t", dtype={"bacdive_id": str}) for p in panel_paths]
    for panel in (core, no_spore):
        validate_panel(panel)
    shared = core.set_index("gtdb_species").join(no_spore.set_index("gtdb_species"),
                                                 lsuffix="_core", rsuffix="_no", how="inner")
    panel_disagreements = {col: int((shared[f"{col}_core"].fillna("<missing>") !=
                                    shared[f"{col}_no"].fillna("<missing>")).sum())
                           for col in ("ncbi_assembly_accession", "assembly_genbank", *cfg.core)}
    # A row's BacDive labels and assembly must stay together. Core wins for its
    # species; no-spore contributes only the additional species.
    df = pd.concat([core, no_spore.loc[~no_spore.gtdb_species.isin(core.gtdb_species)]], ignore_index=True)
    if args.gtdb_metadata:
        if sha256(args.gtdb_metadata) != json.loads((cfg.path("reports") / "run_manifest.json").read_text())["gtdb"]["metadata_sha256"]:
            raise ValueError("GTDB metadata hash differs from the Stage 1 v232 snapshot")
        df = add_family(df, args.gtdb_metadata)
    elif "gtdb_family" in df and df.gtdb_family.notna().all():
        pass
    traits = cfg.core
    validate_panel(df)
    full_union_n = len(df)
    if args.accessions:
        selected = set(args.accessions.read_text().split())
        if not selected or not selected.issubset(set(df.ncbi_assembly_accession)):
            raise ValueError("accession subset is empty or contains accessions outside the selected panel")
        df = df.loc[df.ncbi_assembly_accession.isin(selected)].reset_index(drop=True)
    genome_matrix = load_genomes(df, args.genomes) if args.genomes else None
    folds, distance = benchmark(df, traits, genome_matrix)
    summary, by_distance = summarize(folds, distance)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("folds", folds), ("summary", summary), ("distance", by_distance)):
        table.to_csv(args.out / f"{name}.tsv", sep="\t", index=False)
    audit = {"panel_inputs": {str(p.relative_to(ROOT)): sha256(p) for p in panel_paths},
             "species_union": full_union_n, "species_evaluated": len(df),
             "accession_subset_sha256": sha256(args.accessions) if args.accessions else None,
             "family_available": "gtdb_family" in df and df.gtdb_family.notna().all(),
             "duplicate_species": int(df.gtdb_species.duplicated().sum()),
             "shared_panel_species": len(shared), "panel_disagreements": panel_disagreements,
             "duplicate_refseq_or_genbank": int(df.ncbi_assembly_accession.duplicated().sum()),
             "duplicate_genbank": int(df.assembly_genbank.duplicated().sum()),
             "taxonomy": {r: int(df[f"gtdb_{r}"].nunique()) for r in RANKS if f"gtdb_{r}" in df},
             "traits": {t: {str(k): int(v) for k, v in df[t].value_counts(dropna=False).items()} for t in traits},
             "genomes_loaded": genome_matrix is not None, "seed": SEED, "folds": FOLDS, "repeats": REPEATS,
             "taxonomy_min_count": 5, "python": platform.python_version(), "pandas": pd.__version__,
             "numpy": np.__version__, "config_sha256": cfg.text_sha256}
    audit["pretraining_overlap"] = (accession_overlap(df, args.pretraining_accessions)
                                    if args.pretraining_accessions else {"status": "not_checked_no_manifest"})
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"wrote {args.out}: {len(summary)} benchmark rows; family available={audit['family_available']}")


if __name__ == "__main__":
    main()
