"""Stage 2 leakage-aware baselines on the committed Stage 1 species panels.

Run ``python -m src.stage2 prepare`` once with the two small GTDB taxonomy
snapshots in data/raw/gtdb, then ``python -m src.stage2 benchmark`` offline.
No BacDive label or test-fold statistic is used to construct model features.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config

RANKS = ("genus", "family", "order", "class", "phylum")
TRAITS = ("gram", "shape", "motility", "spore", "oxygen", "temperature")
LABELS = {
    "gram": ("negative", "positive"),
    "shape": ("coccus", "other", "rod"),
    "motility": ("no", "yes"),
    "spore": ("no", "yes"),
    "oxygen": ("aerobe", "anaerobe", "facultative"),
    "temperature": ("meso", "psychro", "thermo"),
}
SOURCE_URL = "https://data.gtdb.ecogenomic.org/releases/release{v}/{v}.0/bac120_taxonomy_r{v}.tsv.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assembly_stem(accession: str) -> str:
    """Ignore the GTDB RS/GB prefix and assembly version, retaining GCA/GCF."""
    return re.sub(r"\.\d+$", "", re.sub(r"^(RS|GB)_", "", accession))


def read_taxonomy(path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    out = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            accession, taxonomy = line.rstrip("\n").split("\t", 1)
            if wanted is None or accession in wanted:
                if accession in out:
                    raise ValueError(f"duplicate GTDB accession: {accession}")
                out[accession] = taxonomy
    return out


def panel_path(root: Path, name: str) -> Path:
    return root / "data/final" / f"{name}_panel_species.tsv"


def mapping_path(root: Path) -> Path:
    return root / "data/final/stage2_taxonomy.tsv"


def prepare(root: Path = ROOT) -> dict:
    """Join family from v232 and test v220 source membership without large genomes."""
    panels = [pd.read_csv(panel_path(root, p), sep="\t") for p in ("core", "no_spore")]
    accessions = set(pd.concat(panels)["gtdb_accession"])
    raw = root / "data/raw/gtdb"
    p232 = raw / "bac120_taxonomy_r232.tsv.gz"
    p220 = raw / "bac120_taxonomy_r220.tsv.gz"
    v232 = read_taxonomy(p232, accessions)
    v220 = read_taxonomy(p220)
    if set(v232) != accessions:
        raise ValueError(f"v232 taxonomy missing {len(accessions - set(v232))} panel accessions")
    for panel in panels:
        for row in panel.itertuples(index=False):
            taxonomy = {part[:1]: part[3:] for part in v232[row.gtdb_accession].split(";")}
            for code, rank in (("g", "genus"), ("o", "order"), ("c", "class"), ("p", "phylum")):
                if taxonomy.get(code) != getattr(row, f"gtdb_{rank}"):
                    raise ValueError(f"v232 {rank} mismatch for {row.gtdb_accession}")
    stems220 = {assembly_stem(x) for x in v220}
    rows = []
    for accession in sorted(accessions):
        fields = v232[accession].split(";")
        family = next((f[3:] for f in fields if f.startswith("f__")), "")
        if not family:
            raise ValueError(f"missing family for {accession}")
        rows.append({"gtdb_accession": accession, "gtdb_family": family,
                     "in_gtdb_v220_exact": accession in v220,
                     "in_gtdb_v220_stem": assembly_stem(accession) in stems220})
    pd.DataFrame(rows).to_csv(mapping_path(root), sep="\t", index=False)
    provenance = {f"gtdb_v{v}": {"url": SOURCE_URL.format(v=v), "sha256": sha256(path)}
                  for v, path in ((232, p232), (220, p220))}
    (root / "reports/stage2_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"rows": len(rows), "exact_v220": sum(r["in_gtdb_v220_exact"] for r in rows),
            "stem_v220": sum(r["in_gtdb_v220_stem"] for r in rows)}


def load_panel(root: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(panel_path(root, name), sep="\t")
    mapping = pd.read_csv(mapping_path(root), sep="\t")
    for key in ("gtdb_species", "gtdb_accession", "ncbi_assembly_accession", "assembly_genbank"):
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f"missing or duplicate {key} in {name} panel")
    if df["assembly_genbank"].map(assembly_stem).duplicated().any():
        raise ValueError(f"duplicate version-stripped GenBank assembly in {name} panel")
    if mapping["gtdb_accession"].duplicated().any():
        raise ValueError("duplicate accession in Stage 2 taxonomy mapping")
    df = df.merge(mapping, on="gtdb_accession", how="left", validate="one_to_one")
    for rank in RANKS:
        if df[f"gtdb_{rank}"].isna().any():
            raise ValueError(f"missing {rank} in {name} panel")
    if df["in_gtdb_v220_exact"].isna().any():
        raise ValueError("Stage 2 taxonomy mapping does not cover panel")
    return df


def audit(core: pd.DataFrame, no_spore: pd.DataFrame) -> dict:
    result = {}
    for name, df in (("core", core), ("no_spore", no_spore)):
        result[name] = {
            "species": len(df),
            "unique_genomes": int(df["ncbi_assembly_accession"].nunique()),
            "gtdb_v220_exact": int(df["in_gtdb_v220_exact"].sum()),
            "gtdb_v220_stem": int(df["in_gtdb_v220_stem"].sum()),
            "taxa": {rank: int(df[f"gtdb_{rank}"].nunique()) for rank in RANKS},
            "labels": {t: df[t].value_counts(dropna=False).rename(index={np.nan: "missing"}).to_dict()
                       for t in TRAITS},
            "label_missing": {t: int(df[t].isna().sum()) for t in TRAITS},
            "match_method": df["match_method"].value_counts().to_dict(),
            "temperature_source": df["temperature_bin_source"].value_counts().to_dict(),
            "shape_other": int((df["shape"] == "other").sum()),
            "microaerophile_folded": int(df["oxygen_microaerophile_folded"].sum()),
        }
    shared = core.merge(no_spore, on="gtdb_species", suffixes=("_core", "_no_spore"))
    result["cross_panel"] = {
        "shared_species": len(shared),
        "different_assembly": int((shared["gtdb_accession_core"] != shared["gtdb_accession_no_spore"]).sum()),
        "different_labels": {t: int((shared[f"{t}_core"] != shared[f"{t}_no_spore"]).sum())
                             for t in TRAITS if t != "spore"},
    }
    return result


def split_indices(df: pd.DataFrame, holdout: str, seed: int, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Repeated random species split or disjoint taxon split; no labels consulted."""
    if holdout != "within" and holdout not in RANKS:
        raise ValueError(holdout)
    rng = np.random.default_rng(seed)
    if holdout == "within":
        test = rng.choice(len(df), size=round(len(df) * fraction), replace=False)
    else:
        groups = df[f"gtdb_{holdout}"].to_numpy()
        shuffled = rng.permutation(np.unique(groups))
        chosen = []
        count = 0
        for group in shuffled:
            chosen.append(group)
            count += int(np.sum(groups == group))
            if count >= round(len(df) * fraction):
                break
        test = np.flatnonzero(np.isin(groups, chosen))
    train = np.setdiff1d(np.arange(len(df)), test)
    if not len(train) or not len(test):
        raise ValueError("empty split")
    if holdout != "within" and set(df.iloc[train][f"gtdb_{holdout}"]) & set(df.iloc[test][f"gtdb_{holdout}"]):
        raise AssertionError("held taxon crossed the split")
    for key in ("gtdb_species", "ncbi_assembly_accession", "assembly_genbank"):
        if set(df.iloc[train][key]) & set(df.iloc[test][key]):
            raise AssertionError(f"{key} crossed the split")
    if set(df.iloc[train]["assembly_genbank"].map(assembly_stem)) & set(
            df.iloc[test]["assembly_genbank"].map(assembly_stem)):
        raise AssertionError("version-stripped GenBank assembly crossed the split")
    return train, test


def nearest_rank(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Closest shared taxonomic rank as a coarse distance proxy, not a tree distance."""
    out = np.full(len(test), "different_phylum", dtype=object)
    for rank in reversed(RANKS):
        known = set(train[f"gtdb_{rank}"])
        out[test[f"gtdb_{rank}"].isin(known).to_numpy()] = f"same_{rank}"
    return out


def prevalence_predict(y_train: pd.Series, n: int, labels: list[str]) -> np.ndarray:
    counts = y_train.value_counts()
    winner = max(labels, key=lambda label: (counts.get(label, 0), -labels.index(label)))
    return np.repeat(winner, n)


def taxonomy_predict(train: pd.DataFrame, test: pd.DataFrame, trait: str, labels: list[str]) -> np.ndarray:
    """Train-only empirical Bayes taxon frequencies; fall back along the taxonomy.

    The fixed prior strength (5) and class-balanced decision rule are specified
    before evaluation. No test labels, categories, or preprocessing enter fit.
    """
    y = train[trait]
    prior = np.array([(y == label).mean() for label in labels])
    counts = np.zeros((len(test), len(labels)))
    # Broad ranks first, then overwrite rows with any more specific taxon
    # observed in training. Category discovery and frequency estimation use
    # train alone; the test taxon is only a lookup key.
    for rank in reversed(RANKS):
        column = f"gtdb_{rank}"
        table = (train.groupby(column)[trait].value_counts().unstack(fill_value=0)
                 .reindex(columns=labels, fill_value=0))
        found = test[column].isin(table.index).to_numpy()
        if found.any():
            counts[found] = table.reindex(test.loc[found, column]).to_numpy(dtype=float)
    posterior = (counts + 5 * prior) / (counts.sum(axis=1, keepdims=True) + 5)
    # Balanced decision threshold: compare conditional frequency to the
    # training prevalence. Unseen train labels cannot be predicted.
    score = np.divide(posterior, prior, out=np.zeros_like(posterior), where=prior > 0)
    return np.asarray(labels)[np.argmax(score, axis=1)]


def kmer_features(fasta: Path, k: int = 4) -> np.ndarray:
    """Canonical 4-mer frequencies from one assembly FASTA; ignores ambiguous bases."""
    opener = gzip.open if fasta.suffix == ".gz" else open
    seq = []
    with opener(fasta, "rt") as stream:
        for line in stream:
            if not line.startswith(">"):
                seq.append(line.strip().upper())
            else:
                seq.append("N" * (k - 1))  # never count across contigs
    bases = {base: i for i, base in enumerate("ACGT")}
    vector = np.zeros(4 ** k, dtype=float)
    code = valid = 0
    for base in "".join(seq):
        if base not in bases:
            code = valid = 0
            continue
        code = ((code << 2) | bases[base]) & (4 ** k - 1)
        valid += 1
        if valid >= k:
            vector[code] += 1
    if not vector.sum():
        raise ValueError(f"no valid {k}-mers in {fasta}")
    return vector / vector.sum()


def kmer_predict(x_train: np.ndarray, y_train: pd.Series, x_test: np.ndarray, labels: list[str]) -> np.ndarray:
    """Training-fold class centroids with cosine similarity; no fitted test statistics."""
    centroids = []
    for label in labels:
        subset = x_train[np.asarray(y_train == label)]
        center = subset.mean(axis=0) if len(subset) else np.zeros(x_train.shape[1])
        norm = np.linalg.norm(center)
        centroids.append(center / norm if norm else center)
    norm = np.linalg.norm(x_test, axis=1, keepdims=True)
    scores = (x_test / np.maximum(norm, 1e-12)) @ np.array(centroids).T
    return np.asarray(labels)[np.argmax(scores, axis=1)]


def load_kmers(df: pd.DataFrame, fasta_dir: Path) -> np.ndarray:
    vectors = []
    for accession in df["ncbi_assembly_accession"]:
        candidates = [fasta_dir / f"{accession}.fna", fasta_dir / f"{accession}.fna.gz"]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"no FASTA for {accession} in {fasta_dir}")
        vectors.append(kmer_features(found))
    return np.stack(vectors)


def metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """Balanced accuracy and macro F1 over every training-panel label."""
    y_true = np.asarray(y_true)
    recalls, f1s = [], []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        fp = np.sum((y_true != label) & (y_pred == label))
        recalls.append(tp / (tp + fn) if tp + fn else np.nan)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)
    return float(np.nanmean(recalls)), float(np.nanmean(f1s))


def benchmark(root: Path = ROOT, repeats: int = 20, fasta_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    core, no_spore = load_panel(root, "core"), load_panel(root, "no_spore")
    audit_info = audit(core, no_spore)
    (root / "reports/stage2_audit.json").write_text(json.dumps(audit_info, indent=2) + "\n")
    records, distance_rows = [], []
    for trait in TRAITS:
        df = (core if trait == "spore" else no_spore).copy()
        if df[trait].isna().any():
            raise ValueError(f"missing {trait} label in evaluation panel")
        labels = list(LABELS[trait])
        if set(df[trait]) != set(labels):
            raise ValueError(f"unexpected or absent {trait} labels")
        features = load_kmers(df, fasta_dir) if fasta_dir is not None else None
        for holdout in ("within", "genus", "family", "order"):
            for repeat in range(repeats):
                train_idx, test_idx = split_indices(df, holdout, 20261007 + repeat)
                train, test = df.iloc[train_idx], df.iloc[test_idx]
                y_test = test[trait].to_numpy()
                distance = nearest_rank(train, test)
                models = {
                    "prevalence": prevalence_predict(train[trait], len(test), labels),
                    "taxonomy": taxonomy_predict(train, test, trait, labels),
                }
                if features is not None:
                    models["kmer4"] = kmer_predict(features[train_idx], train[trait], features[test_idx], labels)
                for model, prediction in models.items():
                    ba, f1 = metrics(y_test, prediction, labels)
                    records.append({"trait": trait, "panel_n": len(df), "split": holdout,
                                    "repeat": repeat, "train_n": len(train), "test_n": len(test),
                                    "test_classes": len(set(y_test)), "model": model,
                                    "balanced_accuracy": ba, "macro_f1": f1})
                    for rank in np.unique(distance):
                        mask = distance == rank
                        sub_ba, sub_f1 = metrics(y_test[mask], prediction[mask], labels)
                        distance_rows.append({"trait": trait, "split": holdout, "repeat": repeat,
                                              "model": model, "nearest_train_rank": rank,
                                              "n": int(mask.sum()), "balanced_accuracy": sub_ba,
                                              "macro_f1": sub_f1})
    details = pd.DataFrame(records)
    details.to_csv(root / "reports/tables/stage2_split_metrics.tsv", sep="\t", index=False)
    distances = pd.DataFrame(distance_rows)
    distances.to_csv(root / "reports/tables/stage2_distance_metrics.tsv", sep="\t", index=False)
    summary = details.groupby(["trait", "split", "model"], as_index=False).agg(
        panel_n=("panel_n", "first"), train_n=("train_n", "median"), test_n=("test_n", "median"),
        min_test_classes=("test_classes", "min"),
        ba_mean=("balanced_accuracy", "mean"), ba_p025=("balanced_accuracy", lambda s: s.quantile(.025)),
        ba_p975=("balanced_accuracy", lambda s: s.quantile(.975)),
        f1_mean=("macro_f1", "mean"), f1_p025=("macro_f1", lambda s: s.quantile(.025)),
        f1_p975=("macro_f1", lambda s: s.quantile(.975)))
    summary.to_csv(root / "reports/tables/stage2_benchmark_summary.tsv", sep="\t", index=False)
    return summary, distances


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.stage2")
    parser.add_argument("command", choices=("prepare", "benchmark"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fasta-dir", type=Path, help="directory of accession.fna[.gz] files for optional 4-mer baseline")
    args = parser.parse_args(argv)
    root = load_config().root
    if args.command == "prepare":
        print(json.dumps(prepare(root)))
    else:
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        summary, _ = benchmark(root, args.repeats, args.fasta_dir)
        print(f"wrote {len(summary)} benchmark summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
