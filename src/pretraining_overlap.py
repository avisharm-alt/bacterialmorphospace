"""Pretraining leakage: which panel genomes were in Evo 2's GTDB training data (OpenGenome2)?

Evo 2's prokaryotic pretraining data is the GTDB release 214.1 species-representative genomes (85,205) plus 28,174 genomes
added from release 220 (species with no 214.1 genome in their cluster). OpenGenome2's `species_metadata.csv` lists
only the GTDB_v220 rows (28,177), NOT the 214.1 base set, so matching panel accessions against that file alone
under-counts. The 214.1 representatives come from GTDB's public `sp_clusters_r214.tsv` (not from the OpenGenome2 shards).

Tiers, from strongest to weakest evidence that Evo 2 saw the genome or a near-copy of it:

- `genome_seen`: the panel genome is itself a 214.1 representative or one of the OpenGenome2 GTDB_v220 genomes.
- `species_seen`: not `genome_seen`, but its species cluster in the panel's GTDB release contains a genome that is in
  the training set (a same-species genome, >= 95% ANI, was trained on).
- `unseen`: no genome of its species cluster is in the training set. This is "unseen at species level", not "unseen
  at genus level": nearly every panel genus was in training.

Only the GTDB/IMG-PR part of the corpus is checked. Metagenomes (MGD, 46% of phase-1 tokens) can contain the same species
and cannot be checked from metadata, so `unseen` is an upper bound on how novel a genome is to the model.

Pure logic, no network. Inputs are plain files; see `main` for the command line.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

TIERS = ("genome_seen", "species_seen", "unseen")
OG2_GTDB_SOURCE = "GTDB_v220"
_PREFIX = re.compile(r"^(?:RS_|GB_)?GC[AF]_")


def assembly_key(acc: str) -> str:
    """Assembly number with version, without the RS_/GB_ and GCA_/GCF_ prefixes: 'RS_GCF_001580535.1' -> '001580535.1'.

    GTDB flips a genome between its GenBank (GCA) and RefSeq (GCF) accession across releases, and the paired accessions
    share the number and version, so the key must not depend on the prefix. The version stays: a different version is a
    different assembly."""
    return _PREFIX.sub("", acc.strip())


def read_sp_clusters(path: str | Path) -> list[tuple[str, str, str, list[str]]]:
    """[(representative key, species, taxonomy string, [member keys])] from a GTDB sp_clusters TSV.

    The members column includes the representative. Fields can be megabytes long (E. coli), so the csv limit is lifted."""
    csv.field_size_limit(1 << 30)
    out = []
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        if header[0] != "Representative genome" or header[-1] != "Clustered genomes":
            raise ValueError(f"{path} does not look like a GTDB sp_clusters file: header {header[:2]}...{header[-1:]}")
        for row in r:
            members = [assembly_key(a) for a in row[-1].split(",") if a]
            out.append((assembly_key(row[0]), row[1], row[2], members))
    return out


def read_og2_gtdb(path: str | Path) -> set[str]:
    """Assembly keys of the GTDB genomes listed in OpenGenome2's species_metadata.csv (the release-220 additions)."""
    m = pd.read_csv(path, usecols=["source_dataset", "accession_or_taxid"], dtype=str)
    return {assembly_key(a) for a in m.loc[m["source_dataset"] == OG2_GTDB_SOURCE, "accession_or_taxid"]}


def training_genomes(r214_clusters: list, og2_keys: set[str]) -> set[str]:
    """The GTDB genomes Evo 2 was trained on: every 214.1 representative, plus the OpenGenome2 release-220 genomes."""
    return {rep for rep, _, _, _ in r214_clusters} | set(og2_keys)


def classify(panel_keys: list[str], train: set[str], clusters: list) -> pd.DataFrame:
    """Tier for each panel genome (see module docstring). `clusters` are the panel's own GTDB release's species clusters.

    A genome missing from `clusters` cannot be placed in a species and is reported as `not_in_release`, never as
    `unseen`. Also returns the cluster's species and genus, whether any genome of that genus was trained on, and the
    number of trained genomes in the cluster."""
    cl_of, trained_in = {}, []
    genus_trained: set[str] = set()
    for i, (_, _, tax, members) in enumerate(clusters):
        for k in members:
            cl_of[k] = i
        n = sum(k in train for k in members)
        trained_in.append(n)
        if n:
            genus_trained.add(_genus(tax))
    rows = []
    for k in panel_keys:
        i = cl_of.get(k)
        if i is None:
            rows.append({"key": k, "tier": "not_in_release", "cluster_species": None, "cluster_genus": None,
                         "genus_seen": None, "n_trained_in_species": None})
            continue
        tier = "genome_seen" if k in train else ("species_seen" if trained_in[i] else "unseen")
        genus = _genus(clusters[i][2])
        rows.append({"key": k, "tier": tier, "cluster_species": clusters[i][1], "cluster_genus": genus,
                     "genus_seen": genus in genus_trained, "n_trained_in_species": trained_in[i]})
    return pd.DataFrame(rows)


def _genus(taxonomy: str) -> str:
    return next((x for x in taxonomy.split(";") if x.startswith("g__")), "g__")


def tier_counts(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Counts and shares of each tier within each level of `by`, with an 'all' row."""
    tiers = list(TIERS) + [t for t in df["tier"].unique() if t not in TIERS]
    ct = pd.crosstab(df[by], df["tier"]).reindex(columns=tiers, fill_value=0)
    ct.loc["all"] = ct.sum()
    ct.insert(0, "n", ct.sum(axis=1))
    for t in tiers:
        ct[f"share_{t}"] = (ct[t] / ct["n"]).round(4)
    ct.index.name = by
    return ct.reset_index()


def unseen_feasibility(df: pd.DataFrame, targets: dict[str, tuple[str, str]], unseen_tiers: tuple[str, ...],
                       phylum_col: str = "gtdb_phylum", genus_col: str = "gtdb_genus") -> pd.DataFrame:
    """Per held-out phylum: how many genomes fall in `unseen_tiers`, how many genera they span, and for every target the
    positive/negative counts among them (a within-phylum AUC needs both) next to the prevalence among the rest."""
    rows = []
    for ph in sorted(df[phylum_col].unique()):
        d = df[df[phylum_col] == ph]
        u, s = d[d["tier"].isin(unseen_tiers)], d[~d["tier"].isin(unseen_tiers)]
        row = {"phylum": ph, "n": len(d), "n_unseen": len(u), "n_unseen_genera": u[genus_col].nunique()}
        for name, (col, pos) in targets.items():
            row[f"{name}_pos_unseen"] = int((u[col] == pos).sum())
            row[f"{name}_neg_unseen"] = int((u[col] != pos).sum())
            row[f"{name}_prev_unseen"] = round(float((u[col] == pos).mean()), 3) if len(u) else None
            row[f"{name}_prev_rest"] = round(float((s[col] == pos).mean()), 3) if len(s) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--og2-metadata", required=True, help="species_metadata.csv from the OpenGenome2 HF repo")
    ap.add_argument("--r214-clusters", required=True, help="GTDB 214.1 auxillary_files/sp_clusters_r214.tsv")
    ap.add_argument("--clusters", required=True, help="sp_clusters.tsv of the panel's GTDB release (v232)")
    ap.add_argument("--panel", default="data/final/core_panel_species.tsv")
    ap.add_argument("--failures", default="reports/tables/evo2_embed_all_failures.tsv")
    ap.add_argument("--out", default="reports/tables")
    ap.add_argument("--phyla", default="Pseudomonadota,Bacillota,Actinomycetota,Bacteroidota")
    a = ap.parse_args(argv)

    targets = {"motility": ("motility", "yes"), "oxygen_aerobe": ("oxygen", "aerobe"),
               "oxygen_facultative": ("oxygen", "facultative"), "shape_rod": ("shape", "rod")}  # as evo2_modal.TARGETS
    r214, latest = read_sp_clusters(a.r214_clusters), read_sp_clusters(a.clusters)
    og2 = read_og2_gtdb(a.og2_metadata)
    train = training_genomes(r214, og2)

    panel = pd.read_csv(a.panel, sep="\t")
    panel["key"] = panel["gtdb_accession"].map(assembly_key)
    panel["in_og2_metadata"] = panel["key"].isin(og2)  # the naive match: lists only the release-220 additions
    panel["r214_representative"] = panel["key"].isin({r for r, _, _, _ in r214})
    failed = set(pd.read_csv(a.failures, sep="\t")["acc"]) if Path(a.failures).exists() else set()
    keep = [p.strip() for p in a.phyla.split(",") if p.strip()]
    panel["embedded"] = panel["gtdb_phylum"].isin(keep) & ~panel["ncbi_assembly_accession"].isin(failed)
    df = panel.merge(classify(list(panel["key"]), train, latest), on="key", how="left")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["bacdive_id", "gtdb_species", "gtdb_genus", "gtdb_phylum", "gtdb_accession", "ncbi_assembly_accession", "embedded",
            "tier", "in_og2_metadata", "r214_representative", "genus_seen", "n_trained_in_species",
            "gram", "shape", "motility", "spore", "oxygen", "temperature"]
    df[cols].to_csv(out / "opengenome2_overlap_genomes.tsv", sep="\t", index=False)
    emb = df[df["embedded"]]
    tier_counts(emb, "gtdb_phylum").to_csv(out / "opengenome2_overlap_by_phylum.tsv", sep="\t", index=False)
    traits = []
    for t in ("motility", "oxygen", "shape", "gram", "spore", "temperature"):
        c = tier_counts(emb, t)
        c.insert(0, "trait", t)
        traits.append(c.rename(columns={t: "label"}))
    pd.concat(traits).to_csv(out / "opengenome2_overlap_by_trait.tsv", sep="\t", index=False)
    feas = []
    for name, tiers in (("unseen_species", ("unseen",)), ("not_genome_seen", ("unseen", "species_seen"))):
        f = unseen_feasibility(emb, targets, tiers)
        f.insert(0, "definition", name)
        feas.append(f)
    pd.concat(feas).to_csv(out / "opengenome2_unseen_feasibility.tsv", sep="\t", index=False)

    sizes = {"r214_representatives": len(r214), "og2_gtdb_v220_rows": len(og2), "og2_rows_also_r214_reps":
             len(og2 & {r for r, _, _, _ in r214}), "training_genomes_union": len(train)}
    print(sizes)
    print(tier_counts(emb, "gtdb_phylum").to_string(index=False))


if __name__ == "__main__":
    main()
