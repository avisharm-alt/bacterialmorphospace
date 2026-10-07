"""Metadata-only queries behind docs/overt_scoping.md. Downloads no scans and no genomes.

1. MorphoSource public API (no key needed for metadata): oVert project media and specimen counts, and every specimen's
   taxon name.
2. NCBI: the four vertebrate `assembly_summary.txt` files (about 6.5 MB), matched on species name. A random sample of the
   names that do not match is re-checked through Entrez, which resolves taxonomic synonyms, to estimate how much the
   name match under-counts.

    python docs/overt_scoping_query.py --out reports/tables/overt_genome_coverage.tsv [--og2-metadata species_metadata.csv]
"""
from __future__ import annotations

import argparse
import collections
import io
import random
import re
import time
from pathlib import Path

import requests

MS = "https://www.morphosource.org/api"
PROJECT = "000368762"  # oVert Thematic Collections Network
NCBI = "https://ftp.ncbi.nlm.nih.gov/genomes"
SUMMARIES = [f"{NCBI}/{db}/{grp}/assembly_summary.txt" for db in ("refseq", "genbank") for grp in ("vertebrate_mammalian", "vertebrate_other")]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = {"Accept": "application/json", "User-Agent": "bacterialmorphospace/0.1 (research; metadata-only scoping)"}
NOT_A_SPECIES = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "null", "nr.", "indet", "indet."}


def species_name(raw: str) -> str | None:
    """'Anolis cybotes cybotes' -> 'Anolis cybotes'; 'ATRETIUM SCHISTOSUM' -> 'Atretium schistosum';
    'Rhea Rhea americana' -> 'Rhea americana'; 'Hisonotus sp.' / 'X NULL' -> None."""
    t = [x for x in raw.replace("(", " ").replace(")", " ").split() if x]
    t = [x.capitalize() if i == 0 and x.isupper() else x.lower() if i > 0 and x.isupper() else x for i, x in enumerate(t)]
    if len(t) >= 3 and t[1][:1].isupper() and t[2][:1].islower():  # repeated or former genus before the binomial
        t = t[1:]
    if len(t) < 2 or not re.fullmatch(r"[A-Z][a-z]+", t[0]) or not re.fullmatch(r"[a-z][a-z-]+\.?", t[1]) or t[1].lower() in NOT_A_SPECIES:
        return None
    return f"{t[0]} {t[1]}"


def ms_get(path: str, **params) -> dict:
    r = requests.get(f"{MS}/{path}", params=params, headers=UA, timeout=180)
    r.raise_for_status()
    return r.json()["response"]


def facet_counts(d: dict, name: str) -> dict:
    return {i["value"]: i["hits"] for f in d["facets"] if f["name"] == name for i in f["items"]}


def specimens() -> list[dict]:
    rows, page = [], 1
    while True:
        d = ms_get("physical-objects", **{"f.project": PROJECT, "per_page": 1000, "page": page})
        rows += d["physical_objects"]
        if not d["pages"].get("next_page"):
            return rows
        page += 1
        time.sleep(1)


def media_species(media_type: str, visibility: str) -> tuple[int, set[str]]:
    """(item count, set of species) for oVert media of one type and visibility; metadata only."""
    items, page = [], 1
    while True:
        d = ms_get("media", **{"f.project": PROJECT, "f.media_type": media_type, "f.publication_status": visibility, "per_page": 1000, "page": page})
        items += d["media"]
        if not d["pages"].get("next_page"):
            break
        page += 1
        time.sleep(1)
    sp = {species_name(n[0]) for i in items if (n := [x for x in (i.get("physical_object_taxonomy_name") or []) if x])}
    return len(items), {s for s in sp if s}


def assembly_index() -> dict[str, dict]:
    """species name -> best assembly info over the four NCBI vertebrate summaries."""
    idx: dict[str, dict] = {}
    for url in SUMMARIES:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        db = "refseq" if "/refseq/" in url else "genbank"
        for line in io.StringIO(r.text):
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            sp = species_name(c[7])
            if sp is None:
                continue
            e = idx.setdefault(sp, {"genbank": 0, "refseq": 0, "reference": False, "chromosome_or_complete": False})
            e[db] += 1
            e["reference"] |= c[4] in ("reference genome", "representative genome")
            e["chromosome_or_complete"] |= c[11] in ("Chromosome", "Complete Genome")
    return idx


def entrez_has_assembly(name: str) -> bool:
    r = requests.get(f"{EUTILS}/esearch.fcgi", params={"db": "assembly", "term": f'"{name}"[Organism]', "retmode": "json"},
                     headers=UA, timeout=60)
    r.raise_for_status()
    return int(r.json()["esearchresult"]["count"]) > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="reports/tables/overt_genome_coverage.tsv")
    ap.add_argument("--entrez-sample", type=int, default=60)
    ap.add_argument("--og2-metadata", help="OpenGenome2 species_metadata.csv (optional): adds the pretraining overlap")
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args()

    media = ms_get("media", **{"f.project": PROJECT, "per_page": 1})
    out = collections.OrderedDict()
    out["media_items"] = media["pages"]["total_count"]
    for k, v in facet_counts(media, "media_type").items():
        out[f"media_type: {k}"] = v
    for k, v in facet_counts(media, "publication_status").items():
        out[f"visibility: {k}"] = v
    for mt in ("Mesh", "Volumetric Image Series"):
        for vis in ("Open Download", "Restricted Download"):
            out[f"{mt} / {vis}"] = ms_get("media", **{"f.project": PROJECT, "f.media_type": mt, "f.publication_status": vis, "per_page": 1})["pages"]["total_count"]
    spec = specimens()
    out["specimens"] = len(spec)
    names = [species_name(n[0]) for r in spec if (n := [x for x in (r.get("taxonomy_name") or []) if x])]
    names = [n for n in names if n]
    sp = sorted(set(names))
    out["specimens_with_species_name"] = len(names)
    out["distinct_species"] = len(sp)
    out["distinct_genera"] = len({n.split()[0] for n in sp})

    idx = assembly_index()
    hit = [s for s in sp if s in idx]
    out["species_with_any_ncbi_assembly (name match)"] = len(hit)
    out["species_with_refseq_assembly"] = sum(idx[s]["refseq"] > 0 for s in hit)
    out["species_with_chromosome_or_complete_assembly"] = sum(idx[s]["chromosome_or_complete"] for s in hit)
    out["species_with_ncbi_reference_or_representative"] = sum(idx[s]["reference"] for s in hit)
    genera_with = {s.split()[0] for s in idx}
    out["genera_with_any_assembly_of_any_species"] = len({s.split()[0] for s in sp} & genera_with)

    for mt, vis in (("Mesh", "Open Download"), ("Volumetric Image Series", "Open Download")):
        n, s = media_species(mt, vis)
        out[f"species with an open {mt.lower()}"] = len(s)
        out[f"species with an open {mt.lower()} and any NCBI assembly"] = len(s & set(idx))
        out[f"species with an open {mt.lower()} and a RefSeq assembly"] = len({x for x in s if idx.get(x, {}).get("refseq")})
    out["specimens whose species has an NCBI assembly"] = sum(n in idx for n in names)

    if a.og2_metadata:
        import pandas as pd

        m = pd.read_csv(a.og2_metadata, usecols=["source_dataset", "phylum", "species"], dtype=str)
        og = set(m.loc[(m["source_dataset"] == "NCBI_Eukaryotic_Genomes") & (m["phylum"] == "Chordata"), "species"])
        out["species with an NCBI assembly that are in OpenGenome2's eukaryote list (by name)"] = len(set(hit) & og)

    miss = [s for s in sp if s not in idx]
    rng = random.Random(a.seed)
    sample = rng.sample(miss, min(a.entrez_sample, len(miss)))
    found = 0
    for s in sample:
        found += entrez_has_assembly(s)
        time.sleep(0.4)  # NCBI: 3 requests/s without an API key
    out["entrez_check: unmatched names sampled"] = len(sample)
    out["entrez_check: of those with an assembly under a synonym"] = found
    out["entrez_check: implied extra species among all unmatched (point estimate)"] = round(len(miss) * found / max(len(sample), 1))

    for k, v in out.items():
        print(f"{k}\t{v}")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("metric\tvalue\n" + "".join(f"{k}\t{v}\n" for k, v in out.items()))


if __name__ == "__main__":
    main()
