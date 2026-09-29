"""Join BacDive strains to GTDB genomes.

Two routes, tried in this order for each strain:

1. **accession**   BacDive lists `INSDC accession` (GCA_..., unversioned) for many strains;
                   it is matched to GTDB `ncbi_genbank_assembly_accession` with the version
                   stripped. GTDB only holds genomes that passed its QC, so a listed assembly can be
                   absent from GTDB - that is reported as its own attrition line, not hidden.
2. **designation** every normalised designation of the strain (culture-collection numbers,
                   strain designation, DSM number) is looked up, by exact key equality, among
                   the normalised `ncbi_strain_identifiers` of GTDB genomes.

                   * "collection" designations (DSM20231) are accepted directly.
                   * "bare" designations (BS107) are accepted only with taxon agreement
                     (`bare_requires_taxon_agreement`), because a bare identifier is not globally
                     unique ("LP1", "B86" recur across unrelated genera). Taxon agreement is exact:
                     the BacDive genus equals the GTDB (suffix-stripped) or NCBI genus of the genome,
                     OR one of BacDive's NCBI tax IDs equals the genome's ncbi_taxid or
                     ncbi_species_taxid (this catches genus renames, e.g. Salinibacterium ->
                     Homoserinimonas).

There is no fuzzy matching anywhere in this module.

When several genomes qualify, one is chosen deterministically (genus agreement, GTDB representative,
RefSeq over GenBank, assembly level, CheckM2 quality, accession) and `n_gtdb_candidates` /
`match_ambiguous_species` record how ambiguous the match was.

`match_stage_table` gives the before/after-normalisation comparison. Every designation step applies
the same acceptance rule (collection codes accepted, bare designations need taxon agreement), so
the steps differ only in how strings are compared, and the last row equals the real match count.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config
from .fetch import GtdbInfo
from .normalise import Normaliser

log = logging.getLogger(__name__)

GTDB_COLUMNS = [
    "accession", "ncbi_genbank_assembly_accession", "ncbi_strain_identifiers", "gtdb_taxonomy",
    "gtdb_representative", "gtdb_type_designation_ncbi_taxa", "ncbi_type_material_designation",
    "ncbi_assembly_level", "ncbi_organism_name", "checkm2_completeness", "checkm2_contamination", "genome_size",
    "ncbi_taxid", "ncbi_species_taxid",
]
_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_SUFFIX_RE = re.compile(r"_[A-Z]{1,3}$")


def _strip_rank(x: str | None) -> str | None:
    if not isinstance(x, str) or "__" not in x:
        return None
    v = x.split("__", 1)[1].strip()
    return v or None


def _ncbi_genus(name) -> str | None:
    if not isinstance(name, str):
        return None
    toks = [t.strip("[]\"'()") for t in name.split()]
    toks = [t for t in toks if t]
    if toks and toks[0].lower() == "candidatus" and len(toks) > 1:
        toks = toks[1:]
    return toks[0].lower() if toks else None


def prepare_gtdb(raw: pd.DataFrame, level_rank: dict[str, int]) -> pd.DataFrame:
    """Add derived columns (ranks, accessions, preference order) to the raw GTDB metadata subset."""
    missing = [c for c in GTDB_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"GTDB metadata is missing expected columns {missing}; the release layout has changed")
    g = raw[GTDB_COLUMNS].copy().reset_index(drop=True)
    g["accession"] = g["accession"].astype(str)
    g["gtdb_prefix"] = g["accession"].str[:3]
    g["assembly_genbank"] = g["ncbi_genbank_assembly_accession"].where(
        g["ncbi_genbank_assembly_accession"].astype(str).str.startswith("GC"), None)
    # GB_ accessions are the GenBank accession themselves
    gb_fallback = g["accession"].where(g["gtdb_prefix"] == "GB_").str[3:]
    g["assembly_genbank"] = g["assembly_genbank"].fillna(gb_fallback)
    g["assembly_refseq"] = g["accession"].where(g["gtdb_prefix"] == "RS_").str[3:]
    g["gca_base"] = g["assembly_genbank"].astype(object).map(lambda a: a.split(".")[0] if isinstance(a, str) else None)
    tax = g["gtdb_taxonomy"].astype(object).map(lambda s: [_strip_rank(x) for x in s.split(";")] if isinstance(s, str) else [None] * 7)
    for i, r in enumerate(_RANKS):
        g[f"gtdb_{r}"] = tax.map(lambda t, i=i: t[i] if i < len(t) else None)
    g["genus_gtdb"] = g["gtdb_genus"].astype(object).map(lambda x: _SUFFIX_RE.sub("", x).lower() if isinstance(x, str) else None)
    g["genus_ncbi"] = g["ncbi_organism_name"].map(_ncbi_genus)
    g["gtdb_is_representative"] = g["gtdb_representative"].astype(str).str.lower().isin(["t", "true", "1"])
    g["gtdb_is_type_strain_of_species"] = g["gtdb_type_designation_ncbi_taxa"].astype(str).eq("type strain of species")
    for c in ("checkm2_completeness", "checkm2_contamination", "genome_size"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    for c in ("ncbi_taxid", "ncbi_species_taxid"):
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("Int64")
    g["assembly_level_rank"] = g["ncbi_assembly_level"].map(level_rank).fillna(0).astype(int)
    quality = g["checkm2_completeness"].fillna(0) - 5 * g["checkm2_contamination"].fillna(100)
    order = np.lexsort((g["accession"].values, -quality.values, -g["assembly_level_rank"].values,
                        -(g["gtdb_prefix"] == "RS_").astype(int).values, -g["gtdb_is_representative"].astype(int).values))
    pref = np.empty(len(g), dtype=np.int64)
    pref[order] = np.arange(len(g))
    g["pref_order"] = pref
    return g


def load_gtdb(cfg: Config, info: GtdbInfo) -> pd.DataFrame:
    raw = pd.read_csv(info.metadata_path, sep="\t", usecols=GTDB_COLUMNS, low_memory=False)
    return prepare_gtdb(raw, cfg["matching"]["assembly_level_rank"])


class GtdbIndex:
    """Lookup structures over the GTDB genomes."""

    def __init__(self, g: pd.DataFrame, norm: Normaliser):
        self.g = g
        self.norm = norm
        self.by_gca: dict[str, list[int]] = defaultdict(list)
        for i, a in enumerate(g["gca_base"].tolist()):
            if isinstance(a, str):
                self.by_gca[a].append(i)
        self.by_key: dict[str, list[int]] = defaultdict(list)       # normalised key -> rows
        self.by_raw_full: dict[str, list[int]] = defaultdict(list)  # whole raw string -> rows (baseline)
        self.by_raw_token: dict[str, list[int]] = defaultdict(list)  # raw split token -> rows (baseline)
        for i, s in enumerate(g["ncbi_strain_identifiers"].astype(object).tolist()):
            if not isinstance(s, str) or not s.strip():
                continue
            self.by_raw_full[s.strip()].append(i)
            for tok in norm.raw_tokens(s):
                self.by_raw_token[tok].append(i)
            for d in norm.parse(s):
                if d.matchable:
                    self.by_key[d.canonical].append(i)
        self.genus_gtdb = g["genus_gtdb"].tolist()
        self.genus_ncbi = g["genus_ncbi"].tolist()
        self.taxid = [None if pd.isna(x) else int(x) for x in g["ncbi_taxid"]]
        self.sp_taxid = [None if pd.isna(x) else int(x) for x in g["ncbi_species_taxid"]]
        self.species = g["gtdb_species"].tolist()
        self.pref = g["pref_order"].tolist()

    def genus_agrees(self, genus: str | None, i: int) -> bool:
        if not isinstance(genus, str) or not genus:  # missing values arrive as NaN floats
            return False
        gl = genus.lower()
        return gl == self.genus_gtdb[i] or gl == self.genus_ncbi[i]

    def taxon_agreement(self, genus: str | None, taxids: set[int], i: int) -> str | None:
        """'genus', 'ncbi_taxid', or None. Exact comparisons only."""
        if self.genus_agrees(genus, i):
            return "genus"
        if taxids and (self.taxid[i] in taxids or self.sp_taxid[i] in taxids):
            return "ncbi_taxid"
        return None


def bacdive_taxids(raw) -> set[int]:
    """NCBI tax IDs listed in a BacDive record (species- and strain-level)."""
    if not isinstance(raw, str) or not raw:
        return set()
    out = set()
    for e in json.loads(raw):
        v = e.get("NCBI tax id") if isinstance(e, dict) else e
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            pass
    return out


def _strain_designations(row, norm: Normaliser):
    """Parsed designations of a strain across all BacDive sources, deduplicated by key."""
    seen, out = set(), []
    dsm = row.dsm_number if isinstance(row.dsm_number, str) and row.dsm_number else None
    sources = [("culture_collection_no", row.culture_collection_no_raw), ("strain_designation", row.strain_designation_raw),
               ("dsm_number", f"DSM {dsm}" if dsm else None)]
    for src, s in sources:
        for d in norm.parse(s):
            if d.matchable and d.canonical not in seen:
                seen.add(d.canonical)
                out.append((d, src))
    return out


def match_strains(traits: pd.DataFrame, idx: GtdbIndex, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match every BacDive strain. Returns (matches, unmatched_log)."""
    norm = idx.norm
    require_taxon = cfg["matching"]["bare_requires_taxon_agreement"]
    g = idx.g
    rows, unmatched = [], []
    cols = ["bacdive_id", "genus", "species", "culture_collection_no_raw", "strain_designation_raw", "dsm_number",
            "bacdive_gca", "ncbi_tax_ids_raw"]
    for r in traits[cols].itertuples(index=False):
        desigs = _strain_designations(r, norm)
        genus = r.genus
        taxids = bacdive_taxids(r.ncbi_tax_ids_raw)
        gcas = r.bacdive_gca.split("|") if isinstance(r.bacdive_gca, str) else []

        def accepted(kind: str, i: int) -> bool:
            if kind == "collection":
                return True
            if kind == "bare":
                return (not require_taxon) or idx.taxon_agreement(genus, taxids, i) is not None
            return False  # unusable

        # -- baselines for the before/after comparison, same acceptance rule -------------------
        raw_strings = [s for s in (r.culture_collection_no_raw, r.strain_designation_raw) if isinstance(s, str) and s.strip()]
        hit_raw_full = False
        for s in raw_strings:
            kinds = {norm.parse(tok)[0].kind if norm.parse(tok) else "unusable" for tok in norm.raw_tokens(s)}
            kind = "collection" if "collection" in kinds else ("bare" if "bare" in kinds else "unusable")
            if any(accepted(kind, i) for i in idx.by_raw_full.get(s.strip(), [])):
                hit_raw_full = True
        hit_raw_tokens = hit_raw_tokens_unchecked = False
        for s in raw_strings:
            for tok in norm.raw_tokens(s):
                hits = idx.by_raw_token.get(tok, [])
                if hits:
                    hit_raw_tokens_unchecked = True
                    parsed = norm.parse(tok)
                    kind = parsed[0].kind if parsed else "unusable"
                    if any(accepted(kind, i) for i in hits):
                        hit_raw_tokens = True

        # -- designation candidates (normalised keys) -----------------------------------------
        coll_rows, bare_rows, bare_rejected = {}, {}, 0
        for d, src in desigs:
            for i in idx.by_key.get(d.canonical, []):
                if d.kind == "collection":
                    coll_rows.setdefault(i, d.canonical)
                elif accepted("bare", i):
                    bare_rows.setdefault(i, d.canonical)
                else:
                    bare_rejected += 1
        desig_rows = {**bare_rows, **coll_rows}  # collection key wins if both hit the same row

        # -- direct accession -------------------------------------------------------------------
        acc_rows = []
        for a in gcas:
            acc_rows += idx.by_gca.get(a, [])
        acc_rows = list(dict.fromkeys(acc_rows))

        if acc_rows:
            method, cand, key, kind = "accession", acc_rows, None, "accession"
        elif desig_rows:
            method, cand, key, kind = "designation", list(desig_rows), None, None
        else:
            method, cand, key, kind = "none", [], None, None

        chosen = None
        if cand:
            chosen = min(cand, key=lambda i: (idx.taxon_agreement(genus, taxids, i) is None, idx.pref[i]))
            if method == "designation":
                key = desig_rows[chosen]
                kind = "collection" if chosen in coll_rows else "bare"
        species_cands = {idx.species[i] for i in cand if idx.species[i]}
        via = idx.taxon_agreement(genus, taxids, chosen) if chosen is not None else None

        # precision check: does the designation route point at the same genome/species as the accession?
        agree = None
        if method == "accession" and desig_rows:
            agree = chosen in desig_rows or bool(species_cands & {idx.species[i] for i in desig_rows})

        rows.append({
            "bacdive_id": r.bacdive_id,
            "match_method": method,
            "match_key": key,
            "match_token_kind": kind,
            "n_gtdb_candidates": len(cand),
            "match_ambiguous_species": len(species_cands) > 1,
            "genus_agree": bool(chosen is not None and idx.genus_agrees(genus, chosen)),
            "taxon_agree": via is not None,
            "taxon_agree_via": via,
            "bacdive_lists_assembly": bool(gcas),
            "bacdive_assembly_in_gtdb": bool(acc_rows),
            "n_designation_keys": len(desigs),
            "designation_route_hit": bool(desig_rows),
            "designation_collection_hit": bool(coll_rows),
            "designation_bare_hit": bool(bare_rows),
            "designation_bare_hit_rejected": bare_rejected > 0 and not bare_rows,
            "designation_agrees_with_accession": agree,
            "baseline_raw_full_hit": hit_raw_full,
            "baseline_raw_token_hit": hit_raw_tokens,
            "baseline_raw_token_hit_unchecked": hit_raw_tokens_unchecked,
            "_gtdb_row": chosen if chosen is not None else -1,
        })

        if chosen is None:
            first = "bacdive_assembly_not_in_gtdb" if gcas else "no_bacdive_assembly"
            if not desigs:
                second = "no_usable_designation"
            elif bare_rejected:
                second = "bare_designation_taxon_mismatch"
            else:
                second = "designations_not_in_gtdb"
            unmatched.append({
                "bacdive_id": r.bacdive_id, "species": r.species, "genus": genus,
                "culture_collection_no_raw": r.culture_collection_no_raw, "strain_designation_raw": r.strain_designation_raw,
                "dsm_number": r.dsm_number, "bacdive_gca": r.bacdive_gca,
                "keys_tried": ";".join(d.canonical for d, _ in desigs), "reason": f"{first};{second}",
            })

    m = pd.DataFrame(rows)
    take = ["accession", "assembly_genbank", "assembly_refseq", "gtdb_phylum", "gtdb_class", "gtdb_order", "gtdb_family",
            "gtdb_genus", "gtdb_species", "gtdb_is_representative", "gtdb_type_designation_ncbi_taxa",
            "ncbi_type_material_designation", "gtdb_is_type_strain_of_species", "ncbi_assembly_level",
            "checkm2_completeness", "checkm2_contamination", "genome_size", "pref_order", "ncbi_organism_name"]
    sel = g.iloc[m["_gtdb_row"].clip(lower=0)][take].reset_index(drop=True)
    sel.columns = ["gtdb_accession" if c == "accession" else c for c in sel.columns]
    has = (m["_gtdb_row"] >= 0).values
    for c in sel.columns:
        sel[c] = sel[c].astype(object).where(has, None)
    m = pd.concat([m.drop(columns="_gtdb_row"), sel], axis=1)
    m["genome_matched"] = has
    m["ncbi_assembly_accession"] = m["assembly_refseq"].where(m["assembly_refseq"].notna(), m["assembly_genbank"])
    for c in ("gtdb_is_representative", "gtdb_is_type_strain_of_species"):
        m[c] = m[c].map(lambda v: bool(v) if v is not None and v == v else False)
    return m, pd.DataFrame(unmatched, columns=["bacdive_id", "species", "genus", "culture_collection_no_raw",
                                               "strain_designation_raw", "dsm_number", "bacdive_gca", "keys_tried", "reason"])


def match_stage_table(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Before/after normalisation: strains with >=1 accepted GTDB genome at each step, per subset.

    Designation route (every step applies the same acceptance rule: collection codes accepted,
    bare designations need taxon agreement; steps are cumulative):
      A  raw whole strings, exact equality                      <- "before normalisation"
      B  + split on the delimiters, raw tokens, exact equality
      C  + normalised keys                                       <- "after normalisation"
    Then
      D  direct accession route alone
      E  final: accession OR normalised designation (== genome_matched)
    Plus an informational row: raw token hits *without* the taxon check, which includes the
    cross-genus collisions the check exists to reject. It is not a match count.
    """
    a = df["baseline_raw_full_hit"]
    b = a | df["baseline_raw_token_hit"]
    c = b | df["designation_route_hit"]
    d = df["bacdive_assembly_in_gtdb"]
    e = d | df["designation_route_hit"]
    if not (e == df["genome_matched"]).all():
        raise AssertionError("final stage does not reproduce genome_matched")
    steps = [("A. designation, raw whole strings (before normalisation)", a),
             ("B. designation, + split into tokens, raw", b),
             ("C. designation, + normalised keys (after normalisation)", c),
             ("D. accession route alone", d),
             ("E. final: accession or normalised designation", e),
             ("(info) raw tokens without the taxon check — includes cross-genus collisions, not a match count",
              df["baseline_raw_token_hit_unchecked"])]
    out = []
    for label, s in steps:
        row = {"step": label}
        for name, mask in masks.items():
            n = int(mask.sum())
            k = int((s & mask).sum())
            row[f"{name} n"] = k
            row[f"{name} %"] = 100 * k / n if n else float("nan")
        out.append(row)
    return pd.DataFrame(out)
