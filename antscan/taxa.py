"""Match Antscan specimens to GAGA genomes and pick the scans to download.

GAGA names carry lab suffixes ("Lasius flavus s1", "Myrmecia croslandi 2n2") and "cf." qualifiers, and a few
Antscan names are misspelt or use the other genus of a recent split. Matching is on a normalised
"genus species" key plus a short hand-checked synonym list.
"""
from __future__ import annotations

import re

import pandas as pd

# GAGA name (normalised) -> Antscan name (normalised). Each checked by hand against AntCat synonymy.
SYNONYMS = {
    "neoponera goeldii": "neoponera goeldi",
    "pheidole nodus": "pheidole noda",
    "rossomyrmex quandratinodus": "rossomyrmex quandratinodum",
    "diacamma indicum": "diacamma indica",
    "myrmicocrypta squomosa uncinata": "myrmicocrypta uncinata",
    "ectomomyrmex javanus": "pachycondyla javana",  # Ectomomyrmex javanus = Pachycondyla javana (genus split)
    "stictoponera bicolor": "gnamptogenys bicolor",  # Stictoponera revived from Gnamptogenys
}
# Antscan spelling errors (normalised) -> correct name
ANTSCAN_TYPOS = {
    "wasmannia auropuctata": "wasmannia auropunctata",
    "wasmannia aeropunctata": "wasmannia auropunctata",
}

_QUALIFIERS = {"cf", "aff", "nr"}
_SUFFIX = re.compile(r"^(s\d+|2n\d+)$")


def norm_name(name: str) -> str:
    """'Lasius cf. flavus s1', 'Lasius.cf.flavus' -> 'lasius flavus'."""
    toks = re.split(r"[\s._]+", str(name).lower())
    toks = [t for t in toks if t and t not in _QUALIFIERS]
    if len(toks) > 2 and _SUFFIX.match(toks[-1]):
        toks = toks[:-1]
    return " ".join(toks)


def gaga_table(species_list_path: str, genecount_columns: list[str]) -> pd.DataFrame:
    g = pd.read_csv(species_list_path, sep="\t")
    g = g.rename(columns={"GAGA ID": "gaga_id", "Species name": "gaga_species", "Subfamily": "gaga_subfamily"})
    g = g[g.gaga_id.isin(genecount_columns)].copy()
    g["key"] = g.gaga_species.map(norm_name).map(lambda k: SYNONYMS.get(k, k))
    # where GAGA has two genomes of one species (s1/s2 or karyotypes), keep the first ID: one genome per taxon
    return g.sort_values("gaga_id").drop_duplicates("key")[["gaga_id", "gaga_species", "gaga_subfamily", "key"]]


def antscan_table(specimens_csv: str) -> pd.DataFrame:
    m = pd.read_csv(specimens_csv)
    m["key"] = (m.Genus.astype(str) + " " + m.Species.astype(str)).map(norm_name)
    m["key"] = m.key.map(lambda k: ANTSCAN_TYPOS.get(k, k))
    m["voxel_um"] = m.voxel_size.astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    return m


def match(specimens_csv: str, species_list_path: str, genecount_columns: list[str]) -> pd.DataFrame:
    """Antscan specimens with a GAGA genome of the same species, one row per specimen."""
    m = antscan_table(specimens_csv)
    g = gaga_table(species_list_path, genecount_columns)
    out = m.merge(g, on="key", how="inner")
    # "Genus sp." on both sides only names the same animal when Antscan itself flags the genome link
    generic = out.key.str.fullmatch(r"\S+ sp")
    return out[~generic | out.genome_associated.notna()].reset_index(drop=True)


def pick_scans(matched: pd.DataFrame, castes=("worker",), per_taxon: int = 2) -> pd.DataFrame:
    """Up to `per_taxon` specimens per GAGA genome and caste.

    Preference: same colony as the sequenced sample, then the standard 2.44 um setting, then unstained.
    """
    d = matched[matched.caste.isin(castes)].copy()
    d["_colony"] = (d.genome_associated == "colony").astype(int)
    d["_std"] = (d.voxel_um == 2.44).astype(int)
    d["_unst"] = (d.stained.astype(str) == "unstained").astype(int)
    d = d.sort_values(["gaga_id", "caste", "_colony", "_std", "_unst", "specimen_code"],
                      ascending=[True, True, False, False, False, True])
    d = d.groupby(["gaga_id", "caste"], group_keys=False).head(per_taxon)
    return d.drop(columns=["_colony", "_std", "_unst"])
