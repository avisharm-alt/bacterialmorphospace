"""Extraction of the eight traits from BacDive v2 strain records.

Field paths were verified against the live API (`/v2/fetch`, `/strain_fields_information`)
and real records in Sept 2026. Paths are relative to `results["<bacdive_id>"]`:

  gram        Morphology > cell morphology > "gram stain"
  shape       Morphology > cell morphology > "cell shape"
  motility    Morphology > cell morphology > "motility"
  spore       Physiology and metabolism > spore formation > "spore formation"
  oxygen      Physiology and metabolism > oxygen tolerance > "oxygen tolerance"
  temperature Culture and growth conditions > culture temp > {type, growth, temperature}
  ph          Culture and growth conditions > culture pH   > {type, ability, pH}
  halophily   Physiology and metabolism > halophily > "halophily level" (curated category)
              and the NaCl growth tests (salt, growth, tested relation, concentration)

Every subsection may be absent, a dict, or a list of dicts (one per reference).

Per trait the output columns are (X = trait name)
  X_raw           every observation, as JSON  [{"ref":..,"value":..}, ...]     nothing is lost
  X_n_obs         number of observations
  X_fine          fine bin        (None if no usable observation or a conflict)
  X               coarse bin      (None if no usable observation or a conflict)
  X_fine_conflict observations disagree at the fine level
  X_conflict      observations disagree at the coarse level -> X is nulled, never guessed
  (numeric traits also) X_opt, X_growth_min, X_growth_max, X_point, X_bin_source

Conflicts are evaluated separately at the fine and coarse level: microaerophile + facultative
anaerobe conflict at the fine level but agree once microaerophile is folded into facultative.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from collections import Counter
from typing import Any

import pandas as pd

from .config import Config
from .fetch import iter_cached_records

CATEGORICAL_SPECS = {
    "gram": (("Morphology", "cell morphology"), "gram stain"),
    "shape": (("Morphology", "cell morphology"), "cell shape"),
    "motility": (("Morphology", "cell morphology"), "motility"),
    "spore": (("Physiology and metabolism", "spore formation"), "spore formation"),
    "oxygen": (("Physiology and metabolism", "oxygen tolerance"), "oxygen tolerance"),
}
NUMERIC_SPECS = {
    "temperature": (("Culture and growth conditions", "culture temp"), "temperature"),
    "ph": (("Culture and growth conditions", "culture pH"), "pH"),
}
GCA_RE = re.compile(r"^(GC[AF]_\d{9})(?:\.\d+)?$")
DOI_DATE_RE = re.compile(r"\.(\d{8})\.")


def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def entries(rec: dict, *path: str) -> list[dict]:
    o: Any = rec
    for k in path:
        if not isinstance(o, dict):
            return []
        o = o.get(k)
    return [e for e in as_list(o) if isinstance(e, dict)]


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ---------------------------------------------------------------------------
# categorical
# ---------------------------------------------------------------------------
def _map_value(trait: str, raw: str, tcfg: dict, collectors: dict) -> str | None:
    fine = tcfg["fine_map"].get(raw.lower())
    if fine is not None:
        return fine
    collectors["unmapped"][(trait, raw)] += 1
    return "other" if tcfg.get("unmapped") == "other" else None


def _resolve(values: list[str]) -> tuple[str | None, bool]:
    """(value, conflict) for a list of already-mapped observations."""
    s = set(values)
    if not s:
        return None, False
    if len(s) == 1:
        return next(iter(s)), False
    return None, True


def extract_categorical(rec: dict, trait: str, tcfg: dict, collectors: dict) -> dict:
    path, key = CATEGORICAL_SPECS[trait]
    obs = []
    for e in entries(rec, *path):
        v = _clean(e.get(key))
        if v is not None:
            obs.append({"ref": e.get("@ref"), "value": v})
    fine_vals = [f for f in (_map_value(trait, o["value"], tcfg, collectors) for o in obs) if f is not None]
    coarse_vals = [tcfg["coarse_map"][f] for f in fine_vals]
    fine, fine_conf = _resolve(fine_vals)
    coarse, coarse_conf = _resolve(coarse_vals)
    out = {
        f"{trait}_raw": json.dumps(obs, ensure_ascii=False) if obs else None,
        f"{trait}_n_obs": len(obs),
        f"{trait}_fine": fine,
        trait: coarse,
        f"{trait}_fine_conflict": fine_conf,
        f"{trait}_conflict": coarse_conf,
    }
    if trait == "oxygen":
        out["oxygen_microaerophile_folded"] = bool(coarse and fine == tcfg.get("folded_flag_fine"))
    return out


# ---------------------------------------------------------------------------
# numeric (temperature, pH)
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"  # signed: psychrophile growth ranges go below 0 C ("-1.8-10")
_VALUE_RE = re.compile(rf"^(?:[<>]=?)?\s*(?P<a>{_NUM})\s*(?:(?:-|to)\s*(?P<b>{_NUM}))?$")
_UNIT_RE = re.compile(r"(?:^ph\s*|\s*(?:°\s*c|°|c|degrees?)\s*$)", re.I)


def parse_numeric(v: Any) -> tuple[float, float] | None:
    """Parse "37", "22-37", "5.0", ">7", "<4.5", "37 °C" into (lo, hi). `<x`/`>x` -> point x."""
    s = _clean(v)
    if s is None:
        return None
    s = _UNIT_RE.sub("", s.replace("–", "-").replace("−", "-")).strip()
    s = re.sub(r"\((-?\d+(?:\.\d+)?)\)", r"\1", s)  # "(-2)-(-1)" -> "-2--1"
    m = _VALUE_RE.match(s)
    if not m:
        return None
    a = float(m.group("a"))
    b = float(m.group("b")) if m.group("b") else a
    if b < 0 <= a:  # "25--30": a non-negative lower bound with a negative upper bound is a typo, not a range
        return None
    return (min(a, b), max(a, b))


def _result(e: dict) -> str | None:
    r = _clean(e.get("growth")) or _clean(e.get("ability"))
    if r is None:
        return None
    r = r.lower()
    if r in ("positive", "yes", "+"):
        return "pos"
    if r in ("negative", "no", "-"):
        return "neg"
    return "other"


def bin_value(x: float, edges: list[float], levels: list[str]) -> str:
    return levels[bisect.bisect_right(edges, x)]


def extract_numeric(rec: dict, trait: str, tcfg: dict, collectors: dict) -> dict:
    """Bin from the optimum if present, else the growth range, else a single growth value."""
    path, key = NUMERIC_SPECS[trait]
    raw_obs, opt_pts, pos_vals, pos_ints, neg_ints = [], [], [], [], []
    for e in entries(rec, *path):
        typ = (_clean(e.get("type")) or "").lower() or None
        res = _result(e)
        rv = _clean(e.get(key))
        raw_obs.append({"ref": e.get("@ref"), "type": typ, "result": res, "value": rv})
        if rv is None:
            continue
        parsed = parse_numeric(rv)
        if parsed is None:
            collectors["unparsed"][(trait, rv)] += 1
            continue
        lo, hi = parsed
        if typ == "optimum" and res in (None, "pos"):
            opt_pts.append((lo + hi) / 2)
        elif typ in ("growth", "minimum", "maximum"):
            if res == "pos":
                pos_vals += [lo, hi]
                pos_ints.append((lo, hi))
            elif res == "neg":
                neg_ints.append((lo, hi))

    edges, levels, cmap = tcfg["fine_edges"], tcfg["fine_levels"], tcfg["coarse_map"]
    out = {
        f"{trait}_raw": json.dumps(raw_obs, ensure_ascii=False) if raw_obs else None,
        f"{trait}_n_obs": len(raw_obs),
        f"{trait}_opt": (sum(opt_pts) / len(opt_pts)) if opt_pts else math.nan,
        f"{trait}_growth_min": min(pos_vals) if pos_vals else math.nan,
        f"{trait}_growth_max": max(pos_vals) if pos_vals else math.nan,
        f"{trait}_point": math.nan,
        f"{trait}_bin_source": "none",
        f"{trait}_fine": None,
        trait: None,
        f"{trait}_fine_conflict": False,
        f"{trait}_conflict": False,
    }
    if opt_pts:
        fine_set = {bin_value(p, edges, levels) for p in opt_pts}
        coarse_set = {cmap[f] for f in fine_set}
        out[f"{trait}_bin_source"] = "optimum"
        out[f"{trait}_point"] = sum(opt_pts) / len(opt_pts)
        out[f"{trait}_fine"], out[f"{trait}_fine_conflict"] = _resolve(list(fine_set))
        out[trait], out[f"{trait}_conflict"] = _resolve(list(coarse_set))
    elif pos_vals:
        lo, hi = min(pos_vals), max(pos_vals)
        distinct = sorted(set(pos_vals))
        out[f"{trait}_bin_source"] = "growth_range" if len(distinct) >= 2 else "growth_single_point"
        point = (lo + hi) / 2
        out[f"{trait}_point"] = point
        # a recorded no-growth interval that overlaps the growth span contradicts it
        contradicted = any(nlo <= hi and nhi >= lo for nlo, nhi in neg_ints)
        if contradicted:
            out[f"{trait}_fine_conflict"] = out[f"{trait}_conflict"] = True
        else:
            fine = bin_value(point, edges, levels)
            out[f"{trait}_fine"], out[trait] = fine, cmap[fine]
    return out


# ---------------------------------------------------------------------------
# halophily (reported, not part of the occupancy analysis)
# ---------------------------------------------------------------------------
def extract_halophily(rec: dict) -> dict:
    ents = entries(rec, "Physiology and metabolism", "halophily")
    levels = [{"ref": e.get("@ref"), "value": _clean(e.get("halophily level"))} for e in ents if _clean(e.get("halophily level"))]
    tests = [
        {"ref": e.get("@ref"), "salt": _clean(e.get("salt")), "growth": _clean(e.get("growth")),
         "tested_relation": _clean(e.get("tested relation")), "concentration": _clean(e.get("concentration"))}
        for e in ents if _clean(e.get("salt")) or _clean(e.get("concentration"))
    ]
    vals = {lv["value"].lower() for lv in levels}
    return {
        "halophily_level_raw": json.dumps(levels, ensure_ascii=False) if levels else None,
        "halophily_level_n_obs": len(levels),
        "halophily_level": next(iter(vals)) if len(vals) == 1 else None,
        "halophily_level_conflict": len(vals) > 1,
        "nacl_tests_raw": json.dumps(tests, ensure_ascii=False) if tests else None,
        "nacl_tests_n": len(tests),
    }


# ---------------------------------------------------------------------------
# identity / genome fields
# ---------------------------------------------------------------------------
def _joined(v: Any) -> str | None:
    if v is None:
        return None
    return ", ".join(str(x) for x in as_list(v) if x not in (None, "")) or None


def extract_identity(bid: int, rec: dict, fetched_at: str) -> dict:
    tax = rec.get("Name and taxonomic classification") or {}
    lit = rec.get("Literature") or {}
    gen = rec.get("General") or {}
    doi = _clean(gen.get("doi"))
    m = DOI_DATE_RE.search(doi or "")
    ts = _clean(tax.get("type strain"))
    lpsn = tax.get("LPSN") if isinstance(tax.get("LPSN"), dict) else {}
    genomes = entries(rec, "Sequence information", "Genome sequences")
    cands, bases = [], []
    for g in genomes:
        acc = _clean(g.get("INSDC accession"))
        mm = GCA_RE.match(acc) if acc else None
        cands.append({"insdc_accession": acc, "assembly_level": _clean(g.get("assembly level")),
                      "score": g.get("score"), "description": _clean(g.get("description"))})
        if mm and mm.group(1) not in bases:
            bases.append(mm.group(1))
    return {
        "bacdive_id": bid,
        "species": _clean(tax.get("species")),
        "genus": _clean(tax.get("genus")),
        "full_scientific_name": _clean(tax.get("full scientific name")),
        "bacdive_phylum": _clean(tax.get("phylum")),
        "lpsn_phylum": _clean(lpsn.get("phylum")),
        "strain_designation_raw": _joined(tax.get("strain designation")),
        "culture_collection_no_raw": _joined(lit.get("culture collection no.")),
        "dsm_number": _clean(gen.get("DSM-Number")),
        "type_strain_bacdive": None if ts is None else ts.lower() == "yes",
        "ncbi_tax_ids_raw": json.dumps(as_list(gen.get("NCBI tax id"))) if gen.get("NCBI tax id") else None,
        "doi": doi,
        "doi_date": m.group(1) if m else None,
        "fetched_at": fetched_at,
        "bacdive_gca": "|".join(bases) if bases else None,
        "bacdive_gca_n": len(bases),
        "genome_candidates_raw": json.dumps(cands, ensure_ascii=False) if cands else None,
    }


def extract_record(bid: int, rec: dict, fetched_at: str, tcfg: dict, collectors: dict) -> dict:
    if rec.get("Genome-based predictions"):
        collectors["predictions"].append(bid)
    row = extract_identity(bid, rec, fetched_at)
    for t in CATEGORICAL_SPECS:
        row.update(extract_categorical(rec, t, tcfg[t], collectors))
    for t in NUMERIC_SPECS:
        row.update(extract_numeric(rec, t, tcfg[t], collectors))
    row.update(extract_halophily(rec))
    return row


def build_trait_table(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Extract every cached BacDive record into one row per strain."""
    tcfg = cfg["traits"]
    collectors = {"unmapped": Counter(), "unparsed": Counter(), "predictions": []}
    rows = [extract_record(bid, rec, ts, tcfg, collectors) for bid, rec, ts in iter_cached_records(cfg)]
    if collectors["predictions"]:
        raise RuntimeError(
            f"{len(collectors['predictions'])} records carry a non-empty 'Genome-based predictions' section "
            f"(first ids: {collectors['predictions'][:5]}). Predictions must not enter the trait table."
        )
    df = pd.DataFrame(rows)
    if df["bacdive_id"].duplicated().any():
        raise RuntimeError("duplicate BacDive IDs across cached batches")
    return df, collectors
