"""Strain-designation normalisation.

Sources write the same strain differently: "DSM 20231", "DSM20231", "DSMZ 20231",
"DSM-20231T", "ATCC 11775 / DSM 30083", "DSM 46234, ATCC 21370, JCM 2849" (BacDive's own
comma-separated culture-collection field). We reduce each designation to a canonical key so
that equal keys mean the same designation, and match on exact key equality only.

Pipeline for one raw string
---------------------------
1. `split_designations`   split on , ; / | =  (configurable) into individual designations.
2. `normalise_designation` per designation:
     a. Unicode NFKD, drop accents, drop (R)/(TM) marks, unify dashes, upper-case.
     b. Drop wrapper words ("type strain", "strain", "str.", "isolate").
     c. Drop the type-strain marker: trailing T, ^T, (T), "(type strain)".
        The bare "T" is only stripped after a recognised collection code + number, because
        "K12T" or "ABT" could be real designations.
     d. Fold collection aliases on the leading code (DSMZ -> DSM, NCIB -> NCIMB).
     e. Keep only [A-Z0-9]: separators (space, hyphen, dot, underscore) are dropped, so
        "DSM 20231", "DSM-20231", "DSM20231" share a key.
     f. Strip leading zeros of a purely numeric remainder after a recognised collection code.
3. `parse_designations` returns `Designation` records with a `kind`:
     "collection"  a recognised collection code followed by an identifier   (DSM20231)
     "bare"        anything else that still looks like an identifier        (BS107)
     "unusable"    too ambiguous to match on (purely numeric, purely alphabetic, a placeholder
                   such as "none", or shorter than 3 characters)
   Only "collection" and "bare" designations are ever used for matching; "bare" ones are
   accepted downstream only with genus agreement, because "BS 107" is not globally unique.

Nothing here is fuzzy: two designations match iff their canonical keys are identical.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

DEFAULT_DELIMITERS = (",", ";", "/", "|", "=")
DEFAULT_ALIASES = {"DSMZ": "DSM", "NCIB": "NCIMB"}
DEFAULT_COLLECTIONS = frozenset(
    {
        "DSM", "ATCC", "NCTC", "NCIMB", "CIP", "JCM", "LMG", "KCTC", "CCUG", "CGMCC", "BCRC", "NBRC",
        "IFO", "VKM", "CECT", "CCM", "CFBP", "IAM", "MTCC", "NRRL", "NCCB", "CBS", "UCC", "CCTCC",
        "KACC", "NZP", "PCM", "TISTR", "VTT", "ICMP", "LMD", "BCCM", "MCCC", "HAMBI", "IMET", "CCAP", "IBRC",
    }
)
PLACEHOLDERS = frozenset({"NONE", "NA", "NAN", "NULL", "UNKNOWN", "UNK", "NOTAVAILABLE", "MISSING", "TYPESTRAIN", "STRAIN"})

_WRAPPER = re.compile(r"^(?:type\s+strain|strain|str\.?|isolate|type)\s+", re.I)
_TYPE_PAREN = re.compile(r"\s*[\(\[]\s*(?:type\s*strain|type|t)\s*[\)\]]\s*$", re.I)
_CARET_T = re.compile(r"\^\s*T$")
_TRAILING_T = re.compile(r"^([\s\-_.:]*\d[\w.\-]*?)\s*T$")
_LEADING_CODE = re.compile(r"^([A-Z]{2,8})(?=[\s\-_.:]*[A-Z0-9])")
_ONLY_ALNUM = re.compile(r"[^A-Z0-9]")


@dataclass(frozen=True)
class Designation:
    raw: str
    canonical: str
    kind: str  # "collection" | "bare" | "unusable"
    collection: str | None = None

    @property
    def matchable(self) -> bool:
        return self.kind != "unusable"


def _ascii_upper(s: str) -> str:
    # Strip (R)/(TM) marks *before* NFKC, which would turn the trademark sign into the letters "TM".
    s = s.replace("\u00ae", "").replace("\u2122", "").replace("\u00a0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub("[\u2010-\u2015\u2212]", "-", s)  # every dash variant -> "-"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper().strip()


def split_designations(s: str | None, delimiters: Iterable[str] = DEFAULT_DELIMITERS) -> list[str]:
    """Split a multi-designation string; empty pieces are dropped, order is preserved."""
    if s is None:
        return []
    s = str(s)
    if not s.strip():
        return []
    pattern = "|".join(re.escape(d) for d in delimiters)
    return [p.strip() for p in re.split(pattern, s) if p and p.strip()]


def _parse_one(raw: str, aliases: dict[str, str], collections: frozenset[str]) -> Designation:
    s = _ascii_upper(raw)
    s = _WRAPPER.sub("", s).strip()
    s = _TYPE_PAREN.sub("", s).strip()
    s = _CARET_T.sub("", s).strip()

    collection = None
    m = _LEADING_CODE.match(s)
    if m:
        code = aliases.get(m.group(1), m.group(1))
        if code in collections:
            collection = code
            rest = s[m.end(1):]
            # bare trailing "T" (type-strain marker): only stripped after "<code> <number>"
            m_t = _TRAILING_T.match(rest)
            if m_t:
                rest = m_t.group(1)
            rest = _ONLY_ALNUM.sub("", rest)
            if rest.isdigit():
                rest = rest.lstrip("0") or "0"
            key = code + rest
            if not rest or not any(c.isdigit() for c in rest):
                return Designation(raw, key, "unusable", collection)
            return Designation(raw, key, "collection", collection)

    key = _ONLY_ALNUM.sub("", s)
    if (
        len(key) < 3
        or key in PLACEHOLDERS
        or not any(c.isdigit() for c in key)
        or not any(c.isalpha() for c in key)
    ):
        return Designation(raw, key, "unusable")
    return Designation(raw, key, "bare")


def parse_designation(raw: str, aliases: dict[str, str] | None = None, collections: Iterable[str] | None = None) -> Designation:
    return _parse_one(
        raw,
        DEFAULT_ALIASES if aliases is None else aliases,
        DEFAULT_COLLECTIONS if collections is None else frozenset(collections),
    )


def normalise_designation(raw: str | None, aliases: dict[str, str] | None = None, collections: Iterable[str] | None = None) -> str:
    """Canonical key of a *single* designation ('' for empty input).

    >>> normalise_designation("DSMZ 20231") == normalise_designation("DSM-20231T") == "DSM20231"
    True
    """
    if raw is None or not str(raw).strip():
        return ""
    return parse_designation(str(raw), aliases, collections).canonical


def parse_designations(
    s: str | None,
    delimiters: Iterable[str] = DEFAULT_DELIMITERS,
    aliases: dict[str, str] | None = None,
    collections: Iterable[str] | None = None,
) -> list[Designation]:
    """Split a multi-designation string and parse every piece (duplicates by key removed)."""
    seen: set[str] = set()
    out: list[Designation] = []
    for piece in split_designations(s, delimiters):
        d = parse_designation(piece, aliases, collections)
        if d.canonical and d.canonical not in seen:
            seen.add(d.canonical)
            out.append(d)
    return out


def canonical_keys(s: str | None, **kw) -> list[str]:
    """Canonical keys of all *matchable* designations in a multi-designation string."""
    return [d.canonical for d in parse_designations(s, **kw) if d.matchable]


class Normaliser:
    """Config-bound, cached front end used by the join stage."""

    def __init__(self, delimiters=DEFAULT_DELIMITERS, aliases=None, collections=None):
        self.delimiters = tuple(delimiters)
        self.aliases = dict(DEFAULT_ALIASES if aliases is None else aliases)
        self.collections = frozenset(DEFAULT_COLLECTIONS if collections is None else collections)
        self._parse = lru_cache(maxsize=None)(self._parse_uncached)

    @classmethod
    def from_config(cls, cfg) -> "Normaliser":
        m = cfg["matching"]
        return cls(m["delimiters"], m["aliases"], m["collections"])

    def _parse_uncached(self, s: str) -> tuple[Designation, ...]:
        return tuple(parse_designations(s, self.delimiters, self.aliases, self.collections))

    def parse(self, s) -> tuple[Designation, ...]:
        return self._parse(s) if isinstance(s, str) and s else ()  # NaN / None -> no designations

    def raw_tokens(self, s) -> list[str]:
        return split_designations(s, self.delimiters) if isinstance(s, str) else []
