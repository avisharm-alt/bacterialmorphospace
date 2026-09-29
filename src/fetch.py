"""Retrieval of BacDive strain records and GTDB metadata, cached on disk.

BacDive API v2 (https://api.bacdive.dsmz.de/): no authentication; `GET /v2/fetch/{id;id;...}`
takes at most 100 IDs per call and silently drops IDs that do not exist. There is no
list-all endpoint, so we sweep the ID space in fixed batches of 100. Each batch is cached
as `data/raw/bacdive/fetch_<startid>_n<count>.json.gz`; a re-run only requests batches that
are not cached, so a crashed crawl resumes where it stopped.

Predictions (`?predictions=1`) are never requested.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .config import Config

log = logging.getLogger(__name__)

_BATCH_RE = re.compile(r"^fetch_(\d{7})_n(\d+)\.json\.gz$")


class FetchError(RuntimeError):
    pass


def user_agent() -> str:
    load_dotenv()
    ua = os.environ.get("BACDIVE_USER_AGENT")
    if ua:
        return ua
    contact = os.environ.get("BACDIVE_CONTACT_EMAIL", "").strip()
    base = "bacterialmorphospace/0.1 (research crawl)"
    return f"{base} contact:{contact}" if contact else base


# ---------------------------------------------------------------------------
# BacDive
# ---------------------------------------------------------------------------
def bacdive_dir(cfg: Config) -> Path:
    d = cfg.path("raw") / "bacdive"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_batches(cfg: Config) -> dict[int, int]:
    """Map start_id -> number of strains returned, for every cached batch."""
    out = {}
    for f in bacdive_dir(cfg).iterdir():
        m = _BATCH_RE.match(f.name)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def batch_file(cfg: Config, start: int, n: int) -> Path:
    return bacdive_dir(cfg) / f"fetch_{start:07d}_n{n}.json.gz"


def _get_json(session: requests.Session, url: str, cfg: Config) -> dict:
    b = cfg["bacdive"]
    last = None
    for attempt in range(b["max_retries"]):
        try:
            r = session.get(url, timeout=b["timeout_seconds"], headers={"Accept": "application/json"})
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r.json()
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            last = repr(e)
        wait = min(300, b["backoff_seconds"] * 2**attempt)
        log.warning("retry %d/%d for %s... (%s); sleeping %ds", attempt + 1, b["max_retries"], url[:80], last, wait)
        time.sleep(wait)
    raise FetchError(f"giving up on {url[:120]}: {last}")


def fetch_batch(session: requests.Session, cfg: Config, start: int) -> int:
    """Fetch one batch, validate and cache it. Returns the number of strains returned."""
    b = cfg["bacdive"]
    ids = list(range(start, start + b["batch_size"]))
    url = f"{b['base_url']}/fetch/" + ";".join(map(str, ids))
    resp = _get_json(session, url, cfg)
    if not isinstance(resp, dict) or "results" not in resp or "count" not in resp:
        raise FetchError(f"unexpected response shape for batch {start}: {str(resp)[:200]}")
    results = resp["results"] or {}
    if isinstance(results, list):  # empty result comes back as []
        if results:
            raise FetchError(f"unexpected list results for batch {start}")
        results = {}
    if resp["count"] != len(results):
        raise FetchError(f"batch {start}: count {resp['count']} != {len(results)} results")
    if resp.get("next"):
        raise FetchError(f"batch {start}: response is paginated ('next' set); not expected for /fetch")
    stray = [k for k in results if not (start <= int(k) < start + b["batch_size"])]
    if stray:
        raise FetchError(f"batch {start}: returned IDs outside the requested range: {stray[:5]}")
    payload = {
        "requested_start": start,
        "requested_end": start + b["batch_size"] - 1,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": b["base_url"],
        "results": results,
    }
    path = batch_file(cfg, start, len(results))
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    time.sleep(b["delay_seconds"])
    return len(results)


def all_starts(cfg: Config) -> list[int]:
    b = cfg["bacdive"]
    return list(range(b["start_id"], b["max_id"] + 1, b["batch_size"]))


def sweep_summary(cfg: Config) -> dict:
    """Describe what is cached: completeness, highest ID found, and the empty ID ranges."""
    b = cfg["bacdive"]
    cached = cached_batches(cfg)
    starts = all_starts(cfg)
    missing = [s for s in starts if s not in cached]
    nonempty = [s for s in starts if cached.get(s, 0) > 0]
    # runs of consecutive empty batches (>= 10 batches = 1,000 IDs), as inclusive ID ranges
    gaps, run_start = [], None
    for s in starts + [None]:
        empty = s is not None and s in cached and cached[s] == 0
        if empty and run_start is None:
            run_start = s
        if not empty and run_start is not None:
            end = (s if s is not None else starts[-1] + b["batch_size"]) - 1
            if (end - run_start + 1) >= 10 * b["batch_size"]:
                gaps.append((run_start, end))
            run_start = None
    max_found = None
    if nonempty:
        with gzip.open(batch_file(cfg, nonempty[-1], cached[nonempty[-1]]), "rt", encoding="utf8") as fh:
            max_found = max(int(k) for k in json.load(fh)["results"])
    return {
        "sweep_range": [b["start_id"], b["max_id"]],
        "batches_expected": len(starts),
        "batches_cached": len(cached),
        "batches_missing": len(missing),
        "sweep_complete": not missing,
        "strains_cached": sum(cached.values()),
        "max_bacdive_id_found": max_found,
        "cap_looks_too_low": bool(max_found and max_found > b["max_id"] - b["warn_if_max_found_within"]),
        "empty_id_ranges_ge_1000": gaps,
    }


def crawl_bacdive(cfg: Config, max_new_batches: int | None = None) -> dict:
    """Sweep every batch of the BacDive ID space up to `max_id`; resumable.

    No early-stop heuristic: the ID space has multi-ten-thousand-ID gaps, so any "N empty
    batches in a row" rule can truncate the crawl silently. Completeness is instead an
    explicit property (`sweep_summary()['sweep_complete']`) that downstream stages assert.
    """
    b = cfg["bacdive"]
    session = requests.Session()
    session.headers["User-Agent"] = user_agent()
    fetched = 0
    while True:
        cached = cached_batches(cfg)
        todo = [s for s in all_starts(cfg) if s not in cached]
        if max_new_batches is not None:
            todo = todo[: max(0, max_new_batches - fetched)]
        if not todo:
            break
        chunk = todo[: b["workers"] * 4]
        with ThreadPoolExecutor(max_workers=b["workers"]) as pool:
            sizes = list(pool.map(lambda st: fetch_batch(session, cfg, st), chunk))
        fetched += len(chunk)
        log.info("fetched %d batches (%d..%d): %d strains this chunk", len(chunk), chunk[0], chunk[-1], sum(sizes))
    summary = sweep_summary(cfg)
    summary["new_batches_fetched"] = fetched
    if summary["cap_looks_too_low"]:
        log.warning("strains found within %d of max_id=%d: raise [bacdive].max_id and re-run", b["warn_if_max_found_within"], b["max_id"])
    return summary


def iter_cached_records(cfg: Config):
    """Yield (bacdive_id:int, record:dict, fetched_at:str) for every cached strain, in ID order."""
    for start, n in sorted(cached_batches(cfg).items()):
        if n == 0:
            continue
        with gzip.open(batch_file(cfg, start, n), "rt", encoding="utf8") as fh:
            payload = json.load(fh)
        for k in sorted(payload["results"], key=int):
            yield int(k), payload["results"][k], payload["fetched_at"]


# ---------------------------------------------------------------------------
# GTDB
# ---------------------------------------------------------------------------
@dataclass
class GtdbInfo:
    release: str
    released: str
    metadata_path: Path
    metadata_sha256: str
    metadata_last_modified: str | None
    source_url: str


def gtdb_dir(cfg: Config) -> Path:
    d = cfg.path("raw") / "gtdb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> str | None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": user_agent()}) as r:
        r.raise_for_status()
        last_mod = r.headers.get("Last-Modified")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(tmp, dest)
    return last_mod


def fetch_gtdb(cfg: Config, refresh: bool = False) -> GtdbInfo:
    """Download GTDB `latest` bac120 metadata + VERSION.txt once; cached thereafter.

    The release identifier is read from VERSION.txt (FILE_DESCRIPTIONS.txt in the same
    directory can lag behind and must not be used for versioning).
    """
    g = cfg["gtdb"]
    d = gtdb_dir(cfg)
    vfile, mfile = d / g["version_file"], d / g["metadata_file"]
    meta_info = d / "download_info.json"
    if refresh or not vfile.exists():
        _download(f"{g['base_url']}/{g['version_file']}", vfile)
    if refresh or not mfile.exists():
        log.info("downloading %s", g["metadata_file"])
        last_mod = _download(f"{g['base_url']}/{g['metadata_file']}", mfile)
        meta_info.write_text(json.dumps({"last_modified": last_mod, "sha256": _sha256(mfile),
                                         "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    info = json.loads(meta_info.read_text()) if meta_info.exists() else {}
    if "sha256" not in info:
        info["sha256"] = _sha256(mfile)
        meta_info.write_text(json.dumps(info))
    lines = [ln.strip() for ln in vfile.read_text().splitlines() if ln.strip()]
    release = lines[0]
    if not re.fullmatch(r"v?\d+(\.\d+)?", release):
        raise FetchError(f"cannot parse GTDB release from VERSION.txt: {lines!r}")
    released = lines[1] if len(lines) > 1 else ""
    return GtdbInfo(release=release, released=released, metadata_path=mfile, metadata_sha256=info["sha256"],
                    metadata_last_modified=info.get("last_modified"),
                    source_url=f"{g['base_url']}/{g['metadata_file']}")
