"""Pipeline stages that read and write data/interim and data/final."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from . import fetch as fetch_mod
from .analysis import add_panel_flags
from .config import Config
from .join import GtdbIndex, load_gtdb, match_strains
from .normalise import Normaliser
from .traits import build_trait_table

log = logging.getLogger(__name__)


def extract(cfg: Config, allow_incomplete: bool = False) -> pd.DataFrame:
    sweep = fetch_mod.sweep_summary(cfg)
    if not sweep["sweep_complete"] and not allow_incomplete:
        raise RuntimeError(f"BacDive sweep incomplete ({sweep['batches_missing']} batches missing); run `fetch` first")
    df, coll = build_trait_table(cfg)
    inter = cfg.path("interim")
    df.to_parquet(inter / "bacdive_traits.parquet", index=False)
    rows = [{"kind": "unmapped", "trait": t, "value": v, "count": n} for (t, v), n in coll["unmapped"].most_common()]
    rows += [{"kind": "unparsed", "trait": t, "value": v, "count": n} for (t, v), n in coll["unparsed"].most_common()]
    pd.DataFrame(rows, columns=["kind", "trait", "value", "count"]).to_csv(inter / "unmapped_values.tsv", sep="\t", index=False)
    (inter / "bacdive_sweep.json").write_text(json.dumps(sweep, indent=1))
    log.info("extracted %d strains", len(df))
    return df


def join(cfg: Config) -> pd.DataFrame:
    inter, final = cfg.path("interim"), cfg.path("final")
    traits = pd.read_parquet(inter / "bacdive_traits.parquet")
    info = fetch_mod.fetch_gtdb(cfg)
    gtdb = load_gtdb(cfg, info)
    log.info("GTDB %s: %d genomes; building index", info.release, len(gtdb))
    idx = GtdbIndex(gtdb, Normaliser.from_config(cfg))
    matches, unmatched = match_strains(traits, idx, cfg)
    df = traits.merge(matches, on="bacdive_id", how="left", validate="one_to_one")
    df = add_panel_flags(df, cfg)
    df["gtdb_release"] = info.release
    df.to_parquet(final / "strains.parquet", index=False)
    unmatched.to_csv(inter / "unmatched_strains.tsv", sep="\t", index=False)
    gtdb_info = {"release": info.release, "released": info.released, "source_url": info.source_url,
                 "metadata_sha256": info.metadata_sha256, "metadata_last_modified": info.metadata_last_modified,
                 "n_genomes": int(len(gtdb)), "n_genomes_with_strain_ids": int(gtdb["ncbi_strain_identifiers"].notna().sum()),
                 "n_designation_keys": len(idx.by_key), "joined_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (inter / "gtdb_info.json").write_text(json.dumps(gtdb_info, indent=1))
    log.info("joined: %d of %d strains matched to a GTDB genome", int(df["genome_matched"].sum()), len(df))
    return df
