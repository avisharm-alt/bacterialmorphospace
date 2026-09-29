"""Command-line entry point.

    python -m src.pipeline fetch     crawl BacDive (resumable, cached) + download GTDB metadata
    python -m src.pipeline extract   cached BacDive JSON -> data/interim/bacdive_traits.parquet
    python -m src.pipeline join      + GTDB -> data/final/strains.parquet, unmatched log
    python -m src.pipeline report    -> reports/attrition_report.md (+ tables, manifest, accession lists)
    python -m src.pipeline all       all of the above; only uncached BacDive batches are requested
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import fetch as fetch_mod
from . import stages
from .config import load_config
from .report import build_report


def cmd_fetch(cfg, args) -> None:
    summary = fetch_mod.crawl_bacdive(cfg, max_new_batches=args.max_new_batches)
    logging.info("bacdive: %s", summary)
    if not args.skip_gtdb:
        info = fetch_mod.fetch_gtdb(cfg, refresh=args.refresh_gtdb)
        logging.info("gtdb release %s (%s)", info.release, info.released)


def cmd_extract(cfg, args) -> None:
    stages.extract(cfg, allow_incomplete=args.allow_incomplete)


def cmd_join(cfg, args) -> None:
    stages.join(cfg)


def cmd_report(cfg, args) -> None:
    build_report(cfg)
    logging.info("wrote %s", cfg.path("reports") / "attrition_report.md")


def cmd_all(cfg, args) -> None:
    cmd_fetch(cfg, args)
    cmd_extract(cfg, args)
    cmd_join(cfg, args)
    cmd_report(cfg, args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="src.pipeline")
    ap.add_argument("--config", default=None, help="path to config.toml (default: repo root)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "all"):
        p = sub.add_parser(name)
        p.add_argument("--max-new-batches", type=int, default=None, help="stop after N uncached batches (testing)")
        p.add_argument("--skip-gtdb", action="store_true")
        p.add_argument("--refresh-gtdb", action="store_true", help="re-download GTDB even if cached")
        p.add_argument("--allow-incomplete", action="store_true", help="extract even if the BacDive sweep is incomplete")
    p = sub.add_parser("extract")
    p.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("join")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "extract": cmd_extract, "join": cmd_join, "report": cmd_report, "all": cmd_all}[args.cmd](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
