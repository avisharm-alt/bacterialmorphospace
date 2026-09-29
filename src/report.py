"""Attrition + occupancy report.

Writes
  reports/attrition_report.md          the deliverable
  reports/tables/*.tsv                 every table behind the report, machine-readable
  reports/run_manifest.json            versions, hashes, counts - what the report was produced from
  data/final/assembly_accessions_*.txt accession lists for `datasets download genome accession --inputfile`
  data/final/<panel>_species.tsv       the analysed sets, compact
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .config import Config
from .join import match_stage_table

log = logging.getLogger(__name__)

MORPH_LABEL = "morphology panel (4)"
NULLS: list[str | None] = [None, "phylum", "class", "order"]
NULL_NAME = {None: "global", "phylum": "phylum", "class": "class", "order": "order"}
VIEWS = [
    # name, binning, dereplication level, temperature source, description
    ("primary", "coarse", "species", "all", "one strain per GTDB species · coarse bins · all temperature sources"),
    ("strain-level", "coarse", "strain", "all", "no dereplication"),
    ("optimum-only", "coarse", "species", "optimum", "temperature bin from a reported optimum only"),
    ("fine bins", "fine", "species", "all", "fine binning (shape 5, oxygen 4, temperature 4)"),
    ("genus-level", "coarse", "genus", "all", "one strain per GTDB genus"),
]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def pct(k, n, digits=1) -> str:
    return "–" if not n else f"{100 * k / n:.{digits}f}%"


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) >= 1:
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if v == 0:
            return "0"
        return f"{v:.{digits}g}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def _git_commit(cfg: Config) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config.toml"], cwd=cfg.root,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_fingerprint(path: Path) -> dict:
    """sha256 of the file bytes and, for tables, of the row contents (independent of writer versions)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    out = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
    t = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t", low_memory=False)
    out["rows"], out["columns"] = int(len(t)), int(t.shape[1])
    out["content_sha256"] = hashlib.sha256(pd.util.hash_pandas_object(t.astype(str), index=False).values.tobytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# attrition / matching tables
# ---------------------------------------------------------------------------
def attrition_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n = len(df)
    g = df["genome_matched"]

    def row(stage, mask, note=""):
        k = int(mask.sum())
        kg = int((mask & g).sum())
        sp = df.loc[mask & g, "gtdb_species"].nunique()
        return {"stage": stage, "strains": k, "% of BacDive": pct(k, n), "with GTDB genome": kg,
                "% with genome": pct(kg, k), "GTDB species": sp, "note": note}

    rows = [row("1. BacDive strains retrieved", pd.Series(True, index=df.index)),
            row("2. ≥1 of the 8 traits populated (raw)", df["any_trait_populated"])]
    for t in A.ALL_TRAITS:
        note = {"temperature": "bin from optimum, else growth range, else single growth point",
                "halophily_level": "curated category only; NaCl tests kept as a column",
                "ph": "outside the analysed panels"}.get(t, "")
        rows.append(row(f"3. {A.TRAIT_LABEL[t]} — usable value", df[f"{t}_usable"], note))
    rows.append(row("3. (NaCl growth tests present, raw column only)", df["nacl_tests_populated"],
                    "not binned; kept in nacl_tests_raw"))
    rows += [
        row(f"4. {MORPH_LABEL} complete", df["morph4_complete"], " + ".join(cfg.morphology)),
        row("5. core panel (6) complete", df["core6_complete"], " + ".join(cfg.core)),
        row("5a. core panel (6), temperature from optimum only", df["core6_complete_optimum_temp"]),
        row("5b. no-spore panel (5) complete", df["nospore5_complete"], " + ".join(cfg.no_spore)),
        row("5c. original 8-trait panel (core + pH + curated halophily)", df["all8_reference_complete"], "for reference; not analysed"),
    ]
    return pd.DataFrame(rows)


def per_trait_detail(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in A.ALL_TRAITS:
        pop, use = int(df[f"{t}_populated"].sum()), int(df[f"{t}_usable"].sum())
        conf = int(df[f"{t}_conflict"].sum()) if f"{t}_conflict" in df else 0
        rows.append({"trait": A.TRAIT_LABEL[t], "any raw observation": pop, "usable (normalised)": use,
                     "lost to conflict": conf,
                     "no binnable value (unmapped, unparsed, or only negative/qualitative entries)": pop - use - conf,
                     "usable with genome": int((df[f"{t}_usable"] & df["genome_matched"]).sum())})
    return pd.DataFrame(rows)


def genome_attrition(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    d = df[mask]
    n = len(d)
    desig = d["match_method"] == "designation"
    rows = [("strains in set", n),
            ("BacDive lists ≥1 GCA assembly", int(d["bacdive_lists_assembly"].sum())),
            ("… of which ≥1 listed assembly is in GTDB", int(d["bacdive_assembly_in_gtdb"].sum())),
            ("… listed assembly not in GTDB (failed GTDB QC, or absent)", int((d["bacdive_lists_assembly"] & ~d["bacdive_assembly_in_gtdb"]).sum())),
            ("matched via direct accession", int((d["match_method"] == "accession").sum())),
            ("matched via designation — collection code (e.g. DSM 20231)", int((desig & (d["match_token_kind"] == "collection")).sum())),
            ("matched via designation — bare designation + taxon agreement", int((desig & (d["match_token_kind"] == "bare")).sum())),
            ("matched, total", int(d["genome_matched"].sum())),
            ("unmatched", int((~d["genome_matched"]).sum()))]
    return pd.DataFrame([{"": k, "strains": v, "%": pct(v, n)} for k, v in rows])


def type_strain_table(sub: pd.DataFrame) -> pd.DataFrame:
    b = sub["type_strain_bacdive"].fillna(False).astype(bool)
    g = sub["gtdb_is_type_strain_of_species"].astype(bool)
    n = len(sub)
    rows = [("type strain per BacDive", int(b.sum())), ("type strain of species per GTDB/NCBI", int(g.sum())),
            ("either", int((b | g).sum())), ("both", int((b & g).sum())),
            ("BacDive says type, GTDB genome not flagged as type", int((b & ~g).sum()))]
    return pd.DataFrame([{"": k, "N": v, "%": pct(v, n)} for k, v in rows])


def capacity_summary(cap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in cap.groupby("size"):
        rows.append({"traits in map": k, "maps": len(g), "cells": f"{g.cells.min()}–{g.cells.max()}",
                     "N (species) range": f"{g.N.min():,}–{g.N.max():,}",
                     "N per cell, median over maps": g.N_per_cell.median(),
                     "median expected count in a cell": g.expected_median.median(),
                     "smallest expected count": g.expected_min.min(),
                     "testable cells (global null), median over maps": f"{100 * g.testable_cell_frac.median():.0f}%"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# occupancy tables
# ---------------------------------------------------------------------------
def merged_cells(res: dict, traits: list[str]) -> pd.DataFrame:
    """One row per cell, with observed/expected/q/flags under every null (primary view)."""
    base = res[None].cell_table[traits + ["observed"]].rename(columns={"observed": "observed (all species)"}).copy()
    for null in NULLS:
        c = res[null].cell_table
        nm = NULL_NAME[null]
        base[f"{nm}: expected"] = c["expected"].values
        base[f"{nm}: q"] = c["q_low"].values
        base[f"{nm}: testable"] = c["testable"].values
        base[f"{nm}: empty✗"] = c["empty_beyond_chance"].values
        if null is not None:
            base[f"{nm}: observed (kept species)"] = c["observed"].values
            base[f"{nm}: emptied by exclusion"] = c["emptied_by_exclusion"].values
    base["global: q (seed 2)"] = res["global_seed2"].cell_table["q_low"].values
    base["global: empty✗ (seed 2)"] = res["global_seed2"].cell_table["empty_beyond_chance"].values
    base["order: q (seed 2)"] = res["order_seed2"].cell_table["q_low"].values
    base["order: empty✗ (seed 2)"] = res["order_seed2"].cell_table["empty_beyond_chance"].values

    def category(r):
        coarser = [NULL_NAME[x] for x in NULLS[:-1] if r[f"{NULL_NAME[x]}: empty✗"]]
        if r["order: empty✗"]:
            stable = " (both seeds)" if r["order: empty✗ (seed 2)"] else " (seed 1 only: borderline)"
            return "SURVIVES order-level shuffle" + stable
        if coarser:
            how = "untestable at order level" if not r["order: testable"] else "tested at order level, not significant"
            if r["order: emptied by exclusion"]:
                how = "occupants only in excluded small orders"
            return f"phylogenetic structure — fires under {'/'.join(coarser)} only; {how}"
        if r["observed (all species)"] == 0 and any(r[f"{NULL_NAME[x]}: testable"] for x in NULLS):
            return "empty, testable, not significant under any null"
        if r["observed (all species)"] == 0:
            return "empty, untestable under every null"
        return ""

    base["category"] = base.apply(category, axis=1)
    return base


def crosstab_grid(cells: pd.DataFrame, traits: list[str], cfg: Config) -> pd.DataFrame:
    """Rows = all traits but oxygen/temperature, columns = oxygen x temperature. Empty cells are marked."""
    ct = ["oxygen", "temperature"]
    rt = [t for t in traits if t not in ct]

    def mark(r):
        if r["observed (all species)"] > 0:
            return f"{int(r['observed (all species)'])}"
        if r["order: empty✗"]:
            return "**0✗**"
        if r["category"].startswith("phylogenetic"):
            return "0†"
        if any(r[f"{NULL_NAME[x]}: testable"] for x in NULLS):
            return "0"
        return "·"

    c = cells.copy()
    c["mark"] = [mark(r) for _, r in c.iterrows()]
    c["row"] = c[rt].astype(str).agg(" / ".join, axis=1)
    c["col"] = c[ct].astype(str).agg(" / ".join, axis=1)
    order_r = [" / ".join(x) for x in pd.MultiIndex.from_product([A.levels_for(cfg, t, "coarse") for t in rt])]
    order_c = [" / ".join(x) for x in pd.MultiIndex.from_product([A.levels_for(cfg, t, "coarse") for t in ct])]
    grid = c.pivot(index="row", columns="col", values="mark").reindex(index=order_r, columns=order_c)
    grid.index.name = " / ".join(rt) + " ↓   " + " / ".join(ct) + " →"
    return grid


def occ_row(label: str, s: dict) -> dict:
    return {"map": label, "null": s["null"], "N used": s["N"], "strata": s["strata_used"],
            "species excluded (small strata)": s["species_excluded"], "cells": s["cells"],
            "occupied": s["occupied"], "expected occupied (null mean)": round(s["null_occupied_mean"], 1),
            "null 95% interval": f"{s['null_occupied_lo']:.0f}–{s['null_occupied_hi']:.0f}",
            "deficit": round(s["null_occupied_mean"] - s["occupied"], 1),
            "P(null ≤ observed)": s["p_fewer_occupied_than_null"]}


def strata_table(sub: pd.DataFrame, min_sizes=(2, 5, 10), default=5) -> pd.DataFrame:
    rows = []
    for lvl in ("phylum", "class", "order"):
        for ms in min_sizes:
            _, _, i = A.assign_strata(sub, lvl, ms)
            rows.append({"rank": lvl, "min species per stratum": f"{ms}{' (used)' if ms == default else ''}",
                         "strata kept": i["strata_used"], "strata excluded": i["strata_excluded"],
                         "species kept": i["N_used"], "species excluded": i["species_excluded"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_report(cfg: Config) -> str:
    final, inter, rep = cfg.path("final"), cfg.path("interim"), cfg.path("reports")
    tables = rep / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(final / "strains.parquet")
    sweep = json.loads((inter / "bacdive_sweep.json").read_text())
    gtdb = json.loads((inter / "gtdb_info.json").read_text())
    unmapped = pd.read_csv(inter / "unmapped_values.tsv", sep="\t")
    unmatched = pd.read_csv(inter / "unmatched_strains.tsv", sep="\t")
    a = cfg["analysis"]
    n = len(df)
    core, morph, nosp = cfg.core, cfg.morphology, cfg.no_spore
    panels = [("core", core, "core panel (6 traits)"), ("no_spore", nosp, "no-spore panel (5 traits)")]

    def save(name, t, index=False):
        t.to_csv(tables / f"{name}.tsv", sep="\t", index=index)
        return t

    # ---------------------------------------------------------------- attrition & matching
    att = save("attrition", attrition_table(df, cfg))
    ptd = save("per_trait_detail", per_trait_detail(df))
    gen_all = genome_attrition(df, pd.Series(True, index=df.index))
    gen_core = genome_attrition(df, df["core6_complete"])
    masks = {"all strains": pd.Series(True, index=df.index), "core panel": df["core6_complete"],
             "no-spore panel": df["nospore5_complete"]}
    stage = save("match_rate_by_normalisation_step", match_stage_table(df, masks))
    reasons = save("unmatched_reasons", unmatched["reason"].value_counts().rename_axis("reason").reset_index(name="strains"))
    agree = df["designation_agrees_with_accession"].dropna().astype(bool)
    bare_only = df["designation_bare_hit"] & ~df["designation_collection_hit"]
    bare_agree = df.loc[bare_only, "designation_agrees_with_accession"].dropna().astype(bool)
    matched = df[df["genome_matched"]]
    n_desig = int((matched.match_method == "designation").sum())
    n_taxid_only = int(((matched.match_method == "designation") & (matched.taxon_agree_via == "ncbi_taxid")).sum())

    cov = A.coverage_matrices(df)
    for k, v in cov.items():
        save(f"coverage_{k}", v, index=True)
    cov_core = A.coverage_matrices(df, core)
    ind = save("coverage_vs_independence", A.independence_check(df, {
        "morphology (4)": morph, "no-spore (5)": nosp, "core (6)": core,
        "core + pH (7)": core + ["ph"], "original 8": core + ["ph", "halophily_level"]}))
    ga = A.genome_availability(df, cfg)
    for k, v in ga.items():
        save(f"genome_availability_{k}", v)

    # ---------------------------------------------------------------- occupancy, both panels
    results: dict[str, dict] = {}
    for pi, (pkey, traits, _) in enumerate(panels):
        results[pkey] = {}
        for vi, (vname, binning, level, tsrc, _) in enumerate(VIEWS):
            sub = A.analysis_set(df, traits, binning, level, tsrc)
            nulls = NULLS if vname == "primary" else [None, "order"]
            results[pkey][vname] = {}
            for ni, null in enumerate(nulls):
                log.info("occupancy %s / %s / %s null", pkey, vname, NULL_NAME[null])
                rng = np.random.default_rng(a["seed"] + 10_000 * pi + 100 * vi + ni)
                results[pkey][vname][null] = A.occupancy(sub, traits, cfg, binning, a["n_perm_full"], rng,
                                                         keep_cells=True, stratify=null)
                save(f"cells_{pkey}_{vname.replace(' ', '_')}_{NULL_NAME[null]}", results[pkey][vname][null].cell_table)
            if vname == "primary":  # Monte Carlo stability: independent seed for the global and order nulls
                for null in (None, "order"):
                    rng = np.random.default_rng(a["seed"] + 777_777 + 10 * pi + (null is None))
                    results[pkey][vname][f"{NULL_NAME[null]}_seed2"] = A.occupancy(
                        sub, traits, cfg, binning, a["n_perm_full"], rng, keep_cells=True, stratify=null)

    prim = {k: A.analysis_set(df, t) for k, t, _ in panels}
    merged = {k: save(f"cells_{k}_all_nulls", merged_cells(results[k]["primary"], t)) for k, t, _ in panels}

    # subsets of the core traits, all four nulls; the two full panels are reused, not recomputed
    pre = {(A.subset_name(t), NULL_NAME[x]): results[k]["primary"][x] for k, t, _ in panels for x in NULLS}
    subset_cells: dict = {}
    grid = A.occupancy_grid(df, cfg, "coarse", "species", "all", precomputed=pre, nulls=tuple(NULLS), cells_out=subset_cells)
    save("occupancy_subsets_all_nulls", grid)
    subset_survivors = []
    for (sname, null), ct in subset_cells.items():
        if null == "order" and ct is not None:
            for _, r in ct[ct["empty_beyond_chance"]].iterrows():
                tr = sname.split(" + ")
                subset_survivors.append({"map": sname, "cell": " / ".join(f"{t}={r[t]}" for t in tr),
                                         "order: expected": r["expected"], "order: q": r["q_low"],
                                         "global: q": subset_cells[(sname, "global")].set_index(tr).loc[tuple(r[t] for t in tr), "q_low"]})
    subset_survivors = save("order_level_survivors_all_maps", pd.DataFrame(
        subset_survivors, columns=["map", "cell", "order: expected", "order: q", "global: q"]))

    # sensitivity views
    views_rows = []
    for k, t, plabel in panels:
        for vname, _, _, _, desc in VIEWS:
            for null, occ in results[k][vname].items():
                if isinstance(null, str) and null.endswith("seed2"):
                    continue
                s = occ.summary
                views_rows.append({"panel": plabel, "view": vname, "null": NULL_NAME[null], "N used": s["N"],
                                   "occupied": s["occupied"], "expected occupied": round(s["null_occupied_mean"], 1),
                                   "testable cells": s["testable_cells"], "empty✗": s["empty_beyond_chance_cells"],
                                   "emptied by exclusion": s.get("cells_emptied_by_exclusion", 0)})
    views = save("views_summary", pd.DataFrame(views_rows))

    # ---------------------------------------------------------------- panels side by side, power
    side = []
    for k, t, plabel in panels:
        r = results[k]["primary"]
        row = {"panel": plabel, "traits": " + ".join(t), "cells": r[None].summary["cells"],
               "strains with genome": len(A.analysis_set(df, t, level="strain")), "N species": len(prim[k]),
               "N genera": len(A.analysis_set(df, t, level="genus")), "species per cell": round(len(prim[k]) / r[None].summary["cells"], 1)}
        for null in NULLS:
            row[f"testable cells ({NULL_NAME[null]})"] = r[null].summary["testable_cells"]
        row["species kept by order null"] = r["order"].summary["N"]
        row["order-level survivors"] = int(merged[k]["order: empty✗"].sum())
        side.append(row)
    side = save("panels_side_by_side", pd.DataFrame(side))
    caps = {k: save(f"power_expected_per_cell_{k}", A.capacity_table(df, cfg, panel=t)) for k, t, _ in panels}

    # occupied vs expected
    mname = A.subset_name(morph)
    ove = [occ_row(MORPH_LABEL, grid[(grid.traits == mname) & (grid.null == NULL_NAME[x])].iloc[0].to_dict()) for x in NULLS]
    ove += [occ_row(plabel, results[k]["primary"][x].summary) for k, _, plabel in panels for x in NULLS]
    ove = save("occupied_vs_expected", pd.DataFrame(ove))
    sub_occ = grid.pivot_table(index="traits", columns="null", values=["occupied", "null_occupied_mean", "p_fewer_occupied_than_null"],
                               aggfunc="first")

    # per-phylum occupancy
    phy_occ = {k: save(f"phylum_occupancy_{k}", A.phylum_occupancy(prim[k], t, cfg, n_perm=a["n_perm_phylum_occupancy"],
                                                                      seed=a["seed"] + 3)) for k, t, _ in panels}

    # positive controls
    pcs = []
    for pc in a.get("positive_controls", []):
        k = pc["panel"]
        t = core if k == "core" else nosp
        m = merged[k]
        sel = np.logical_and.reduce([m[x] == pc["cell"][x] for x in t])
        r = m[sel].iloc[0]
        rows = [{"null": NULL_NAME[x], "species used": results[k]["primary"][x].summary["N"],
                 "observed": int(r["observed (all species)"]) if x is None else int(r[f"{NULL_NAME[x]}: observed (kept species)"]),
                 "expected": r[f"{NULL_NAME[x]}: expected"], "testable": r[f"{NULL_NAME[x]}: testable"],
                 "q": r[f"{NULL_NAME[x]}: q"], "empty beyond chance": r[f"{NULL_NAME[x]}: empty✗"]} for x in NULLS]
        passed = bool(r["global: empty✗"]) and not bool(r["order: empty✗"])
        pcs.append({"spec": pc, "table": pd.DataFrame(rows), "passed": passed, "row": r})

    # survivors detail
    surv = {}
    for k, t, plabel in panels:
        m = merged[k]
        for _, r in m[m["order: empty✗"]].iterrows():
            cell = {x: r[x] for x in t}
            vrows = []
            for vname, *_ in VIEWS:
                for null in (None, "order"):
                    ct = results[k][vname][null].cell_table
                    if vname == "fine bins":
                        vrows.append({"view": vname, "null": NULL_NAME[null], "note": "cell not defined at fine resolution (bins split)"})
                        continue
                    rr = ct[np.logical_and.reduce([ct[x] == cell[x] for x in t])].iloc[0]
                    vrows.append({"view": vname, "null": NULL_NAME[null], "N used": results[k][vname][null].summary["N"],
                                  "observed": int(rr["observed_all_species"]), "expected": rr["expected"],
                                  "testable": rr["testable"], "q": rr["q_low"], "empty beyond chance": rr["empty_beyond_chance"]})
            for lvl in a.get("survivor_diagnostic_levels", []):
                key = f"diag_{lvl}"
                if key not in results[k]["primary"]:
                    log.info("survivor diagnostic %s / %s null", k, lvl)
                    results[k]["primary"][key] = A.occupancy(prim[k], t, cfg, "coarse", a["n_perm_full"],
                                                             np.random.default_rng(a["seed"] + 424_242), keep_cells=True, stratify=lvl)
                    save(f"cells_{k}_primary_{lvl}", results[k]["primary"][key].cell_table)
                occ = results[k]["primary"][key]
                ct = occ.cell_table
                rr = ct[np.logical_and.reduce([ct[x] == cell[x] for x in t])].iloc[0]
                vrows.append({"view": f"primary (diagnostic: {lvl} shuffle)", "null": lvl, "N used": occ.summary["N"],
                              "observed": int(rr["observed_all_species"]), "expected": rr["expected"], "testable": rr["testable"],
                              "q": rr["q_low"], "empty beyond chance": rr["empty_beyond_chance"]})
            contrib = A.cell_expected_by_stratum(prim[k], t, cfg, cell, "order")
            # which families inside the top contributing orders carry each constituent trait level
            fam_rows = []
            for o, e_o in zip(contrib["order"], contrib["expected in cell"]):
                so = prim[k][prim[k]["gtdb_order"] == o]
                carriers = {x: set(so.loc[so[x] == cell[x], "gtdb_family"].fillna("unassigned")) for x in t}
                together = set.intersection(*carriers.values()) if carriers else set()
                rare = min(t, key=lambda x: (so[x] == cell[x]).mean())
                fam_rows.append({"order": o, "expected in cell": e_o, "species": len(so), "families": so["gtdb_family"].nunique(),
                                 "rarest constituent level in this order": f"{rare}={cell[rare]}",
                                 "families carrying that level": ", ".join(sorted(carriers[rare])) or "none",
                                 "families carrying every constituent level": ", ".join(sorted(together)) or "none"})
            wide = df[np.logical_and.reduce([df[x] == cell[x] for x in t])]
            surv[(k, " / ".join(f"{x}={cell[x]}" for x in t))] = (pd.DataFrame(vrows), contrib, r, pd.DataFrame(fam_rows), wide)
            save(f"survivor_{k}_{'_'.join(str(cell[x]) for x in t)}_order_contributions", contrib)

    # ---------------------------------------------------------------- deliverables & manifest
    keep = ["bacdive_id", "species", "gtdb_species", "gtdb_genus", "gtdb_order", "gtdb_class", "gtdb_phylum", "gtdb_accession",
            "ncbi_assembly_accession", "assembly_genbank", "assembly_refseq", "match_method", "type_strain_bacdive",
            "gtdb_is_type_strain_of_species"] + [c for t in core for c in (t, f"{t}_fine")] + [
            "temperature_bin_source", "temperature_point", "oxygen_microaerophile_folded", "ph", "ph_bin_source", "halophily_level"]
    for k, _, _ in panels:
        prim[k][keep].to_csv(final / f"{k}_panel_species.tsv", sep="\t", index=False)
        (final / f"assembly_accessions_{k}_panel_species.txt").write_text("\n".join(prim[k]["ncbi_assembly_accession"].astype(str)) + "\n")
    (final / "assembly_accessions_all_matched.txt").write_text(
        "\n".join(matched["ncbi_assembly_accession"].dropna().astype(str).drop_duplicates()) + "\n")
    doi_dates = df["doi_date"].value_counts()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(cfg), "config_sha256": cfg.text_sha256,
        "regenerate": "python -m src.pipeline all",
        "bacdive": {"api": cfg["bacdive"]["base_url"], "api_version": "v2",
                    "release_stamp_from_record_dois": doi_dates.index[0] if len(doi_dates) else None,
                    "doi_date_counts": {str(k): int(v) for k, v in doi_dates.items()},
                    "fetched_from": df["fetched_at"].min(), "fetched_to": df["fetched_at"].max(),
                    "predictions_requested": False, **sweep},
        "gtdb": gtdb,
        "outputs": {"data/final/strains.parquet": file_fingerprint(final / "strains.parquet"),
                    "data/interim/unmatched_strains.tsv": file_fingerprint(inter / "unmatched_strains.tsv")},
        "counts": {"strains": n, "matched": int(df["genome_matched"].sum()),
                   **{f"{k}_species": len(prim[k]) for k, _, _ in panels},
                   **{f"{k}_order_level_survivors": int(merged[k]["order: empty✗"].sum()) for k, _, _ in panels}},
        "analysis": dict(a),
    }
    (rep / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    # ---------------------------------------------------------------- markdown
    R: list[str] = []
    w = R.append
    bd_stamp = manifest["bacdive"]["release_stamp_from_record_dois"]
    n_core = int(df["core6_complete"].sum())
    n_core_g = int((df["core6_complete"] & df["genome_matched"]).sum())
    pr = {k: results[k]["primary"] for k, _, _ in panels}
    n_surv = {k: int(merged[k]["order: empty✗"].sum()) for k, _, _ in panels}
    n_glob = {k: int(merged[k]["global: empty✗"].sum()) for k, _, _ in panels}
    ts_core = type_strain_table(prim["core"]).set_index("")

    w("# BacDive × GTDB: attrition and trait-space occupancy\n")
    w(f"_Generated {manifest['generated_at']} · code `{manifest['git_commit']}` · config sha256 `{cfg.text_sha256[:12]}`_\n")
    w("## Snapshot\n")
    w(md_table(pd.DataFrame([
        {"source": "BacDive", "version": f"API v2, record DOIs stamped `{bd_stamp}` (BacDive exposes no release number)",
         "retrieved": f"{manifest['bacdive']['fetched_from'][:10]} → {manifest['bacdive']['fetched_to'][:10]}",
         "detail": f"IDs {sweep['sweep_range'][0]:,}–{sweep['sweep_range'][1]:,} swept in full ({sweep['batches_cached']:,} batches of 100); "
                   f"highest ID found {sweep['max_bacdive_id_found']:,}; genome-based predictions not requested"},
        {"source": "GTDB", "version": f"{gtdb['release']} ({gtdb['released']})", "retrieved": gtdb["joined_at"][:10],
         "detail": f"bac120_metadata.tsv.gz, {gtdb['n_genomes']:,} genomes, sha256 `{gtdb['metadata_sha256'][:12]}`"},
    ])))
    w("\nEmpty BacDive ID ranges ≥1,000 IDs: " + ", ".join(f"{a_:,}–{b_:,}" for a_, b_ in sweep["empty_id_ranges_ge_1000"]) + ".\n")

    # headline -------------------------------------------------------------------------------
    oc, on = pr["core"], pr["no_spore"]
    w("## Headline\n")
    w(f"- **Data.** {n:,} BacDive strains. {n_core:,} have the 6-trait core panel, {n_core_g:,} of them with a GTDB genome, "
      f"giving **{len(prim['core']):,} species**. Dropping spore formation gives the 5-trait no-spore panel: "
      f"**{len(prim['no_spore']):,} species** ({len(prim['no_spore']) / len(prim['core']):.1f}×) on half the cells "
      f"({on[None].summary['cells']} vs {oc[None].summary['cells']}).")
    pcs_core = [p for p in pcs if p["spec"]["panel"] == "core"]
    if pcs_core:
        p0 = pcs_core[0]
        w(f"- **Positive control {'passed' if p0['passed'] else 'FAILED'}.** {p0['spec']['name']} is a known phylogenetic constraint, not a finding. "
          f"The global null flags it (expected {p0['row']['global: expected']:.1f}, observed 0). Under the order-level shuffle it is "
          f"{'untestable' if not p0['row']['order: testable'] else 'not significant'} (expected {p0['row']['order: expected']:.1f}), "
          "as it should be once the null knows which clades carry which traits (§7).")
    w(f"- **The occupancy deficit is mostly phylogenetic.** Core panel: {oc[None].summary['occupied']} cells occupied vs "
      f"{oc[None].summary['null_occupied_mean']:.0f} expected under the global null (P = {oc[None].summary['p_fewer_occupied_than_null']:.2g}). "
      f"The phylum, class and order shuffles give {oc['phylum'].summary['null_occupied_mean']:.0f}, {oc['class'].summary['null_occupied_mean']:.0f} "
      f"and {oc['order'].summary['null_occupied_mean']:.0f} (order: {oc['order'].summary['occupied']} observed among kept species, "
      f"P = {oc['order'].summary['p_fewer_occupied_than_null']:.2g}). The no-spore panel behaves the same "
      f"({on[None].summary['occupied']} vs {on[None].summary['null_occupied_mean']:.0f} global; "
      f"{on['order'].summary['occupied']} vs {on['order'].summary['null_occupied_mean']:.0f} at order level, "
      f"P = {on['order'].summary['p_fewer_occupied_than_null']:.2g}) (§8).")
    def cells_word(x):
        return f"{x} cell{'s' if x != 1 else ''}"
    for k, _, plabel in panels:
        glob_only = int((merged[k]["global: empty✗"] & ~merged[k]["order: empty✗"]).sum())
        glob_txt = (f"{glob_only} of the {cells_word(n_glob[k])} flagged by the global null do not survive at order level: phylogenetic structure"
                    if glob_only else f"the global null flags {cells_word(n_glob[k])}, all of which also survive at order level")
        w(f"- **{plabel}: {cells_word(n_surv[k])} survive{'s' if n_surv[k] == 1 else ''} the order-level shuffle.** {glob_txt[0].upper() + glob_txt[1:]} (§10).")
        for (kk, c), v in surv.items():
            if kk != k:
                continue
            vt, _, r, _, wide = v
            vv = vt.set_index(["view", "null"])
            notes = []
            for view_, null_ in (("optimum-only", "order"), ("genus-level", "order")):
                if (view_, null_) in vv.index:
                    x = vv.loc[(view_, null_)]
                    notes.append(f"{view_}: {'survives' if x['empty beyond chance'] else ('untestable' if not x['testable'] else f'not significant, q = {x.q:.2g}')}")
            diag = [(nl, x) for (vw, nl), x in vv.iterrows() if str(vw).startswith("primary (diagnostic")]
            dtxt = "; ".join(f"a {nl}-level shuffle {'also flags it' if x['empty beyond chance'] else ('makes it untestable' if not x['testable'] else 'does not flag it')} "
                             f"(expected {x['expected']:.1f})" for nl, x in diag)
            w(f"  - `{c}`: expected {r['order: expected']:.1f} under the order shuffle, observed 0, q = {r['order: q']:.2g} "
              f"(independent seed {r['order: q (seed 2)']:.2g}). Fragile: {'; '.join(notes)}. Diagnostic beyond the three requested ranks: {dtxt}. "
              f"{len(wide)} of all {n:,} BacDive strains {'has' if len(wide) == 1 else 'have'} the combination. "
              + ("**Treat as family-level structure until shown otherwise (§11).**" if any(not x["empty beyond chance"] for _, x in diag)
                 else "**Survives the finer diagnostic too (§11).**"))
    w(f"- **Samples are type strains.** {ts_core.loc['either', '%']} of the core-panel species are represented by a type strain (§12).\n")

    # 1 attrition -----------------------------------------------------------------------------
    w("## 1. Attrition\n")
    w("Counts are strains. \"Usable\" means a normalised value exists: at least one observation, mappable to a bin, "
      "and no conflict between references (conflicting observations null the value rather than picking one).\n")
    w(md_table(att))
    w("\n### 1a. Per-trait losses\n")
    w(md_table(ptd))
    w("\nUnmapped and unparsed raw values (all kept in the `*_raw` columns; full list in `data/interim/unmapped_values.tsv`):\n")
    w(md_table(unmapped.head(20)))
    src_counts = prim["core"]["temperature_bin_source"].value_counts()
    w("\n### 1b. Temperature bin sources, core panel × genome × species\n")
    w(md_table(pd.DataFrame({"temperature_bin_source": src_counts.index, "N": src_counts.values,
                             "%": [pct(v, len(prim["core"])) for v in src_counts.values]})))
    fold_conf = int((df["oxygen_conflict"] & df["oxygen_raw"].fillna("").str.contains("microaerophile")).sum())
    w(f"\nOxygen: {int(prim['core']['oxygen_microaerophile_folded'].sum())} core-panel species are \"facultative\" only because microaerophile "
      f"was folded in (`oxygen_microaerophile_folded`). Across all strains, {fold_conf:,} of the {int(df['oxygen_conflict'].sum()):,} coarse "
      "oxygen conflicts involve a microaerophile observation.\n")

    # 2 matching ------------------------------------------------------------------------------
    w("## 2. Genome matching\n")
    w("Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality of "
      "normalised strain designations. A designation with no culture-collection code (e.g. `BS 107`) is accepted only with **taxon agreement**: "
      "the BacDive genus equals the GTDB or NCBI genus of the genome, or one of BacDive's NCBI tax IDs equals the genome's `ncbi_taxid` or "
      "`ncbi_species_taxid`. There is no fuzzy matching anywhere.\n")
    w("### 2a. What normalisation bought\n")
    w("Strains with ≥1 accepted GTDB genome. Steps A–C use the designation route alone and apply the same acceptance rule, "
      "so they differ only in how strings are compared. Step E equals the final match count.\n")
    w(md_table(stage))
    w("\nSplitting multi-designation strings does most of the work (A→B). Normalisation proper adds B→C. The last row is what a naive raw-token "
      "join would report, and includes cross-genus collisions (`LP1`, `B86`, `M120` recur across unrelated genera).\n")
    w("### 2b. Genome attrition, all strains\n")
    w(md_table(gen_all))
    w("\n### 2c. Genome attrition, core-panel-complete strains\n")
    w(md_table(gen_core))
    w("\n### 2d. Match quality checks\n")
    w(md_table(pd.DataFrame([
        {"check": "designation route agrees with accession route (strains where both fire)", "value": pct(int(agree.sum()), len(agree), 2), "N": len(agree)},
        {"check": "bare-designation hits agree with accession route (only bare keys hit, accession also present)",
         "value": pct(int(bare_agree.sum()), len(bare_agree), 2), "N": len(bare_agree)},
        {"check": "matched genome's GTDB/NCBI genus equals BacDive genus", "value": pct(int(matched['genus_agree'].sum()), len(matched)), "N": len(matched)},
        {"check": "designation matches accepted via NCBI tax ID only (genus differs)", "value": pct(n_taxid_only, n_desig), "N": n_taxid_only},
        {"check": "candidates span >1 GTDB species (ambiguous)", "value": pct(int(matched['match_ambiguous_species'].sum()), len(matched)), "N": len(matched)},
    ])))
    w("\nGenus disagreement on accession-route matches mostly reflects GTDB reclassification (split genera) rather than wrong matches; those strains are kept.\n")
    w("### 2e. Unmatched strains\n")
    w(f"All {len(unmatched):,} are listed with the keys tried in `data/interim/unmatched_strains.tsv` (gitignored; regenerate with `python -m src.pipeline join`).\n")
    w(md_table(reasons))

    # 3 coverage ------------------------------------------------------------------------------
    w("\n## 3. Trait coverage co-occurrence\n")
    w("Strains with usable values for both traits (diagonal = single-trait coverage), over all BacDive strains.\n")
    w(md_table(cov["counts"], index=True))
    w("\nP(column usable | row usable):\n")
    w(md_table(cov_core["conditional"].round(3), index=True))
    w("\nLift = P(both) / (P(row)·P(column)); 1 = independent coverage:\n")
    w(md_table(cov_core["lift"].round(2), index=True))
    w("\n### 3a. Panel completeness vs independent coverage\n")
    w(md_table(ind))
    w("\nCoverage is nested: strains described with one morphology trait usually have the others, because the traits come together from "
      "species descriptions. Spore formation is the binding constraint.\n")

    # 4 genome availability ------------------------------------------------------------------
    w("## 4. Does genome availability track trait coverage?\n")
    w(f"Base rate over all {n:,} strains: {pct(int(df['bacdive_lists_assembly'].sum()), n)} list a GCA accession in BacDive. "
      "Fisher p-values below 1e-300 underflow and are shown as <1e-300.\n")
    gs = ga["sets"].copy()
    for c in [c for c in gs.columns if c.endswith("Fisher p")]:
        gs[c] = gs[c].map(lambda v: "<1e-300" if v == 0 else ("–" if pd.isna(v) else f"{v:.2g}"))
    w(md_table(gs))
    w("\nBy number of usable core traits:\n")
    w(md_table(ga["by_n_traits"]))

    # 5 panels side by side ------------------------------------------------------------------
    w("\n## 5. Two panels: with and without spore formation\n")
    w(f"Spore formation is the coverage bottleneck ({pct(int(df['spore_usable'].sum()), n)} of strains) and drove 3 of the 4 global-null hits in the first run. "
      "The no-spore panel drops it. Both are analysed identically: one strain per GTDB species, coarse bins, all temperature sources.\n")
    w(md_table(side))
    sc, sn = side.iloc[0], side.iloc[1]
    w(f"\nDoes dropping spore formation buy power? It more than doubles N ({sc['N species']:,} → {sn['N species']:,} species) and halves the cells, "
      f"so species per cell rise {sn['species per cell'] / sc['species per cell']:.1f}×. The *number* of cells the order-level null can test barely "
      f"moves ({sc['testable cells (order)']} → {sn['testable cells (order)']}), but the *fraction* does ({100 * sc['testable cells (order)'] / sc['cells']:.0f}% → "
      f"{100 * sn['testable cells (order)'] / sn['cells']:.0f}%). The core panel's extra cells are mostly unreachable at this N. The no-spore panel "
      f"produced {sn['order-level survivors']} order-level survivor(s), the core panel {sc['order-level survivors']}.\n")
    w("\nExpected species per cell, per map size (global null, species level). Read this before the results: it says where the null has power.\n")
    for k, _, plabel in panels:
        w(f"\n**{plabel}**\n")
        w(md_table(capacity_summary(caps[k])))
    w("\n### 5a. GTDB phyla (species level)\n")
    ph = pd.concat({plabel: prim[k]["gtdb_phylum"].value_counts() for k, _, plabel in panels}, axis=1).fillna(0).astype(int)
    ph = ph.sort_values(ph.columns[1], ascending=False)
    w(md_table(ph.rename_axis("GTDB phylum"), index=True))

    # 6 nulls ---------------------------------------------------------------------------------
    w("\n## 6. The nulls\n")
    w("- **Global.** Each trait column is permuted across all species, which preserves every marginal and destroys all association. "
      "It cannot tell a biological constraint from the fact that traits are each fixed in different clades.")
    w("- **Stratified (phylum, class, order).** Each trait column is permuted only among species of the same GTDB taxon, preserving "
      "every taxon's own trait frequencies. Expected count per cell = Σ over taxa n·∏ p(trait level within the taxon). A cell that "
      "is still emptier than chance under the order-level shuffle is not explained by which orders carry which traits. "
      "**That is the only category treated as a candidate.**")
    excl = {k: merged[k][merged[k]["order: emptied by exclusion"] & (merged[k]["order: expected"] >= a["min_expected_for_test"])]
            for k, _, _ in panels}
    ex_txt = ""
    for k, t_, plabel in panels:
        if len(excl[k]):
            r0 = excl[k].sort_values("observed (all species)", ascending=False).iloc[0]
            ex_txt = (f" In this run it matters: `{' / '.join(f'{x}={r0[x]}' for x in t_)}` ({plabel}) holds {int(r0['observed (all species)'])} "
                      "species, all in orders below the size threshold, so it is empty among kept species and testable under the order null. "
                      "It is labelled \"emptied by exclusion\", not flagged.")
            break
    w(f"- **Small strata.** Taxa with fewer than {a['min_stratum_size']} species in the analysed set cannot be meaningfully shuffled. "
      "Their species are excluded from that null and counted below, not silently kept. Exclusion is never allowed to manufacture "
      "emptiness: a cell is \"empty beyond chance\" only if it is empty among *all* species. The excluded species are exactly the "
      "phylogenetically unusual ones, so this guard is not cosmetic." + ex_txt)
    w(f"- **Testability.** A cell is tested only if its expected count under that null is ≥ {a['min_expected_for_test']:g}. "
      f"BH correction is applied within each map over its testable cells; {a['n_perm_full']:,} permutations per full map; "
      "the global and order nulls are re-run with an independent seed.\n")
    for k, _, plabel in panels:
        w(f"**Species kept per null, {plabel}**\n")
        w(md_table(save(f"strata_{k}", strata_table(prim[k], default=a["min_stratum_size"]))))
        w("")

    # 7 positive control ---------------------------------------------------------------------
    w("## 7. Positive control\n")
    for p in pcs:
        spec = p["spec"]
        w(f"**{spec['name']}** — `" + " / ".join(f"{k}={v}" for k, v in spec["cell"].items()) + "`\n")
        w(spec["rationale"] + "\n")
        w("This cell was the strongest hit of the first, global-only run. It is reported here as a check on the method, **not as a finding**. "
          "Expected behaviour: the global null flags it, and a phylogeny-aware null does not, because its emptiness follows from "
          "which clades carry which traits.\n")
        w(md_table(p["table"]))
        w(f"\n**Result: {'passed' if p['passed'] else 'FAILED'}.** "
          + (f"The global null detects the known constraint, and the stratified nulls absorb it: the expected count falls from "
             f"{p['row']['global: expected']:.1f} (global) to {p['row']['phylum: expected']:.1f} (phylum), "
             f"{p['row']['class: expected']:.1f} (class) and {p['row']['order: expected']:.1f} (order), because within any one taxon "
             "the constituent traits barely co-occur."
             if p["passed"] else "Inspect before trusting any other cell.") + "\n")

    # 8 occupied vs expected -----------------------------------------------------------------
    w("## 8. Occupied vs expected cells\n")
    w("Number of occupied cells against the number the null predicts for the same species. Under the global null a deficit means the "
      "traits co-occur more tightly than independence allows. Under a stratified null the deficit left over is what the taxon's own trait "
      "frequencies cannot explain. Stratified rows use the species kept by that null (small strata excluded), for both observed and expected.\n")
    w(md_table(ove))
    w("\nAll 4- and 5-trait subsets of the core traits (full table: `reports/tables/occupancy_subsets_all_nulls.tsv`):\n")
    so = pd.DataFrame({"map": sub_occ.index})
    for x in NULLS:
        nm = NULL_NAME[x]
        so[f"{nm}: occupied / expected"] = [f"{int(sub_occ.loc[i, ('occupied', nm)])} / {sub_occ.loc[i, ('null_occupied_mean', nm)]:.0f}" for i in sub_occ.index]
        so[f"{nm}: P"] = [sub_occ.loc[i, ("p_fewer_occupied_than_null", nm)] for i in sub_occ.index]
    w(md_table(so))

    # 9 per-phylum occupancy -----------------------------------------------------------------
    w("\n## 9. Per-phylum occupancy\n")
    w("Cells each phylum occupies on its own, cells that only it occupies, and the cumulative share of all occupied cells, "
      "adding phyla from largest to smallest. \"Within-phylum null\" permutes traits among that phylum's species only.\n")
    for k, _, plabel in panels:
        po = phy_occ[k]
        top3 = po.iloc[:3]
        tot_occ = pr[k][None].summary["occupied"]
        w(f"\n**{plabel}** — {tot_occ} cells occupied in total. The three largest phyla ({', '.join(top3.phylum)}; "
          f"{pct(int(top3.N.sum()), int(po.N.sum()))} of species) occupy {int(top3['cumulative occupied (this + larger phyla)'].iloc[-1])} of them "
          f"({top3['cumulative share of all occupied'].iloc[-1] * 100:.0f}%). "
          f"{po.iloc[0].phylum} alone occupies {int(po.iloc[0]['cells occupied'])} ({100 * po.iloc[0]['cells occupied'] / tot_occ:.0f}%).\n")
        dd = po.dropna(subset=["occupied (within-phylum null mean)"]).copy()
        dd["deficit"] = dd["occupied (within-phylum null mean)"] - dd["cells occupied"]
        top = dd.sort_values("deficit", ascending=False).iloc[0]
        w(f"Largest within-phylum deficit: {top.phylum}, {int(top['cells occupied'])} cells occupied vs {top['occupied (within-phylum null mean)']:.0f} "
          "expected from its own trait frequencies. Other phyla sit near their own expectation.\n")
        w(md_table(po.round({"occupied (within-phylum null mean)": 1, "cumulative share of all occupied": 2})))

    # 10 cell-level results --------------------------------------------------------------------
    w("\n## 10. Cell-level results under all four nulls\n")
    w("Every cell that is empty and testable under at least one null, or flagged under at least one. `q` is BH-adjusted within the map; "
      "`–` means the cell is untestable under that null (expected below threshold). Full tables, every cell: `reports/tables/cells_<panel>_all_nulls.tsv`.\n")
    show_cols = lambda t: (t + ["observed (all species)", "global: expected", "global: q", "phylum: expected", "phylum: q",  # noqa: E731
                                "class: expected", "class: q", "order: expected", "order: q", "global: q (seed 2)", "order: q (seed 2)", "category"])
    for k, t, plabel in panels:
        m = merged[k]
        testable_any = np.logical_or.reduce([m[f"{NULL_NAME[x]}: testable"] for x in NULLS])
        flagged_any = np.logical_or.reduce([m[f"{NULL_NAME[x]}: empty✗"] for x in NULLS])
        sel = m[((m["observed (all species)"] == 0) & testable_any) | flagged_any].copy()
        for x in NULLS:
            nm = NULL_NAME[x]
            sel.loc[~sel[f"{nm}: testable"], f"{nm}: q"] = np.nan
        sel = sel.sort_values("global: expected", ascending=False)
        w(f"\n### 10{'a' if k == 'core' else 'b'}. {plabel} ({pr[k][None].summary['cells']} cells, {len(prim[k]):,} species)\n")
        w(md_table(sel[show_cols(t)]) if len(sel) else "No cell is empty and testable under any null.\n")
        w("\nCross-tabulation (species counts). **0✗** survives the order-level shuffle · 0† fires under a coarser null only "
          "(phylogenetic structure) · 0 empty, testable, not significant · `·` empty and untestable under every null.\n")
        w(md_table(crosstab_grid(m, t, cfg), index=True))

    # 11 survivors ------------------------------------------------------------------------------
    w("\n## 11. Cells that survive the order-level shuffle\n")
    if not surv:
        w("None, in either full panel.\n")
    for (k, cname), (vt, contrib, r, fams, wide) in surv.items():
        plabel = dict((kk, lab) for kk, _, lab in panels)[k]
        w(f"### `{cname}` ({plabel})\n")
        w(f"Observed 0 among {len(prim[k]):,} species. Expected {r['global: expected']:.1f} under the global null and "
          f"{r['order: expected']:.1f} under the order-level shuffle (q = {r['order: q']:.2g}; independent seed q = {r['order: q (seed 2)']:.2g}).\n")
        w("Robustness across views:\n")
        w(md_table(vt))
        tot = contrib["expected in cell"].sum()
        w(f"\nOrders the order-level null expects to populate this cell (top 10 of {len(contrib)} by expected count; each order's share of "
          f"species with each trait level). The top order carries {100 * contrib['expected in cell'].iloc[0] / tot:.0f}% of the expectation.\n")
        w(md_table(contrib.head(10).round(3)))
        split = fams.loc[fams["families carrying every constituent level"] == "none", "expected in cell"].sum()
        w(f"\nInside the contributing orders, do the constituent trait levels ever occur in the same family? "
          f"{100 * split / tot:.0f}% of the order-level expectation comes from orders in which **no single family** carries every "
          "constituent level; there the combination is only \"expected\" because the order mixes families. Top 5 orders:\n")
        w(md_table(fams.head(5).round(3)))
        w(f"\nAcross **all** {n:,} BacDive strains, with or without a genome, {len(wide)} strain{' has' if len(wide) == 1 else 's have'} this combination"
          + (": " + ", ".join(f"{s_} (BacDive {b_}{', genome' if g_ else ', no genome'})" for s_, b_, g_ in
                                zip(wide["species"], wide["bacdive_id"], wide["genome_matched"])) if 0 < len(wide) <= 10 else "") + ".\n")
        diag = [row_ for _, row_ in vt.iterrows() if str(row_["view"]).startswith("primary (diagnostic")]
        for d in diag:
            if not d["empty beyond chance"]:
                w(f"**The {d['null']}-level diagnostic absorbs it:** expected {d['expected']:.2f} under a {d['null']} shuffle "
                  f"({'untestable' if not d['testable'] else 'not significant'}). Read this as phylogenetic structure at {d['null']} "
                  "level unless a finer analysis says otherwise.\n")
            else:
                w(f"**It also survives the {d['null']}-level diagnostic** (expected {d['expected']:.2f}, q = {d['q']:.2g}).\n")
        w("\nWhat surviving the order shuffle does and does not mean: within orders that contain every constituent trait, the combination "
          "is rarer than independence predicts. It can still be phylogenetic structure *below* order (family, genus), and the whole panel "
          "is described type strains (§12). It is a candidate, not a result.\n")
    w("### 11a. Order-level survivors in lower-dimensional maps\n")
    w("All 4- and 5-trait subsets of the core traits, order-level null. Lower-dimensional maps have more species per cell, so they can test "
      "cells the full panels cannot. Not corrected across maps.\n")
    w(md_table(subset_survivors) if len(subset_survivors) else "None.\n")

    # 12 sensitivities ------------------------------------------------------------------------
    w("\n## 12. Sensitivity views\n")
    w("Global and order-level nulls in each view. Genus-level keeps one strain per GTDB genus, so its order-level strata are small and "
      "many species are excluded.\n")
    w(md_table(views))
    w("\n### 12a. Type strains\n")
    w(md_table(ts_core))

    # 13 caveats ------------------------------------------------------------------------------
    w("\n## 13. Caveats\n")
    w(f"- **Type-strain sampling.** {ts_core.loc['either', '%']} of core-panel species are represented by a type strain. The map describes "
      "*described, culturable, formally characterised* species: organisms someone isolated in pure culture and wrote up under a standard "
      "protocol. Uncultured lineages (most candidate phyla, most of the tree by genome count) are absent. Trait values come from protocol-driven "
      "species descriptions: growth tested at standard temperatures, a fixed test battery. The marginals are those of the described world, "
      "and a cell can be empty because no one has isolated and described such an organism.")
    w("- **Exchangeability below order.** The order-level shuffle removes structure between orders but not within them. Families and "
      "genera share traits, so a survivor can still be phylogenetic structure at a finer rank. The genus-level view is a crude check, not a fix.")
    w("- **Stratified nulls lose power.** Testable cells fall from global to order level (§5). A cell that stops firing at order level "
      "can be untestable rather than explained. §10 says which.")
    w(f"- **Temperature source.** Across BacDive most temperature values are single cultivation temperatures "
      f"({int((df.temperature_bin_source == 'growth_single_point').sum()):,} single-point vs {int((df.temperature_bin_source == 'optimum').sum()):,} optimum). "
      f"In the core-panel set {pct(int(src_counts.get('optimum', 0)), len(prim['core']))} of bins come from an optimum. The optimum-only view (§12) removes the rest.")
    w("- **Multiple testing.** BH is applied within each map over its testable cells only. The subset maps are not jointly corrected.")
    w(f"- **Resolution.** Permutation p-values cannot go below 1/(n_perm+1) = {1 / (a['n_perm_full'] + 1):.1e} for the full maps.")
    w("\n## Files\n")
    w("- `data/final/strains.parquet` (gitignored): every BacDive strain with raw and normalised traits, bin sources, conflicts, genome match and panel flags. "
      "Regenerate with `python -m src.pipeline all`; checksums in `reports/run_manifest.json` → `outputs`.")
    w("- `data/final/{core,no_spore}_panel_species.tsv`: the analysed sets (one strain per GTDB species).")
    w("- `data/final/assembly_accessions_{core,no_spore}_panel_species.txt`, `assembly_accessions_all_matched.txt`: for "
      "`datasets download genome accession --inputfile …` (RefSeq GCF where GTDB uses RefSeq, else GenBank GCA).")
    w("- `data/interim/unmatched_strains.tsv` (gitignored), `data/interim/unmapped_values.tsv`: audit logs.")
    w("- `reports/tables/*.tsv`: every table above, plus per-cell results for every panel × view × null.")
    w("- `reports/run_manifest.json`: versions, hashes and counts.")
    text = "\n".join(R) + "\n"
    (rep / "attrition_report.md").write_text(text)
    return text
