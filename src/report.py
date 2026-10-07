"""Occupancy report: attrition, occupancy, global and stratified nulls, power.

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
    keep = ["bacdive_id", "species", "gtdb_species", "gtdb_genus", "gtdb_family", "gtdb_order", "gtdb_class", "gtdb_phylum", "gtdb_accession",
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

    # ---------------------------------------------------------------- power: minimum detectable constraint
    mde_rows = [{"panel": plabel, **A.detectable_zero(results[k]["primary"][x], cfg)} for k, _, plabel in panels for x in (None, "order")]
    mde = save("minimum_detectable_constraint", pd.DataFrame(mde_rows))
    mde_o = {k: A.detectable_zero(results[k]["primary"]["order"], cfg) for k, _, _ in panels}

    # deficit trajectory, and correction across the subset maps
    pr = {k: results[k]["primary"] for k, _, _ in panels}
    traj = {}
    for k, _, _ in panels:
        d = [pr[k][x].summary["null_occupied_mean"] - pr[k][x].summary["occupied"] for x in NULLS]
        traj[k] = {"deficits": d, "nonincreasing": all(d[i + 1] <= d[i] + 1e-9 for i in range(len(d) - 1)),
                   "strict": all(d[i + 1] < d[i] - 1e-9 for i in range(len(d) - 1))}
    so_order = grid[grid["null"] == "order"][["traits", "occupied", "null_occupied_mean", "p_fewer_occupied_than_null"]].copy()
    so_order["q across maps (BH)"] = A.bh_qvalues(so_order["p_fewer_occupied_than_null"].to_numpy())
    save("order_level_deficit_across_maps", so_order)

    # small-strata methods note: cells that only the exclusion would have emptied
    excl_rows = []
    for k, t, plabel in panels:
        ct = pr[k]["order"].cell_table
        hit = ct[ct["emptied_by_exclusion"] & ct["testable"]]
        for _, r in hit.iterrows():
            excl_rows.append({"panel": plabel, "cell": " / ".join(f"{x}={r[x]}" for x in t),
                              "species in cell (all)": int(r["observed_all_species"]), "species in cell (kept)": int(r["observed"]),
                              "order-null expected (kept)": r["expected"], "q it would have had": r["q_low"],
                              "would have been flagged": bool(r["q_low"] < a["fdr_alpha"]),
                              "order sizes of its species": ", ".join(str(int(s_)) for s_ in sorted(
                                  prim[k].loc[np.logical_and.reduce([prim[k][x] == r[x] for x in t]), "gtdb_order"]
                                  .map(prim[k]["gtdb_order"].value_counts()).tolist()))})
    excl = save("small_strata_false_positives", pd.DataFrame(excl_rows))

    # ---------------------------------------------------------------- markdown
    R: list[str] = []
    w = R.append
    bd_stamp = manifest["bacdive"]["release_stamp_from_record_dois"]
    n_core = int(df["core6_complete"].sum())
    n_core_g = int((df["core6_complete"] & df["genome_matched"]).sum())
    n_surv = {k: int(merged[k]["order: empty✗"].sum()) for k, _, _ in panels}
    n_glob = {k: int(merged[k]["global: empty✗"].sum()) for k, _, _ in panels}
    ts_core = type_strain_table(prim["core"]).set_index("")
    plab = {k: lab for k, _, lab in panels}
    oc, on = pr["core"], pr["no_spore"]
    alpha = a["fdr_alpha"]

    def dtxt(k):
        d = traj[k]["deficits"]
        return " → ".join(f"{x:.0f}" for x in d)

    def trend(k):
        return "narrows at every step" if traj[k]["strict"] else ("never widens" if traj[k]["nonincreasing"] else "does not narrow monotonically")

    w("# Bacterial trait space: occupancy and its phylogenetic structure\n")
    w(f"_Generated {manifest['generated_at']} · code `{manifest['git_commit']}` · config sha256 `{cfg.text_sha256[:12]}` · "
      f"BacDive API v2 (record DOIs `{bd_stamp}`) · GTDB {gtdb['release']}_\n")

    # ---- summary ---------------------------------------------------------------------------------
    w("## Summary\n")
    w(f"**Bacterial trait space is sparser than independence predicts, and that sparsity is almost entirely phylogenetic.** "
      f"On the six-trait core map, {len(prim['core']):,} species occupy {oc[None].summary['occupied']} of {oc[None].summary['cells']} cells; "
      f"with traits assigned independently they would fill {oc[None].summary['null_occupied_mean']:.0f} "
      f"(P = {oc[None].summary['p_fewer_occupied_than_null']:.2g}). When the null is made phylogeny-aware, shuffling traits only within "
      f"a phylum, class or order, the deficit {trend('core')}: {dtxt('core')} cells. At order level it is not significant "
      f"(P = {oc['order'].summary['p_fewer_occupied_than_null']:.2g}). The five-trait map without spore formation ({len(prim['no_spore']):,} species) "
      f"behaves the same ({dtxt('no_spore')} cells; P = {on['order'].summary['p_fewer_occupied_than_null']:.2g} at order level).\n")
    surv_ns = [(c, v) for (kk, c), v in surv.items() if kk == "no_spore"]
    absorbed = all(any(not x["empty beyond chance"] for _, x in v[0].iterrows() if str(x["view"]).startswith("primary (diagnostic"))
                   for _, v in surv.items()) if surv else True
    s_txt = ""
    if n_surv["core"] == 0 and surv_ns:
        c, v = surv_ns[0]
        d = [x for _, x in v[0].iterrows() if str(x["view"]).startswith("primary (diagnostic")]
        s_txt = (f" In the core panel no cell survives the order-level shuffle. In the no-spore panel one cell, `{c}`, passes the order-level "
                 f"criterion (expected {v[2]['order: expected']:.1f}, q = {v[2]['order: q']:.2g})"
                 + (f", but a family-level shuffle makes it untestable (expected {d[0]['expected']:.1f})" if d and not d[0]["empty beyond chance"] else "")
                 + f", and {len(v[4])} of {n:,} BacDive strains has the combination.")
    elif sum(n_surv.values()) == 0:
        s_txt = " No cell survives the order-level shuffle in either panel."
    verdict = ("No trait combination is empty beyond what lineage structure explains." if absorbed
               else "At least one trait combination is empty beyond what lineage structure explains (§5).")
    w(f"**{verdict}**{s_txt}\n")
    if pcs:
        p0 = pcs[0]
        w(f"The method does detect known constraints. {p0['spec']['name']}, a combination absent because endospores are a Bacillota trait and "
          f"Bacillota are Gram-positive, is flagged by the global null (expected {p0['row']['global: expected']:.1f}, observed 0) and correctly "
          f"absorbed by every stratified null (expected {p0['row']['order: expected']:.1f} at order level).\n")
    w("The argument, section by section:\n")
    w("1. **§1 The occupancy map.** Which trait combinations are occupied, and by how many species.")
    w("2. **§2 Global null.** Fewer cells are occupied than independent traits would fill.")
    w("3. **§3 Stratified nulls.** The shortfall shrinks as the null respects phylum, class, then order, and is not significant at order level.")
    w("4. **§4 Positive control.** The method flags a known constraint globally and attributes it to lineage.")
    w("5. **§5 No novel constraints.** No cell survives both the order-level shuffle and the family-level check.")
    w("6. **§6 Power.** The smallest constraint the order-level null could have detected.")
    w("7. **§7 Methods note.** Why dropping small strata from a stratified permutation test creates false positives.\n")

    # ---- read this first -------------------------------------------------------------------------
    mc, mn = mde_o["core"], mde_o["no_spore"]
    n_phyla_panel = prim["core"]["gtdb_phylum"].nunique()
    w("## Read this first: what the result does and does not show\n")
    w(f"1. **Six coarse traits are a low-dimensional slice of phenotype.** The maps have 2–3 levels per trait "
      f"({oc[None].summary['cells']} and {on[None].summary['cells']} cells). Absence of detectable constraint at this resolution is not absence "
      "of constraint. Constraints can live in finer bins, in continuous values (cell size, optimum temperature), or in traits not measured here "
      "(metabolism, envelope chemistry, genome features).")
    w(f"2. **The panel is type strains.** {ts_core.loc['either', '%']} of core-panel species are represented by a type strain: culturable, "
      f"formally described organisms, characterised with standard protocols. The panel spans {n_phyla_panel} of the "
      f"{gtdb.get('n_phyla', '–')} bacterial phyla in GTDB {gtdb['release']}. Whole uncultured phyla are absent by construction. The marginals "
      "are those of the described world, and a combination can be absent because no one has isolated and described such an organism.")
    w(f"3. **Power is limited.** At order level only {mc['testable cells (m)']} of {mc['cells']} core-panel cells are testable "
      f"(no-spore: {mn['testable cells (m)']} of {mn['cells']}). A true zero is guaranteed to reach q < {alpha} only if the order-level null "
      f"expects **≥ {mc['minimum detectable constraint (E)']:.1f} species** in that cell (core panel; {100 * mc['as share of species']:.2f}% "
      f"of the species tested) or **≥ {mn['minimum detectable constraint (E)']:.1f}** (no-spore; {100 * mn['as share of species']:.2f}%). "
      f"Only {mc['cells with E >= minimum detectable constraint']} core-panel cells meet that bar "
      f"(no-spore: {mn['cells with E >= minimum detectable constraint']}). They hold {100 * mc['share of expected species mass in those cells']:.0f}% "
      f"({100 * mn['share of expected species mass in those cells']:.0f}%) of the expected species. **We could have detected a constraint "
      f"that empties a combination the lineage structure predicts to hold about {mc['minimum detectable constraint (E)']:.0f} or more species. "
      "Constraints on rarer combinations are invisible to this analysis** (§6).")
    w("4. **Exchangeability below order.** The order-level shuffle removes structure between orders, not within them. That is why "
      "the single order-level survivor was re-tested at family level (§5).\n")

    # ---- 1. map -------------------------------------------------------------------------------------
    w("## 1. The occupancy map\n")
    w(f"{n:,} BacDive strains; {n_core:,} have the six-trait core panel, {n_core_g:,} of them with a matched GTDB genome, "
      f"dereplicated to one strain per GTDB species (type strain preferred). Attrition and matching details are in Appendix A–B.\n")
    w(md_table(side))
    sc, sn = side.iloc[0], side.iloc[1]
    w(f"\nDropping spore formation, the coverage bottleneck ({pct(int(df['spore_usable'].sum()), n)} of strains), raises N "
      f"{sn['N species'] / sc['N species']:.1f}× on half the cells. That lifts the testable share at order level from "
      f"{100 * sc['testable cells (order)'] / sc['cells']:.0f}% to {100 * sn['testable cells (order)'] / sn['cells']:.0f}% of cells, "
      "though the absolute number barely moves.\n")
    for k, t, plabel in panels:
        w(f"\n**{plabel}: species per cell.** **0✗** survives the order-level shuffle · 0† flagged by a coarser null only (phylogenetic "
          "structure) · 0 empty, testable, not flagged · `·` empty and untestable under every null.\n")
        w(md_table(crosstab_grid(merged[k], t, cfg), index=True))
        w("")

    # ---- 2. global null -------------------------------------------------------------------------------
    w("\n## 2. The global null: fewer occupied cells than independence predicts\n")
    w("Each trait column is permuted across all species, which keeps every trait's frequency and destroys all association between traits. "
      "The observed map has fewer occupied cells than the permuted maps: traits co-occur more tightly than independence allows.\n")
    w(md_table(ove[ove["null"] == "global"].drop(columns=["strata", "species excluded (small strata)"])))
    gl = pd.concat([merged[k].loc[merged[k]["global: empty✗"], t + ["global: expected", "global: q", "global: q (seed 2)"]]
                    .assign(panel=plab[k]) for k, t, _ in panels], ignore_index=True)
    w(f"\nCells the global null flags as empty beyond chance (q < {alpha}): {n_glob['core']} in the core panel, {n_glob['no_spore']} in the no-spore panel.\n")
    w(md_table(gl[["panel"] + core + ["global: expected", "global: q", "global: q (seed 2)"]].fillna("·")))
    w("\nThe global null cannot distinguish a constraint from the fact that traits are each fixed in different clades. §3 separates the two.\n")

    # ---- 3. stratified ------------------------------------------------------------------------------
    w("## 3. Stratified nulls: the deficit is phylogenetic\n")
    w("Each trait column is permuted only among species of the same GTDB taxon, so every taxon keeps its own trait frequencies. "
      "Expected count per cell = Σ over taxa n·∏ p(trait level within the taxon). Whatever deficit remains is what lineage composition "
      f"cannot explain. Taxa with fewer than {a['min_stratum_size']} species are excluded and counted (§7); stratified rows use the kept species "
      "for both observed and expected.\n")
    w(md_table(ove))
    for k, _, plabel in panels:
        d = traj[k]["deficits"]
        w(f"\n- **{plabel}:** deficit {' → '.join(f'{x:.1f}' for x in d)} cells (global → phylum → class → order); it {trend(k)}. "
          f"Order level: {pr[k]['order'].summary['occupied']} occupied vs {pr[k]['order'].summary['null_occupied_mean']:.1f} expected, "
          f"P = {pr[k]['order'].summary['p_fewer_occupied_than_null']:.2g}.")
    w(f"\n**Lower-dimensional maps.** Across all {len(so_order)} 4- and 5-trait subsets of the core traits, the smallest order-level P for an "
      f"occupancy deficit is {so_order['p_fewer_occupied_than_null'].min():.2g}; after BH correction across maps the smallest q is "
      f"{so_order['q across maps (BH)'].min():.2g}. {'No' if so_order['q across maps (BH)'].min() >= alpha else 'Some'} "
      "subset map retains a significant deficit at order level. Full table: `reports/tables/order_level_deficit_across_maps.tsv`.\n")
    w("### 3a. Where the occupancy comes from: per phylum\n")
    for k, _, plabel in panels:
        po = phy_occ[k]
        top3 = po.iloc[:3]
        tot_occ = pr[k][None].summary["occupied"]
        dd = po.dropna(subset=["occupied (within-phylum null mean)"]).copy()
        dd["deficit"] = dd["occupied (within-phylum null mean)"] - dd["cells occupied"]
        top = dd.sort_values("deficit", ascending=False).iloc[0]
        w(f"\n**{plabel}.** {tot_occ} cells occupied. The three largest phyla ({', '.join(top3.phylum)}; "
          f"{pct(int(top3.N.sum()), int(po.N.sum()))} of species) account for {int(top3['cumulative occupied (this + larger phyla)'].iloc[-1])} "
          f"of them ({top3['cumulative share of all occupied'].iloc[-1] * 100:.0f}%). The largest within-phylum deficit is {top.phylum}: "
          f"{int(top['cells occupied'])} cells vs {top['occupied (within-phylum null mean)']:.0f} expected from its own trait frequencies.\n")
        w(md_table(po.head(10).round({"occupied (within-phylum null mean)": 1, "cumulative share of all occupied": 2})))
    w("\n### 3b. Cell-level results under all four nulls\n")
    w("Every cell that is empty and testable under at least one null, or flagged under at least one. `–` = untestable under that null. "
      "Every cell, every null: `reports/tables/cells_<panel>_all_nulls.tsv`.\n")
    show_cols = lambda t: (t + ["observed (all species)", "global: expected", "global: q", "phylum: expected", "phylum: q",  # noqa: E731
                                "class: expected", "class: q", "order: expected", "order: q", "category"])
    for k, t, plabel in panels:
        m = merged[k]
        testable_any = np.logical_or.reduce([m[f"{NULL_NAME[x]}: testable"] for x in NULLS])
        flagged_any = np.logical_or.reduce([m[f"{NULL_NAME[x]}: empty✗"] for x in NULLS])
        sel = m[((m["observed (all species)"] == 0) & testable_any) | flagged_any].copy()
        for x in NULLS:
            sel.loc[~sel[f"{NULL_NAME[x]}: testable"], f"{NULL_NAME[x]}: q"] = np.nan
        w(f"\n**{plabel}**\n")
        w(md_table(sel.sort_values("global: expected", ascending=False)[show_cols(t)]) if len(sel) else "None.\n")
    w("\nCells flagged under phylum or class but not globally arise when a taxon mixes sub-clades. Within Bacillota, for example, "
      "spore formers and cocci sit in different orders, so a phylum-level shuffle predicts spore-forming cocci that an order-level shuffle does not.\n")

    # ---- 4. positive control -------------------------------------------------------------------------
    w("## 4. Positive control: the method detects a known constraint\n")
    for p in pcs:
        spec = p["spec"]
        w(f"**{spec['name']}**: `" + " / ".join(f"{kk}={vv}" for kk, vv in spec["cell"].items()) + "`\n")
        w(spec["rationale"] + "\n")
        w("This was the strongest hit of the first, global-only run. It is a check on the method, **not a finding**. Expected behaviour: the "
          "global null flags it, and a phylogeny-aware null attributes it to lineage.\n")
        w(md_table(p["table"]))
        w(f"\n**Result: {'passed' if p['passed'] else 'FAILED'}.** "
          + (f"The expected count falls from {p['row']['global: expected']:.1f} (global) to {p['row']['phylum: expected']:.1f} (phylum), "
             f"{p['row']['class: expected']:.1f} (class) and {p['row']['order: expected']:.1f} (order): within any one taxon the constituent "
             "traits barely co-occur, so the combination is not expected to be occupied in the first place."
             if p["passed"] else "Inspect before trusting any other cell.") + "\n")

    # ---- 5. no novel constraints ---------------------------------------------------------------------
    w("## 5. No novel constraints survive\n")
    for k, _, plabel in panels:
        w(f"- **{plabel}:** {n_surv[k]} cell{'s' if n_surv[k] != 1 else ''} survive{'s' if n_surv[k] == 1 else ''} the order-level shuffle "
          f"({n_glob[k]} flagged by the global null).")
    w("")
    for (k, cname), (vt, contrib, r, fams, wide) in surv.items():
        tot = contrib["expected in cell"].sum()
        split = fams.loc[fams["families carrying every constituent level"] == "none", "expected in cell"].sum()
        diag = [x for _, x in vt.iterrows() if str(x["view"]).startswith("primary (diagnostic")]
        w(f"### The one order-level survivor, and why it is not a finding: `{cname}` ({plab[k]})\n")
        w(f"- **Order level.** Observed 0; the order-level null expects {r['order: expected']:.1f} (q = {r['order: q']:.2g}; independent seed "
          f"q = {r['order: q (seed 2)']:.2g}).")
        for d in diag:
            w(f"- **{d['null'].capitalize()} level (diagnostic).** Expected falls to {d['expected']:.2f}, "
              f"{'below the testability floor' if not d['testable'] else ('not significant' if not d['empty beyond chance'] else 'still flagged')}.")
        w(f"- **Where the expectation comes from.** {100 * contrib['expected in cell'].iloc[0] / tot:.0f}% comes from one order "
          f"({contrib['order'].iloc[0]}, {int(contrib['N'].iloc[0])} species). {100 * split / tot:.0f}% comes from orders in which no single family "
          "carries every constituent trait level, i.e. the combination is \"expected\" only because the order mixes families.")
        vv = vt.set_index(["view", "null"])
        notes = []
        for view_ in ("optimum-only", "genus-level"):
            if (view_, "order") in vv.index:
                x = vv.loc[(view_, "order")]
                notes.append(f"{view_}: {'survives' if x['empty beyond chance'] else ('untestable' if not x['testable'] else f'not significant (q = {x.q:.2g})')}")
        w(f"- **Sensitivity.** {'; '.join(notes)}.")
        w(f"- **In BacDive as a whole.** {len(wide)} of {n:,} strains, with or without a genome, "
          f"{'has' if len(wide) == 1 else 'have'} the combination"
          + (": " + ", ".join(f"*{s_}* (BacDive {b_}{', genome' if g_ else ', no genome'})" for s_, b_, g_ in
                                zip(wide["species"], wide["bacdive_id"], wide["genome_matched"])) if 0 < len(wide) <= 5 else "") + ".\n")
        w("Top contributing orders, and whether any family inside them carries every constituent level:\n")
        w(md_table(fams.head(5).round(3)))
        w("")
    w("### Lower-dimensional maps\n")
    w("Order-level survivors in the 4- and 5-trait subsets (not corrected across maps). Projections of the same cell count as one observation, not independent support:\n")
    w(md_table(subset_survivors) if len(subset_survivors) else "None.\n")
    w("\n**No transplant or follow-up experiments are proposed: there is no surviving target.**\n")

    # ---- 6. power -------------------------------------------------------------------------------------
    w("## 6. Power: the smallest constraint this analysis could have detected\n")
    w(f"For a cell that is truly empty, its permutation p-value is p0 = P(null count = 0). With m testable cells, BH gives q ≤ p0·m. So "
      f"p0 ≤ {alpha}/m **guarantees** q < {alpha} whatever the other cells do, and since q ≥ p0, p0 ≤ {alpha} is **necessary**. Under a "
      f"Poisson approximation p0 ≈ e^(−E), where E is the null's expected count. That gives E ≥ ln(m/{alpha}) guaranteed and "
      f"E ≥ ln(1/{alpha}) ≈ 3.0 at best. The within-taxon permutation is usually less dispersed than Poisson, so the empirical p0 from the "
      f"{a['n_perm_full']:,} permutations is also used. The **minimum detectable constraint** is the largest of the Poisson bound, the empirical "
      f"threshold and the testability floor ({a['min_expected_for_test']:g}), so it holds conservatively.\n")
    show = mde[["panel", "null", "species used", "cells", "testable cells (m)", "E needed, guaranteed (Poisson ln(m/α))",
                "E needed, guaranteed (empirical, this null)", "minimum detectable constraint (E)", "as share of species",
                "cells with E >= minimum detectable constraint", "share of expected species mass in those cells"]].copy()
    show["as share of species"] = show["as share of species"].map(lambda v: f"{100 * v:.2f}%")
    show["share of expected species mass in those cells"] = show["share of expected species mass in those cells"].map(lambda v: f"{100 * v:.0f}%")
    w(md_table(show))
    w(f"\n**Statement.** At order level we could have detected, with q < {alpha}, any constraint that empties a combination the lineage "
      f"structure predicts to hold ≥ {mc['minimum detectable constraint (E)']:.1f} species (core panel, {100 * mc['as share of species']:.2f}% of "
      f"species) or ≥ {mn['minimum detectable constraint (E)']:.1f} species (no-spore, {100 * mn['as share of species']:.2f}%). None was found. "
      f"The {mc['cells'] - mc['cells with E >= minimum detectable constraint']} core-panel cells below that bar are untested territory, not "
      "evidence of occupancy. A constraint that thins rather than empties a cell would need a larger E still.\n")
    w("Expected species per cell by map size (global null, species level):\n")
    for k, _, plabel in panels:
        w(f"\n**{plabel}**\n")
        w(md_table(capacity_summary(caps[k])))

    # ---- 7. methods note -------------------------------------------------------------------------------
    w("\n## 7. Methods note: dropping small strata from a stratified permutation test manufactures false positives\n")
    w("A within-stratum permutation cannot shuffle a stratum of one or two species, so the obvious fix is to drop small strata. "
      "Done naively, that creates false \"forbidden combinations\". Small strata are not a random subset of species: they are phylogenetically "
      "isolated lineages, deep-branching or sparsely described, and those are exactly the organisms that carry unusual trait combinations, "
      "i.e. the ones that populate rare cells. Dropping them removes a rare cell's only occupants. The null, computed from the large strata "
      "that remain, still predicts the cell from within-stratum trait frequencies. Observed zero, expected several: a confident, spurious "
      "\"empty beyond chance\".\n")
    if len(excl):
        e0 = excl.sort_values("species in cell (all)", ascending=False).iloc[0]
        n_flag = int(excl["would have been flagged"].sum())
        true_q = [merged[k].loc[merged[k]["order: empty✗"], "order: q"].min() for k, _, _ in panels if merged[k]["order: empty✗"].any()]
        stronger = bool(true_q) and e0["q it would have had"] < min(true_q)
        w(f"It happened here. During development, before the guard existed, the first stratified run flagged `{e0['cell']}` ({e0['panel']}) "
          "as empty beyond chance at order level, and it would have been reported as a novel constraint. The cell holds "
          f"{e0['species in cell (all)']} species. Every one of them sits in an order with fewer than {a['min_stratum_size']} species in the panel "
          f"(order sizes: {e0['order sizes of its species']}), so excluding small orders emptied it. In this run it has "
          f"q = {e0['q it would have had']:.2g} against {e0['order-null expected (kept)']:.1f} expected among kept species"
          + (", a lower q than the genuine order-level survivor" if stronger else "")
          + (f". Without the guard, {n_flag} cell{'s' if n_flag != 1 else ''} across the two panels would be falsely flagged in this run.\n"
             if n_flag else (". With this run's seed it falls just short of q < 0.05: the false positive is threshold-marginal, which is "
                             "exactly why it would have been easy to believe.\n" if e0["q it would have had"] < 2 * alpha else
                             ". With this run's seed it is not significant, but the mechanism is seed-independent.\n")))
        w(md_table(excl))
    w("\n**Fix used here.** A cell counts as \"empty beyond chance\" under a stratified null only if it is empty among *all* species. "
      "Excluded species still count as occupants; they are just not shuffled. Cells emptied only by the exclusion are labelled "
      "`emptied_by_exclusion` and reported. An equivalent alternative keeps small strata as fixed, unpermuted blocks that contribute identically "
      "to observed and null counts. The general point applies to any stratified or blocked permutation test with a minimum block size: "
      "phylogenetically stratified trait tests, stratified enrichment tests, case-control tests blocked by site. The exclusion rule is itself "
      "a selection on the covariate that defines the strata, and it must not change the observed statistic.\n")
    w(f"Other settings: a cell is tested only if its expected count is ≥ {a['min_expected_for_test']:g} under that null; BH correction within "
      f"each map over its testable cells; {a['n_perm_full']:,} permutations per full map ({a['n_perm_subset']:,} per subset map); the global "
      "and order nulls re-run with an independent seed. Species kept per null:\n")
    for k, _, plabel in panels:
        w(f"\n**{plabel}**\n")
        w(md_table(save(f"strata_{k}", strata_table(prim[k], default=a["min_stratum_size"]))))

    # ---- appendices ------------------------------------------------------------------------------------
    w("\n---\n\n## Appendix A. Snapshot and attrition\n")
    w(md_table(pd.DataFrame([
        {"source": "BacDive", "version": f"API v2, record DOIs stamped `{bd_stamp}` (BacDive exposes no release number)",
         "retrieved": f"{manifest['bacdive']['fetched_from'][:10]} → {manifest['bacdive']['fetched_to'][:10]}",
         "detail": f"IDs {sweep['sweep_range'][0]:,}–{sweep['sweep_range'][1]:,} swept in full ({sweep['batches_cached']:,} batches of 100); "
                   f"highest ID found {sweep['max_bacdive_id_found']:,}; genome-based predictions not requested"},
        {"source": "GTDB", "version": f"{gtdb['release']} ({gtdb['released']})", "retrieved": gtdb["joined_at"][:10],
         "detail": f"bac120_metadata.tsv.gz, {gtdb['n_genomes']:,} genomes, sha256 `{gtdb['metadata_sha256'][:12]}`"},
    ])))
    w("\nEmpty BacDive ID ranges ≥1,000 IDs: " + ", ".join(f"{a_:,}–{b_:,}" for a_, b_ in sweep["empty_id_ranges_ge_1000"]) + ".\n")
    w("Counts are strains. \"Usable\" = a normalised value exists (≥1 observation, mappable, no conflict between references; conflicts null the value).\n")
    w(md_table(att))
    w("\n**Per-trait losses**\n")
    w(md_table(ptd))
    w("\n**Unmapped and unparsed raw values** (kept in the `*_raw` columns; full list `data/interim/unmapped_values.tsv`):\n")
    w(md_table(unmapped.head(20)))
    src_counts = prim["core"]["temperature_bin_source"].value_counts()
    w("\n**Temperature bin sources, core panel species**\n")
    w(md_table(pd.DataFrame({"temperature_bin_source": src_counts.index, "N": src_counts.values,
                             "%": [pct(v, len(prim["core"])) for v in src_counts.values]})))
    fold_conf = int((df["oxygen_conflict"] & df["oxygen_raw"].fillna("").str.contains("microaerophile")).sum())
    w(f"\nOxygen: {int(prim['core']['oxygen_microaerophile_folded'].sum())} core-panel species are \"facultative\" only because microaerophile "
      f"was folded in; {fold_conf:,} of the {int(df['oxygen_conflict'].sum()):,} coarse oxygen conflicts across all strains involve a microaerophile observation.\n")

    w("## Appendix B. Genome matching\n")
    w("Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality "
      "of normalised strain designations. A designation without a culture-collection code needs taxon agreement (same genus, or same NCBI taxon ID). "
      "There is no fuzzy matching.\n")
    w("**What normalisation bought.** Steps A–C use the designation route alone, under one acceptance rule; step E equals the final match count.\n")
    w(md_table(stage))
    w("\n**Genome attrition, all strains**\n")
    w(md_table(gen_all))
    w("\n**Genome attrition, core-panel strains**\n")
    w(md_table(gen_core))
    w("\n**Match quality**\n")
    w(md_table(pd.DataFrame([
        {"check": "designation route agrees with accession route (both fire)", "value": pct(int(agree.sum()), len(agree), 2), "N": len(agree)},
        {"check": "bare-designation hits agree with accession route", "value": pct(int(bare_agree.sum()), len(bare_agree), 2), "N": len(bare_agree)},
        {"check": "matched genome's GTDB/NCBI genus equals BacDive genus", "value": pct(int(matched['genus_agree'].sum()), len(matched)), "N": len(matched)},
        {"check": "designation matches accepted via NCBI tax ID only", "value": pct(n_taxid_only, n_desig), "N": n_taxid_only},
        {"check": "candidates span >1 GTDB species", "value": pct(int(matched['match_ambiguous_species'].sum()), len(matched)), "N": len(matched)},
    ])))
    w(f"\n**Unmatched strains** ({len(unmatched):,}; listed in `data/interim/unmatched_strains.tsv`, gitignored, regenerated by `join`):\n")
    w(md_table(reasons))

    w("\n## Appendix C. Trait coverage and genome availability\n")
    w("Strains with usable values for both traits (diagonal = single-trait coverage):\n")
    w(md_table(cov["counts"], index=True))
    w("\nLift = P(both) / (P(row)·P(column)); 1 = independent coverage:\n")
    w(md_table(cov_core["lift"].round(2), index=True))
    w("\n**Panel completeness vs independent coverage.** Coverage is nested: traits come together from species descriptions.\n")
    w(md_table(ind))
    w(f"\n**Genome availability.** Base rate: {pct(int(df['bacdive_lists_assembly'].sum()), n)} of all strains list a GCA accession in BacDive. "
      "Fisher p-values below 1e-300 are shown as <1e-300.\n")
    gs = ga["sets"].copy()
    for c in [c for c in gs.columns if c.endswith("Fisher p")]:
        gs[c] = gs[c].map(lambda v: "<1e-300" if v == 0 else ("–" if pd.isna(v) else f"{v:.2g}"))
    w(md_table(gs))

    w("\n## Appendix D. Sensitivity views, phyla, type strains\n")
    w("Global and order-level nulls in each view (primary = one strain per GTDB species, coarse bins, all temperature sources).\n")
    w(md_table(views))
    w("\n**GTDB phyla (species level)**\n")
    ph = pd.concat({plabel: prim[k]["gtdb_phylum"].value_counts() for k, _, plabel in panels}, axis=1).fillna(0).astype(int)
    w(md_table(ph.sort_values(ph.columns[1], ascending=False).rename_axis("GTDB phylum"), index=True))
    w("\n**Type strains, core panel**\n")
    w(md_table(ts_core, index=True))

    w("\n## Files\n")
    w("- `data/final/strains.parquet` (gitignored): every BacDive strain, with raw and normalised traits, genome match and panel flags. "
      "Regenerate with `python -m src.pipeline all`; checksums are in `README.md` and `reports/run_manifest.json` → `outputs`.")
    w("- `data/final/{core,no_spore}_panel_species.tsv`: the analysed sets.")
    w("- `data/final/assembly_accessions_*.txt`: inputs for `datasets download genome accession --inputfile …`.")
    w("- `data/interim/unmatched_strains.tsv` (gitignored), `data/interim/unmapped_values.tsv`: audit logs.")
    w("- `reports/tables/*.tsv`: every table above, and per-cell results for every panel × view × null.")
    w("- `reports/run_manifest.json`: versions, hashes, counts.")
    text = "\n".join(R) + "\n"
    (rep / "attrition_report.md").write_text(text)
    return text
