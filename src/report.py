"""Attrition + occupancy report.

Writes
  reports/attrition_report.md          the deliverable
  reports/tables/*.tsv                 every table behind the report, machine-readable
  reports/run_manifest.json            versions, hashes, counts - what the report was produced from
  data/final/assembly_accessions_*.txt accession lists for `datasets download genome accession --inputfile`
  data/final/core_panel_species.tsv    the primary analysis set, compact
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import analysis as A
from .config import Config
from .join import match_stage_table

log = logging.getLogger(__name__)

CORE_LABEL = "core panel (6)"
MORPH_LABEL = "morphology panel (4)"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def pct(k, n, digits=1) -> str:
    return "–" if not n else f"{100 * k / n:.{digits}f}%"


def fmt(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "–"
        if float(v).is_integer() and abs(v) >= 1:
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if v == 0:
            return "0"
        return f"{v:.{digits}g}"
    return "" if v is None else str(v)


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


# ---------------------------------------------------------------------------
# sections
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
                "ph": "outside the core panel"}.get(t, "")
        rows.append(row(f"3. {A.TRAIT_LABEL[t]} — usable value", df[f"{t}_usable"], note))
    rows.append(row("3. (NaCl growth tests present, raw column only)", df["nacl_tests_populated"],
                    "not binned; kept in nacl_tests_raw"))
    rows += [
        row(f"4. {MORPH_LABEL} complete", df["morph4_complete"], " + ".join(cfg.morphology)),
        row(f"5. {CORE_LABEL} complete", df["core6_complete"], " + ".join(cfg.core)),
        row(f"5a. {CORE_LABEL}, temperature from optimum only", df["core6_complete_optimum_temp"]),
        row("5b. original 8-trait panel (core + pH + curated halophily)", df["all8_reference_complete"], "for reference; not analysed"),
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
    rows = [("strains in set", n),
            ("BacDive lists ≥1 GCA assembly", int(d["bacdive_lists_assembly"].sum())),
            ("… of which ≥1 listed assembly is in GTDB (others: failed GTDB QC or absent)", int(d["bacdive_assembly_in_gtdb"].sum())),
            ("… listed assembly not in GTDB", int((d["bacdive_lists_assembly"] & ~d["bacdive_assembly_in_gtdb"]).sum())),
            ("matched via direct accession", int((d["match_method"] == "accession").sum())),
            ("matched via designation — collection code (e.g. DSM 20231)", int(((d["match_method"] == "designation") & (d["match_token_kind"] == "collection")).sum())),
            ("matched via designation — bare designation + genus agreement", int(((d["match_method"] == "designation") & (d["match_token_kind"] == "bare")).sum())),
            ("matched, total", int(d["genome_matched"].sum())),
            ("unmatched", int((~d["genome_matched"]).sum()))]
    return pd.DataFrame([{"": k, "strains": v, "%": pct(v, n)} for k, v in rows])


def phylum_table(sub: pd.DataFrame) -> pd.DataFrame:
    c = sub["gtdb_phylum"].value_counts()
    return pd.DataFrame({"GTDB phylum": c.index, "N": c.values, "%": [pct(v, len(sub)) for v in c.values]})


def type_strain_table(sub: pd.DataFrame) -> pd.DataFrame:
    b = sub["type_strain_bacdive"].fillna(False).astype(bool)
    g = sub["gtdb_is_type_strain_of_species"].astype(bool)
    n = len(sub)
    rows = [("type strain per BacDive", int(b.sum())), ("type strain of species per GTDB/NCBI", int(g.sum())),
            ("either", int((b | g).sum())), ("both", int((b & g).sum())),
            ("BacDive says type, GTDB genome not flagged as type", int((b & ~g).sum()))]
    return pd.DataFrame([{"": k, "N": v, "%": pct(v, n)} for k, v in rows])


def crosstab_grid(cells: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """24 x 9 rendering of the 216-cell core map: rows gram/shape/motility/spore, cols oxygen/temperature."""
    rt, ct = ["gram", "shape", "motility", "spore"], ["oxygen", "temperature"]

    def mark(r):
        if r.observed > 0:
            return f"{int(r.observed)}{' ↓' if r.emptier_than_chance else ''}"
        if r.empty_beyond_chance:
            return "**0✗**"
        return "0" if r.testable else "·"

    c = cells.copy()
    c["mark"] = [mark(r) for r in c.itertuples()]
    c["row"] = c[rt].astype(str).agg(" / ".join, axis=1)
    c["col"] = c[ct].astype(str).agg(" / ".join, axis=1)
    order_r = [" / ".join(x) for x in pd.MultiIndex.from_product([A.levels_for(cfg, t, "coarse") for t in rt])]
    order_c = [" / ".join(x) for x in pd.MultiIndex.from_product([A.levels_for(cfg, t, "coarse") for t in ct])]
    grid = c.pivot(index="row", columns="col", values="mark").reindex(index=order_r, columns=order_c)
    grid.index.name = " / ".join(rt) + " ↓   " + " / ".join(ct) + " →"
    return grid


def capacity_summary(cap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in cap.groupby("size"):
        rows.append({"traits in panel": k, "panels": len(g), "cells": f"{g.cells.min()}–{g.cells.max()}",
                     "N (species) range": f"{g.N.min():,}–{g.N.max():,}",
                     "N per cell, median over panels": g.N_per_cell.median(),
                     "median expected count in a cell": g.expected_median.median(),
                     "smallest expected count": g.expected_min.min(),
                     "testable cells (expected ≥ threshold), median over panels": f"{100 * g.testable_cell_frac.median():.0f}%"})
    return pd.DataFrame(rows)


VIEWS = [
    # name, binning, level, temperature source, description
    ("primary", "coarse", "species", "all", "one strain per GTDB species · coarse bins · all temperature sources"),
    ("strain-level", "coarse", "strain", "all", "no dereplication (sensitivity)"),
    ("optimum-only", "coarse", "species", "optimum", "temperature bin from a reported optimum only"),
    ("fine bins", "fine", "species", "all", "fine binning (shape 5, oxygen 4, temperature 4)"),
    ("genus-level", "coarse", "genus", "all", "one strain per GTDB genus (extra sensitivity for congener clustering)"),
]


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
    core, morph = cfg.core, cfg.morphology

    def save(name, t, index=False):
        t.to_csv(tables / f"{name}.tsv", sep="\t", index=index)
        return t

    # --- counts --------------------------------------------------------------------------------
    att = save("attrition", attrition_table(df, cfg))
    ptd = save("per_trait_detail", per_trait_detail(df))
    gen_all = genome_attrition(df, pd.Series(True, index=df.index))
    gen_core = genome_attrition(df, df["core6_complete"])
    masks = {"all strains": pd.Series(True, index=df.index), "morph. panel": df["morph4_complete"], "core panel": df["core6_complete"]}
    stage = save("match_rate_by_normalisation_step", match_stage_table(df, masks))
    reasons = unmatched["reason"].value_counts().rename_axis("reason").reset_index(name="strains")
    save("unmatched_reasons", reasons)
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
        "morphology (4)": morph, "core (6)": core,
        "core + pH (7)": core + ["ph"], "original 8": core + ["ph", "halophily_level"]}))
    ga = A.genome_availability(df, cfg)
    for k, v in ga.items():
        save(f"genome_availability_{k}", v)

    cap = save("power_expected_per_cell", A.capacity_table(df, cfg))
    capsum = capacity_summary(cap)

    prim = A.analysis_set(df, core, "coarse", "species", "all")
    prim_strain = A.analysis_set(df, core, "coarse", "strain", "all")
    phy = save("core_panel_species_phyla", phylum_table(prim))
    phy_strain = phylum_table(prim_strain)
    ts = type_strain_table(prim)
    src_counts = prim["temperature_bin_source"].value_counts()
    folded = int(prim["oxygen_microaerophile_folded"].sum())
    fold_conf_only = int((df["oxygen_conflict"] & df["oxygen_raw"].fillna("").str.contains("microaerophile")).sum())

    # --- occupancy ------------------------------------------------------------------------------
    grids, full_cells = {}, {}
    for j, (name, binning, level, tsrc, _) in enumerate(VIEWS):
        log.info("occupancy view %s", name)
        sub = A.analysis_set(df, core, binning, level, tsrc)
        occ = A.occupancy(sub, core, cfg, binning, a["n_perm_full"], np.random.default_rng(a["seed"] + 7 * j), keep_cells=True)
        full_cells[name] = occ
        grids[name] = A.occupancy_grid(df, cfg, binning, level, tsrc, seed_offset=1000 * j, full=occ)
        save(f"occupancy_subsets_{name.replace(' ', '_')}", grids[name])
        save(f"full_map_cells_{name.replace(' ', '_')}", occ.cell_table)

    # Monte Carlo stability: re-run the primary full-map null with an independent seed
    prim_sub = A.analysis_set(df, core, "coarse", "species", "all")
    rerun = A.occupancy(prim_sub, core, cfg, "coarse", a["n_perm_full"], np.random.default_rng(a["seed"] + 99991), keep_cells=True)
    pct_ = full_cells["primary"].cell_table
    pct_["empty_beyond_chance_seed2"] = rerun.cell_table["empty_beyond_chance"].values
    pct_["q_low_seed2"] = rerun.cell_table["q_low"].values
    save("full_map_cells_primary", pct_)

    pc = full_cells["primary"].cell_table
    comp = pc[core + ["observed", "expected", "p_low", "q_low", "q_low_seed2", "testable", "empty_beyond_chance",
                      "empty_beyond_chance_seed2", "emptier_than_chance"]].copy()
    for name in ("strain-level", "optimum-only", "genus-level"):
        o = full_cells[name].cell_table.set_index(core)
        comp = comp.join(o[["observed", "expected", "q_low", "empty_beyond_chance", "testable"]]
                         .rename(columns=lambda c: f"{name}: {c}"), on=core)
    save("full_map_views_compared", comp)

    views_summary = pd.DataFrame([{"view": name, "description": desc, "N": full_cells[name].summary["N"],
                                   "cells": full_cells[name].summary["cells"],
                                   "occupied": full_cells[name].summary["occupied"],
                                   "occupied (null mean)": full_cells[name].summary["null_occupied_mean"],
                                   "testable cells": full_cells[name].summary["testable_cells"],
                                   "empty & testable": full_cells[name].summary["empty_testable_cells"],
                                   "empty beyond chance (q<α)": full_cells[name].summary["empty_beyond_chance_cells"],
                                   "occupied but depleted (q<α)": full_cells[name].summary["depleted_occupied_cells"]}
                                  for name, _, _, _, desc in VIEWS])
    save("full_map_views_summary", views_summary)

    # --- final deliverables -----------------------------------------------------------------------
    keep = ["bacdive_id", "species", "gtdb_species", "gtdb_genus", "gtdb_phylum", "gtdb_accession", "ncbi_assembly_accession",
            "assembly_genbank", "assembly_refseq", "match_method", "type_strain_bacdive", "gtdb_is_type_strain_of_species"] + \
        [c for t in core for c in (t, f"{t}_fine")] + ["temperature_bin_source", "temperature_point", "oxygen_microaerophile_folded",
                                                      "ph", "ph_bin_source", "halophily_level"]
    prim[keep].to_csv(final / "core_panel_species.tsv", sep="\t", index=False)
    (final / "assembly_accessions_core_panel_species.txt").write_text("\n".join(prim["ncbi_assembly_accession"].astype(str)) + "\n")
    (final / "assembly_accessions_all_matched.txt").write_text(
        "\n".join(matched["ncbi_assembly_accession"].dropna().astype(str).drop_duplicates()) + "\n")

    # --- manifest ----------------------------------------------------------------------------------
    doi_dates = df["doi_date"].value_counts()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(cfg), "config_sha256": cfg.text_sha256,
        "bacdive": {"api": cfg["bacdive"]["base_url"], "api_version": "v2",
                    "release_stamp_from_record_dois": doi_dates.index[0] if len(doi_dates) else None,
                    "doi_date_counts": {str(k): int(v) for k, v in doi_dates.items()},
                    "fetched_from": df["fetched_at"].min(), "fetched_to": df["fetched_at"].max(),
                    "predictions_requested": False, **sweep},
        "gtdb": gtdb,
        "counts": {"strains": n, "matched": int(df["genome_matched"].sum()), "core6_complete": int(df["core6_complete"].sum()),
                   "core6_complete_with_genome": int((df["core6_complete"] & df["genome_matched"]).sum()),
                   "core6_species": len(prim)},
        "analysis": dict(a),
    }
    (rep / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    # --- markdown -----------------------------------------------------------------------------------
    R = []
    w = R.append
    bd_stamp = manifest["bacdive"]["release_stamp_from_record_dois"]
    n_core = int(df["core6_complete"].sum())
    n_core_g = int((df["core6_complete"] & df["genome_matched"]).sum())
    ps = full_cells["primary"].summary
    n_stable = int((pc["empty_beyond_chance"] & pc["empty_beyond_chance_seed2"]).sum())
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

    w("## Headline\n")
    w(f"- **{n:,}** BacDive strains; **{n_core:,}** ({pct(n_core, n)}) have the full 6-trait core panel; "
      f"**{n_core_g:,}** of those have a GTDB genome, collapsing to **{len(prim):,} GTDB species** — the primary analysis set.")
    w(f"- The original 8-trait panel (adding pH and curated halophily) is complete for **{int(df['all8_reference_complete'].sum()):,}** strains, "
      f"**{int((df['all8_reference_complete'] & df['genome_matched']).sum()):,}** with a genome.")
    w(f"- Primary map: {ps['cells']} cells, {ps['occupied']} occupied (null expectation {ps['null_occupied_mean']:.1f}); "
      f"{ps['testable_cells']} cells are testable (expected ≥ {a['min_expected_for_test']:g}); "
      f"**{ps['empty_beyond_chance_cells']} empty cells are emptier than the marginals predict** (BH q < {a['fdr_alpha']}); "
      f"{n_stable} of them are flagged under two independent permutation seeds. "
      f"{ps['depleted_occupied_cells']} occupied cells are significantly depleted.")
    w(f"- Coverage is strongly nested: the 6-trait panel is complete {ind.set_index('panel').loc['core (6)', 'ratio']:,.0f}× more often than "
      "independent coverage would give. Genome availability is also enriched among well-phenotyped strains (§6).\n")

    w("## 1. Attrition\n")
    w("Counts are strains. \"Usable\" means a normalised value exists: at least one observation, mappable to a bin, "
      "and no conflict between references (conflicting observations null the value rather than picking one).\n")
    w(md_table(att))
    w("\n### 1a. Per-trait losses\n")
    w(md_table(ptd))
    w("\nUnmapped and unparsed raw values (all kept in the `*_raw` columns; full list in `data/interim/unmapped_values.tsv`):\n")
    w(md_table(unmapped.head(20)))

    w(f"\n### 1b. Temperature bin sources, {CORE_LABEL} × genome × species\n")
    w(md_table(pd.DataFrame({"temperature_bin_source": src_counts.index, "N": src_counts.values,
                             "%": [pct(v, len(prim)) for v in src_counts.values]})))
    w(f"\nOxygen: {folded} of these species are \"facultative\" only because microaerophile was folded in "
      f"(`oxygen_microaerophile_folded`). Across all strains, {fold_conf_only:,} of the {int(df['oxygen_conflict'].sum()):,} "
      "coarse oxygen conflicts involve a microaerophile observation (e.g. anaerobe + microaerophile, which conflicts once "
      "microaerophile is read as facultative).\n")

    w("## 2. Genome matching\n")
    w("Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality of "
      "normalised strain designations. A designation with no culture-collection code (e.g. `BS 107`) is accepted only with **taxon agreement**: "
      "the BacDive genus equals the GTDB or NCBI genus of the genome, or one of BacDive's NCBI tax IDs equals the genome's `ncbi_taxid` or "
      "`ncbi_species_taxid`. The tax-ID criterion recovers genus renames. There is no fuzzy matching anywhere.\n")
    w("### 2a. What normalisation bought\n")
    w("Strains with ≥1 accepted GTDB genome. Steps A–C use the designation route alone and apply the same acceptance rule, "
      "so they differ only in how strings are compared. Step E equals the final match count.\n")
    w(md_table(stage))
    w("\nThe last row shows what a naive raw-token join would report. The gap between it and step B is cross-genus collisions "
      "(`LP1`, `B86`, `M120` recur across unrelated genera), which the taxon rule rejects.\n")
    w("### 2b. Genome attrition, all strains\n")
    w(md_table(gen_all))
    w(f"\n### 2c. Genome attrition, {CORE_LABEL}-complete strains\n")
    w(md_table(gen_core))
    w("\n### 2d. Match quality checks\n")
    w(md_table(pd.DataFrame([
        {"check": "designation route agrees with accession route (strains where both fire)", "value": pct(int(agree.sum()), len(agree), 2), "N": len(agree)},
        {"check": "matched genome's GTDB/NCBI genus equals BacDive genus", "value": pct(int(matched['genus_agree'].sum()), len(matched)), "N": len(matched)},
        {"check": "… accession-route matches", "value": pct(int(matched.loc[matched.match_method == 'accession', 'genus_agree'].sum()), int((matched.match_method == 'accession').sum())), "N": int((matched.match_method == 'accession').sum())},
        {"check": "bare-designation hits agree with accession route (only bare keys hit, accession also present)",
         "value": pct(int(bare_agree.sum()), len(bare_agree), 2), "N": len(bare_agree)},
        {"check": "designation matches accepted via NCBI tax ID only (genus differs)",
         "value": pct(n_taxid_only, n_desig), "N": n_taxid_only},
        {"check": "candidates span >1 GTDB species (ambiguous)", "value": pct(int(matched['match_ambiguous_species'].sum()), len(matched)), "N": len(matched)},
    ])))
    w("\nGenus disagreement on accession-route matches mostly reflects GTDB reclassification (e.g. split genera) rather than wrong matches; "
      "those strains are kept.\n")
    w("### 2e. Unmatched strains\n")
    w(f"All {len(unmatched):,} are listed with the keys tried in `data/interim/unmatched_strains.tsv`.\n")
    w(md_table(reasons))

    w("\n## 3. Trait coverage co-occurrence\n")
    w("Strains with usable values for both traits (diagonal = single-trait coverage), over all BacDive strains.\n")
    w(md_table(cov["counts"], index=True))
    w("\nP(column usable | row usable):\n")
    w(md_table(cov_core["conditional"].round(3), index=True))
    w("\nLift = P(both) / (P(row)·P(column)); 1 = independent coverage:\n")
    w(md_table(cov_core["lift"].round(2), index=True))
    w("\n### 3a. Panel completeness vs independent coverage\n")
    w(md_table(ind))
    w("\nCoverage is nested rather than independent: strains described with one morphology trait usually have the others, "
      "because the traits come together from species descriptions. Spore formation is the binding constraint.\n")

    w("## 4. Power: expected strains per cell, before reading results\n")
    w(f"Species-level, coarse bins. \"Testable\" means expected count ≥ {a['min_expected_for_test']:g} under independence. "
      f"Below that an empty cell carries no information (at expected = 1 the null itself is empty {100 * np.exp(-1):.0f}% of the time).\n")
    w(md_table(capsum))
    w("\nPer-panel detail: `reports/tables/power_expected_per_cell.tsv`.\n")

    w(f"## 5. The {CORE_LABEL} + genome set\n")
    w(f"{len(prim):,} GTDB species (from {len(prim_strain):,} strains).\n")
    w("### 5a. GTDB phyla (species level)\n")
    w(md_table(phy))
    w("\nStrain level: " + ", ".join(f"{r['GTDB phylum']} {r['N']}" for _, r in phy_strain.iterrows()) + ".\n")
    w("### 5b. Type strains (species level)\n")
    w(md_table(ts))

    w("\n## 6. Does genome availability track trait coverage?\n")
    w(f"Base rate over all {n:,} strains: {pct(int(df['bacdive_lists_assembly'].sum()), n)} list a GCA accession in BacDive. "
      "The ~24% quoted during planning came from a 2,011-strain sample. Fisher p-values below 1e-300 underflow and are shown as <1e-300.\n")
    gs = ga["sets"].copy()
    for c in [c for c in gs.columns if c.endswith("Fisher p")]:
        gs[c] = gs[c].map(lambda v: "<1e-300" if v == 0 else ("–" if pd.isna(v) else f"{v:.2g}"))
    w(md_table(gs))
    w("\nBy number of usable core traits:\n")
    w(md_table(ga["by_n_traits"]))

    w("\n## 7. Occupancy of trait subsets\n")
    w(f"Each row is a trait subset, analysed on the species with a genome that are complete for that subset. "
      f"Every trait column is permuted independently ({a['n_perm_subset']:,} permutations; {a['n_perm_full']:,} for the full panel), "
      "which preserves the marginals exactly. \"Occupied (null)\" is the mean number of occupied cells under that null, "
      "so occupancy fractions can be read against sparsity. \"Empty beyond chance\" counts cells with 0 observed that are testable "
      f"and have BH q < {a['fdr_alpha']}.\n")
    cols = ["panel", "traits", "N", "cells", "mean_per_cell", "occupied", "occupied_frac", "null_occupied_mean", "null_occupied_frac",
            "testable_cells", "empty_testable_cells", "empty_beyond_chance_cells", "depleted_occupied_cells"]
    ren = {"mean_per_cell": "N/cell", "occupied_frac": "occ. frac", "null_occupied_mean": "occupied (null)", "null_occupied_frac": "occ. frac (null)",
           "testable_cells": "testable", "empty_testable_cells": "empty & testable", "empty_beyond_chance_cells": "empty beyond chance",
           "depleted_occupied_cells": "occupied, depleted"}
    for g_ in grids.values():
        for c in ("mean_per_cell", "occupied_frac", "null_occupied_mean", "null_occupied_frac"):
            g_[c] = g_[c].round(2 if "frac" in c else 1)
    g0 = grids["primary"]
    w("### 7a. Morphology map (primary view)\n")
    w(md_table(g0[g0["morphology_map"]][cols].rename(columns=ren)))
    w("\n### 7b. All 4- and 5-trait subsets and the full panel (primary view)\n")
    w(md_table(g0[cols].rename(columns=ren)))
    w("\n### 7c. Same, other views: empty-beyond-chance cells per subset\n")
    cmp_ = pd.DataFrame({"traits": g0["traits"]})
    for name, *_ in VIEWS:
        g = grids[name]
        cmp_[f"{name} N"] = g["N"].values
        cmp_[f"{name} empty✗"] = g["empty_beyond_chance_cells"].values
    w(md_table(cmp_))

    w("\n## 8. Full 6-trait map\n")
    w(md_table(views_summary))
    w(f"\n### 8a. Cross-tabulation, primary view (N = {ps['N']:,} species, {ps['cells']} cells)\n")
    w("Cell = species count. **0✗** = empty and emptier than chance (testable, q < α). `0` = empty, testable, not significant. "
      "`·` = empty and untestable (expected < threshold, so emptiness is uninformative). ↓ = occupied but significantly depleted.\n")
    w(md_table(crosstab_grid(pc, cfg), index=True))
    eb = pc[pc["empty_beyond_chance"]].sort_values("expected", ascending=False)
    w("\n### 8b. Cells empty beyond what the marginals predict (primary view)\n")
    if len(eb):
        c2 = comp[comp["empty_beyond_chance"]].sort_values("expected", ascending=False)
        show = c2[core + ["expected", "q_low", "q_low_seed2", "strain-level: observed", "strain-level: empty_beyond_chance",
                          "optimum-only: observed", "optimum-only: expected", "optimum-only: empty_beyond_chance",
                          "genus-level: expected", "genus-level: empty_beyond_chance"]]
        w(md_table(show))
    else:
        w("None.\n")
    dep = pc[pc["emptier_than_chance"] & (pc["observed"] > 0)].sort_values("q_low")
    w("\n### 8c. Occupied but significantly depleted cells (primary view)\n")
    w(md_table(dep[core + ["observed", "expected", "q_low"]]) if len(dep) else "None.\n")
    w("\nBecause the marginals are fixed, a structural gap forces compensating depletion in partner cells, so depleted cells "
      "need not be constraints themselves.\n")
    w("\n### 8d. Does strictness change the map?\n")
    both = comp[comp["empty_beyond_chance"] | comp["optimum-only: empty_beyond_chance"].fillna(False).astype(bool)]
    w(f"Cells flagged in the primary view: {int(comp['empty_beyond_chance'].sum())}; in the optimum-only view: "
      f"{int(comp['optimum-only: empty_beyond_chance'].fillna(False).astype(bool).sum())}; in both: "
      f"{int((comp['empty_beyond_chance'] & comp['optimum-only: empty_beyond_chance'].fillna(False).astype(bool)).sum())}. "
      f"The optimum-only view has N = {full_cells['optimum-only'].summary['N']:,} and {full_cells['optimum-only'].summary['testable_cells']} testable cells "
      f"(primary: {ps['testable_cells']}), so a cell dropping out of the optimum-only list is as likely to be lost power as a changed answer. "
      "`reports/tables/full_map_views_compared.tsv` lists every cell in every view.\n")
    if len(both):
        w(md_table(both[core + ["observed", "expected", "empty_beyond_chance", "optimum-only: observed", "optimum-only: expected",
                                "optimum-only: testable", "optimum-only: empty_beyond_chance"]]))

    w("\n## 9. Caveats\n")
    w("- **Exchangeability.** The permutation null treats species as exchangeable. They are not: congeners share traits, "
      "and BacDive over-samples culturable, clinically and industrially relevant lineages. A cell emptier than the marginals predict "
      "can reflect phylogenetic clustering or sampling as much as a biological constraint. The genus-level view is a crude check, not a fix.")
    w("- **Selection.** Complete panels come almost entirely from type-strain species descriptions (§5b). The map describes described "
      "type strains, not bacterial diversity at large.")
    w(f"- **Temperature source.** Across all of BacDive most temperature values are single cultivation temperatures, not optima "
      f"({int((df.temperature_bin_source == 'growth_single_point').sum()):,} single-point vs {int((df.temperature_bin_source == 'optimum').sum()):,} optimum). "
      f"In the analysed core-panel set, however, {pct(int(src_counts.get('optimum', 0)), len(prim))} of temperature bins come from an optimum "
      f"and only {pct(int(src_counts.get('growth_single_point', 0)), len(prim))} from a single point (§1b). A single cultivation temperature "
      "is usually a routine 28–37 °C, which bins as mesophile by construction. The optimum-only view (§8d) removes these.")
    w("- **Null power.** With 216 cells and N in the low thousands, many cells are untestable (§4). An empty untestable cell is not evidence of anything.")
    w("- **Multiple testing.** BH is applied within each map over its testable cells only. The subset grids are not jointly corrected.")
    w(f"- **Resolution.** Permutation p-values cannot go below 1/(n_perm+1) = {1 / (a['n_perm_full'] + 1):.1e} for the full map.")
    w("\n## Files\n")
    w("- `data/final/strains.parquet`: every BacDive strain, with raw and normalised traits, bin sources, conflicts, the genome match and panel flags.")
    w("- `data/final/core_panel_species.tsv`: the primary analysis set.")
    w("- `data/final/assembly_accessions_core_panel_species.txt`: accessions for `datasets download genome accession --inputfile …` "
      "(RefSeq GCF where GTDB uses RefSeq, else GenBank GCA). `assembly_accessions_all_matched.txt` holds all matched strains.")
    w("- `data/interim/unmatched_strains.tsv`, `data/interim/unmapped_values.tsv`: audit logs.")
    w("- `reports/tables/*.tsv`: every table above, plus full per-cell results for each view.")
    w("- `reports/run_manifest.json`: versions, hashes and counts.")
    text = "\n".join(R) + "\n"
    (rep / "attrition_report.md").write_text(text)
    return text
