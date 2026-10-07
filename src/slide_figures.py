"""Figures for the slide outline (`reports/slides_2026-10-09.md`), drawn from committed tables only.

    python -m src.slide_figures                 # writes reports/figures/*.png
    python -m src.slide_figures --lopo reports/tables/evo2_lopo_motility.json   # adds the LOPO figure

The data preparation is separate from the drawing so it can be tested without a display. Colours follow the dataviz
reference palette (light mode, validated): blue/orange/aqua/yellow for series in fixed order, a one-hue blue ramp for
ordered tiers. Text uses ink tokens, never the series colour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ("#1c5cab", "#5598e7", "#86b6ef")  # genome_seen, species_seen, unseen: dark -> light
NULL_ORDER = ("global", "phylum", "class", "order")
TIER_LABELS = {"genome_seen": "genome in training set", "species_seen": "same-species genome in training set",
               "unseen": "no genome of the species in training set"}


# ---------------------------------------------------------------------------
# Data preparation (no matplotlib)
# ---------------------------------------------------------------------------
def attrition_rows(path: str | Path) -> list[tuple[str, int]]:
    """The five bars of the attrition funnel, from reports/tables/attrition.tsv."""
    t = pd.read_csv(path, sep="\t")
    stage = lambda prefix: t[t["stage"].str.startswith(prefix)].iloc[0]  # noqa: E731
    core = stage("5. core panel (6) complete")
    return [("BacDive strains retrieved", int(stage("1. BacDive").strains)),
            ("at least one of the traits recorded", int(stage("2. ").strains)),
            ("all six core traits recorded", int(core.strains)),
            ("... with a matched GTDB genome", int(core["with GTDB genome"])),
            ("... one per GTDB species (analysed)", int(core["GTDB species"]))]


def parse_interval(s: str) -> tuple[float, float]:
    """'110–124' (en dash, as written in occupied_vs_expected.tsv) -> (110.0, 124.0)."""
    lo, hi = s.replace("–", "-").split("-")
    return float(lo), float(hi)


def occupancy_nulls(path: str | Path, map_name: str = "core panel (6 traits)") -> pd.DataFrame:
    """Observed vs expected occupied cells under each null for one map, in global -> order sequence."""
    t = pd.read_csv(path, sep="\t")
    t = t[t["map"] == map_name].set_index("null").loc[list(NULL_ORDER)].reset_index()
    iv = t["null 95% interval"].map(parse_interval)
    return pd.DataFrame({"null": t["null"], "observed": t["occupied"], "expected": t["expected occupied (null mean)"],
                         "lo": [a for a, _ in iv], "hi": [b for _, b in iv], "p": t["P(null ≤ observed)"],
                         "n_species": t["N used"]})


def tier_shares(path: str | Path) -> pd.DataFrame:
    """Share of each pretraining tier per phylum plus 'all', from opengenome2_overlap_by_phylum.tsv."""
    t = pd.read_csv(path, sep="\t")
    t = t.rename(columns={t.columns[0]: "group"})
    for tier in ("genome_seen", "species_seen", "unseen"):
        t[f"share_{tier}"] = t[tier] / t["n"]  # from the counts: the TSV shares are rounded to 4 decimals
    return t[["group", "n", "share_genome_seen", "share_species_seen", "share_unseen"]]


def overt_rows(path: str | Path) -> list[tuple[str, int]]:
    m = dict(pd.read_csv(path, sep="\t").itertuples(index=False))
    return [("species in oVert", int(m["distinct_species"])),
            ("... with any NCBI assembly", int(m["species_with_any_ncbi_assembly (name match)"])),
            ("... with an open volume and an assembly", int(m["species with an open volumetric image series and any NCBI assembly"])),
            ("... with an open mesh and an assembly", int(m["species with an open mesh and any NCBI assembly"]))]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _title(fig, title: str, sub: str | None = None) -> None:
    """Title and subtitle anchored to the figure's left edge, so a narrow axes cannot clip them."""
    fig.text(0.012, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.012, 0.905, sub, fontsize=9.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.87))


def _hbars(rows: list[tuple[str, int]], out: str | Path, title: str, sub: str, color: str = BLUE) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    ax.barh(labels, vals, height=0.5, color=color)
    top = max(vals)
    for i, v in enumerate(vals):
        ax.text(v + top * 0.01, i, f"{v:,}", va="center", fontsize=11, color=INK)  # value at the bar tip
    ax.set_xlim(0, top * 1.14)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    _title(fig, title, sub)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_attrition(tables: str | Path, out: str | Path) -> None:
    _hbars(attrition_rows(Path(tables) / "attrition.tsv"), out, "From 102,187 BacDive strains to 2,580 analysed species",
           "Strains retained at each step. BacDive API v2 snapshot 2026-09, GTDB v232.")


def fig_overt(tables: str | Path, out: str | Path) -> None:
    _hbars(overt_rows(Path(tables) / "overt_genome_coverage.tsv"), out, "oVert: few species have both a genome and an open 3D file",
           "Species counts from MorphoSource and NCBI metadata, 2026-10-07 (names matched).", color=BLUE)


def fig_occupancy(tables: str | Path, out: str | Path) -> None:
    plt = _plt()
    d = occupancy_nulls(Path(tables) / "occupied_vs_expected.tsv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(d))
    for i, r in d.iterrows():
        ax.plot([i, i], [r.lo, r.hi], color=BLUE, lw=2, solid_capstyle="round")
        ax.scatter([i], [r.expected], s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([i], [r.observed], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.annotate(f"observed {int(d.observed.iloc[-1])}", (3, d.observed.iloc[-1]), xytext=(12, -14), textcoords="offset points", fontsize=10, color=INK)
    ax.annotate(f"expected {d.expected.iloc[-1]:.0f}", (3, d.expected.iloc[-1]), xytext=(12, 4), textcoords="offset points", fontsize=10, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\nP = {p:.4f}" if p < 0.01 else f"{n}\nP = {p:.2f}" for n, p in zip(d["null"], d["p"])])
    ax.set_xlim(-0.4, 3.9)
    ax.set_ylim(70, 130)
    ax.set_ylabel("occupied cells (of 216)")
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("traits shuffled within ...")
    ax.scatter([], [], s=70, color=ORANGE, label="observed")
    ax.scatter([], [], s=70, color=BLUE, label="expected under the null, with 95% interval")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    _title(fig, "The empty-cell deficit shrinks as the null respects lineage",
           "Core map, six traits, 2,580 species. P = chance of this few occupied cells or fewer.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_pretraining(tables: str | Path, out: str | Path) -> None:
    plt = _plt()
    d = tier_shares(Path(tables) / "opengenome2_overlap_by_phylum.tsv")
    d = pd.concat([d[d["group"] == "all"], d[d["group"] != "all"].sort_values("share_unseen", ascending=False)]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = ("share_genome_seen", "share_species_seen", "share_unseen")
    left = [0.0] * len(d)
    gap = 0.004
    for c, color, tier in zip(cols, RAMP, TIER_LABELS):
        w = list(d[c])
        ax.barh(d["group"], [max(v - gap, 0) for v in w], left=[l + gap / 2 for l in left], height=0.52, color=color, label=TIER_LABELS[tier])
        for i, (l, v) in enumerate(zip(left, w)):
            if v >= 0.06:  # only label segments wide enough to hold the text
                ax.text(l + v / 2, i, f"{v:.0%}" if v >= 0.1 else f"{v:.1%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if tier == "genome_seen" else INK)
        left = [l + v for l, v in zip(left, w)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{g}  (n = {int(n):,})" for g, n in zip(d["group"], d["n"])])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=9, handlelength=1.2)
    _title(fig, "81% of the genomes we embed with Evo 2 were in its GTDB training data",
           "Embedded four-phylum genomes (2,435), tiered by overlap with Evo 2's GTDB training genomes.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


# display order, top to bottom; models absent from a result file are skipped
LOPO_MODELS = (("pfam", "Pfam (whole genome)"), ("pfam_windows", "Pfam (Evo 2 windows)"), ("evo2", "Evo 2"),
               ("kmer_genome", "k-mer (whole genome)"), ("kmer_windows", "k-mer (Evo 2 windows)"), ("gc", "GC only"))


def lopo_points(d: dict) -> list[dict]:
    """One row per (held-out phylum or macro mean, model) from an `evo2_lopo_<target>*.json` result."""
    present = set(d["models"])
    rows = []
    for k, _ in LOPO_MODELS:
        if k not in present:
            continue
        for h in d["phyla"]:
            m = d["per_phylum"][h]["models"][k]
            rows.append({"group": h, "model": k, "auc": m["auc"], "lo": m["ci"][0], "hi": m["ci"][1]})
        m = d["macro"]["models"][k]
        rows.append({"group": "macro mean", "model": k, "auc": m["auc"], "lo": m["ci"][0], "hi": m["ci"][1]})
    return rows


def fig_lopo(json_path: str | Path, out: str | Path, target: str = "motility") -> None:
    """Small multiples: one panel per held-out phylum plus the macro mean, one row per model (identity is the row label,
    so a single colour is enough). Descriptive: nothing here is a test."""
    import json

    plt = _plt()
    d = json.load(open(json_path))
    pts = pd.DataFrame(lopo_points(d))
    labels = dict(LOPO_MODELS)
    models = [k for k, _ in LOPO_MODELS if k in set(pts["model"])]
    groups = list(dict.fromkeys(pts["group"]))
    lo, hi = max(0.0, pts["lo"].min() - 0.03), min(1.0, pts["hi"].max() + 0.03)
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 4.6), sharey=True, sharex=True)
    for ax, g in zip(axes, groups):
        s = pts[pts["group"] == g].set_index("model").loc[models]
        ys = list(range(len(models)))[::-1]
        ax.hlines(ys, s["lo"], s["hi"], color=BLUE, lw=2)
        ax.scatter(s["auc"], ys, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axvline(0.5, color=INK2, lw=1)
        ax.set_xlim(lo, hi)
        ax.xaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(models) - 0.25)  # headroom so the top row's value label clears the panel title
        ax.set_title(g, fontsize=10.5, color=INK, loc="left", pad=14)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        for y, v in zip(ys, s["auc"]):
            ax.text(v, y + 0.28, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    axes[0].set_yticks(list(range(len(models)))[::-1])
    axes[0].set_yticklabels([labels[k] for k in models])
    fig.supxlabel("AUC inside the held-out phylum (0.5 = chance)", fontsize=10, color=INK2)
    _title(fig, f"Leave-one-phylum-out, {target}: AUC inside the held-out phylum",
           "Descriptive. Trained on the other three phyla; bars are 95% genus-cluster bootstrap intervals.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--lopo", help="evo2_lopo_<target>.json; adds fig_lopo.png when given")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_attrition(a.tables, out / "fig_attrition.png")
    fig_occupancy(a.tables, out / "fig_occupancy_nulls.png")
    fig_pretraining(a.tables, out / "fig_pretraining_overlap.png")
    if (Path(a.tables) / "overt_genome_coverage.tsv").exists():
        fig_overt(a.tables, out / "fig_overt_coverage.png")
    if a.lopo:
        fig_lopo(a.lopo, out / "fig_lopo.png")
    print("wrote", ", ".join(sorted(p.name for p in out.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
