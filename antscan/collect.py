"""Download and reduce the chosen Antscan scans, in parallel, resumably (one .npz per specimen).

    python -m antscan.collect --out /mnt/project-files/data/antscan/shapes --castes worker --per-taxon 2
"""
from __future__ import annotations

import argparse
import glob
import gzip
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import mesh, taxa

log = logging.getLogger("antscan.collect")
DATA = "/mnt/project-files/data/antscan"


def genecount_columns(path: str) -> list[str]:
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")[1:]


def selection(data: str, castes: tuple[str, ...], per_taxon: int) -> pd.DataFrame:
    cols = genecount_columns(f"{data}/gaga/Orthogroup_table_N0_genecount.tsv.gz")
    m = taxa.match(f"{data}/antscan_specimens_with_genome_flags.csv", f"{data}/gaga/GAGA_species_list.txt", cols)
    pick = taxa.pick_scans(m, castes, per_taxon)
    idx = pd.read_csv(f"{data}/biomedisa_index.csv")
    idx = idx[idx.thumb.str.startswith("/media/")].drop_duplicates("specimen_code")
    idx["stl"] = idx.thumb.str.replace(r"\.png$", ".stl", regex=True)
    return pick.merge(idx[["specimen_code", "page_id", "stl"]], on="specimen_code", how="inner")


def selection_all(data: str, castes: tuple[str, ...], per_taxon: int) -> pd.DataFrame:
    """Every Antscan taxon, genome or not: the shape prior for pretraining (no genome needed)."""
    m = taxa.antscan_table(f"{data}/antscan_specimens_with_genome_flags.csv")
    m = m[m.caste.isin(castes)].copy()
    m["_std"] = (m.voxel_um == 2.44).astype(int)
    m = m.sort_values(["taxon_code", "caste", "_std", "specimen_code"], ascending=[True, True, False, True])
    m = m.groupby(["taxon_code", "caste"], group_keys=False).head(per_taxon).drop(columns="_std")
    idx = pd.read_csv(f"{data}/biomedisa_index.csv")
    idx = idx[idx.thumb.str.startswith("/media/")].drop_duplicates("specimen_code")
    idx["stl"] = idx.thumb.str.replace(r"\.png$", ".stl", regex=True)
    return m.merge(idx[["specimen_code", "page_id", "stl"]], on="specimen_code", how="inner")


def _one(row: dict, out: str, tmp: str, require: str | None = None) -> tuple[str, str]:
    dest = f"{out}/{row['specimen_code']}.npz"
    if os.path.exists(dest) and (require is None or f"desc__{require}" in np.load(dest).files):
        return row["specimen_code"], "cached"
    r = mesh.fetch_and_process(row["stl"], row["voxel_um"], tmpdir=tmp)
    part = dest + ".part.npz"
    np.savez_compressed(part, points=r.points, solid32=r.solid32, core32=r.core32,
                        n_triangles=r.n_triangles, kept_area_frac=r.kept_area_frac,
                        volume_mm3=r.volume_mm3, area_mm2=r.area_mm2, voxel_um=row["voxel_um"],
                        stl_unit_mm=mesh.stl_unit_mm(row["voxel_um"]),
                        **{f"desc__{k}": v for k, v in r.desc.items()})
    os.replace(part, dest)
    return row["specimen_code"], "ok"


def fix_units(out: str) -> int:
    """Rescale records written before `stl_unit_mm` existed (6.11 um scans were scaled by 6.11 too many)."""
    n = 0
    for path in glob.glob(f"{out}/*.npz"):
        z = dict(np.load(path))
        if "stl_unit_mm" in z:
            continue
        unit = mesh.stl_unit_mm(float(z["voxel_um"]))
        f = unit / (float(z["voxel_um"]) / 1000.0)  # correct / used
        if not np.isclose(f, 1.0):
            z["points"] = (z["points"] * f).astype(np.float32)
            z["volume_mm3"] = z["volume_mm3"] * f**3
            z["area_mm2"] = z["area_mm2"] * f**2
            for k in ("desc__log_size", "desc__core_log_size"):
                z[k] = z[k] + np.log(f)
            n += 1
        z["stl_unit_mm"] = unit
        np.savez_compressed(path + ".part.npz", **z)
        os.replace(path + ".part.npz", path)
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=f"{DATA}/shapes")
    ap.add_argument("--castes", nargs="+", default=["worker"])
    ap.add_argument("--per-taxon", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--all-taxa", action="store_true", help="all Antscan taxa, not only those with a GAGA genome")
    ap.add_argument("--require", default=None, help="redo records lacking this descriptor (e.g. thickness)")
    ap.add_argument("--tmp", default=None, help="local scratch for the STL while it is processed")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    os.makedirs(a.out, exist_ok=True)
    log.info("rescaled %d older records", fix_units(a.out))
    if a.all_taxa:
        sel = selection_all(a.data, tuple(a.castes), a.per_taxon)
        sel.to_csv(f"{a.out}/selection_all_taxa.csv", index=False)
        log.info("%d scans selected (%d taxa)", len(sel), sel.taxon_code.nunique())
    else:
        sel = selection(a.data, tuple(a.castes), a.per_taxon)
        sel.to_csv(f"{a.out}/selection_{'_'.join(a.castes)}.csv", index=False)
        log.info("%d scans selected (%d genomes)", len(sel), sel.gaga_id.nunique())
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, r, a.out, a.tmp, a.require): r["specimen_code"] for r in sel.to_dict("records")}
        for f in as_completed(futs):
            done += 1
            try:
                code, status = f.result()
                log.info("[%d/%d] %s %s", done, len(sel), code, status)
            except Exception as e:
                log.error("[%d/%d] %s FAILED: %s", done, len(sel), futs[f], e)


if __name__ == "__main__":
    main()
