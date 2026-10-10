"""Segment every Keio phase-contrast image on Modal (CPU), one container per plate-row zip from the BioImage Archive.

    python -m keio_g2p.modal_segment <out_dir>

Writes <out_dir>/outlines/<plate>_<row>.npz holding, per well, an (n_cells, 64, 2) float16 array of canonical outlines.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import modal

BASES = {"151": "https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/S-BSSTxxx151/S-BSST151/Files",
         "192": "https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/192/S-BSST192/Files"}
app = modal.App("keio-segment")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl")
         .pip_install("numpy>=1.26", "scipy>=1.11", "scikit-image==0.25.2", "tifffile")
         .add_local_python_source("keio_g2p"))


@app.function(image=image, cpu=2, memory=4096, timeout=3600, retries=2)
def run_zip(study: str, name: str) -> tuple[str, bytes]:
    import re
    import subprocess
    import warnings
    import zipfile

    import numpy as np
    import tifffile

    from keio_g2p.segment import segment
    warnings.filterwarnings("ignore")
    subprocess.run(["curl", "-s", "-f", "--retry", "5", "-o", "/tmp/z.zip", f"{BASES[study]}/{name}"], check=True)
    z = zipfile.ZipFile("/tmp/z.zip")
    wells: dict[str, list] = {}
    for n in z.namelist():
        m = re.search(r"([A-H]\d{3})xy\d+c1\.tif$", n)
        if not m:
            continue
        try:
            wells.setdefault(m.group(1), []).extend(segment(tifffile.imread(io.BytesIO(z.read(n)))))
        except Exception:  # noqa: BLE001 - one unreadable image should not lose the row
            continue
    buf = io.BytesIO()
    np.savez_compressed(buf, **{w: np.asarray(c, dtype=np.float16).reshape(-1, 64, 2) for w, c in wells.items()})
    return name, buf.getvalue()


def zip_list() -> list[tuple[str, str]]:
    import json
    import urllib.request
    out = []
    for study in BASES:
        acc = f"S-BSST{study}"
        d = json.load(urllib.request.urlopen(f"https://www.ebi.ac.uk/biostudies/files/{acc}/{acc}.json"))
        names = []

        def walk(x):
            if isinstance(x, dict):
                if isinstance(x.get("path"), str) and x["path"].endswith(".zip"):
                    names.append(x["path"])
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(d)
        out += [(study, n) for n in sorted(set(names)) if "_c1_" in n]
    return out


def main(out: str) -> None:
    od = Path(out) / "outlines"
    od.mkdir(parents=True, exist_ok=True)
    todo = [(s, n) for s, n in zip_list() if not (od / n.replace("_c1_", "_").replace(".zip", ".npz")).exists()]
    print(f"{len(todo)} zips to segment", flush=True)
    with modal.enable_output(), app.run():
        for i, (name, blob) in enumerate(run_zip.starmap(todo, return_exceptions=True, order_outputs=False)):
            if isinstance(name, Exception):
                print("failed:", name, flush=True)
                continue
            (od / name.replace("_c1_", "_").replace(".zip", ".npz")).write_bytes(blob)
            if i % 10 == 0:
                print(f"{i + 1}/{len(todo)} {name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
