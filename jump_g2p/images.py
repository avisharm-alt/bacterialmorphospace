"""Fetch JUMP Cell Painting field images for a well from the public cellpainting-gallery bucket."""
from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"


@lru_cache(maxsize=1)
def plate_batches(raw: str) -> dict[str, str]:
    p = pd.read_csv(Path(raw) / "plate.csv.gz")
    p = p[p["Metadata_Source"] == "source_13"]
    return dict(zip(p["Metadata_Plate"], p["Metadata_Batch"]))


def _list(prefix: str) -> list[str]:
    url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
    x = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r"<Key>([^<]+)</Key>", x)


def field(raw: str, plate: str, well: str, site: int = 1) -> np.ndarray:
    """(5, H, W) float32 image stack, channels C01..C05, each scaled to its 0.5–99.8 percentiles."""
    import tifffile
    batch = plate_batches(raw)[plate]
    pre = f"cpg0016-jump/source_13/images/{batch}/images/{plate}/{plate}_{well}_T0001F{site:03d}"
    keys = sorted(k for k in _list(pre) if k.endswith(".tif"))
    chans = []
    for k in keys[:5]:
        a = tifffile.imread(io.BytesIO(urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(k)}", timeout=120).read()))
        a = a.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.8])
        chans.append(np.clip((a - lo) / (hi - lo + 1e-6), 0, 1))
    return np.stack(chans)


def composite(stack: np.ndarray) -> np.ndarray:
    """RGB view: C01 blue, C02 green, C04 red (a fixed false-colour choice for display only)."""
    rgb = np.stack([stack[3], stack[1], stack[0]], -1)
    return np.clip(rgb, 0, 1)


