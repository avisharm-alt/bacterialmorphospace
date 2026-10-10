"""Real vs generated crops for held-out genes.

    python -m jump_g2p.imagediff.figure <sel_dir> [GENE ...]

Without genes, shows the held-out genes where the conditioned model beat the gene-free one by most (DINO energy
distance). False colour: C01 (DNA) blue, C02 green, C04 red; all panels share one scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rgb(t):
    x = t.astype(np.float32) / 255
    return np.clip(np.stack([x[3], x[1], x[0]], -1) * 1.3, 0, 1)


def main(sel: str, wanted: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel_p = Path(sel)
    genes = pd.read_csv(sel_p / "genes.csv")
    pg = pd.read_csv(sel_p / "results" / "per_gene.csv")
    if not wanted:
        pg["gain"] = pg["dino_ed_free"] - pg["dino_ed_cond"]
        wanted = pg.sort_values("gain", ascending=False)["symbol"].head(6).tolist()
    panels = []
    for f in sorted((sel_p / "folds").glob("fold*.npz")):
        z = np.load(f, allow_pickle=True)
        for j, g in enumerate(z["test_genes"]):
            sym = genes.at[int(g), "symbol"]
            if sym in wanted:
                real = z["examples_real"][z["examples_real_gene"] == g][:4]
                panels.append((sym, real, z["gen_cond"][j * 8:j * 8 + 4], z["gen_free"][j * 8:j * 8 + 4],
                               z["gen_shuffled"][j * 8:j * 8 + 4]))
    panels.sort(key=lambda p: wanted.index(p[0]))
    fig, ax = plt.subplots(len(panels), 16, figsize=(16 * 1.1, len(panels) * 1.25))
    ax = np.atleast_2d(ax)
    for r, (sym, *groups) in enumerate(panels):
        for gi, (name, tiles) in enumerate(zip(("real", "generated: gene", "gene-free", "other gene's code"), groups)):
            for c in range(4):
                a = ax[r, gi * 4 + c]
                if c < len(tiles):
                    a.imshow(rgb(tiles[c]))
                a.set_xticks([]); a.set_yticks([])
                if r == 0 and c == 0:
                    a.set_title(name, loc="left", fontsize=9)
        ax[r, 0].set_ylabel(sym, fontsize=9)
    plt.tight_layout(pad=0.2)
    out = sel_p / "results" / "examples.png"
    plt.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
