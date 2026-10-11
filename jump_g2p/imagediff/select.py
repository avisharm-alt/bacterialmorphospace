"""Pick the gene subset, image folds and gene codes for the image diffusion model.

    python -m jump_g2p.imagediff.select <raw_dir> <homolog_pairs.tsv> <out_dir> [--n-genes=300] [--codes=network|chain|depmap]

Folds: the 10 chromosome-arm folds of jump_g2p.predictability, merged in pairs into 5 image folds, with every paralog
of a test gene purged from the training set. Gene codes are exactly those of the working profile model
(jump_g2p.diffusion_cv): cross-fitted DepMap ridge + STRING + co-essentiality predictions of the gene's mean profile
PCs, computed from all 7,968 imaged genes (not only the image subset), with the test genes' own data never used.
--codes=chain swaps in jump_g2p.diffusion_cv's sequence -> predicted DepMap codes (depmap: the same code built from the
measured DepMap profile); the gene subset, folds and manifest are unchanged.

Image subset: the most reproducible knockouts (split-half reliability), since most knockouts look like controls,
plus non-targeting control wells on the same plates (used only for plate intensity normalisation).
Writes manifest.csv (one row per well to fetch) and codes.npz (per fold: code of every gene, train/test masks).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import data
from ..diffusion_cv import K, chain_inputs, coessentiality, cross_fitted
from ..predictability import splits_for

N_IMG_FOLDS = 5
CTRL_PER_PLATE = 3
SEED = 0


def main(raw: str, homologs: str, out: str, n_genes: int = 300, codes_kind: str = "network") -> None:
    raw_p, out_p = Path(raw), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    jd = data.load(raw_p)
    g = jd.genes
    sym2i = {s: i for i, s in enumerate(g["symbol"])}
    w = jd.wells[jd.wells["Metadata_Symbol"].isin(sym2i)]
    W = w[jd.feats].to_numpy(dtype=np.float32)
    wg = w["Metadata_Symbol"].map(sym2i).to_numpy()
    if codes_kind in ("chain", "depmap"):
        inp = chain_inputs(g, raw_p, Path(homologs).parent, homologs)
        inp["chain"]["true_depmap"] = codes_kind == "depmap"
    else:
        D, has = data.depmap(raw_p, g["gene_id"].tolist())
        inp = {"depmap": D, "string": data.string_matrix(raw_p, g["symbol"].tolist()), "coess": coessentiality(D, has)}
    arms = g["arm"].replace("", "unknown").to_numpy()
    splits = splits_for(g, pd.read_csv(homologs, sep="\t", header=None))

    # gene subset: most reproducible knockouts with >= 4 wells
    cand = g[(g["n_wells"] >= 4)].sort_values("reliability", ascending=False)
    chosen = np.sort(cand.index[:n_genes].to_numpy())
    print(f"chosen {len(chosen)} genes, reliability >= {g.loc[chosen, 'reliability'].min():.3f}", flush=True)

    codes, test_mask, train_mask = [], [], []
    for k in range(N_IMG_FOLDS):
        te = np.concatenate([splits[2 * k][1], splits[2 * k + 1][1]])
        tr = np.intersect1d(splits[2 * k][0], splits[2 * k + 1][0])   # purges paralogs of both halves
        tr = np.setdiff1d(tr, te)
        trw = np.isin(wg, tr)
        pca = PCA(K, whiten=True, random_state=SEED).fit(W[trw])
        X = pca.transform(W).astype(np.float32)
        T = np.zeros((len(g), K), dtype=np.float32)
        np.add.at(T, wg, X)
        T /= np.bincount(wg, minlength=len(g))[:, None]
        code = cross_fitted(inp, T, tr, te, arms)
        r2 = 1 - ((T[te] - code[te]) ** 2).sum(0) / ((T[te] - T[te].mean(0)) ** 2).sum(0)
        print(f"fold {k}: {len(te)} test genes, {np.isin(chosen, te).sum()} in image subset; code R² PC1-3 "
              f"{np.round(r2[:3], 3)}", flush=True)
        codes.append(code); test_mask.append(np.isin(np.arange(len(g)), te)); train_mask.append(np.isin(np.arange(len(g)), tr))

    sel = w[w["Metadata_Symbol"].isin(set(g.loc[chosen, "symbol"]))]
    plates = sel["Metadata_Plate"].unique()
    p, _ = data.load_profiles(raw_p)
    ctrl = p[(p["Metadata_Symbol"] == "non-targeting") & p["Metadata_Plate"].isin(plates)]
    ctrl = ctrl.sample(frac=1, random_state=SEED).groupby("Metadata_Plate").head(CTRL_PER_PLATE)
    man = pd.concat([
        pd.DataFrame({"plate": sel["Metadata_Plate"], "well": sel["Metadata_Well"], "symbol": sel["Metadata_Symbol"],
                      "gene_index": sel["Metadata_Symbol"].map(sym2i)}),
        pd.DataFrame({"plate": ctrl["Metadata_Plate"], "well": ctrl["Metadata_Well"], "symbol": "non-targeting",
                      "gene_index": -1}),
    ]).reset_index(drop=True)
    man.to_csv(out_p / "manifest.csv", index=False)
    g.assign(in_subset=g.index.isin(chosen)).to_csv(out_p / "genes.csv", index=False)
    np.savez_compressed(out_p / "codes.npz", codes=np.stack(codes), test=np.stack(test_mask), train=np.stack(train_mask),
                        chosen=chosen)
    info = {"n_genes": int(len(chosen)), "n_gene_wells": int(len(sel)), "n_control_wells": int(len(ctrl)),
            "n_plates": int(len(plates)), "K": K, "folds": N_IMG_FOLDS, "codes": codes_kind}
    (out_p / "select_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    n = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--n-genes=")), 300)
    kind = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--codes=")), "network")
    main(*a[:3], n_genes=n, codes_kind=kind)
