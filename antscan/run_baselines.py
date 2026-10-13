"""Held-out-genus test: does the genome beat the family tree at predicting worker 3D shape?

    python -m antscan.run_baselines --shapes /mnt/project-files/data/antscan/shapes \
        --out /mnt/project-files/data/antscan/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import genome as gn
from . import mesh

DATA = "/mnt/project-files/data/antscan"


def load_shapes(shape_dir: str, selection_csv: str) -> tuple[pd.DataFrame, list[str]]:
    sel = pd.read_csv(selection_csv)
    rows, cols = [], None
    for code in sorted(sel.specimen_code):
        path = f"{shape_dir}/{code}.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        if not all(f"desc__{k}" in z.files for keys in ev.BLOCKS.values() for k in keys):
            continue
        desc = {k[len("desc__"):]: z[k] for k in z.files if k.startswith("desc__")}
        v, names = mesh.flat_descriptor(desc, tuple(k for keys in ev.BLOCKS.values() for k in keys))
        cols = cols or names
        rows.append([os.path.basename(path)[:-4], float(z["kept_area_frac"])] + list(v))
    df = pd.DataFrame(rows, columns=["specimen_code", "kept_area_frac"] + cols)
    df = df.merge(sel[["specimen_code", "gaga_id", "Genus", "Subfamily", "caste", "voxel_um"]], on="specimen_code")
    return df, cols


def standardise(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / np.clip(X.std(0), 1e-9, None)


def noise_ceiling(spec: pd.DataFrame, cols: list[str], w: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    """How well one worker predicts its conspecific: the best any species-level model can hope for."""
    pairs = [g for _, g in spec.groupby("gaga_id") if len(g) >= 2]
    if not pairs:
        return {}
    A = np.array([(g.iloc[0][cols].to_numpy(float) - mu) / sd for g in pairs])
    B = np.array([(g.iloc[1][cols].to_numpy(float) - mu) / sd for g in pairs])
    base = ((A - 0) ** 2 * w).sum() + ((B - 0) ** 2 * w).sum()
    sse = ((A - B) ** 2 * w).sum() * 2
    # predicting a single worker by the other is noisier than predicting a species mean; the species-mean
    # ceiling is ~ 1 - (within-species variance / 2) / between-species variance
    within = ((A - B) ** 2 * w).sum() / 2 / len(pairs)
    total = (np.concatenate([A, B]) ** 2 * w).sum() / (2 * len(pairs))
    return {"n_pairs": len(pairs), "one_worker_predicts_other_R2": float(1 - sse / base),
            "within_species_share_of_variance": float(within / total)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=f"{DATA}/shapes")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--gaga", default=f"{DATA}/gaga")
    ap.add_argument("--out", default=f"{DATA}/results")
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--mix-perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    spec, cols = load_shapes(a.shapes, a.selection or f"{a.shapes}/selection_worker.csv")
    spec = spec[spec.caste == "worker"].dropna(subset=cols)
    raw = spec[cols].to_numpy(float)
    mu_s, sd_s = raw.mean(0), np.clip(raw.std(0), 1e-9, None)
    w = ev.block_weights(cols)
    ceiling = noise_ceiling(spec, cols, w, mu_s, sd_s)

    sp = spec.groupby("gaga_id").agg({**{c: "mean" for c in cols}, "Genus": "first", "Subfamily": "first",
                                      "specimen_code": "count"}).rename(columns={"specimen_code": "n_scans"})
    ids = list(sp.index)
    Y = standardise(sp[cols].to_numpy(float))
    groups = sp.Genus.to_numpy()
    subfam = sp.Subfamily.to_numpy()

    tree_text = open(f"{a.gaga}/GAGA_dated_phylogeny_onlydates_newick.tre").read()
    C = gn.phylo_covariance(tree_text, ids).to_numpy()
    K_tree = ev.normalise_kernel(C + 1e-6 * np.eye(len(ids)))
    counts = gn.read_genecounts(f"{a.gaga}/Orthogroup_table_N0_genecount.tsv.gz", ids)
    G = gn.genome_features(counts).to_numpy()
    K_gen = ev.linear_kernel(G)

    base = ev.mean_predict(Y, groups)
    preds = {"mean": base, "subfamily": ev.group_mean_predict(Y, groups, subfam)}
    picks = {}
    preds["tree"], picks["tree"] = ev.cv_predict([K_tree], Y, w, groups)
    preds["genome"], picks["genome"] = ev.cv_predict([K_gen], Y, w, groups)
    mixes = [m * K_gen + (1 - m) * K_tree for m in ev.MIX]
    preds["genome+tree"], picks["genome+tree"] = ev.cv_predict(mixes, Y, w, groups)

    scores = {k: ev.r2(p, Y, w, base) for k, p in preds.items()}
    blocks = {k: ev.block_r2(p, Y, base, cols) for k, p in preds.items()}

    # nulls: genome codes shuffled across species (same kernel, broken link to the animal)
    null_gen, null_mix, null_tree = [], [], []
    for i in range(a.perms):
        p = rng.permutation(len(ids))
        Kp = K_gen[np.ix_(p, p)]
        null_gen.append(ev.r2(ev.cv_predict([Kp], Y, w, groups)[0], Y, w, base))
        Kt = K_tree[np.ix_(p, p)]
        null_tree.append(ev.r2(ev.cv_predict([Kt], Y, w, groups)[0], Y, w, base))
        if i < a.mix_perms:
            mp = [m * Kp + (1 - m) * K_tree for m in ev.MIX]
            null_mix.append(ev.r2(ev.cv_predict(mp, Y, w, groups)[0], Y, w, base))
    pval = lambda obs, null: float((1 + np.sum(np.array(null) >= obs)) / (1 + len(null)))

    # genome vs tree, species by species (paired), and bootstrap over genera for the difference
    e = {k: ev.per_species_errors(p, Y, w) for k, p in preds.items()}
    ug = np.unique(groups)
    boot = []
    for _ in range(2000):
        gs = rng.choice(ug, len(ug))
        idx = np.concatenate([np.flatnonzero(groups == g) for g in gs])
        eb = e["mean"][idx].sum()
        boot.append((1 - e["genome"][idx].sum() / eb) - (1 - e["tree"][idx].sum() / eb))
    boot = np.array(boot)

    metrics = {
        "n_species": len(ids), "n_genera": int(len(ug)), "n_scans": int(len(spec)),
        "n_subfamilies": int(len(np.unique(subfam))), "n_genome_features": int(G.shape[1]),
        "n_shape_features": len(cols), "noise_ceiling": ceiling,
        "R2_vs_training_mean": scores, "R2_by_block": blocks,
        "null_genome_shuffled": {"mean": float(np.mean(null_gen)), "sd": float(np.std(null_gen)),
                                 "p_genome": pval(scores["genome"], null_gen), "n": len(null_gen)},
        "null_tree_shuffled": {"mean": float(np.mean(null_tree)), "sd": float(np.std(null_tree)),
                               "p_tree": pval(scores["tree"], null_tree), "n": len(null_tree)},
        "null_genome_shuffled_plus_tree": {"mean": float(np.mean(null_mix)), "sd": float(np.std(null_mix)),
                                           "p_genome_adds_to_tree": pval(scores["genome+tree"], null_mix),
                                           "n": len(null_mix)},
        "genome_minus_tree_R2": {"estimate": scores["genome"] - scores["tree"],
                                 "bootstrap_over_genera_95CI": [float(np.percentile(boot, 2.5)),
                                                                float(np.percentile(boot, 97.5))],
                                 "share_of_species_genome_closer": float(np.mean(e["genome"] < e["tree"]))},
        "mix_weight_on_genome_chosen": [float(ev.MIX[k]) for _, k, _ in picks["genome+tree"]],
    }
    with open(f"{a.out}/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    per = pd.DataFrame({"gaga_id": ids, "genus": groups, "subfamily": subfam, "n_scans": sp.n_scans.values,
                        **{f"err_{k}": v for k, v in e.items()}})
    per.to_csv(f"{a.out}/per_species_errors.csv", index=False)
    np.savez_compressed(f"{a.out}/predictions.npz", ids=np.array(ids), cols=np.array(cols), Y=Y,
                        **{f"pred_{k.replace('+', '_')}": v for k, v in preds.items()})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
