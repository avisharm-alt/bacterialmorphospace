"""DGRP wing pipeline: Procrustes, readers, relatedness folds, line means, ridge baseline, end-to-end on simulated data."""
import json

import numpy as np
import pandas as pd
import pytest

from drosophila_wings import io, model, pipeline, procrustes, relatedness, shape
from drosophila_wings.simulate import simulate, write_plink


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_gpa_removes_position_scale_and_rotation_but_not_reflection():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(12, 2))
    X = np.stack([base @ _rot(rng.uniform(-3, 3)) * rng.uniform(0.5, 5) + rng.normal(size=2) for _ in range(6)])
    Y, cs, ref = procrustes.gpa(X)
    assert np.allclose(Y, Y[0], atol=1e-8)
    assert np.isclose((ref ** 2).sum(), 1)
    mirrored = np.concatenate([X, (base * [-1, 1])[None]])
    Ym, _, _ = procrustes.gpa(mirrored)
    assert not np.allclose(Ym[-1], Ym[0], atol=1e-3)


def test_tangent_coordinates_are_orthogonal_to_consensus():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(10, 2))
    X = base + rng.normal(scale=0.05, size=(30, 10, 2))
    Y, _, ref = procrustes.gpa(X)
    T = procrustes.tangent(Y, ref)
    u = ref.reshape(-1)
    assert np.allclose((T - u) @ u, 0, atol=1e-12)


@pytest.mark.parametrize("raw, want", [("RAL-21", "21"), ("line_021", "21"), ("DGRP_908", "908"), (21, "21"), (21.0, "21")])
def test_line_id(raw, want):
    assert io.line_id(raw) == want


@pytest.mark.parametrize("cols", [["x1", "y1", "x2", "y2", "x3", "y3"], ["LM1x", "LM1y", "LM2x", "LM2y", "LM3x", "LM3y"],
                                  ["X01", "X02", "X03", "Y01", "Y02", "Y03"], ["P1_X", "P1_Y", "P2_X", "P2_Y", "P3_X", "P3_Y"]])
def test_landmark_columns_pair_by_index(cols):
    xs, ys = io.landmark_columns(["Line", "Sex", "CS"] + cols)
    assert len(xs) == len(ys) == 3
    assert all(x.lower().replace("x", "") == y.lower().replace("y", "") for x, y in zip(xs, ys))


def test_plink_round_trip_with_missing_and_odd_sample_count(tmp_path):
    rng = np.random.default_rng(3)
    G = rng.choice([0, 1, 2, -1], size=(7, 40), p=[0.45, 0.05, 0.45, 0.05]).astype(np.int8)
    write_plink(tmp_path / "t", [f"line_{i}" for i in range(7)], G, ["2L"] * 40, np.arange(40))
    g = io.read_plink(tmp_path / "t", maf=0, max_missing=1)
    keep = io._filter_block(G, 0, 1)
    assert g.lines == [str(i) for i in range(7)]
    assert np.array_equal(g.G, G[:, keep])


def test_vcf_and_tgeno_readers(tmp_path):
    vcf = tmp_path / "t.vcf"
    vcf.write_text("##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tline_21\tline_26\tline_28\n"
                   "2L\t10\ts1\tA\tG\t.\tPASS\t.\tGT:GQ\t0/0:9\t1/1:9\t./.:0\n"
                   "2L\t20\ts2\tA\tG,T\t.\tPASS\t.\tGT\t0/0\t1/1\t2/2\n"
                   "2L\t30\ti1\tAT\tA\t.\tPASS\t.\tGT\t0/0\t1/1\t1/1\n"
                   "2L\t40\ts3\tC\tT\t.\tPASS\t.\tGT\t1\t0\t1\n")
    g = io.read_vcf(vcf, maf=0, max_missing=1)
    assert g.lines == ["21", "26", "28"] and list(g.variants["id"]) == ["s1", "s3"]
    assert g.G.T.tolist() == [[0, 2, -1], [2, 0, 2]]
    tg = tmp_path / "t.tgeno"
    tg.write_text("chr pos id ref alt refc altc qual cov line_21 line_26\n2L 10 s1 A G 1 1 99 10 0 2\n2L 20 s2 A G 1 1 99 10 - 2\n")
    t = io.read_tgeno(tg, maf=0, max_missing=1)
    assert t.G.T.tolist() == [[0, 2], [-1, 2]]


def test_related_groups_and_group_folds_keep_relatives_together():
    rng = np.random.default_rng(4)
    G = (rng.random((40, 3000)) < 0.5).astype(np.int8) * 2
    G[1] = np.where(rng.random(3000) < 0.9, G[0], G[1])
    G[2] = np.where(rng.random(3000) < 0.9, G[1], G[2])  # 0-1-2 chain
    G[11] = np.where(rng.random(3000) < 0.9, G[10], G[11])
    groups = relatedness.related_groups(relatedness.grm(G), 0.2)
    assert groups[0] == groups[1] == groups[2] and groups[10] == groups[11] and groups[0] != groups[10]
    assert len(np.unique(groups)) == 40 - 3
    folds = relatedness.group_kfold(groups, 5, rng)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    assert np.bincount(folds).max() - np.bincount(folds).min() <= 3


def test_line_means_weight_sexes_equally():
    meta = pd.DataFrame({"line": ["1"] * 4, "sex": ["F", "M", "M", "M"]})
    V = np.array([[0.0], [2.0], [2.0], [2.0]])
    lm, n = shape.line_means(V, meta, ["sex"], min_wings=1)
    assert lm.loc["1", 0] == 1.0 and n.loc["1"] == 4


def test_ridge_beats_mean_with_signal_and_not_without():
    rng = np.random.default_rng(5)
    # few independent loci, so 120 unrelated lines can learn them (with thousands, expected R^2 is near 0)
    G = (rng.random((120, 100)) < 0.5).astype(np.int8) * 2
    Z = relatedness.standardise(G)
    K = Z @ Z.T / Z.shape[1]
    groups = np.arange(120)
    signal = Z @ rng.normal(size=(100, 6)) / np.sqrt(100)
    for Y, should_win in ((signal + rng.normal(scale=0.5, size=signal.shape), True), (rng.normal(size=(120, 6)), False)):
        res = model.cross_validate(Y, K, groups, rng, models=("mean", "snp"))
        r2 = model.metrics(Y, res.pred["snp"], res.pred["mean"])["r2_vs_mean"]
        assert (r2 > 0.1) if should_win else (r2 < 0.05)


def test_covariate_design_one_hot(tmp_path):
    p = tmp_path / "cov.csv"
    pd.DataFrame({"line": ["line_1", "line_2", "line_3"], "wolbachia": ["y", "n", "y"], "In(2L)t": ["ST", "INV", None]}).to_csv(p, index=False)
    X = io.design_matrix(io.load_covariates([p]), ["1", "2", "3"])
    assert list(X["wolbachia_Y"]) == [1, 0, 1] and X.shape == (3, 2)


def test_pipeline_end_to_end_on_simulated_data(tmp_path):
    from drosophila_wings.simulate import write_dataset
    paths = write_dataset(tmp_path / "raw", np.random.default_rng(6), n_lines=50, wings_per_line=(12, 20))
    pipeline.main(["--data-dir", str(tmp_path), "all", "--wings", str(paths["wings"]), "--genotypes", str(paths["genotypes"]),
                   "--covariates", str(paths["covariates"]), "--repeats", "1", "--n-perm", "3", "--simulated"])
    m = json.loads((tmp_path / "results" / "baseline_metrics.json").read_text())
    assert m["n_lines"] == 50 and m["covariates"]
    assert "related-grouped | snp" in m["summary"]
    report = (tmp_path / "results" / "baseline_report.md").read_text()
    assert "SIMULATED" in report


def test_simulated_relatives_are_grouped():
    sim = simulate(np.random.default_rng(7), n_lines=60, n_snps=5000)
    K = relatedness.grm(sim["G"])
    groups = relatedness.related_groups(K, 0.05)
    lines = [str(x) for x in sim["ids"]]
    for a, b in sim["related_pairs"]:
        assert groups[lines.index(str(a))] == groups[lines.index(str(b))]


@pytest.mark.parametrize("raw, want", [("F", "F"), ("female", "F"), ("M", "M"), ("probablyM", "M"), ("male", "M")])
def test_sex_code(raw, want):
    assert io.sex_code(raw) == want
