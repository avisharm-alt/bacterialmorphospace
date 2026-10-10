"""Step 3 wing diffusion: scores, genotype codes, sampler sanity, and a tiny end-to-end run on simulated data."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drosophila_wings import generative as gm  # noqa: E402
from drosophila_wings.diffusion import ConditionalDiffusion, DiffusionConfig  # noqa: E402


def test_energy_distance_is_unbiased_and_sees_shifts():
    rng = np.random.default_rng(0)
    same = [gm.energy_distance(rng.normal(size=(60, 5)), rng.normal(size=(200, 5))) for _ in range(200)]
    assert abs(np.mean(same)) < 0.02
    assert gm.energy_distance(rng.normal(size=(200, 5)) + 1, rng.normal(size=(200, 5))) > 0.5


def test_cell_summary_weights_sexes_equally():
    X = np.concatenate([np.zeros((90, 2)), np.ones((10, 2))])
    sex = np.r_[np.zeros(90), np.ones(10)]
    mean, var, dist = gm.cell_summary(X, sex, (0, 1))
    assert np.allclose(mean, 0.5) and var == 0 and np.allclose(dist, 0)


def test_genotype_codes_project_training_lines_like_their_own_pcs():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40, 500))
    K = Z @ Z.T / 500
    train = np.arange(30)
    C = gm.genotype_codes(K, train, 5)
    s, U = np.linalg.eigh(K[np.ix_(train, train)])
    direct = U[:, ::-1][:, :5] * np.sqrt(s[::-1][:5])
    direct = (direct - direct.mean(0)) / direct.std(0)
    assert np.allclose(np.abs(C[train]), np.abs(direct), atol=1e-8)
    assert np.allclose(C[train].std(0), 1)


def test_shape_space_round_trip():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6))
    sp = gm.ShapeSpace(V, var_kept=1.0)
    assert np.allclose(sp.decode(sp.encode(V)), V)
    assert np.allclose(sp.encode(V).var(0, ddof=1), 1)


def test_diffusion_learns_a_condition_dependent_shift():
    rng = np.random.default_rng(3)
    g = rng.choice([-1.0, 1.0], size=4000)
    X = rng.normal(size=(4000, 3)) * 0.3 + 2 * g[:, None]
    m = ConditionalDiffusion(3, 1, DiffusionConfig(steps=1500, width=64, depth=2, batch=256, sample_steps=50,
                                                   cond_noise=0.0, ema=0.99), device="cpu")
    m.fit(X, g[:, None], np.zeros(4000))
    hi = m.sample(np.ones((500, 1)), np.zeros(500), w=1)
    lo = m.sample(-np.ones((500, 1)), np.zeros(500), w=1)
    free = m.sample(np.ones((500, 1)), np.zeros(500), w=0)
    assert hi.mean() > 1.2 and lo.mean() < -1.2
    assert abs(free.mean()) < 0.8 and free.std() > 1.2  # genotype-free samples cover both modes


def test_diffusion_cv_end_to_end_on_simulated_data(tmp_path):
    from drosophila_wings import diffusion_cv, pipeline
    from drosophila_wings.simulate import write_dataset
    paths = write_dataset(tmp_path / "raw", np.random.default_rng(6), n_lines=24, wings_per_line=(12, 16))
    pipeline.main(["--data-dir", str(tmp_path), "all", "--wings", str(paths["wings"]), "--genotypes", str(paths["genotypes"]),
                   "--repeats", "1", "--n-perm", "0", "--outer-k", "3", "--simulated"])
    diffusion_cv.main(["--data-dir", str(tmp_path), "run", "--backend", "local", "--steps", "60", "--width", "32",
                       "--depth", "1", "--sample-steps", "10", "--n-gen", "24", "--geno-pcs", "3"])
    out = tmp_path / "results" / "diffusion"
    m = json.loads((out / "diffusion_metrics.json").read_text())
    methods = {r["method"] for r in m["summary_test_lines"]}
    assert methods == {"mean+resid", "pooled", "diffusion-nogeno", "diffusion-geno", "diffusion-shuffled", "oracle-mean+resid",
                       "diffusion-resid-nogeno", "diffusion-resid-geno", "oracle-mean+diffusion-resid-nogeno",
                       "oracle-mean+diffusion-resid-geno"}
    oracle = next(r for r in m["summary_test_lines"] if r["method"] == "oracle-mean+resid")
    assert oracle["mean_r2_vs_train_mean"] > 0.9
    assert "Held-out lines" in (out / "diffusion_report.md").read_text()
