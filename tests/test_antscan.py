import numpy as np

from antscan import evaluate as ev
from antscan import genome as gn
from antscan import mesh
from antscan.taxa import norm_name


def test_phylo_covariance_shared_path():
    C = gn.phylo_covariance("((A:1,B:1):2,C:3);", ["A", "B", "C"]).to_numpy()
    assert np.allclose(C, [[3, 2, 0], [2, 3, 0], [0, 0, 3]])
    D = gn.patristic(gn.phylo_covariance("((A:1,B:1):2,C:3);", ["A", "B", "C"])).to_numpy()
    assert D[0, 1] == 2 and D[0, 2] == 6


def _cube_triangles(s=1.0):
    v = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]], float)
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return v[np.array(f)]


def test_mesh_volume_area_and_components():
    cube = _cube_triangles(2.0)
    assert np.isclose(abs(mesh.signed_volume(cube)), 8.0)
    assert np.isclose(mesh.triangle_areas(cube).sum(), 24.0)
    speck = _cube_triangles(0.1) + 10
    kept, frac = mesh.largest_component(np.concatenate([cube, speck]))
    assert len(kept) == 12 and frac > 0.99


def test_name_normalisation():
    assert norm_name("Lasius cf. flavus s1") == "lasius flavus"
    assert norm_name("Lasius.cf.flavus") == "lasius flavus"
    assert norm_name("Myrmecia croslandi 2n2") == "myrmecia croslandi"


def test_kernel_ridge_recovers_signal_and_mean_baseline():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    Y = X @ rng.normal(size=(5, 3))
    K = ev.linear_kernel(X)
    groups = np.repeat(np.arange(20), 2)
    w = np.ones(3) / 3
    pred, _ = ev.cv_predict([K], Y, w, groups)
    assert ev.r2(pred, Y, w, ev.mean_predict(Y, groups)) > 0.8
