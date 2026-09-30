"""Validation of src/phylo.py against independent computations (brute force, closed forms, Monte Carlo)."""

import itertools

import numpy as np
import pytest

from src import phylo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def random_newick(n_tips: int, rng, scale: float = 0.3) -> str:
    """Random strictly binary tree by repeated joining; tips t0..t{n-1}."""
    items = [f"t{i}:{rng.uniform(0.02, scale):.4f}" for i in range(n_tips)]
    while len(items) > 1:
        i, j = sorted(rng.choice(len(items), 2, replace=False))
        b, a = items.pop(j), items.pop(i)
        items.append(f"({a},{b}):{rng.uniform(0.02, scale):.4f}")
    return items[0].rsplit(":", 1)[0] + ";"


def gillespie_tree(tree, Q, root_state, rng):
    """Slow reference simulator that also records the true number of state changes. Returns (node states, n_changes)."""
    S = Q.shape[0]
    st = np.zeros(tree.n_nodes, dtype=int)
    st[tree.root] = root_state
    changes = 0
    for i in range(tree.n_nodes - 1, -1, -1):
        if tree.parent[i] < 0:
            continue
        s, t, left = st[tree.parent[i]], 0.0, tree.blen[i]
        while True:
            rate = -Q[s, s]
            dt = rng.exponential(1 / rate) if rate > 0 else np.inf
            if dt >= left:
                break
            left -= dt
            p = Q[s].copy()
            p[s] = 0
            s = rng.choice(S, p=p / p.sum())
            changes += 1
        st[i] = s
    return st, changes


def brute_force_loglik(tree, model, theta, obs):
    """Sum over every assignment of states to the internal nodes."""
    Q = model.Q(theta)
    P = phylo.transition_matrices(tree, Q)
    pi = phylo.stationary(Q)
    tip_lik = model.tip_lik(obs)
    tip_row = {t: r for r, t in enumerate(tree.tips)}
    internal = [i for i in range(tree.n_nodes) if tree.children[i, 0] >= 0]
    total = 0.0
    for assign in itertools.product(range(model.S), repeat=len(internal)):
        st = dict(zip(internal, assign))
        p = pi[st[tree.root]]
        for i in range(tree.n_nodes):
            par = tree.parent[i]
            if par < 0:
                continue
            if i in st:
                p *= P[i, st[par], st[i]]
            else:  # tip: sum over its hidden state (only matters for HRM ambiguity)
                p *= (P[i, st[par], :] * tip_lik[tip_row[i]]).sum()
        total += p
    return np.log(total)


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------
def test_tree_parsing_pruning_and_post_order():
    t = phylo.tree_from_newick("(((a:1,b:2):3,c:4):5,(d:6,e:7):8);")
    assert t.n_nodes == 9 and len(t.tips) == 5 and t.root == 8
    assert all(t.parent[i] > i for i in range(t.n_nodes - 1))  # children come before parents
    assert t.height[t.root] == 3 and sorted(t.labels) == ["a", "b", "c", "d", "e"]
    p = phylo.tree_from_newick("(((a:1,b:2):3,c:4):5,(d:6,e:7):8);", keep={"a", "c", "d"})
    assert sorted(p.labels) == ["a", "c", "d"] and p.n_nodes == 5  # exact lengths are checked in the next test
    with pytest.raises(ValueError):
        phylo.tree_from_newick("(a:1,b:2,c:3);")  # polytomy
    with pytest.raises(ValueError):
        phylo.tree_from_newick("(a:1,b:2);", keep={"a", "zzz"})


def test_pruning_merges_the_lengths_of_suppressed_nodes():
    # dropping b leaves (a,b) with one child, so its 3 units merge into a's branch: a = 1 + 3
    p = phylo.tree_from_newick("(((a:1,b:2):3,c:4):5,d:6);", keep={"a", "c", "d"})
    lens = {l: float(p.blen[i]) for l, i in zip(p.labels, p.tips)}
    assert lens == {"a": 4.0, "c": 4.0, "d": 6.0}
    internal = [k for k in p.children[p.root] if p.children[k, 0] >= 0][0]
    assert p.blen[internal] == 5.0
    # dropping c as well collapses the whole clade: a's branch absorbs 1 + 3 + 5
    q = phylo.tree_from_newick("(((a:1,b:2):3,c:4):5,d:6);", keep={"a", "d"})
    assert {l: float(q.blen[i]) for l, i in zip(q.labels, q.tips)} == {"a": 9.0, "d": 6.0}


# ---------------------------------------------------------------------------
# rate models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,kind", [(2, "ER"), (2, "ARD"), (3, "ER"), (3, "SYM"), (3, "ARD"), (3, "ORD"), (2, "HRM"), (3, "HRM")])
def test_rate_matrices_are_valid_generators_with_a_stationary_distribution(k, kind):
    m = phylo.RateModel(k, kind)
    rng = np.random.default_rng(0)
    th = rng.uniform(-2, 1.5, m.npar)
    if kind == "HRM":
        th[m.n_rates] = abs(th[m.n_rates])  # log m >= 0
    Q = m.Q(th)
    assert Q.shape == (m.S, m.S) and np.allclose(Q.sum(1), 0) and np.all(Q - np.diag(np.diag(Q)) >= 0)
    pi = phylo.stationary(Q)
    assert pi.sum() == pytest.approx(1) and np.all(pi >= 0) and np.allclose(pi @ Q, 0, atol=1e-10)
    if kind == "ORD":
        assert Q[0, 2] == 0 and Q[2, 0] == 0  # only adjacent transitions
    if kind == "SYM":
        assert np.allclose(Q - np.diag(np.diag(Q)), (Q - np.diag(np.diag(Q))).T)


# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,kind", [(2, "ER"), (2, "ARD"), (3, "SYM"), (3, "ARD"), (3, "ORD"), (2, "HRM")])
def test_likelihood_matches_brute_force_enumeration_on_small_trees(k, kind):
    rng = np.random.default_rng(1)
    m = phylo.RateModel(k, kind)
    for trial in range(3):
        tree = phylo.tree_from_newick(random_newick(5, rng))
        th = rng.uniform(-1.5, 1.0, m.npar)
        if kind == "HRM":
            th[m.n_rates] = abs(th[m.n_rates])
        obs = rng.integers(0, k, len(tree.tips))
        assert phylo.Likelihood(tree, m, obs)(th) == pytest.approx(brute_force_loglik(tree, m, th, obs), abs=1e-9)


def test_two_tip_likelihood_has_a_closed_form():
    tree = phylo.tree_from_newick("(a:0.3,b:0.7);")
    m = phylo.RateModel(3, "ARD")
    th = np.log([0.5, 1.2, 0.3, 0.8, 2.0, 0.1])
    Q = m.Q(th)
    pi = phylo.stationary(Q)
    from scipy.linalg import expm
    Pa, Pb = expm(Q * 0.3), expm(Q * 0.7)
    for i, j in itertools.product(range(3), repeat=2):
        obs = np.zeros(2, dtype=int)
        obs[list(tree.labels).index("a")], obs[list(tree.labels).index("b")] = i, j
        assert phylo.Likelihood(tree, m, obs)(th) == pytest.approx(np.log((pi * Pa[:, i] * Pb[:, j]).sum()), abs=1e-12)


def test_hrm_with_equal_classes_reduces_to_ard():
    rng = np.random.default_rng(2)
    tree = phylo.tree_from_newick(random_newick(40, rng))
    ard, hrm = phylo.RateModel(2, "ARD"), phylo.RateModel(2, "HRM")
    obs = rng.integers(0, 2, 40)
    th = np.array([-0.4, 0.3])
    th_h = np.concatenate([th, [0.0, -1.0, -1.0]])  # m = 1: both classes run at the same rates
    assert phylo.Likelihood(tree, hrm, obs)(th_h) == pytest.approx(phylo.Likelihood(tree, ard, obs)(th), abs=1e-8)


def test_pruning_is_stable_on_a_deep_tree():
    rng = np.random.default_rng(3)
    tree = phylo.tree_from_newick(random_newick(400, rng, scale=0.6))
    m = phylo.RateModel(2, "ARD")
    ll = phylo.Likelihood(tree, m, rng.integers(0, 2, 400))(np.array([0.5, -0.5]))
    assert np.isfinite(ll) and ll < 0


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def test_simulator_reproduces_the_closed_form_tip_distribution_including_parameter_draws():
    tree = phylo.tree_from_newick("(a:0.3,b:0.7);")
    m = phylo.RateModel(3, "ARD")
    draws = np.array([np.log([0.5, 1.2, 0.3, 0.8, 2.0, 0.1]), np.log([2.0, 0.2, 1.0, 0.4, 0.5, 1.5])])
    rng = np.random.default_rng(4)
    R = 60_000
    sim = phylo.simulate_tips(tree, m, draws, R, rng)  # (2 tips, 2*R)
    assert sim.shape == (2, 2 * R)
    from scipy.linalg import expm
    exp = np.zeros((3, 3))
    for th in draws:
        Q = m.Q(th)
        pi, Pa, Pb = phylo.stationary(Q), expm(Q * 0.3), expm(Q * 0.7)
        exp += np.einsum("x,xi,xj->ij", pi, Pa, Pb) / len(draws)
    ia, ib = list(tree.labels).index("a"), list(tree.labels).index("b")
    obs = np.zeros((3, 3))
    np.add.at(obs, (sim[ia], sim[ib]), 1)
    obs /= 2 * R
    sd = np.sqrt(exp * (1 - exp) / (2 * R))
    assert np.all(np.abs(obs - exp) < 5 * sd + 1e-4)


def test_hrm_simulation_returns_observable_states_only():
    tree = phylo.tree_from_newick(random_newick(30, np.random.default_rng(0)))
    m = phylo.RateModel(3, "HRM")
    th = np.concatenate([np.log(np.full(6, 0.5)), [1.0, -1.0, -1.0]])
    sim = phylo.simulate_tips(tree, m, th, 200, np.random.default_rng(1))
    assert sim.max() <= 2 and sim.min() >= 0


# ---------------------------------------------------------------------------
# stochastic mapping
# ---------------------------------------------------------------------------
def _cherry(t):
    tree = phylo.tree_from_newick(f"(x:{t},y:0.001);")
    return tree, list(tree.labels).index("x")


def test_branch_sampler_matches_conditioned_gillespie():
    """Number of changes on a branch given its endpoints: bridge sampler vs rejection from unconditional simulation."""
    m = phylo.RateModel(3, "ARD")
    Q = m.Q(np.log([0.9, 0.4, 1.6, 0.7, 0.5, 1.1]))
    t, a, b = 0.8, 0, 2
    tree, ix = _cherry(t)
    node = tree.tips[ix]
    rng = np.random.default_rng(5)
    states = np.zeros(tree.n_nodes, dtype=int)
    states[tree.root], states[node] = a, b
    states[tree.tips[1 - ix]] = a
    counts = []
    for _ in range(6000):
        br, tm, fr, to = phylo.sample_branch_histories(tree, Q, states, rng)
        sel = br == node
        counts.append(int(sel.sum()))
        if sel.any():
            assert np.all((tm[sel] >= 0) & (tm[sel] <= t)) and fr[sel][0] == a and to[sel][-1] == b  # chain runs a -> ... -> b
    ref = []
    while len(ref) < 6000:
        s, left, n = a, t, 0
        while True:
            rate = -Q[s, s]
            dt = rng.exponential(1 / rate)
            if dt >= left:
                break
            left -= dt
            p = Q[s].copy()
            p[s] = 0
            s = rng.choice(3, p=p / p.sum())
            n += 1
        if s == b:
            ref.append(n)
    hi = max(max(counts), max(ref)) + 1
    f1, f2 = np.bincount(counts, minlength=hi) / len(counts), np.bincount(ref, minlength=hi) / len(ref)
    assert 0.5 * np.abs(f1 - f2).sum() < 0.04  # total variation distance between the two distributions
    assert np.mean(counts) == pytest.approx(np.mean(ref), abs=0.08)


def test_branch_sampler_with_identical_endpoints_can_still_change_and_return():
    m = phylo.RateModel(2, "ARD")
    Q = m.Q(np.log([2.0, 2.0]))
    tree, ix = _cherry(1.5)
    node = tree.tips[ix]
    states = np.zeros(tree.n_nodes, dtype=int)
    rng = np.random.default_rng(6)
    n = [int((phylo.sample_branch_histories(tree, Q, states, rng)[0] == node).sum()) for _ in range(3000)]
    assert max(n) >= 2 and all(x % 2 == 0 for x in n)  # a -> b -> a: an even number of changes


def test_node_state_sampler_matches_the_exact_posterior():
    rng = np.random.default_rng(7)
    tree = phylo.tree_from_newick(random_newick(5, rng))
    m = phylo.RateModel(2, "ARD")
    th = np.log([1.3, 0.6])
    obs = np.array([0, 1, 1, 0, 1])
    Q = m.Q(th)
    P = phylo.transition_matrices(tree, Q)
    pi = phylo.stationary(Q)
    internal = [i for i in range(tree.n_nodes) if tree.children[i, 0] >= 0]
    tip_row = {t: r for r, t in enumerate(tree.tips)}
    post = {}
    for assign in itertools.product(range(2), repeat=len(internal)):
        st = dict(zip(internal, assign))
        p = pi[st[tree.root]]
        for i in range(tree.n_nodes):
            par = tree.parent[i]
            if par < 0:
                continue
            p *= P[i, st[par], st[i]] if i in st else P[i, st[par], obs[tip_row[i]]]
        post[assign] = p
    z = sum(post.values())
    exact_root1 = sum(p for a, p in post.items() if a[internal.index(tree.root)] == 1) / z
    freq = np.mean([sample_root(tree, m, th, obs, rng) for _ in range(6000)])
    assert freq == pytest.approx(exact_root1, abs=0.02)


def sample_root(tree, m, th, obs, rng):
    return phylo.sample_node_states(tree, m, th, obs, rng)[0][tree.root]


def test_mapped_changes_are_unbiased_for_the_true_number_of_changes():
    """Average over datasets: E[#changes in a stochastic map | tips] equals E[#true changes] when Q is known."""
    rng = np.random.default_rng(8)
    m = phylo.RateModel(3, "ARD")
    th = np.log([0.8, 0.3, 0.5, 1.2, 0.4, 0.9])
    Q = m.Q(th)
    pi = phylo.stationary(Q)
    tree = phylo.tree_from_newick(random_newick(30, rng, scale=0.5))
    true, mapped = [], []
    for _ in range(250):
        st, n_true = gillespie_tree(tree, Q, rng.choice(3, p=pi), rng)
        obs = st[tree.tips]
        mp = phylo.stochastic_map(tree, m, th, obs, rng)
        true.append(n_true)
        mapped.append(len(mp["branch"]))
    diff = np.array(mapped) - np.array(true)
    assert abs(diff.mean()) < 3 * diff.std(ddof=1) / np.sqrt(len(diff)) + 0.05


def test_mapped_history_is_consistent_with_the_node_states():
    rng = np.random.default_rng(9)
    tree = phylo.tree_from_newick(random_newick(60, rng))
    m = phylo.RateModel(3, "ARD")
    th = np.log([0.6, 0.4, 0.5, 0.7, 0.3, 0.9])
    obs = rng.integers(0, 3, 60)
    mp = phylo.stochastic_map(tree, m, th, obs, rng)
    assert np.array_equal(mp["states"][tree.tips], obs)  # tips keep their observed states
    for node in np.unique(mp["branch"]):
        sel = np.flatnonzero(mp["branch"] == node)
        sel = sel[np.argsort(mp["time"][sel])]
        chain = [mp["states"][tree.parent[node]]] + list(mp["to"][sel])
        assert list(mp["from"][sel]) == chain[:-1] and chain[-1] == mp["states"][node]


# ---------------------------------------------------------------------------
# fitting and MCMC
# ---------------------------------------------------------------------------
def test_ml_fit_recovers_rates_on_average():
    rng = np.random.default_rng(10)
    m = phylo.RateModel(2, "ARD")
    th_true = np.log([1.0, 0.5])
    tree = phylo.tree_from_newick(random_newick(200, rng, scale=0.4))
    Q = m.Q(th_true)
    pi = phylo.stationary(Q)
    errs = []
    for _ in range(8):
        st, _ = gillespie_tree(tree, Q, rng.choice(2, p=pi), rng)
        f = phylo.fit_ml(phylo.Likelihood(tree, m, st[tree.tips]), starts=2, hessian=False)
        errs.append(f["theta"] - th_true)
    errs = np.array(errs)
    assert np.all(np.abs(np.median(errs, axis=0)) < 0.35) and np.all(np.abs(errs) < 1.6)


def test_mcmc_posterior_matches_numerical_integration_on_a_grid():
    rng = np.random.default_rng(11)
    tree = phylo.tree_from_newick(random_newick(60, rng, scale=0.5))
    m = phylo.RateModel(2, "ER")
    Q = m.Q(np.array([np.log(1.0)]))
    st, _ = gillespie_tree(tree, Q, 0, rng)
    lik = phylo.Likelihood(tree, m, st[tree.tips])
    grid = np.linspace(*phylo.LOG_RATE_BOUNDS, 1500)
    lp = np.array([m.log_prior(np.array([g])) + lik(np.array([g])) for g in grid])
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean_exact = (w * grid).sum()
    sd_exact = np.sqrt((w * (grid - mean_exact) ** 2).sum())
    fit = phylo.fit_ml(lik, starts=2)
    ch = phylo.run_mcmc(lik, fit["theta"], fit["hessian"], 4000, rng)
    draws = ch["chain"][:, 0]
    assert draws.mean() == pytest.approx(mean_exact, abs=0.15)
    assert draws.std() == pytest.approx(sd_exact, rel=0.25)
    assert 0.1 < ch["acceptance"] < 0.7 and ch["ess"][0] > 100


def test_newick_roundtrip_preserves_structure_and_lengths():
    rng = np.random.default_rng(12)
    t = phylo.tree_from_newick(random_newick(25, rng))
    t2 = phylo.tree_from_newick(t.newick)
    assert sorted(t2.labels) == sorted(t.labels) and t2.n_nodes == t.n_nodes
    assert np.allclose(sorted(t2.blen), sorted(t.blen))
    sub = phylo.tree_from_newick(t.newick, keep=set(t.labels[:10]))
    assert len(sub.tips) == 10


# ---------------------------------------------------------------------------
# src/evolnull.py pieces that do not need the real panel
# ---------------------------------------------------------------------------
from src import evolnull as ev  # noqa: E402


def test_cell_statistics_flags_only_testable_cells_that_are_emptier_than_the_null():
    rng = np.random.default_rng(0)
    R = 4000
    # cell 0: expected ~10, observed 0 -> flagged; cell 1: expected ~10, observed 10 -> not; cell 2: expected ~0.2 -> untestable
    sim = np.column_stack([rng.poisson(10, R), rng.poisson(10, R), rng.poisson(0.2, R), rng.poisson(10, R)])
    obs = np.array([0, 10, 0, 25])
    st = ev.cell_statistics(obs, sim)
    assert st["testable"].tolist() == [True, True, False, True]
    assert st["flag"].tolist() == [True, False, False, False]  # cell 3 is fuller than expected, cell 2 is untestable
    assert st["empty_flag"].tolist() == [True, False, False, False]
    assert st["p_low"][0] == pytest.approx(1 / (R + 1), abs=1e-3) and np.isnan(st["q_low"][2])
    assert st["p_high"][3] < 0.001
    assert st["bonferroni_count"][2] == -1 and st["bonferroni_count"][0] >= 0


def test_cell_statistics_gives_uniform_p_values_when_observed_comes_from_the_null():
    rng = np.random.default_rng(1)
    R, C = 3000, 60
    sim = rng.poisson(12, (R, C))
    flagged = 0
    ps = []
    for _ in range(60):  # the "observed" table is one more draw from the same null
        obs = rng.poisson(12, C)
        st = ev.cell_statistics(obs, sim)
        flagged += int(st["flag"].any())
        ps += list(st["p_low"][st["testable"]])
    assert flagged <= 8  # BH at 5% over 60 datasets: expect ~3, allow generous slack
    assert abs(np.mean(np.array(ps) < 0.05) - 0.05) < 0.03


def test_hamming1_neighbours_counts_and_symmetry():
    sizes = (2, 3, 2)
    nb = ev.hamming1_neighbours(sizes)
    assert all(len(n) == sum(s - 1 for s in sizes) for n in nb)  # 1 + 2 + 1 = 4 neighbours for every cell
    for c, ns in enumerate(nb):
        assert c not in ns and all(c in nb[j] for j in ns)
        for j in ns:
            a, b = np.unravel_index(c, sizes), np.unravel_index(j, sizes)
            assert sum(x != y for x, y in zip(a, b)) == 1


def test_surprising_absences_ranks_empty_cells_by_neighbour_convergence():
    sizes = (2, 2)  # cells 0..3; neighbours of 0 are 1 and 2; of 3 are 1 and 2
    obs = np.array([0, 5, 5, 0])
    origins = np.array([[0, 30, 40, 0], [0, 34, 44, 0]])  # maps x cells
    sa = ev.surprising_absences(obs, origins, sizes, top=5)
    assert set(sa["cell"]) == {0, 3} and sa["neighbour_origins"].tolist() == [32 + 42, 32 + 42]  # mean origins 32 and 42
    assert sa["occupied_neighbours"].tolist() == [2, 2]
    obs2 = np.array([0, 5, 0, 0])
    sa2 = ev.surprising_absences(obs2, origins, sizes, top=5)
    assert sa2.iloc[0]["cell"] in (0, 3) and set(sa2["cell"]) == {0, 2, 3}


def test_map_origins_counts_entries_into_cells_on_a_tree_with_two_traits():
    """Two independent binary traits on a small tree: origin counts are nonnegative, occupied non-root cells are entered >= once."""
    rng = np.random.default_rng(3)
    tree = phylo.tree_from_newick(random_newick(40, rng, scale=0.5))
    ks = {"a": 2, "b": 2}
    fits = {t: {"draws": {"ARD": [np.log([1.2, 0.8]).tolist()] * 5}} for t in ks}
    codes = {t: rng.integers(0, 2, 40) for t in ks}
    sizes = (2, 2)
    o = ev.map_origins(tree, ks, fits, codes, sizes, n_maps=12, seed=1, traits=["a", "b"])
    assert o.shape == (12, 4) and o.min() >= 0
    occupied = np.bincount(np.ravel_multi_index(tuple(codes[t] for t in ks), sizes), minlength=4) > 0
    assert (o.sum(0)[occupied] > 0).all() and o.sum() > 0


def test_build_comparison_labels_disagreements_and_explains_them():
    import pandas as pd
    n = 4
    grid = pd.DataFrame({"a": list("wxyz"), "cell": list("wxyz")})
    obs = np.array([0, 0, 0, 3])

    def st(flag, exp=10.0, sd=2.0):
        return {"expected": np.full(n, exp), "sd": np.full(n, sd), "p_low": np.full(n, .01), "q_low": np.full(n, .01),
                "testable": np.ones(n, bool), "flag": np.array(flag)}
    evo = {"ARD": st([True, True, False, False]), "HRM": st([True, False, False, False])}

    def perm(flag, exp=10.0):
        return pd.DataFrame({"expected": np.full(n, exp), "q_low": np.full(n, .01), "testable": np.ones(n, bool), "emptier_than_chance": np.array(flag)})
    p = {"global": perm([True, True, True, False]), "phylum": perm([False] * 4), "class": perm([False] * 4), "order": perm([False, False, True, False], exp=4.0)}
    sd = {k: np.full(n, 1.0) for k in p}
    c = ev.build_comparison(grid, obs, evo, p, sd)
    assert c["evo_robust_emptier"].tolist() == [True, False, False, False]  # flagged under BOTH evolutionary nulls
    assert c["category"].tolist() == ["flagged by both", "permutation + one evolutionary null", "permutation only", "neither"]
    assert c.loc[2, "note"].startswith("the tree makes this count far more variable")  # evolutionary SD 2.0 > 1.5 x the shuffle's 1.0
    assert "SD 2.0 vs 1.0 under the order shuffle" in c.loc[2, "note"]
    assert "order-level shuffle already expects only 4.0" in c.loc[1, "note"]  # evolutionary expects 10.0 > 1.3 x 4.0
    assert c.loc[3, "note"] == "" and c.loc[0, "note"] == ""


def test_power_summary_and_bins_show_overdispersion_against_poisson():
    rng = np.random.default_rng(4)
    R = 5000
    lam = np.array([0.5, 2, 5, 12, 30])
    sim = np.column_stack([rng.negative_binomial(2, 2 / (2 + m), R) for m in lam])  # over-dispersed counts with mean m
    st = ev.cell_statistics(np.array([0, 0, 0, 0, 0]), sim)
    pw = ev.power_summary(st)
    assert pw["cells"] == 5 and pw["testable"] == int(st["testable"].sum())
    assert pw["median_overdispersion_var_over_mean"] > 1.5
    tab = ev.power_by_expected(st)
    assert (tab["median_P_null_empty"] >= tab["poisson_exp_minus_E"] - 1e-9).all()  # a clade-structured null is emptier than Poisson


# ---------------------------------------------------------------------------
# fast transition matrices and multi-chain MCMC
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,kind", [(2, "ARD"), (3, "ER"), (3, "SYM"), (3, "ARD"), (3, "ORD"), (2, "HRM"), (3, "HRM")])
def test_eigen_transition_matrices_agree_with_expm(k, kind):
    rng = np.random.default_rng(20)
    tree = phylo.tree_from_newick(random_newick(60, rng, scale=0.8))
    m = phylo.RateModel(k, kind)
    before = dict(phylo._fallbacks)
    for _ in range(25):
        th = rng.uniform(-6, 4.5, m.npar)
        if kind == "HRM":
            th[m.n_rates] = abs(th[m.n_rates])
        Q = m.Q(th)
        fast = phylo.transition_matrices(tree, Q)
        ref = phylo.transition_matrices(tree, Q, force_expm=True)
        assert np.abs(fast - ref).max() < 1e-7  # whichever path ran, it must agree with scipy's expm
        assert np.allclose(fast.sum(2), 1, atol=1e-8) and fast.min() >= 0
    assert phylo._fallbacks["eig"] > before["eig"]  # the fast path is actually being used


def test_transition_matrices_fall_back_to_expm_for_a_defective_generator():
    tree = phylo.tree_from_newick("(a:0.3,b:0.5);")
    Q = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]])  # Jordan block: not diagonalisable
    before = dict(phylo._fallbacks)
    P = phylo.transition_matrices(tree, Q)
    assert np.abs(P - phylo.transition_matrices(tree, Q, force_expm=True)).max() < 1e-12
    assert phylo._fallbacks["expm"] > before["expm"]


def test_likelihood_is_unchanged_by_the_fast_path():
    rng = np.random.default_rng(21)
    tree = phylo.tree_from_newick(random_newick(80, rng))
    m = phylo.RateModel(3, "ARD")
    th = rng.uniform(-2, 1.5, m.npar)
    obs = rng.integers(0, 3, 80)
    fast = phylo.Likelihood(tree, m, obs)(th)
    Q = m.Q(th)
    L, ls = phylo.conditional_likelihoods(tree, phylo.transition_matrices(tree, Q, force_expm=True), m.tip_lik(obs))
    assert fast == pytest.approx(ls + np.log(phylo.stationary(Q) @ L[tree.root]), abs=1e-9)


def test_multichain_mcmc_reports_convergence_and_matches_the_grid_posterior():
    rng = np.random.default_rng(22)
    tree = phylo.tree_from_newick(random_newick(60, rng, scale=0.5))
    m = phylo.RateModel(2, "ER")
    st, _ = gillespie_tree(tree, m.Q(np.array([0.0])), 0, rng)
    lik = phylo.Likelihood(tree, m, st[tree.tips])
    grid = np.linspace(*phylo.LOG_RATE_BOUNDS, 1500)
    lp = np.array([m.log_prior(np.array([g])) + lik(np.array([g])) for g in grid])
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean_exact = (w * grid).sum()
    fit = phylo.fit_ml(lik, starts=2)
    r = phylo.run_chains(lik, fit["theta"], fit["hessian"], 2500, rng, n_chains=3)
    assert len(r["chains"]) == 3 and r["rhat"][0] < 1.1 and r["ess"][0] > 200
    assert r["chain"][:, 0].mean() == pytest.approx(mean_exact, abs=0.12)


def test_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(23)
    good = [rng.normal(0, 1, (500, 1)) for _ in range(3)]
    bad = [rng.normal(0, 1, (500, 1)), rng.normal(0, 1, (500, 1)), rng.normal(4, 1, (500, 1))]
    assert phylo.gelman_rubin(good)[0] < 1.05 and phylo.gelman_rubin(bad)[0] > 1.5
