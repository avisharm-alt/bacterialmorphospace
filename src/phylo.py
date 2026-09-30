"""Discrete-trait evolution on a fixed tree: Mk likelihood, ML fit, MCMC, forward simulation, stochastic mapping.

Pure numpy/scipy (tree parsing uses treeswift). Written to be validated, not trusted: see tests/test_phylo.py and
`python -m src.evolnull validate`, which check the likelihood against brute-force enumeration, the simulator and
the branch sampler against closed forms and Monte Carlo, and parameter recovery / posterior coverage on the real tree.

Model
-----
A trait with k states evolves along the tree as a continuous-time Markov chain with rate matrix Q (per unit branch
length; GTDB branch lengths are substitutions per site, not time). The root state is drawn from the stationary
distribution of Q, and the same assumption is used for the likelihood, for simulation and for stochastic mapping,
so all three describe one generative model.

  ER   one shared rate
  SYM  one rate per unordered pair of states
  ARD  a separate rate for every ordered pair ("all rates different")
  ORD  adjacent-state transitions only, symmetric (for naturally ordered traits)
  HRM  ARD with a hidden fast/slow rate class: rate matrix Q for class 0 and m*Q (m >= 1) for class 1, switching
       between classes at rates s01, s10. It lets some clades evolve the trait faster than others, which a single
       rate matrix cannot, and it is fitted and simulated on the expanded 2k-state space (tips are ambiguous
       between the two classes).

Nodes are indexed in post-order: children before parents, the root is the last node.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import gammaln

LOG_RATE_BOUNDS = (-10.0, 5.0)  # rates between 4.5e-5 and 148 per unit branch length


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
@dataclass
class Tree:
    parent: np.ndarray  # (N,) parent index, -1 for the root
    blen: np.ndarray  # (N,) branch length above each node (0 for the root)
    children: np.ndarray  # (N, 2) child indices, -1 for tips
    tips: np.ndarray  # node indices of the tips, in post-order
    labels: list  # tip labels aligned to `tips`
    height: np.ndarray
    depth: np.ndarray
    root: int
    newick: str = ""  # the (pruned) tree as Newick, so it can be saved, re-read and sub-sampled

    @property
    def n_nodes(self) -> int:
        return len(self.parent)

    @property
    def levels_up(self) -> list:
        """Internal nodes grouped by height (1, 2, ...): every group depends only on lower groups."""
        return [np.flatnonzero(self.height == h) for h in range(1, int(self.height.max()) + 1)]

    @property
    def levels_down(self) -> list:
        """Non-root nodes grouped by depth (1, 2, ...): every group's parents are in shallower groups."""
        return [np.flatnonzero(self.depth == d) for d in range(1, int(self.depth.max()) + 1)]


def tree_from_newick(newick: str, keep: set | None = None) -> Tree:
    """Parse a binary Newick tree, optionally pruned to the tip labels in `keep` (unifurcations are suppressed)."""
    import treeswift as ts

    t = ts.read_tree_newick(newick.strip())
    if keep is not None:
        missing = set(keep) - {n.label for n in t.traverse_leaves()}
        if missing:
            raise ValueError(f"{len(missing)} requested tips are not in the tree, e.g. {sorted(missing)[:3]}")
        t = t.extract_tree_with(set(keep), suppress_unifurcations=True)
    nodes = list(t.traverse_postorder())
    idx = {id(n): i for i, n in enumerate(nodes)}
    n = len(nodes)
    parent, blen, children = np.full(n, -1), np.zeros(n), np.full((n, 2), -1)
    for nd in nodes:
        i = idx[id(nd)]
        blen[i] = nd.edge_length or 0.0
        kids = nd.children
        if kids and len(kids) != 2:
            raise ValueError("the tree must be strictly binary")
        for j, c in enumerate(kids):
            children[i, j] = idx[id(c)]
            parent[idx[id(c)]] = i
    tips = np.flatnonzero(children[:, 0] < 0)
    height, depth = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for i in range(n):
        if children[i, 0] >= 0:
            height[i] = 1 + max(height[children[i, 0]], height[children[i, 1]])
    for i in range(n - 1, -1, -1):
        if parent[i] >= 0:
            depth[i] = depth[parent[i]] + 1
    if np.any(blen[parent >= 0] < 0):
        raise ValueError("negative branch length")
    return Tree(parent, blen, children, tips, [nodes[i].label for i in tips], height, depth, n - 1,
                newick=t.newick() if n < 20000 else "")


# ---------------------------------------------------------------------------
# Rate models
# ---------------------------------------------------------------------------
class RateModel:
    """Maps unconstrained log-parameters to a rate matrix over `S` states of which `k` are observable."""

    def __init__(self, k: int, kind: str):
        if kind not in ("ER", "SYM", "ARD", "ORD", "HRM"):
            raise ValueError(kind)
        if kind == "ORD" and k < 3:
            raise ValueError("ORD needs >= 3 states")
        self.k, self.kind = k, kind
        self.S = 2 * k if kind == "HRM" else k
        self.n_rates = {"ER": 1, "SYM": k * (k - 1) // 2, "ARD": k * (k - 1), "ORD": k - 1, "HRM": k * (k - 1)}[kind]
        self.npar = self.n_rates + (3 if kind == "HRM" else 0)
        lo, hi = LOG_RATE_BOUNDS
        self.lower = np.full(self.npar, lo)
        self.upper = np.full(self.npar, hi)
        if kind == "HRM":
            self.lower[self.n_rates] = 0.0  # log m >= 0: class 1 is the faster one, which removes label switching
        self.pairs = [(i, j) for i in range(k) for j in range(k) if i != j]

    def _base(self, rates: np.ndarray) -> np.ndarray:
        k, Q = self.k, np.zeros((self.k, self.k))
        if self.kind == "ER":
            Q[:] = rates[0]
        elif self.kind == "SYM":
            it = iter(rates)
            for i in range(k):
                for j in range(i + 1, k):
                    Q[i, j] = Q[j, i] = next(it)
        elif self.kind in ("ARD", "HRM"):
            for (i, j), r in zip(self.pairs, rates):
                Q[i, j] = r
        elif self.kind == "ORD":
            for i in range(k - 1):
                Q[i, i + 1] = Q[i + 1, i] = rates[i]
        np.fill_diagonal(Q, 0.0)
        return Q

    def Q(self, theta: np.ndarray) -> np.ndarray:
        rates = np.exp(theta[: self.n_rates])
        base = self._base(rates)
        if self.kind != "HRM":
            np.fill_diagonal(base, -base.sum(1))
            return base
        k = self.k
        m, s01, s10 = np.exp(theta[self.n_rates: self.n_rates + 3])
        Q = np.zeros((2 * k, 2 * k))
        Q[:k, :k], Q[k:, k:] = base, m * base
        Q[np.arange(k), np.arange(k) + k] = s01
        Q[np.arange(k) + k, np.arange(k)] = s10
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(1))
        return Q

    def observed(self, full_state):
        """Observable state of an expanded state (HRM index = class * k + state)."""
        return np.asarray(full_state) % self.k

    def tip_lik(self, obs: np.ndarray) -> np.ndarray:
        L = np.zeros((len(obs), self.S))
        L[np.arange(len(obs)), obs] = 1.0
        if self.kind == "HRM":
            L[np.arange(len(obs)), obs + self.k] = 1.0  # a tip does not reveal its hidden rate class
        return L

    def log_prior(self, theta: np.ndarray) -> float:
        """Weakly informative: Normal(0, 2) on every log-rate (95% of the mass between 0.02 and 55 per unit length)."""
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        return float(-0.5 * np.sum((theta / 2.0) ** 2))


def stationary(Q: np.ndarray) -> np.ndarray:
    """pi with pi Q = 0, sum(pi) = 1."""
    S = Q.shape[0]
    A = np.vstack([Q.T, np.ones(S)])
    b = np.zeros(S + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


def transition_matrices(tree: Tree, Q: np.ndarray) -> np.ndarray:
    return expm(Q[None] * tree.blen[:, None, None])


def conditional_likelihoods(tree: Tree, P: np.ndarray, tip_lik: np.ndarray):
    """Felsenstein pruning. Returns (normalised partial likelihoods (N, S), total log scale)."""
    S = P.shape[1]
    L = np.zeros((tree.n_nodes, S))
    L[tree.tips] = tip_lik
    log_scale = 0.0
    c0, c1 = tree.children[:, 0], tree.children[:, 1]
    for lev in tree.levels_up:
        a = np.einsum("nij,nj->ni", P[c0[lev]], L[c0[lev]])
        b = np.einsum("nij,nj->ni", P[c1[lev]], L[c1[lev]])
        v = a * b
        s = v.max(1, keepdims=True)
        L[lev] = v / s
        log_scale += float(np.log(s).sum())
    return L, log_scale


class Likelihood:
    def __init__(self, tree: Tree, model: RateModel, obs: np.ndarray):
        self.tree, self.model, self.obs = tree, model, np.asarray(obs)
        self.tip_lik = model.tip_lik(self.obs)
        self.evals = 0

    def __call__(self, theta: np.ndarray) -> float:
        self.evals += 1
        Q = self.model.Q(theta)
        P = transition_matrices(self.tree, Q)
        L, ls = conditional_likelihoods(self.tree, P, self.tip_lik)
        val = float(stationary(Q) @ L[self.tree.root])
        return ls + np.log(val) if val > 0 else -np.inf


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_ml(lik: Likelihood, starts: int = 4, seed: int = 0, hessian: bool = True) -> dict:
    m, rng = lik.model, np.random.default_rng(seed)
    bounds = list(zip(m.lower, m.upper))

    def nll(th):
        v = lik(th)
        return -v if np.isfinite(v) else 1e12

    best = None
    for s in range(starts):
        th0 = rng.uniform(np.log(0.05), np.log(3.0), m.npar)
        th0 = np.clip(th0, m.lower + 1e-6, m.upper - 1e-6)
        r = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    out = {"theta": best.x, "loglik": -best.fun, "aic": 2 * m.npar + 2 * best.fun, "npar": m.npar, "kind": m.kind,
           "at_bound": bool(np.any(best.x <= m.lower + 1e-3) or np.any(best.x >= m.upper - 1e-3))}
    if hessian:
        out["hessian"] = numerical_hessian(nll, best.x, m)
    return out


def numerical_hessian(f, x: np.ndarray, model: RateModel, h: float = 5e-3) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    step = np.full(n, h)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / step[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * step[i] * step[j])
    return H


def run_mcmc(lik: Likelihood, theta0: np.ndarray, hessian: np.ndarray | None, n_iter: int, rng: np.random.Generator,
             burn: float = 0.3) -> dict:
    """Adaptive random-walk Metropolis on the log-rates. Proposal covariance starts from the inverse Hessian and the
    step size is tuned toward 25% acceptance during burn-in only, so the retained draws come from a fixed kernel."""
    m = lik.model
    d = m.npar
    cov = np.eye(d) * 0.05
    if hessian is not None:
        try:
            w, V = np.linalg.eigh(hessian)
            w = np.clip(w, 1e-2, None)  # a flat direction gets a wide (not infinite) proposal
            cov = V @ np.diag(1.0 / w) @ V.T
        except np.linalg.LinAlgError:
            pass
    cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(d)
    scale = 2.38 ** 2 / d
    chol = np.linalg.cholesky(cov)

    def logpost(th):
        lp = m.log_prior(th)
        return -np.inf if not np.isfinite(lp) else lp + lik(th)

    th, lp = np.asarray(theta0, dtype=float).copy(), None
    lp = logpost(th)
    chain, acc_recent, n_burn = np.zeros((n_iter, d)), [], int(burn * n_iter)
    acc_total = 0
    for it in range(n_iter):
        prop = th + np.sqrt(scale) * chol @ rng.standard_normal(d)
        lpp = logpost(prop)
        ok = np.log(rng.random()) < lpp - lp
        if ok:
            th, lp = prop, lpp
        chain[it] = th
        if it >= n_burn:
            acc_total += ok
        acc_recent.append(ok)
        if it < n_burn and (it + 1) % 100 == 0:
            r = np.mean(acc_recent[-100:])
            scale *= np.exp(np.clip(r - 0.25, -0.2, 0.2) * 2)
    post = chain[n_burn:]
    return {"chain": post, "acceptance": acc_total / max(len(post), 1), "ess": effective_sample_size(post)}


def effective_sample_size(x: np.ndarray) -> np.ndarray:
    """Per-column ESS from the initial positive autocorrelation sequence."""
    n, d = x.shape
    out = np.zeros(d)
    for j in range(d):
        v = x[:, j] - x[:, j].mean()
        var = v @ v / n
        if var <= 0:
            out[j] = n
            continue
        f = np.fft.rfft(v, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n] / (n * var)
        tau = 1.0
        for lag in range(1, n // 2):
            if ac[lag] < 0.05:
                break
            tau += 2 * ac[lag]
        out[j] = n / tau
    return out


def thin(chain: np.ndarray, n_draws: int) -> np.ndarray:
    idx = np.linspace(0, len(chain) - 1, n_draws).astype(int)
    return chain[idx]


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------
def simulate_tips(tree: Tree, model: RateModel, thetas: np.ndarray, reps_per_draw: int, rng: np.random.Generator) -> np.ndarray:
    """Observable tip states, shape (n_tips, n_draws * reps_per_draw). Each parameter draw is used for `reps_per_draw`
    independent replicates, so parameter uncertainty is carried into the null; root states come from pi(Q)."""
    thetas = np.atleast_2d(thetas)
    out = []
    for th in thetas:
        Q = model.Q(th)
        cum = np.cumsum(transition_matrices(tree, Q), axis=2)
        st = np.zeros((tree.n_nodes, reps_per_draw), dtype=np.int8)
        st[tree.root] = rng.choice(model.S, size=reps_per_draw, p=stationary(Q))
        for nodes in tree.levels_down:
            u = rng.random((len(nodes), reps_per_draw))
            c = cum[nodes[:, None], st[tree.parent[nodes]]]  # (nodes, reps, S)
            st[nodes] = np.minimum((u[..., None] > c).sum(2), model.S - 1).astype(np.int8)
        out.append(model.observed(st[tree.tips]).astype(np.int8))
    return np.concatenate(out, axis=1)


# ---------------------------------------------------------------------------
# Stochastic mapping
# ---------------------------------------------------------------------------
def sample_node_states(tree: Tree, model: RateModel, theta: np.ndarray, obs: np.ndarray, rng: np.random.Generator):
    """Joint draw of all node states given the tip data (exact, root ~ posterior under pi(Q)). Returns (states, Q, P)."""
    Q = model.Q(theta)
    P = transition_matrices(tree, Q)
    L, _ = conditional_likelihoods(tree, P, model.tip_lik(obs))
    st = np.zeros(tree.n_nodes, dtype=int)
    w = stationary(Q) * L[tree.root]
    st[tree.root] = rng.choice(model.S, p=w / w.sum())
    for nodes in tree.levels_down:
        w = P[nodes, st[tree.parent[nodes]], :] * L[nodes]
        w = w / w.sum(1, keepdims=True)
        st[nodes] = (rng.random((len(nodes), 1)) > np.cumsum(w, axis=1)).sum(1).clip(0, model.S - 1)
    return st, Q, P


def sample_branch_histories(tree: Tree, Q: np.ndarray, states: np.ndarray, rng: np.random.Generator):
    """Exact endpoint-conditioned CTMC paths on every branch (uniformization / bridge sampling).

    For a branch of length t from a to b: draw the number of uniformised events N with
    P(N=n) proportional to Poisson(n; lam*t) * (R^n)_{ab}, R = I + Q/lam, then the intermediate states of the
    R-chain bridge, then N uniform event times. Only real state changes are returned.
    Returns arrays (branch node index, time above the child node's parent end, from, to).
    """
    S = Q.shape[0]
    nodes = np.flatnonzero(tree.parent >= 0)
    t = tree.blen[nodes]
    a, b = states[tree.parent[nodes]], states[nodes]
    lam = float(max(-np.diag(Q).min(), 1e-9))
    R = np.eye(S) + Q / lam
    lt_max = lam * float(t.max())
    nmax = int(lt_max + 6 * np.sqrt(lt_max) + 12)
    Rp = np.empty((nmax + 1, S, S))
    Rp[0] = np.eye(S)
    for n in range(1, nmax + 1):
        Rp[n] = Rp[n - 1] @ R
    ns = np.arange(nmax + 1)
    lt = lam * t
    with np.errstate(divide="ignore", invalid="ignore"):
        logpois = -lt[:, None] + ns[None, :] * np.log(lt)[:, None] - gammaln(ns + 1)[None, :]
    logpois[:, 0] = -lt  # 0 * log(0) := 0
    w = np.exp(logpois) * np.clip(Rp[:, a, b].T, 0, None)
    w = w / w.sum(1, keepdims=True)
    n_ev = (rng.random((len(nodes), 1)) > np.cumsum(w, axis=1)).sum(1).clip(0, nmax)
    total = int(n_ev.sum())
    if total == 0:
        return (np.empty(0, int),) * 4
    seg = np.repeat(np.arange(len(nodes)), n_ev)
    u = rng.random(total)
    order = np.lexsort((u, seg))
    u_sorted = u[order]
    start = np.cumsum(n_ev) - n_ev
    cur = a.copy()
    ev_b, ev_t, ev_f, ev_to = [], [], [], []
    for i in range(1, int(n_ev.max()) + 1):
        act = np.flatnonzero(n_ev >= i)
        rem = n_ev[act] - i
        wgt = R[cur[act]] * Rp[rem, :, b[act]]  # (act, S): R[cur, j] * (R^rem)[j, b]
        s = wgt.sum(1, keepdims=True)
        wgt = np.where(s > 0, wgt / np.where(s > 0, s, 1), np.eye(S)[cur[act]])
        nxt = (rng.random((len(act), 1)) > np.cumsum(wgt, axis=1)).sum(1).clip(0, S - 1)
        chg = nxt != cur[act]
        if chg.any():
            sel = act[chg]
            ev_b.append(nodes[sel])
            ev_t.append(u_sorted[start[sel] + i - 1] * t[sel])
            ev_f.append(cur[sel])
            ev_to.append(nxt[chg])
        cur[act] = nxt
    if not ev_b:
        return (np.empty(0, int),) * 4
    return np.concatenate(ev_b), np.concatenate(ev_t), np.concatenate(ev_f), np.concatenate(ev_to)


def stochastic_map(tree: Tree, model: RateModel, theta: np.ndarray, obs: np.ndarray, rng: np.random.Generator) -> dict:
    """One stochastic character map: node states and every state change (branch, time, from, to)."""
    states, Q, _ = sample_node_states(tree, model, theta, obs, rng)
    br, tm, fr, to = sample_branch_histories(tree, Q, states, rng)
    return {"states": states, "branch": br, "time": tm, "from": fr, "to": to}
