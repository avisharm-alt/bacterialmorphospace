"""Unit tests for src/embed_core.py: no network, no GPU, no Modal."""

import gzip
import math
import os

import pytest

from src import embed_core as core


# --- NCBI paths ---------------------------------------------------------------------------
def test_ncbi_parent_url_refseq_and_genbank():
    assert core.ncbi_parent_url("GCF_000005845.2").endswith("/genomes/all/GCF/000/005/845/")
    assert core.ncbi_parent_url("GCA_001580535.1").endswith("/genomes/all/GCA/001/580/535/")


@pytest.mark.parametrize("bad", ["GCF_000005845", "GCX_000005845.2", "", "GCF_5845.2"])
def test_ncbi_parent_url_rejects_unversioned_or_malformed(bad):
    with pytest.raises(ValueError):
        core.ncbi_parent_url(bad)


def test_find_assembly_dir_picks_exact_accession_not_prefix_neighbour():
    html = ('<a href="GCF_000005845.2_ASM584v2/">x</a> <a href="GCF_000005845.20_Other/">y</a> '
            '<a href="GCF_000005846.1_Z/">z</a>')
    assert core.find_assembly_dir(html, "GCF_000005845.2") == "GCF_000005845.2_ASM584v2"
    assert core.find_assembly_dir(html, "GCF_000005847.1") is None


def test_parse_md5_file():
    txt = "AbC123  ./GCF_1_x_genomic.fna.gz\ndef456  ./README.txt\n"
    assert core.parse_md5_file(txt) == {"GCF_1_x_genomic.fna.gz": "abc123", "README.txt": "def456"}


# --- download verification: the silent zero-byte failure --------------------------------------
def _fasta_gz(path, seq_len=150_000):  # random DNA gzips to ~2 bits/base; must clear the 20 kB floor
    import random

    rng = random.Random(0)
    seq = "".join(rng.choice("ACGT") for _ in range(seq_len))  # random so gzip cannot shrink it below the floor
    with gzip.open(path, "wt") as f:
        f.write(">c1 test\n" + "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80)) + "\n")
    return path


def test_verify_accepts_good_file(tmp_path):
    p = _fasta_gz(tmp_path / "g.fna.gz")
    core.verify_download(p, expected_bytes=p.stat().st_size, expected_md5=core.md5_of(p))


def test_verify_rejects_zero_byte_file(tmp_path):
    p = tmp_path / "empty.fna.gz"
    p.write_bytes(b"")
    with pytest.raises(IOError, match="floor"):
        core.verify_download(p)


def test_verify_rejects_truncated_file(tmp_path):
    p = _fasta_gz(tmp_path / "g.fna.gz")
    full = p.stat().st_size
    p.write_bytes(p.read_bytes()[: full // 2])
    with pytest.raises(IOError):
        core.verify_download(p, expected_bytes=full)


def test_verify_rejects_truncated_gzip_even_without_announced_length(tmp_path):
    p = _fasta_gz(tmp_path / "g.fna.gz", seq_len=200_000)
    p.write_bytes(p.read_bytes()[:-500])
    # header may still decode; the point is that some check rejects it or the read raises
    with pytest.raises((IOError, EOFError)):
        core.verify_download(p)
        core.read_contigs(p)


def test_verify_rejects_md5_mismatch_and_non_fasta(tmp_path):
    p = _fasta_gz(tmp_path / "g.fna.gz")
    with pytest.raises(IOError, match="md5"):
        core.verify_download(p, expected_md5="0" * 32)
    q = tmp_path / "junk.gz"
    with gzip.open(q, "wb") as f:
        f.write(b"<html>" + bytes(range(256)) * 200 + os.urandom(30_000))  # an HTML error page, not FASTA
    with pytest.raises(IOError, match="FASTA"):
        core.verify_download(q)


def test_read_contigs_keeps_only_usable_in_file_order(tmp_path):
    p = tmp_path / "m.fna.gz"
    with gzip.open(p, "wt") as f:
        f.write(">a desc\nacgtacgt\nAC\n>tiny\nAAAA\n>b\nCCCCCCCC\n")
    usable, total, n = core.read_contigs(p, min_len=8)
    assert usable == [("a", "ACGTACGTAC"), ("b", "CCCCCCCC")]  # uppercased, tiny dropped, file order kept
    assert (total, n) == (10 + 4 + 8, 3)


def test_read_contigs_rejects_empty_fasta(tmp_path):
    p = tmp_path / "e.fna.gz"
    with gzip.open(p, "wt") as f:
        f.write("")
    with pytest.raises(ValueError, match="no FASTA"):
        core.read_contigs(p)


# --- windows across all contigs ---------------------------------------------------------------
W = core.WINDOW


def _axis_start(lens, window):
    """Axis coordinate (usable contigs laid end to end) of a (contig, start) window."""
    offs, acc = {}, 0
    for i, L in enumerate(lens):
        if L >= W:
            offs[i] = acc
            acc += L
    return offs[window[0]] + window[1]


def test_pool_windows_single_contig_is_even_and_in_bounds():
    ws = core.pool_windows([100_000], 10)
    starts = [s for _, s in ws]
    assert all(c == 0 for c, _ in ws) and len(set(starts)) == 10
    assert 0 <= min(starts) and max(starts) <= 100_000 - W
    gaps = [b - a for a, b in zip(starts, starts[1:])][1:-1]  # end windows are clamped inside the contig
    assert max(gaps) - min(gaps) <= 1
    assert core.pool_windows([100_000], 1) == [(0, int(min(max(50_000 - W / 2, 0), 100_000 - W)))]


def test_pool_windows_never_crosses_a_contig_boundary():
    lens = [30_000, 9_000, 8_192, 500_000, 20_000]
    for n in (1, 7, 50, 100):
        for c, s in core.pool_windows(lens, n):
            assert 0 <= s and s + W <= lens[c]


def test_pool_windows_skips_contigs_shorter_than_a_window():
    lens = [5_000, 200_000, 8_191, 100_000, 100]
    used = {c for c, _ in core.pool_windows(lens, 100)}
    assert used == {1, 3}


def test_pool_windows_allocates_in_proportion_to_length():
    lens = [800_000, 150_000, 50_000]
    counts = [0, 0, 0]
    for c, _ in core.pool_windows(lens, 100):
        counts[c] += 1
    assert counts == [80, 15, 5] or all(abs(a - b) <= 1 for a, b in zip(counts, [80, 15, 5]))


def test_pool_windows_does_not_discard_the_rest_of_a_fragmented_draft():
    """The bias being fixed: longest-contig-only sampling would put 0 of 100 windows on the other 20 contigs."""
    lens = [1_000_000] + [50_000] * 20  # longest contig holds half of the usable sequence
    on_other = sum(1 for c, _ in core.pool_windows(lens, 100) if c != 0)
    assert 45 <= on_other <= 55


def test_pool_windows_rejects_genome_with_no_usable_contig():
    with pytest.raises(ValueError, match="no contig"):
        core.pool_windows([W - 1, 100], 5)


def test_usable_len_counts_only_contigs_that_can_host_a_window():
    assert core.usable_len([W - 1, W, 2 * W, 10]) == 3 * W


@pytest.mark.parametrize("n", [10, 25, 50, 100])
def test_subset_indices_distinct_span_and_evenly_spaced(n):
    idx = core.subset_indices(100, n)
    assert len(idx) == len(set(idx)) == n and idx[0] == 0 and idx[-1] == 99
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert max(gaps) - min(gaps) <= 1  # within one pool step of perfectly even


def test_subset_windows_stay_within_half_a_pool_step_of_an_independent_draw():
    lens, n, pool = [3_000_000], 25, 100
    pool_w = core.pool_windows(lens, pool)
    sub = [_axis_start(lens, pool_w[i]) for i in core.subset_indices(pool, n)]
    ideal = [_axis_start(lens, w) for w in core.pool_windows(lens, n)]
    step = (lens[0] - 1) / (pool - 1)
    assert max(abs(a - b) for a, b in zip(sub, ideal)) <= step / 2 + 1


def test_subset_keeps_per_contig_allocation_close_to_an_independent_draw():
    lens = [700_000, 200_000, 60_000, 40_000]
    pool_w = core.pool_windows(lens, 100)
    for n in (10, 25, 50):
        sub = [pool_w[i][0] for i in core.subset_indices(100, n)]
        ind = [w[0] for w in core.pool_windows(lens, n)]
        for c in range(len(lens)):
            assert abs(sub.count(c) - ind.count(c)) <= 1


def test_subset_indices_bounds():
    with pytest.raises(ValueError):
        core.subset_indices(10, 11)
    assert core.subset_indices(10, 10) == list(range(10))


def test_depth_seconds_uses_only_that_depths_windows():
    times = [float(i) for i in range(100)]
    assert core.depth_seconds(times, 100, 100) == sum(times)
    assert core.depth_seconds(times, 100, 10) == sum(times[i] for i in core.subset_indices(100, 10))


# --- sampling --------------------------------------------------------------------------------
def test_pool_groups_pools_small_phyla():
    ph = ["A"] * 10 + ["B"] * 9 + ["C"]
    assert core.pool_groups(ph) == ["A"] * 10 + ["other"] * 10


def test_allocate_caps_big_groups_and_fills_exactly():
    sizes = {"big1": 813, "big2": 733, "small": 12, "tiny": 3}
    a = core.allocate(sizes, 100)
    assert sum(a.values()) == 100
    assert a["tiny"] == 3 and a["small"] == 12  # small groups fully represented
    assert abs(a["big1"] - a["big2"]) <= 1  # remainder split, not proportional


def test_allocate_total_larger_than_population():
    assert core.allocate({"a": 3, "b": 2}, 50) == {"a": 3, "b": 2}


def test_panel_allocation_matches_expected_shape():
    sizes = {"Pseudomonadota": 813, "Bacillota": 733, "Actinomycetota": 481, "Bacteroidota": 409,
             "Desulfobacterota": 25, "Deinococcota": 21, "Acidobacteriota": 16, "Chloroflexota": 16,
             "Campylobacterota": 13, "Verrucomicrobiota": 12, "other": 41}
    a = core.allocate(sizes, 200)
    assert sum(a.values()) == 200 and max(a.values()) <= 22
    assert a["Verrucomicrobiota"] == 12 and a["Campylobacterota"] == 13


def test_stratified_sample_is_deterministic_and_unique():
    rows = [{"ncbi_assembly_accession": f"GCF_{i:09d}.1", "group": "g%d" % (i % 4)} for i in range(400)]
    a = core.stratified_sample(rows, 40, seed=1)
    b = core.stratified_sample(rows, 40, seed=1)
    c = core.stratified_sample(rows, 40, seed=2)
    assert a == b and a != c and len(a) == 40
    assert len({r["ncbi_assembly_accession"] for r in a}) == 40


# --- cost ------------------------------------------------------------------------------------
def test_estimate_reflects_the_recipe_numbers():
    e = core.estimate_gpu(200, 100, n_containers=1)
    assert 4.0 < e["gpu_hours"] < 4.6 and 8.5 < e["usd"] < 10  # the sweep is NOT a "couple of dollars"
    assert core.estimate_gpu(2580, 25)["usd"] > 29  # 25 windows across the panel already eats a $30 credit


# --- evaluation ------------------------------------------------------------------------------
def test_evaluate_depth_learns_signal_and_scaling_matters():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 300
    groups = np.array([f"g{i % 6}" for i in range(n)])
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 40)) * rng.uniform(0.01, 1000, 40)  # wildly different feature scales
    X[:, 0] += y * 3 * (np.std(X[:, 0]))  # informative feature
    res = core.evaluate_depth(X, y, groups, n_splits=3)
    assert res["folds_scored"] == 3 and res["auc_mean"] > 0.8


def test_evaluate_depth_skips_single_class_folds_and_counts_them():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    groups = np.array(["a"] * 30 + ["b"] * 30 + ["c"] * 30)
    y = np.array([0] * 30 + [0, 1] * 15 + [0, 1] * 15)  # group a is uniformly negative
    X = rng.normal(size=(90, 5))
    res = core.evaluate_depth(X, y, groups, n_splits=3)
    assert res["folds_total"] == 3 and res["folds_scored"] == 2


def test_select_depth_one_se_rule():
    res = [
        {"depth": 10, "auc_mean": 0.80, "auc_std": 0.06, "folds_scored": 4},
        {"depth": 25, "auc_mean": 0.84, "auc_std": 0.05, "folds_scored": 4},
        {"depth": 50, "auc_mean": 0.86, "auc_std": 0.05, "folds_scored": 4},
        {"depth": 100, "auc_mean": 0.87, "auc_std": 0.05, "folds_scored": 4},
    ]
    sel = core.select_depth(res)  # SE = 0.025; threshold 0.845 -> 50 is the smallest that qualifies
    assert sel["best_depth"] == 100 and sel["selected_depth"] == 50 and math.isclose(sel["one_se"], 0.025)


# --- remote return values must be plain Python (the local process has no torch/numpy) -----------
def _pickle_globals(obj):
    """Module-qualified names a pickle of `obj` would need at load time (empty for builtin primitives)."""
    import pickle
    import pickletools

    names, strings = [], []
    for op, arg, _ in pickletools.genops(pickle.dumps(obj)):
        if op.name in ("GLOBAL", "INST"):
            names.append(arg)
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            names.append(tuple(strings[-2:]))
    return names


def test_plain_turns_str_subclass_from_torch_into_builtin_str(monkeypatch):
    import sys
    import types

    fake_torch = types.ModuleType("fake_torch_version")  # stands in for the module the local side lacks
    fake_torch.TorchVersion = type("TorchVersion", (str,), {"__module__": "fake_torch_version"})
    monkeypatch.setitem(sys.modules, "fake_torch_version", fake_torch)
    v = fake_torch.TorchVersion("2.8.0+cu128")  # torch.__version__ is a str subclass exactly like this
    assert _pickle_globals(v)  # the trap: it pickles by reference to the (absent) module
    out = core.plain({"torch": v, "nested": [v, {"k": v}]})
    assert out == {"torch": "2.8.0+cu128", "nested": ["2.8.0+cu128", {"k": "2.8.0+cu128"}]}
    assert _pickle_globals(out) == []
    assert type(out["torch"]) is str


def test_plain_converts_numpy_scalars_and_int_float_subclasses():
    np = pytest.importorskip("numpy")
    out = core.plain({"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True), "d": (1, 2.0, "x", None)})
    assert out == {"a": 1.5, "b": 3, "c": True, "d": [1, 2.0, "x", None]}
    assert all(type(v) in (float, int, bool, list) for v in out.values())
    assert _pickle_globals(out) == []


def test_plain_rejects_non_primitives_with_the_path():
    class Device:
        pass

    Device.__module__ = "torch"
    with pytest.raises(TypeError, match=r"return\['gpu'\]\[1\]"):
        core.plain({"gpu": ["ok", Device()]})
    with pytest.raises(TypeError):
        core.plain({"arr": object()})


def test_evaluate_depth_result_is_plain():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    import numpy as np

    rng = np.random.default_rng(0)
    groups = np.array([f"g{i % 4}" for i in range(80)])
    res = core.evaluate_depth(rng.normal(size=(80, 6)), rng.integers(0, 2, 80), groups, 4)
    assert core.plain(res) == res and _pickle_globals(core.plain(res)) == []


def test_every_remote_function_returns_through_plain_call():
    """Static audit of src/evo2_modal.py: no @app.function / @modal.method may return a raw value."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parent.parent / "src" / "evo2_modal.py").read_text())
    remote = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                if isinstance(target, ast.Attribute) and target.attr in ("function", "method"):
                    remote.append(node)
    assert {f.name for f in remote} >= {"verify_env", "prime_weights", "download_genome", "embed", "evaluate_sweep"}
    for f in remote:
        rets = [n for n in ast.walk(f) if isinstance(n, ast.Return)]
        assert len(rets) == 1, f"{f.name} must have a single return"
        call = rets[0].value
        assert isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_plain_call", (
            f"{f.name} returns without going through _plain_call (raw values may not unpickle locally)")


def test_plain_call_reraises_foreign_exceptions_as_builtin_runtimeerror():
    pytest.importorskip("modal")
    from src import evo2_modal

    class HubError(Exception):
        pass

    HubError.__module__ = "huggingface_hub.errors"

    def boom():
        raise HubError("401 unauthorized")

    with pytest.raises(RuntimeError, match="HubError: 401 unauthorized") as ei:
        evo2_modal._plain_call(boom)
    assert type(ei.value) is RuntimeError
    assert evo2_modal._plain_call(lambda: {"v": 1}) == {"v": 1}
