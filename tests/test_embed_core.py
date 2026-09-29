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


# --- failure visibility (these reproduce the two reported bugs against a stand-in for Modal) ----------
class _FakeEmbed:
    """Stands in for `Embedder().embed.map(...)`: returns one outcome per input, in input order."""

    def __init__(self, outcomes):
        self.outcomes = outcomes

    def map(self, accs, **kw):
        assert kw.get("return_exceptions") is True  # one failure must not cancel its siblings
        assert kw.get("order_outputs", True) is True  # ordered, so results can be paired with accessions
        return iter(self.outcomes[: len(accs)])


def _patch_embedder(monkeypatch, outcomes):
    evo2_modal = pytest.importorskip("modal") and __import__("src.evo2_modal", fromlist=["x"])
    fake = _FakeEmbed(outcomes)
    monkeypatch.setattr(evo2_modal, "Embedder", lambda: type("E", (), {"embed": fake})())
    return evo2_modal


def _ok(acc, s=80.0):
    return {"acc": acc, "ok": True, "status": "embedded", "gpu_seconds": s, "task_id": "t1", "load_s": 100.0}


def test_failures_are_always_paired_with_their_accession_and_first_is_printed_in_full(monkeypatch, capsys):
    accs = ["GCF_1.1", "GCF_2.1", "GCF_3.1", "GCF_4.1"]
    boom = RuntimeError("remote function failed:\nTraceback (most recent call last):\n  File x.py\nCudaError: out of memory")
    m = _patch_embedder(monkeypatch, [_ok(accs[0]), boom, _ok(accs[2]),
                                      {"acc": accs[3], "ok": False, "error": "Traceback...\nValueError: no contig"}])
    fails, spent = m._run_embedding(accs, [10], 1, max_usd=50, save_windows=True)
    assert [f["acc"] for f in fails] == ["GCF_2.1", "GCF_4.1"]  # never "?"
    out = capsys.readouterr().out
    assert "FIRST FAILURE, GCF_2.1" in out and "CudaError: out of memory" in out  # real traceback, verbatim
    assert "cancelled by user" not in out and spent > 0


def test_systemic_failure_aborts_by_name_after_three_leading_failures(monkeypatch, capsys):
    accs = [f"GCF_{i}.1" for i in range(8)]
    m = _patch_embedder(monkeypatch, [RuntimeError("model failed to load: OSError bad checkpoint")] * 8)
    with pytest.raises(SystemExit) as e:
        m._run_embedding(accs, [10], 1, max_usd=50, save_windows=True)
    assert e.value.code == 3
    out = capsys.readouterr().out
    assert "ABORT (not a Modal cancellation)" in out and "FIRST FAILURE, GCF_0.1" in out
    assert "bad checkpoint" in out and "embedding stage: 0 ok, 3 failed of 3 returned" in out


def test_one_bad_genome_among_successes_does_not_abort(monkeypatch):
    accs = [f"GCF_{i}.1" for i in range(6)]
    outcomes = [_ok(a) for a in accs]
    outcomes[1] = {"acc": accs[1], "ok": False, "error": "ValueError: no contig >= 8192"}
    outcomes[2] = outcomes[3] = outcomes[4] = {"acc": "x", "ok": False, "error": "ValueError: short"}
    m = _patch_embedder(monkeypatch, outcomes)
    fails, _ = m._run_embedding(accs, [10], 1, max_usd=50, save_windows=True)  # successes came first: no abort
    assert len(fails) == 4


def test_budget_stop_is_announced_explicitly_and_spend_is_printed(monkeypatch, capsys):
    accs = [f"GCF_{i}.1" for i in range(10)]
    m = _patch_embedder(monkeypatch, [_ok(a, s=1500.0) for a in accs])  # ~$0.87 per genome
    with pytest.raises(SystemExit) as e:
        m._run_embedding(accs, [10], 1, max_usd=2.0, save_windows=True)
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "BUDGET STOP" in out and "--max-usd 2.0" in out and "GPU spend ~$" in out


def test_evaluate_sweep_refuses_genomes_that_were_never_embedded(tmp_path, monkeypatch):
    m = _patch_embedder(monkeypatch, [])
    monkeypatch.setattr(m, "EMB_DIR", str(tmp_path))
    monkeypatch.setattr(m, "out_vol", type("V", (), {"reload": staticmethod(lambda: None)})())
    rows = [{"acc": f"GCF_{i}.1", "motility": "yes", "group": "g"} for i in range(3)]
    with pytest.raises(RuntimeError, match=r"0 of 3 genomes embedded"):
        m._evaluate_sweep(rows, [10, 25], 3)
    # npy present but meta missing is still refused, with the right count
    for r in rows[:2]:
        for n in (10, 25):
            (tmp_path / f"{r['acc']}__n{n}.npy").write_bytes(b"x")
    (tmp_path / "GCF_0.1__meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"1 of 3 genomes embedded"):
        m._evaluate_sweep(rows, [10, 25], 3)
    with pytest.raises(RuntimeError, match=r"0 of 0 genomes embedded"):
        m._evaluate_sweep([], [10], 3)


def test_plain_call_keeps_the_original_traceback():
    m = _patch_embedder(pytest.MonkeyPatch(), [])

    def deep():
        raise ValueError("the real cause")

    with pytest.raises(RuntimeError) as ei:
        m._plain_call(deep)
    assert "the real cause" in str(ei.value) and "in deep" in str(ei.value) and "Traceback" in str(ei.value)


# --- diagnostics: run the real remote code path against a fake volume --------------------------------
def _fake_volume(tmp_path, misalign=False, n=36, dim=48, seed=0):
    """Synthetic genomes whose embeddings really encode GC, plus phylum structure. Returns the sweep rows."""
    import json
    import random

    np = pytest.importorskip("numpy")
    rng, py = np.random.default_rng(seed), random.Random(seed)
    emb, gen = tmp_path / "embeddings", tmp_path / "genomes"
    emb.mkdir(), gen.mkdir()
    phyla = ["A", "B", "C", "D", "E", "F"]
    centres = {p: rng.normal(size=dim) for p in phyla}
    gc_dir = rng.normal(size=dim)
    rows, vectors = [], {}
    for i in range(n):
        acc, ph = f"GCF_{i:09d}.1", phyla[i % 6]
        gc_target = 0.3 + 0.4 * rng.random()
        contigs = {f"ctg{k}": "".join("GC" if py.random() < gc_target else "AT" for _ in range(6000)) for k in range(3)}
        with gzip.open(gen / f"{acc}.fna.gz", "wt") as f:
            for cid, s in contigs.items():
                f.write(f">{cid} x\n{s}\n")
        wins = [[f"ctg{k % 3}", 0] for k in range(100)]
        pool = np.stack([centres[ph] + 8 * (gc_target - .5) * gc_dir + 0.3 * rng.normal(size=dim) for _ in range(100)]).astype(np.float32)
        vectors[acc] = pool
        rows.append({"acc": acc, "motility": "yes" if rng.random() < 0.5 else "no", "group": ph})
        json.dump({"acc": acc, "pool_n": 100, "windows": wins, "sampling": core.SAMPLING}, open(emb / f"{acc}__meta.json", "w"))
    accs = [r["acc"] for r in rows]
    for j, acc in enumerate(accs):
        src = vectors[accs[(j + 1) % n]] if misalign else vectors[acc]  # misalign: genome j gets genome j+1's embeddings
        np.save(emb / f"{acc}__windows.npy", src)
        for d in (10, 25):
            np.save(emb / f"{acc}__n{d}.npy", src[core.subset_indices(100, d)].mean(axis=0).astype(np.float32))
    return rows


def _patched(monkeypatch, tmp_path):
    m = _patch_embedder(monkeypatch, [])
    monkeypatch.setattr(m, "EMB_DIR", str(tmp_path / "embeddings"))
    monkeypatch.setattr(m, "GENOME_DIR", str(tmp_path / "genomes"))
    monkeypatch.setattr(m, "out_vol", type("V", (), {"reload": staticmethod(lambda: None)})())
    return m


def test_diagnose_passes_when_embeddings_are_correctly_paired(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    m = _patched(monkeypatch, tmp_path)
    rows = _fake_volume(tmp_path)
    d = m._diagnose_sweep(rows, [10, 25], 3, 4, 1)
    a = d["alignment"]
    assert a["all_meta_acc_match"] and a["all_windows_on_own_genome"] and a["all_npy_equal_mean_of_own_windows"]
    assert len(a["audit"]) == 5 and all(x["windows_lie_on_this_genomes_contigs"] for x in a["audit"])
    assert d["pairing_control"]["spearman_r"] > 0.6 > abs(d["pairing_control"]["shuffled_pairing_r"])
    assert d["depths"]["n10"]["fold_report"]["classes"] == [False, True]
    assert set(d["depths"]["n10"]) >= {"null_global", "null_within_group", "ungrouped_stratified_cv", "real"}
    assert 0 <= d["gc_baseline"]["mean_gc"] <= 1


def test_diagnose_detects_embeddings_paired_to_the_wrong_genome(tmp_path, monkeypatch):
    """If rows were misaligned, the GC pairing control must collapse, even though every file 'exists'."""
    pytest.importorskip("sklearn")
    m = _patched(monkeypatch, tmp_path)
    rows = _fake_volume(tmp_path, misalign=True)
    d = m._diagnose_sweep(rows, [10], 3, 3, 1)
    assert d["pairing_control"]["spearman_r"] < 0.35  # correct pairing gave > 0.6 on the same data


def test_diagnose_entrypoint_prints_every_requested_section(tmp_path, monkeypatch, capsys):
    pytest.importorskip("sklearn")
    m = _patched(monkeypatch, tmp_path)
    rows = _fake_volume(tmp_path)
    canned = m._diagnose_sweep(rows, [10, 25], 3, 3, 1)
    stub = type("F", (), {"remote": staticmethod(lambda *a, **k: canned)})()
    monkeypatch.setattr(m, "diagnose_sweep", stub)
    genomes = tmp_path / "g.tsv"
    # the fixture has 6 species per phylum (< 10), so the panel's own pooling maps every phylum to "other"
    genomes.write_text("acc\tgroup\tmotility\n" + "".join(f"{r['acc']}\tother\t{r['motility']}\n" for r in rows))
    panel = tmp_path / "panel.tsv"
    panel.write_text("ncbi_assembly_accession\tassembly_genbank\tgtdb_phylum\tmotility\n" + "".join(
        f"{r['acc']}\t\t{r['group']}\t{r['motility']}\n" for r in rows))
    m.diagnose.info.raw_f(panel=str(panel), genomes=str(genomes), depths="10,25", n_perm=3, n_splits=3, reports=str(tmp_path / "rep"))
    out = capsys.readouterr().out
    for needle in ("36/36 match", "LABEL ALIGNMENT", "pairing control", "POSITIVE CLASS", "classes_ = [False, True]",
                   "FOLD STRUCTURE", "prev_test", "SHUFFLED-LABEL NULLS", "shuffle within phylum", "GC-CONTENT-ONLY"):
        assert needle in out, needle
    assert (tmp_path / "rep" / "tables" / "evo2_sweep_diagnostics.json").exists()


# --- trait choice: within-phylum variance ------------------------------------------------------------
def test_variance_decomposition_extremes():
    determined = core.variance_decomposition(["a"] * 10 + ["b"] * 10, ["g1"] * 10 + ["g2"] * 10)
    assert determined["within_share"] == 0 and determined["off_majority"] == 0
    free = core.variance_decomposition(["a", "b"] * 20, ["g1"] * 20 + ["g2"] * 20)
    assert free["within_share"] == pytest.approx(1.0) and free["off_majority"] == 20


def test_trait_targets_flags_only_groups_with_enough_of_both_classes():
    rows, groups = [], []
    for g, n, k in (("A", 100, 50), ("B", 100, 0), ("C", 40, 10), ("D", 20, 10)):  # (group, size, n motile)
        for i in range(n):
            rows.append({"motility": "yes" if i < k else "no", "shape": "rod" if i % 5 else "coccus"})
            groups.append(g)
    t = {x["target"]: x for x in core.trait_targets(rows, ["motility", "shape"], groups, min_class=10)}
    mot = t["motility: yes vs no"]
    assert mot["per_group"]["A"]["testable"] and not mot["per_group"]["B"]["testable"]  # B has no motile genomes
    assert not mot["per_group"]["C"]["testable"]  # 10 motile < 15
    assert not mot["per_group"]["D"]["testable"]  # only 20 genomes
    assert mot["testable_groups"] == 1 and mot["testable_genomes"] == 100
    assert "shape: coccus vs rod" in t  # a two-class trait gives one target, minority class positive


def test_auc_se_shrinks_with_n_and_is_nan_without_both_classes():
    assert core.auc_se(50, 50) > core.auc_se(500, 500) > 0
    assert core.auc_se(100, 100) == pytest.approx(0.0388, abs=0.003)
    assert math.isnan(core.auc_se(1, 50)) and math.isnan(core.auc_se(0, 50))


# --- reliability estimators against known truth --------------------------------------------------------
def _pools(rho=0.0, g=50, p=100, d=120, sb=1.0, sw=3.0, seed=0):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    mu = 40 * rng.normal(size=d) + sb * rng.normal(size=(g, d))  # a big shared component + genome-specific means
    e = rng.normal(size=(g, p, d))
    x = np.zeros_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, p):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    return (mu[:, None, :] + sw * x).astype(np.float32)


def test_cosine_report_shows_shared_component_and_its_removal():
    c = core.cosine_report(_pools(g=40).mean(1))
    assert c["raw"] > 0.99 and abs(c["centred"]) < 0.1 and abs(c["centred_scaled"]) < 0.1


def test_reliability_recovers_the_true_within_between_ratio_for_independent_windows():
    pools = _pools()
    r = core.reliability_curve(pools, [1, 5, 10, 25], [1, 5, 10, 25], [1, 5, 10, 25, 100], reps=6, n_boot=20, seed=1)
    for n in (1, 5, 10, 25):
        truth = (3.0 ** 2 / n) / 1.0
        for kind in ("random", "systematic"):
            assert r[kind][n]["ratio"] == pytest.approx(truth, rel=0.25), (kind, n)
        lo, hi = r["systematic"][n]["ratio_ci"]
        assert lo <= r["systematic"][n]["ratio"] <= hi
        assert r["model"][n]["ratio"] == pytest.approx(truth, rel=0.25)
    assert r["model"][100]["extrapolated"] and not r["model"][10]["extrapolated"]
    ratios = [r["systematic"][n]["ratio"] for n in (1, 5, 10, 25)]
    assert ratios == sorted(ratios, reverse=True)  # more windows, less within-genome variance
    rel = [r["systematic"][n]["reliability"] for n in (1, 5, 10, 25)]
    assert rel == sorted(rel) and 0 < rel[0] < rel[-1] < 1


def test_even_spacing_beats_random_windows_only_when_windows_are_spatially_correlated():
    iid = core.reliability_curve(_pools(rho=0.0), [10], [10], [10], reps=6, n_boot=5, seed=1)["systematic"][10]["systematic_gain"]
    ar = core.reliability_curve(_pools(rho=0.9), [10], [10], [10], reps=6, n_boot=5, seed=1)["systematic"][10]["systematic_gain"]
    assert iid == pytest.approx(1.0, abs=0.15) and ar > 2.0


def test_reliability_skips_depths_that_cannot_have_disjoint_halves():
    r = core.reliability_curve(_pools(g=20, d=30), [50, 60, 100], [50, 60, 100, 30], [100], reps=2, n_boot=3)
    assert set(r["random"]) == {50} and set(r["systematic"]) == {50}  # 60 & 100: 2n > pool; 30: does not divide 100


def test_flattens_at_picks_the_knee_and_reports_noise_only_curves_as_flat():
    xs = [1, 2, 5, 10, 20, 50]
    assert core.flattens_at(xs, [.2, .5, .8, .9, .91, .92])["n"] == 10
    assert core.flattens_at(xs, [9, 4, 2, 1, .9, .9], higher_is_better=False)["n"] == 10  # needs <= 1.3 (95% of the drop)
    flat = core.flattens_at(xs, [.70, .71, .70, .72, .71, .70], noise=0.02)
    assert flat["flat_within_noise"] and flat["n"] is None


def test_gc_quality_is_high_when_embeddings_encode_gc_and_low_when_they_do_not():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    gc = rng.uniform(0.3, 0.7, 60)
    groups = np.array([f"g{i % 6}" for i in range(60)])
    signal = np.outer(gc, rng.normal(size=30)) + 0.05 * rng.normal(size=(60, 30))
    assert core.gc_quality(signal, gc, groups)["spearman"] > 0.9
    assert abs(core.gc_quality(rng.normal(size=(60, 30)), gc, groups)["spearman"]) < 0.5


def test_reliability_run_on_a_fake_volume_and_the_entrypoint_prints_everything(tmp_path, monkeypatch, capsys):
    pytest.importorskip("sklearn")
    m = _patched(monkeypatch, tmp_path)
    rows = _fake_volume(tmp_path)
    d = m._reliability_run(rows, 1, 3, 5)
    assert d["n_genomes"] == 36 and d["grid"] == [1, 2, 4, 5, 10, 20, 25, 50, 100]
    sysm = d["curve"]["systematic"]
    assert sysm[1]["ratio"] > sysm[50]["ratio"] and set(sysm) == {1, 2, 4, 5, 10, 20, 25, 50}
    assert set(d["anchors"]) == set(d["grid"]) and len(d["anchors"][100]["spearman"]) == 1
    assert d["curve"]["model"][100]["extrapolated"]
    # a missing window pool is refused with a count, before any work
    (tmp_path / "embeddings" / f"{rows[0]['acc']}__windows.npy").unlink()
    with pytest.raises(RuntimeError, match=r"35 of 36 genomes have a saved window pool"):
        m._reliability_run(rows, 1, 3, 5)

    stub = type("F", (), {"remote": staticmethod(lambda *a, **k: d)})()
    monkeypatch.setattr(m, "reliability_run", stub)
    g = tmp_path / "g.tsv"
    g.write_text("acc\tgroup\tmotility\n" + "".join(f"{r['acc']}\tother\t{r['motility']}\n" for r in rows))
    panel = tmp_path / "panel.tsv"
    lines = ["ncbi_assembly_accession\tassembly_genbank\tgtdb_phylum\tgram\tshape\tmotility\tspore\toxygen\ttemperature\n"]
    for i in range(120):
        lines.append(f"GCF_{i:09d}.1\t\t{'P1' if i % 2 else 'P2'}\t{'negative' if i % 3 else 'positive'}\t{'rod' if i % 4 else 'coccus'}\t"
                     f"{'yes' if i % 5 < 2 else 'no'}\t{'yes' if i % 7 == 0 else 'no'}\t{'aerobe' if i % 3 else 'anaerobe'}\t{'meso' if i % 9 else 'thermo'}\n")
    panel.write_text("".join(lines))
    m.reliability.info.raw_f(panel=str(panel), genomes=str(g), reports=str(tmp_path / "rep"))
    out = capsys.readouterr().out
    for needle in ("WITHIN-PHYLUM VARIANCE BY TRAIT", "motility: yes vs no", "testable groups", "SAMPLING-DEPTH RELIABILITY",
                   "raw", "mean-centred", "within/between", "GC Spearman", "WHERE EACH CURVE FLATTENS",
                   "split-half reliability", "motility AUC (ungrouped)", "even spacing vs random windows"):
        assert needle in out, needle
    assert (tmp_path / "rep" / "tables" / "evo2_depth_reliability.json").exists()
    assert (tmp_path / "rep" / "tables" / "trait_within_phylum_variance.tsv").exists()
