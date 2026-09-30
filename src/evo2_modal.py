"""Evo 2 7B genome embeddings on Modal: sampling-depth sweep and production run.

    source ~/venvs/modal/bin/activate
    modal run -m src.evo2_modal::sweep --dry-run          # cost estimate only, spends nothing
    modal run -m src.evo2_modal::sweep                    # the sampling-depth experiment
    modal run -m src.evo2_modal::embed_all --n-windows N  # production run (resumable)

Recipe (see README, "Evo 2 on Modal"): Evo 2 7B does not need transformer-engine, but a
PARTIALLY installed TE raises RuntimeError that evo2's `except ImportError` does not catch
(evo2 issue #201). A fully ABSENT TE degrades to HAS_TE = False. So TE is never installed here,
which is why the image is a plain CUDA base and not an NGC one. `evo2` and `vtx` are installed
with --no-deps, because evo2's metadata lists transformer_engine>=2.0.0.

Only the `evo2_7b_base` config has use_fp8_input_projections: False (8k context).
"""

from __future__ import annotations

import csv
import os
import time
import traceback
from pathlib import Path

import modal

from src import embed_core as core

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
HF_DIR = "/hf"  # HF_HOME: evo2 also merges the shards into <HF_HOME>/evo2_7b_base.pt (~14 GB)
OUT_DIR = "/out"
GENOME_DIR = f"{OUT_DIR}/genomes"
EMB_DIR = f"{OUT_DIR}/embeddings"

hf_vol = modal.Volume.from_name("evo2-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("evo2-embeddings", create_if_missing=True)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
# flash-attn prebuilt wheel: Dao-AILab/flash-attention release v2.8.3, torch 2.8, CUDA 12,
# cxx11abiTRUE, CPython 3.11. Both cp311 and cp312 assets exist for this tag (checked by HTTP
# request); the filename is not guessed.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)

# Fails the image build if any transformer-engine distribution is present, or if torch's C++ ABI
# does not match the flash-attn wheel.
_BUILD_CHECK = (
    "import importlib.metadata as m, importlib.util as u, torch;"
    "bad=[d.metadata['Name'] for d in m.distributions() "
    "if 'transformer' in d.metadata['Name'].lower() and 'engine' in d.metadata['Name'].lower()];"
    "assert not bad, f'transformer-engine is installed: {bad}';"
    "assert u.find_spec('transformer_engine') is None, 'transformer_engine is importable';"
    "assert torch.__version__.startswith('2.8.0'), torch.__version__;"
    "assert torch._C._GLIBCXX_USE_CXX11_ABI, 'torch is cxx11abiFALSE but the flash-attn wheel is TRUE';"
    "print('build check ok: torch', torch.__version__, 'cuda', torch.version.cuda, 'no transformer-engine')"
)

gpu_image = (
    # Plain CUDA 12.8 base. Not NGC: NGC images ship TE, and TE must be completely absent.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "curl")  # gcc for triton's runtime stubs
    # --no-deps is essential: a normal `pip install evo2` pulls transformer_engine.
    .pip_install("evo2==0.3.0", "vtx==1.1.0", extra_options="--no-deps")
    .pip_install(
        "einops==0.8.1", "biopython", "huggingface_hub", "scikit-learn", "pandas", "pyarrow",
        # not in the recipe because Colab preinstalls them; vtx imports them, evo2 uses requests
        "numpy", "pyyaml", "packaging", "rich", "tqdm", "requests",
    )
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(FLASH_ATTN_WHEEL, extra_options="--no-deps")
    .run_commands(f'python -c "{_BUILD_CHECK}"')
    .run_commands("pip list 2>/dev/null | grep -i -E 'transformer[-_]engine' && exit 1 || echo 'pip list: no transformer-engine'")
    .env({"HF_HOME": HF_DIR})
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "huggingface_hub", "numpy", "pandas", "scikit-learn")
    .env({"HF_HOME": HF_DIR})
)

GPU = "A100-40GB"
MAX_GPUS = int(os.environ.get("EVO2_MAX_GPUS", "4"))  # parallel A100s; total GPU-hours (cost) are the same

app = modal.App("evo2-embed")

PANEL = "data/final/core_panel_species.tsv"
SELECTED_DEPTH_JSON = "reports/tables/evo2_sweep_selected_depth.json"
# binary prediction targets: name -> (panel column, positive class)
TARGETS = {"motility": ("motility", "yes"), "oxygen_aerobe": ("oxygen", "aerobe"),
           "oxygen_facultative": ("oxygen", "facultative"), "shape_rod": ("shape", "rod")}
UA = "bacterialmorphospace/0.1 (research; genome download)"


# ---------------------------------------------------------------------------
# Remote functions
#
# Everything that leaves a container is pickled back to the local process, which has no torch,
# numpy or huggingface_hub. So each remote function is a thin wrapper that returns
# `core.plain(...)` (builtin str/int/float/bool/None/dict/list only) and re-raises any failure as a
# builtin RuntimeError carrying the original type and message; the real work is in the `_impl`.
# ---------------------------------------------------------------------------
def _plain_call(fn, *args, **kwargs):
    try:
        return core.plain(fn(*args, **kwargs))
    except Exception:  # noqa: BLE001
        # A builtin RuntimeError unpickles anywhere, but its message must carry the original traceback
        # (`from None` would drop it), because the local side only ever sees this string.
        raise RuntimeError("remote function failed:\n" + traceback.format_exc()) from None


@app.function(image=gpu_image, gpu=GPU, timeout=900)
def verify_env() -> dict:
    return _plain_call(_verify_env)


def _verify_env() -> dict:
    """Runs on the GPU image: TE absent, HAS_TE False, flash-attn importable, right torch/ABI."""
    import importlib.metadata as md
    import importlib.util

    import torch

    te = [d.metadata["Name"] for d in md.distributions()
          if "transformer" in d.metadata["Name"].lower() and "engine" in d.metadata["Name"].lower()]
    assert not te, f"transformer-engine distributions installed: {te}"
    assert importlib.util.find_spec("transformer_engine") is None
    import flash_attn
    from vortex.model import layers

    assert layers.HAS_TE is False, "vortex found transformer_engine"
    from evo2 import Evo2  # noqa: F401  (import only; loading happens in Embedder)

    return {
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda),  # TorchVersion is a str subclass that pickles via torch "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "flash_attn": str(flash_attn.__version__), "gpu": str(torch.cuda.get_device_name(0)),
        "transformer_engine_dists": [str(t) for t in te], "HAS_TE": bool(layers.HAS_TE),
        "evo2": md.version("evo2"), "vtx": md.version("vtx"),
    }


@app.function(image=cpu_image, volumes={HF_DIR: hf_vol}, timeout=3600, cpu=2, memory=8192)
def prime_weights() -> dict:
    return _plain_call(_prime_weights)


def _prime_weights() -> dict:
    """Put the merged evo2_7b_base.pt on the volume from a CPU container, so no GPU second is spent on it.

    Mirrors evo2.Evo2.load_evo2_model exactly (same repo, same shard names, same merged path), so
    the first GPU load finds "existing merged file" and goes straight to load_checkpoint.
    """
    from huggingface_hub import constants, hf_hub_download, snapshot_download

    filename = f"{core.MODEL_NAME}.pt"
    final = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
    if os.path.exists(final) and os.path.getsize(final) > 12e9:
        hf_hub_download(repo_id=f"arcinstitute/{core.MODEL_NAME}", filename="config.json")
        return {"status": "cached", "path": final, "gb": os.path.getsize(final) / 1e9}

    repo_dir = snapshot_download(repo_id=f"arcinstitute/{core.MODEL_NAME}")
    parts, i = [], 0
    while os.path.exists(os.path.join(repo_dir, f"{filename}.part{i}")):
        parts.append(os.path.join(repo_dir, f"{filename}.part{i}"))
        i += 1
    if not parts and os.path.exists(os.path.join(repo_dir, filename)):
        final = os.path.join(repo_dir, filename)
    elif parts:
        with open(final, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while block := f.read(8 << 20):
                        out.write(block)
        for p in parts:  # as evo2 does after merging
            real = os.path.realpath(p)
            for q in {real, p}:
                if os.path.exists(q):
                    os.remove(q)
    else:
        raise FileNotFoundError(f"no {filename} or shards in {repo_dir}")
    size = os.path.getsize(final)
    if size < 12e9:  # bf16 7B is ~14 GB
        raise IOError(f"{final} is only {size / 1e9:.1f} GB; expected ~14 GB")
    hf_vol.commit()
    return {"status": "downloaded", "path": final, "gb": size / 1e9}


def _fetch_genome(acc: str, dest: str) -> int:
    """Download one assembly's genomic FASTA to `dest`, verified. Returns bytes."""
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "identity"})

    def get(url, **kw):
        last = None
        for attempt in range(4):
            try:
                r = s.get(url, timeout=120, **kw)
                r.raise_for_status()
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise last

    parent = core.ncbi_parent_url(acc)
    d = core.find_assembly_dir(get(parent).text, acc)
    if d is None:
        raise FileNotFoundError(f"{acc} not found under {parent} (suppressed or superseded assembly?)")
    fname = f"{d}_genomic.fna.gz"
    md5 = None
    try:
        md5 = core.parse_md5_file(get(f"{parent}{d}/md5checksums.txt").text).get(fname)
    except Exception:  # noqa: BLE001  md5 file is a bonus; size + gzip checks still apply
        pass

    tmp = dest + ".part"
    for attempt in range(3):
        try:
            with get(f"{parent}{d}/{fname}", stream=True) as r:
                announced = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            core.verify_download(tmp, expected_bytes=announced, expected_md5=md5)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=900, max_containers=8,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def download_genome(acc: str, alt_acc: str = "") -> dict:
    return _plain_call(_download_genome, acc, alt_acc)


def _download_genome(acc: str, alt_acc: str = "") -> dict:
    """NCBI download with verification (see core.verify_download). Tries the GenBank accession if RefSeq fails."""
    os.makedirs(GENOME_DIR, exist_ok=True)
    dest = f"{GENOME_DIR}/{acc}.fna.gz"
    if os.path.exists(dest):
        try:
            core.verify_download(dest)
            return {"acc": acc, "ok": True, "cached": True, "source": acc}
        except IOError:
            os.remove(dest)  # a bad file from an earlier run: never trust it
    errors = []
    for cand in dict.fromkeys(a for a in (acc, alt_acc) if a):
        try:
            n = _fetch_genome(cand, dest)
            out_vol.commit()
            return {"acc": acc, "ok": True, "cached": False, "source": cand, "bytes": n}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cand}: {e}")
    return {"acc": acc, "ok": False, "error": " | ".join(errors)}


def _save_npy(path: str, arr) -> None:
    import numpy as np

    with open(path + ".tmp", "wb") as f:
        np.save(f, arr)
    os.replace(path + ".tmp", path)  # a checkpoint is either whole or absent


@app.cls(image=gpu_image, gpu=GPU, volumes={HF_DIR: hf_vol, OUT_DIR: out_vol}, timeout=1800,
         scaledown_window=60, max_containers=MAX_GPUS, retries=modal.Retries(max_retries=2, initial_delay=5.0))
class Embedder:
    @modal.enter()
    def load(self):
        """Never raises. A crash here kills the container before any input runs, and the caller then sees
        only Modal's generic "cancelled by user or a failure" for every input. Instead the traceback is
        kept and returned by every `embed` call, and the steps are printed so they appear in the app logs."""
        import faulthandler

        faulthandler.enable()  # a native crash (CUDA/triton segfault) dumps its stack to the container log
        self.task_id = os.environ.get("MODAL_TASK_ID", "local")
        self.model, self.load_error = None, None
        t0 = time.perf_counter()
        try:
            print("[enter] importing evo2", flush=True)
            from evo2 import Evo2

            print(f"[enter] loading {core.MODEL_NAME}", flush=True)
            self.model = Evo2(core.MODEL_NAME)
            import torch

            print(f"[enter] loaded in {time.perf_counter() - t0:.0f}s; "
                  f"CUDA memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        except Exception:  # noqa: BLE001
            self.load_error = traceback.format_exc()
            print("[enter] MODEL LOAD FAILED:\n" + self.load_error, flush=True)
        self.load_s = time.perf_counter() - t0

    def _embed_windows(self, seqs: list[str], windows: list[tuple[int, int]], batch_size: int):
        import numpy as np
        import torch

        tok = self.model.tokenizer
        vecs, times = [], []
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            ids = torch.tensor([tok.tokenize(seqs[c][s:s + core.WINDOW]) for c, s in chunk],
                               dtype=torch.int).to("cuda:0")
            torch.cuda.synchronize()
            t = time.perf_counter()
            _, emb = self.model(ids, return_embeddings=True, layer_names=[core.LAYER])
            x = emb[core.LAYER]
            if x.shape[-1] != core.EMB_DIM and x.shape[1] == core.EMB_DIM:
                x = x.transpose(1, 2)
            assert x.shape[-1] == core.EMB_DIM, f"unexpected embedding shape {tuple(x.shape)}"
            v = x.float().mean(dim=1)  # mean-pool within window -> (B, 4096)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            v = v.cpu().numpy()
            if not np.isfinite(v).all():
                raise FloatingPointError("non-finite embedding")
            vecs.append(v)
            times += [dt / len(chunk)] * len(chunk)
        return np.concatenate(vecs), times

    @modal.method()
    def embed(self, acc: str, depths: list[int], batch_size: int = 1, save_windows: bool = True) -> dict:
        return _plain_call(self._embed, acc, depths, batch_size, save_windows)

    def _embed(self, acc: str, depths: list[int], batch_size: int = 1, save_windows: bool = True) -> dict:
        import json

        import numpy as np

        t_call = time.perf_counter()
        base = {"acc": acc, "task_id": self.task_id, "load_s": self.load_s}
        try:
            if self.load_error:
                raise RuntimeError("the model failed to load in this container:\n" + self.load_error)
            os.makedirs(EMB_DIR, exist_ok=True)
            pool_n = max(depths)
            npy = {n: f"{EMB_DIR}/{acc}__n{n}.npy" for n in depths}
            win_path, meta_path = f"{EMB_DIR}/{acc}__windows.npy", f"{EMB_DIR}/{acc}__meta.json"
            missing = [n for n in depths if not os.path.exists(npy[n])]
            if not missing:
                return {**base, "ok": True, "status": "skipped", "gpu_seconds": 0.0}

            status = "embedded"
            meta = json.loads(open(meta_path).read()) if os.path.exists(meta_path) else None
            if (meta and meta.get("sampling") == core.SAMPLING and os.path.exists(win_path)
                    and np.load(win_path, mmap_mode="r").shape[0] == pool_n):
                pool = np.load(win_path)  # a pool from an earlier run: derive the missing depths for free
                status = "derived"
            else:
                # windows are drawn across ALL contigs >= one window, in proportion to contig length
                contigs, total_len, n_contigs = core.read_contigs(f"{GENOME_DIR}/{acc}.fna.gz")
                lens = [len(s) for _, s in contigs]
                windows = core.pool_windows(lens, pool_n)
                seqs = [s for _, s in contigs]
                pool, w_times = self._embed_windows(seqs, windows, batch_size)
                non_acgt = [sum(ch not in "ACGT" for ch in seqs[c][s:s + core.WINDOW]) / core.WINDOW for c, s in windows]
                per_contig = [sum(1 for c, _ in windows if c == i) for i in range(len(seqs))]
                meta = {"acc": acc, "sampling": core.SAMPLING, "pool_n": pool_n, "batch_size": batch_size,
                        "total_len": total_len, "n_contigs": n_contigs, "n_contigs_used": len(contigs),
                        "usable_len": core.usable_len(lens), "longest_contig": max(lens),
                        "windows": [[contigs[c][0], s] for c, s in windows], "windows_per_contig": per_contig,
                        "window_times": w_times, "max_non_acgt_frac": max(non_acgt)}
                if save_windows:
                    _save_npy(win_path, pool.astype(np.float32))
                with open(meta_path + ".tmp", "w") as f:
                    json.dump(meta, f)
                os.replace(meta_path + ".tmp", meta_path)
            for n in missing:
                idx = core.subset_indices(pool_n, n)
                _save_npy(npy[n], pool[idx].mean(axis=0).astype(np.float32))
            out_vol.commit()
            return {**base, "ok": True, "status": status, "gpu_seconds": time.perf_counter() - t_call,
                    "usable_len": meta["usable_len"], "n_contigs_used": meta["n_contigs_used"],
                    "max_non_acgt_frac": meta["max_non_acgt_frac"]}
        except Exception:  # noqa: BLE001  deterministic failures are reported (with traceback), not retried
            print(f"[embed] {acc} FAILED:\n{traceback.format_exc()}", flush=True)
            return {**base, "ok": False, "error": traceback.format_exc(), "gpu_seconds": time.perf_counter() - t_call}


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=1800, cpu=2, memory=8192)
def evaluate_sweep(rows: list[dict], depths: list[int], n_splits: int) -> dict:
    return _plain_call(_evaluate_sweep, rows, depths, n_splits)


def _require_embedded(rows: list[dict], depths: list[int]) -> None:
    """Precondition, checked before any work: every genome has all its depth .npy files and a meta file."""
    lacking = [r["acc"] for r in rows
               if not (os.path.exists(f"{EMB_DIR}/{r['acc']}__meta.json")
                       and all(os.path.exists(f"{EMB_DIR}/{r['acc']}__n{n}.npy") for n in depths))]
    if not rows or lacking:
        raise RuntimeError(f"{len(rows) - len(lacking)} of {len(rows)} genomes embedded: {len(lacking)} lack an .npy "
                           f"for depths {depths} or a __meta.json on the volume (e.g. {lacking[:3]}). "
                           "The embed stage did not produce these; nothing was evaluated.")


def _evaluate_sweep(rows: list[dict], depths: list[int], n_splits: int) -> dict:
    """Per-depth GroupKFold AUC on `motility`, plus timings from the per-genome meta files."""
    import json

    import numpy as np

    out_vol.reload()
    _require_embedded(rows, depths)
    y = np.array([r["motility"] == "yes" for r in rows])
    groups = np.array([r["group"] for r in rows])
    bad, results, timing = [], [], {}
    metas = {r["acc"]: json.load(open(f"{EMB_DIR}/{r['acc']}__meta.json")) for r in rows}
    stale = [a for a, m in metas.items() if m.get("sampling") != core.SAMPLING or m["pool_n"] != max(depths)]
    if stale:
        raise RuntimeError(f"{len(stale)} genomes were embedded with a different sampling scheme or pool size "
                           f"(e.g. {stale[:3]}); delete their files from the volume and re-run")
    for n in depths:
        X = np.stack([np.load(f"{EMB_DIR}/{r['acc']}__n{n}.npy") for r in rows])
        if X.shape[1] != core.EMB_DIM or not np.isfinite(X).all():
            bad.append(n)
        res = core.evaluate_depth(X, y, groups, n_splits)
        secs = [core.depth_seconds(m["window_times"], m["pool_n"], n) for m in metas.values()]
        s_gen = float(np.mean(secs))
        # windows must overlap when ALL usable sequence (contigs >= one window) is shorter than n windows
        overlap = float(np.mean([m["usable_len"] < n * core.WINDOW for m in metas.values()]))
        results.append({"depth": n, "n_genomes": len(rows), **res, "frac_overlapping": overlap, "s_per_genome": s_gen,
                        "hours_2580": core.project_hours(s_gen), "usd_2580": core.usd(s_gen * core.N_PANEL)})
    ms = list(metas.values())
    return {"results": results, "invalid_depths": bad, "n_positive": int(y.sum()), "n_groups": int(len(set(groups))),
            "max_non_acgt_frac": float(max(m["max_non_acgt_frac"] for m in ms)),
            "median_usable_kb": float(np.median([m["usable_len"] for m in ms]) / 1e3),
            "median_longest_contig_kb": float(np.median([m["longest_contig"] for m in ms]) / 1e3),
            "median_contigs_used": float(np.median([m["n_contigs_used"] for m in ms])),
            "median_frac_of_genome_usable": float(np.median([m["usable_len"] / m["total_len"] for m in ms]))}


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=3600, cpu=4, memory=8192)
def diagnose_sweep(rows: list[dict], depths: list[int], n_splits: int, n_perm: int, seed: int) -> dict:
    return _plain_call(_diagnose_sweep, rows, depths, n_splits, n_perm, seed)


def _fasta_ids(path: str) -> set:
    import gzip

    with gzip.open(path, "rt") as f:
        return {line[1:].split()[0] for line in f if line.startswith(">")}


def _diagnose_sweep(rows: list[dict], depths: list[int], n_splits: int, n_perm: int, seed: int) -> dict:
    """CPU-only audit of a sweep result: pairing, positive class, fold structure, label-shuffle nulls, GC baseline."""
    import json
    import random

    import numpy as np
    from scipy.stats import spearmanr

    out_vol.reload()
    _require_embedded(rows, depths)
    accs = [r["acc"] for r in rows]
    y = np.array([r["motility"] == "yes" for r in rows])
    groups = np.array([r["group"] for r in rows])
    top = max(depths)

    # 1. alignment: every embedding is read by its own accession and traced back to its own genome
    audit, ok_meta, ok_fasta, ok_pool = [], True, True, True
    pick = set(random.Random(seed).sample(range(len(rows)), min(5, len(rows))))
    for i, r in enumerate(rows):
        acc = r["acc"]
        meta = json.load(open(f"{EMB_DIR}/{acc}__meta.json"))
        pool = np.load(f"{EMB_DIR}/{acc}__windows.npy")
        ids = _fasta_ids(f"{GENOME_DIR}/{acc}.fna.gz")
        w_ids = {w[0] for w in meta["windows"]}
        m_ok, f_ok = meta["acc"] == acc, w_ids <= ids
        p_ok = all(np.allclose(np.load(f"{EMB_DIR}/{acc}__n{n}.npy"), pool[core.subset_indices(meta["pool_n"], n)].mean(axis=0), atol=1e-5)
                   for n in depths)
        ok_meta, ok_fasta, ok_pool = ok_meta and m_ok, ok_fasta and f_ok, ok_pool and p_ok
        if i in pick:
            audit.append({"row": i, "acc": acc, "label_motility": r["motility"], "group": r["group"],
                          "loaded": [f"{acc}__n{n}.npy" for n in depths], "meta_acc": meta["acc"],
                          "meta_acc_matches": m_ok, "n_windows": len(meta["windows"]),
                          "windows_lie_on_this_genomes_contigs": f_ok, "npy_equals_mean_of_its_windows": p_ok})
    Xtop = np.stack([np.load(f"{EMB_DIR}/{a}__n{top}.npy") for a in accs])
    Z = Xtop / np.linalg.norm(Xtop, axis=1, keepdims=True)
    cos = Z @ Z.T
    np.fill_diagonal(cos, -1)
    alignment = {"n_checked": len(rows), "all_meta_acc_match": ok_meta, "all_windows_on_own_genome": ok_fasta,
                 "all_npy_equal_mean_of_own_windows": ok_pool, "max_cosine_between_different_genomes": float(cos.max()),
                 "n_distinct_vectors": int(len({tuple(np.round(v, 5)) for v in Xtop})), "audit": audit}

    # labels
    by_group = {g: {"n": int((groups == g).sum()), "prevalence_motile": float(y[groups == g].mean())} for g in sorted(set(groups))}

    # GC (from the FASTA) : baseline feature, and target for the pairing control
    gc = np.array([core.gc_content(f"{GENOME_DIR}/{a}.fna.gz") for a in accs])
    gc_res = core.evaluate_depth(gc[:, None], y, groups, n_splits)
    gc_folds = core.fold_report(gc[:, None], y, groups, n_splits)
    gc_block = {"spearman_gc_vs_motility": float(spearmanr(gc, y).statistic), "cv": gc_res,
                "fold_aucs": [f["auc"] for f in gc_folds["folds"]], "mean_gc": float(gc.mean()),
                "gc_range": [float(gc.min()), float(gc.max())]}

    per_depth = {}
    for n in depths:
        X = np.stack([np.load(f"{EMB_DIR}/{a}__n{n}.npy") for a in accs])
        real = core.evaluate_depth(X, y, groups, n_splits)
        nulls = {k: core.permutation_null(X, y, groups, n_splits, n_perm, seed, within_groups=(k == "within_group"), n_jobs=4)
                 for k in ("global", "within_group")}
        pct = {k: float(np.mean(np.asarray(v["values"]) <= real["auc_mean"])) for k, v in nulls.items()}
        per_depth[f"n{n}"] = {
            "real": real, "fold_report": core.fold_report(X, y, groups, n_splits),
            "strong_regularisation_C0.01_auc": core.fold_report(X, y, groups, n_splits, C=0.01)["auc_mean"],
            "ungrouped_stratified_cv": core.ungrouped_auc(X, y, n_splits, seed=seed),
            "null_global": {k: v for k, v in nulls["global"].items() if k != "values"},
            "null_within_group": {k: v for k, v in nulls["within_group"].items() if k != "values"},
            "share_of_null_at_or_below_real": pct}
    pairing = core.pairing_control(Xtop, gc, groups, n_splits, seed=seed)
    return {"alignment": alignment, "labels": {"n": len(rows), "prevalence_motile": float(y.mean()), "by_group": by_group},
            "positive_class": "y is a bool array (motility == 'yes'); classes_ order is reported per depth in fold_report.classes",
            "gc_baseline": gc_block, "pairing_control": {"depth": top, **pairing}, "depths": per_depth, "n_perm": n_perm}


DEPTH_GRID = [1, 2, 4, 5, 10, 20, 25, 50, 100]  # divisors of the 100-window pool, so every depth is an evenly spaced subset


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=3600, cpu=4, memory=16384)
def reliability_run(rows: list[dict], seed: int, reps: int, n_boot: int, within_groups: bool = False) -> dict:
    return _plain_call(_reliability_run, rows, seed, reps, n_boot, within_groups)


def _reliability_run(rows: list[dict], seed: int, reps: int, n_boot: int, within_groups: bool = False) -> dict:
    """Free (CPU, existing checkpoints) sampling-depth study built from the saved 100-window pools."""
    import json

    import numpy as np
    from joblib import Parallel, delayed

    out_vol.reload()
    accs = [r["acc"] for r in rows]
    missing = [a for a in accs if not (os.path.exists(f"{EMB_DIR}/{a}__windows.npy") and os.path.exists(f"{EMB_DIR}/{a}__meta.json"))]
    if not rows or missing:
        raise RuntimeError(f"{len(accs) - len(missing)} of {len(accs)} genomes have a saved window pool + meta "
                           f"(missing e.g. {missing[:3]}); the sweep saves them, embed_all does not")
    metas = [json.load(open(f"{EMB_DIR}/{a}__meta.json")) for a in accs]
    stale = [a for a, m in zip(accs, metas) if m.get("sampling") != core.SAMPLING]
    if stale:
        raise RuntimeError(f"{len(stale)} genomes were sampled with a different scheme (e.g. {stale[:3]})")
    pools = np.stack([np.load(f"{EMB_DIR}/{a}__windows.npy") for a in accs])  # (genomes, 100, 4096)
    g, pool_n, _ = pools.shape
    grid = [n for n in DEPTH_GRID if n <= pool_n and pool_n % n == 0]
    y = np.array([r["motility"] == "yes" for r in rows])
    groups = np.array([r["group"] for r in rows])
    gc = np.array([core.gc_content(f"{GENOME_DIR}/{a}.fna.gz") for a in accs])

    curve = core.reliability_curve(pools, grid, grid, grid, reps=reps, n_boot=n_boot, seed=seed,
                                   groups=groups if within_groups else None)

    # anchors: GC prediction (out-of-phylum) and ungrouped motility AUC, on the same evenly spaced subsets
    tasks = []
    for n in grid:
        stride = pool_n // n
        offsets = sorted({int(o) for o in np.linspace(0, stride - 1, min(5, stride))})
        for o in offsets:
            tasks.append((n, o, pools[:, np.arange(n) * stride + o].mean(1)))

    def one(n, o, X):
        return n, o, core.gc_quality(X, gc, groups), core.ungrouped_auc(X, y, 5, repeats=2, seed=seed)

    res = Parallel(n_jobs=4)(delayed(one)(*t) for t in tasks)
    anchors = {}
    for n, o, gq, mo in res:
        a = anchors.setdefault(n, {"spearman": [], "r2": [], "motility_auc": [], "motility_cv_sd": []})
        a["spearman"].append(gq["spearman"]), a["r2"].append(gq["r2"])
        a["motility_auc"].append(mo["auc_mean"]), a["motility_cv_sd"].append(mo["auc_sd_over_repeats"])
    return {"grid": grid, "n_genomes": g, "pool_n": pool_n, "curve": curve, "anchors": anchors,
            "gc_range": [float(gc.min()), float(gc.max())]}


FEAT_DIR = f"{OUT_DIR}/features"


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=3600, cpu=2, memory=4096, max_containers=8)
def seqfeat_batch(accs: list[str], n_windows: int) -> dict:
    return _plain_call(_seqfeat_batch, accs, n_windows)


def _seqfeat_batch(accs: list[str], n_windows: int) -> dict:
    """Tetranucleotide + mono-nucleotide counts for a batch of genomes: whole genome, and the exact windows Evo 2 saw.

    One checkpoint per genome (`features/<acc>__seqfeat_n<N>.npy`, shape (2, 260)); existing files are skipped.
    """
    import json

    out_vol.reload()
    os.makedirs(FEAT_DIR, exist_ok=True)
    done = skipped = 0
    errors = []
    for acc in accs:
        dest = f"{FEAT_DIR}/{acc}__seqfeat_n{n_windows}.npy"
        if os.path.exists(dest):
            skipped += 1
            continue
        try:
            meta = json.load(open(f"{EMB_DIR}/{acc}__meta.json"))
            w = meta["windows"]
            if meta["pool_n"] != n_windows:  # sweep genomes: depth-N vectors are an evenly spaced subset of the 100-pool
                w = [w[i] for i in core.subset_indices(meta["pool_n"], n_windows)]
            _save_npy(dest, core.sequence_features(f"{GENOME_DIR}/{acc}.fna.gz", w))
            done += 1
        except Exception:  # noqa: BLE001
            errors.append([acc, traceback.format_exc()[-300:]])
    out_vol.commit()
    return {"done": done, "skipped": skipped, "errors": errors}


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=10800, cpu=8, memory=16384)
def lopo_run(rows: list[dict], n_windows: int, n_perm: int, n_boot: int, seed: int) -> dict:
    return _plain_call(_lopo_run, rows, n_windows, n_perm, n_boot, seed)


def _lopo_run(rows: list[dict], n_windows: int, n_perm: int, n_boot: int, seed: int) -> dict:
    """Leave-one-phylum-out, scored within the held-out phylum: Evo 2 vs tetranucleotides vs GC vs the trivial floor."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    out_vol.reload()
    ok = [r for r in rows if os.path.exists(f"{EMB_DIR}/{r['acc']}__n{n_windows}.npy")
          and os.path.exists(f"{FEAT_DIR}/{r['acc']}__seqfeat_n{n_windows}.npy")]
    if not rows or len(ok) < 0.98 * len(rows):
        miss = [r["acc"] for r in rows if r not in ok]
        raise RuntimeError(f"{len(ok)} of {len(rows)} genomes have both a depth-{n_windows} embedding and sequence features "
                           f"(need >= 98%; missing e.g. {miss[:3]}); run embed_all and seqfeat first")
    emb = np.stack([np.load(f"{EMB_DIR}/{r['acc']}__n{n_windows}.npy") for r in ok])
    sf = np.stack([np.load(f"{FEAT_DIR}/{r['acc']}__seqfeat_n{n_windows}.npy") for r in ok])  # (G, 2, 260)
    features = {"evo2": emb,
                "kmer_genome": np.stack([core.tetra_frequencies(f[0]) for f in sf]),
                "kmer_windows": np.stack([core.tetra_frequencies(f[1]) for f in sf]),
                "gc": np.array([[core.gc_from_counts(f[0])] for f in sf])}
    y = np.array([r["y"] if "y" in r else r["motility"] == "yes" for r in ok])
    grid = list(core.LOPO_C_GRID)
    models = {"evo2": {"features": "evo2", "grid": grid, "null": True},
              "kmer_genome": {"features": "kmer_genome", "grid": grid, "null": True},
              "kmer_windows": {"features": "kmer_windows", "grid": grid, "null": True},
              "gc": {"features": "gc", "grid": grid, "null": True},
              "evo2_C1": {"features": "evo2", "grid": [1.0], "null": False}}  # the pre-specified pipeline, untuned
    compare = [("evo2", "kmer_genome"), ("evo2", "kmer_windows"), ("evo2", "gc"), ("evo2_C1", "kmer_genome")]
    with threadpool_limits(limits=1):  # one BLAS thread per worker thread, no oversubscription
        res = core.lopo_evaluate(features, y, [r["phylum"] for r in ok], [r["genus"] for r in ok], models, compare,
                                 n_perm=n_perm, n_boot=n_boot, seed=seed, n_jobs=8)
    seen = sf[:, 1, 256:].sum(1) / sf[:, 0, 256:].sum(1)
    res.update({"n_genomes": len(ok), "n_requested": len(rows), "missing": [r["acc"] for r in rows if r not in ok],
                "median_fraction_of_genome_seen_by_evo2": float(np.median(seen)),
                "median_genome_mb": float(np.median(sf[:, 0, 256:].sum(1)) / 1e6), "n_windows": n_windows})
    return res


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=14400, cpu=8, memory=16384)
def nearclade_run(rows: list[dict], n_windows: int, n_splits: int, n_perm: int, n_perm_order: int, n_boot: int, seed: int) -> dict:
    return _plain_call(_nearclade_run, rows, n_windows, n_splits, n_perm, n_perm_order, n_boot, seed)


def _nearclade_run(rows: list[dict], n_windows: int, n_splits: int, n_perm: int, n_perm_order: int, n_boot: int, seed: int) -> dict:
    """Leave-genus-out inside each phylum on the saved depth-N embeddings, AUC overall and by distance to the nearest training genome."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    out_vol.reload()
    ok = [r for r in rows if os.path.exists(f"{EMB_DIR}/{r['acc']}__n{n_windows}.npy")
          and os.path.exists(f"{FEAT_DIR}/{r['acc']}__seqfeat_n{n_windows}.npy")]
    if not rows or len(ok) < 0.98 * len(rows):
        raise RuntimeError(f"{len(ok)} of {len(rows)} genomes have both a depth-{n_windows} embedding and sequence features (need >= 98%)")
    emb = np.stack([np.load(f"{EMB_DIR}/{r['acc']}__n{n_windows}.npy") for r in ok])
    sf = np.stack([np.load(f"{FEAT_DIR}/{r['acc']}__seqfeat_n{n_windows}.npy") for r in ok])
    features = {"evo2": emb, "kmer_genome": np.stack([core.tetra_frequencies(f[0]) for f in sf]),
                "kmer_windows": np.stack([core.tetra_frequencies(f[1]) for f in sf]),
                "gc": np.array([[core.gc_from_counts(f[0])] for f in sf])}
    grid = list(core.LOPO_C_GRID)
    models = {k: {"features": k, "grid": grid, "null": True} for k in features}
    compare = [("evo2", "kmer_genome"), ("evo2", "kmer_windows"), ("evo2", "gc"), ("evo2", "tax_prior")]
    with threadpool_limits(limits=1):
        res = core.near_clade_evaluate(
            features, np.array([r["y"] for r in ok]), [r["phylum"] for r in ok], [r["genus"] for r in ok],
            {"family": [r["family"] for r in ok], "order": [r["order"] for r in ok], "class": [r["class"] for r in ok]},
            models, compare, n_splits=n_splits, n_perm=n_perm, n_perm_order=n_perm_order, n_boot=n_boot, seed=seed, n_jobs=8)
    res.update({"n_genomes": len(ok), "n_requested": len(rows), "n_windows": n_windows,
                "median_fraction_of_genome_seen_by_evo2": float(np.median(sf[:, 1, 256:].sum(1) / sf[:, 0, 256:].sum(1)))})
    return res


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def _read_panel(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    need = {"ncbi_assembly_accession", "assembly_genbank", "gtdb_phylum", "motility"}
    if missing := need - set(rows[0]):
        raise SystemExit(f"{path} lacks columns {sorted(missing)}")
    return rows


def _existing(subdir: str) -> set[str]:
    try:
        return {Path(e.path).name for e in out_vol.listdir(subdir)}
    except Exception:  # noqa: BLE001  directory not created yet
        return set()


MAX_LEADING_FAILURES = 3  # this many failures before ANY success means something systemic, not a bad genome


def _run_embedding(accs: list[str], depths: list[int], batch_size: int, max_usd: float, save_windows: bool):
    """Embed `accs` on the GPUs, printing spend as it goes. Returns (failures, spent_usd).

    Results come back in input order (`order_outputs=True`), so every result, including a raised
    exception, is paired with its accession by position: a failure can never be anonymous.
    `return_exceptions=True` stops one failing input from cancelling its siblings. The first failure
    is printed in full the moment it arrives, before anything else can obscure it.

    Two deliberate stops, each announced by name (never as a generic cancellation):
      * BUDGET STOP: estimated spend exceeded `max_usd`.
      * ABORT: the first MAX_LEADING_FAILURES results all failed. That is systemic (model load, image,
        GPU), so continuing would only bill for identical failures. In-flight inputs are cancelled by
        the local app exiting; that is this abort, not a fault.
    """
    gpu_s, loads, failures, ok_n, done, streak = 0.0, {}, [], 0, 0, 0
    t0 = time.time()
    print(f"embedding {len(accs)} genomes on up to {MAX_GPUS} x {GPU} (guard: --max-usd {max_usd})", flush=True)
    results = Embedder().embed.map(accs, return_exceptions=True,
                                   kwargs={"depths": depths, "batch_size": batch_size, "save_windows": save_windows})
    try:
        for acc, res in zip(accs, results):
            done += 1
            if isinstance(res, BaseException):  # infrastructure failure: the call never returned a result
                err = "".join(traceback.format_exception(type(res), res, res.__traceback__)).strip()
                res = {"acc": acc, "ok": False, "error": f"{type(res).__name__}: {res}\n(local side)\n{err}"}
            else:
                gpu_s += res.get("gpu_seconds", 0.0)
                if "task_id" in res:
                    loads[res["task_id"]] = res.get("load_s", 0.0)
            spent = core.usd(gpu_s + sum(loads.values()))
            if res["ok"]:
                ok_n += 1
                streak = 0
            else:
                streak += 1
                failures.append({"acc": acc, "error": res.get("error", "unknown")})
                if len(failures) == 1:
                    print(f"\nFIRST FAILURE, {acc}:\n{res.get('error', 'unknown')}\n", flush=True)
                else:
                    print(f"  FAILED {acc}: {str(res.get('error', '')).strip().splitlines()[-1][:200]}", flush=True)
            if done % 5 == 0 or done == len(accs) or not res["ok"]:
                print(f"  [{done}/{len(accs)}] {time.time() - t0:5.0f}s | ok {ok_n} failed {len(failures)} | "
                      f"GPU spend ~${spent:5.2f} on {len(loads)} containers", flush=True)
            if ok_n == 0 and streak >= MAX_LEADING_FAILURES:
                print(f"\nABORT (not a Modal cancellation): the first {streak} inputs all failed and none succeeded, "
                      "so this is systemic and further inputs would bill for the same failure. Cause is in the "
                      "FIRST FAILURE above and in the container log lines starting '[enter]' / '[embed]'.")
                raise SystemExit(3)
            if spent > max_usd:
                print(f"\nBUDGET STOP: estimated GPU spend ${spent:.2f} exceeded --max-usd {max_usd} after "
                      f"{done}/{len(accs)} genomes. In-flight calls are cancelled by this stop. Checkpoints are kept; "
                      "re-run the same command to resume.")
                raise SystemExit(2)
    finally:
        print(f"embedding stage: {ok_n} ok, {len(failures)} failed of {done} returned "
              f"({len(accs) - done} not returned), GPU spend ~${core.usd(gpu_s + sum(loads.values())):.2f}", flush=True)
    return failures, core.usd(gpu_s + sum(loads.values()))


def _prepare(rows_needed: list[dict], skip_checks: bool):
    if not skip_checks:
        print("env check:", verify_env.remote())
        print("weights:", prime_weights.remote())
    print(f"downloading {len(rows_needed)} genomes (CPU containers, verified)...")
    ok, bad = set(), []
    for r in download_genome.map([r["ncbi_assembly_accession"] for r in rows_needed],
                                 [r["assembly_genbank"] for r in rows_needed]):
        (ok.add(r["acc"]) if r["ok"] else bad.append(r))
    if bad:
        print(f"WARNING: {len(bad)} downloads failed and are excluded:")
        for b in bad[:10]:
            print("  ", b["acc"], b["error"][:200])
    return ok, bad


def _write_tsv(path: str, rows: list[dict], cols: list[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.4g}" if isinstance(r[c], float) else r[c] for c in cols])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sweep(panel: str = PANEL, n_genomes: int = 200, depths: str = "10,25,50,100", seed: int = 20260929,
          max_usd: float = 4.0, n_splits: int = 5, batch_size: int = 1, reports: str = "reports",
          dry_run: bool = False, skip_checks: bool = False):
    """Sampling-depth experiment: stratified genomes x depths, motility AUC under phylum-grouped CV."""
    ds = sorted({int(x) for x in depths.split(",")})
    rows = _read_panel(panel)
    for r, g in zip(rows, core.pool_groups([r["gtdb_phylum"] for r in rows])):
        r["group"] = g
    sample = core.stratified_sample(rows, n_genomes, seed)

    alloc: dict[str, int] = {}
    for r in sample:
        alloc[r["group"]] = alloc.get(r["group"], 0) + 1
    print(f"sample: {len(sample)} genomes over {len(alloc)} phylum groups (phyla with <{core.MIN_PHYLUM_SPECIES} species -> 'other')")
    for g, k in sorted(alloc.items(), key=lambda kv: -kv[1]):
        print(f"  {g:24s} {k:4d}")
    print(f"  motility yes/no: {sum(r['motility'] == 'yes' for r in sample)}/{sum(r['motility'] == 'no' for r in sample)}")

    est = core.estimate_gpu(len(sample), max(ds), n_containers=MAX_GPUS)
    print(f"\nestimate: one pool of {max(ds)} windows/genome serves depths {ds}: "
          f"{est['gpu_hours']:.2f} GPU-h ~ ${est['usd']:.2f}  (budget guard --max-usd {max_usd})")
    if dry_run:
        return
    if est["usd"] > max_usd:
        cheaper = core.estimate_gpu(len(sample) // 2, max(ds), n_containers=MAX_GPUS)["usd"]
        raise SystemExit(f"Estimate ${est['usd']:.2f} exceeds --max-usd {max_usd}. Raise the cap, or shrink the run: "
                         f"--n-genomes {len(sample) // 2} (~${cheaper:.2f}) or --depths 10,25,50 (pool of 50).")

    have = _existing("embeddings")
    needed = [r for r in sample if not all(f"{r['ncbi_assembly_accession']}__n{n}.npy" in have for n in ds)]
    print(f"{len(sample) - len(needed)} genomes already checkpointed; {len(needed)} to do")
    ok, bad = _prepare(needed, skip_checks) if needed else (set(), [])
    todo = [r["ncbi_assembly_accession"] for r in needed if r["ncbi_assembly_accession"] in ok]
    fails, spent = _run_embedding(todo, ds, batch_size, max_usd, save_windows=True) if todo else ([], 0.0)
    # Positive confirmation, not exclusion: evaluate only genomes whose files are on the volume.
    have = _existing("embeddings")
    final = [{"acc": r["ncbi_assembly_accession"], "motility": r["motility"], "group": r["group"]}
             for r in sample
             if f"{r['ncbi_assembly_accession']}__meta.json" in have
             and all(f"{r['ncbi_assembly_accession']}__n{n}.npy" in have for n in ds)]
    print(f"\n{len(final)} of {len(sample)} genomes embedded on the volume "
          f"({len(fails)} embed failures, {len(bad)} download failures)")
    if len(final) < 0.9 * len(sample):
        for f in fails[:5]:
            print(f"  {f['acc']}: {str(f['error']).strip().splitlines()[-1][:200]}")
        raise SystemExit(f"Refusing to evaluate: only {len(final)} of {len(sample)} genomes are embedded "
                         "(need >= 90%). Fix the failures above and re-run the same command; checkpoints are kept.")
    print(f"evaluating {len(final)} genomes...")
    ev = evaluate_sweep.remote(final, ds, n_splits)
    if ev["invalid_depths"]:
        raise SystemExit(f"invalid embeddings at depths {ev['invalid_depths']}")

    sel = core.select_depth(ev["results"])
    cols = ["depth", "n_genomes", "auc_mean", "auc_std", "folds_scored", "folds_total", "frac_overlapping",
            "s_per_genome", "hours_2580", "usd_2580"]
    _write_tsv(f"{reports}/tables/evo2_sampling_depth_sweep.tsv", ev["results"], cols)
    _write_tsv(f"{reports}/tables/evo2_sweep_genomes.tsv", final, ["acc", "group", "motility"])
    core.dump_json({**sel, "results": ev["results"], "n_genomes": len(final), "seed": seed, "n_splits": n_splits,
                    "sampling": core.SAMPLING, "sample_stats": {k: v for k, v in ev.items() if k != "results"}},
                   SELECTED_DEPTH_JSON.replace("reports", reports, 1))

    print(f"\n{'depth':>6} {'AUC':>15} {'folds':>6} {'overlap':>8} {'s/genome':>9} {'h for 2,580':>12} {'$ for 2,580':>12}")
    for r in ev["results"]:
        print(f"{r['depth']:>6} {r['auc_mean']:>8.3f} ± {r['auc_std']:.3f} {r['folds_scored']:>3}/{r['folds_total']} "
              f"{r['frac_overlapping']:>7.0%} {r['s_per_genome']:>9.1f} {r['hours_2580']:>12.1f} {r['usd_2580']:>12.0f}")
    print("overlap = share of genomes whose total usable sequence (all contigs >= 8,192 bp) is shorter than n windows")
    print(f"sampled genomes: median {ev['median_contigs_used']:.0f} usable contigs, median usable {ev['median_usable_kb']:.0f} kb "
          f"(longest contig alone: {ev['median_longest_contig_kb']:.0f} kb), "
          f"{ev['median_frac_of_genome_usable']:.1%} of each genome usable")
    print(f"\nselected depth (one-SE rule): {sel['selected_depth']}  (best mean AUC at {sel['best_depth']}, SE {sel['one_se']:.3f})")
    print(f"sweep GPU spend ~${spent:.2f} this invocation (Modal dashboard is authoritative)")


@app.local_entrypoint()
def embed_all(panel: str = PANEL, n_windows: int = 0, max_usd: float = 25.0, batch_size: int = 1,
              limit: int = 0, reports: str = "reports", dry_run: bool = False, skip_checks: bool = False,
              phyla: str = ""):
    """Production run at one depth. Resumable: existing checkpoints are skipped.

    --phyla restricts to a comma-separated list of GTDB phyla (exact names); default is the whole panel.
    """
    if n_windows <= 0:
        import json

        p = SELECTED_DEPTH_JSON.replace("reports", reports, 1)
        if not Path(p).exists():
            raise SystemExit(f"pass --n-windows, or run `sweep` first to produce {p}")
        n_windows = json.load(open(p))["selected_depth"]
        print(f"using sweep-selected depth {n_windows}")
    rows = _read_panel(panel)
    if phyla:
        keep = {x.strip() for x in phyla.split(",") if x.strip()}
        unknown = keep - {r["gtdb_phylum"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown phyla {sorted(unknown)}; names are GTDB phyla exactly as in the panel")
        rows = [r for r in rows if r["gtdb_phylum"] in keep]
        print("phyla: " + ", ".join(f"{p} {sum(r['gtdb_phylum'] == p for r in rows)}" for p in sorted(keep)) + f"  ({len(rows)} genomes)")
    if limit:
        rows = rows[:limit]
    have = _existing("embeddings")
    needed = [r for r in rows if f"{r['ncbi_assembly_accession']}__n{n_windows}.npy" not in have]
    est = core.estimate_gpu(len(needed), n_windows, n_containers=MAX_GPUS)
    print(f"{len(rows) - len(needed)}/{len(rows)} done; {len(needed)} to embed at n={n_windows}: "
          f"{est['gpu_hours']:.1f} GPU-h ~ ${est['usd']:.0f} (--max-usd {max_usd})")
    if dry_run or not needed:
        return
    if est["usd"] > max_usd:
        raise SystemExit(f"Estimate ${est['usd']:.0f} exceeds --max-usd {max_usd}; raise it deliberately or lower --n-windows.")
    ok, bad = _prepare(needed, skip_checks)
    todo = [r["ncbi_assembly_accession"] for r in needed if r["ncbi_assembly_accession"] in ok]
    fails, spent = _run_embedding(todo, [n_windows], batch_size, max_usd, save_windows=False)
    allf = [{"acc": b["acc"], "error": b["error"]} for b in bad] + [{"acc": f.get("acc"), "error": f.get("error")} for f in fails]
    if allf:
        _write_tsv(f"{reports}/tables/evo2_embed_all_failures.tsv", allf, ["acc", "error"])
    print(f"done: {len(todo) - len(fails)} embedded, {len(allf)} failed (see reports/tables/evo2_embed_all_failures.tsv); "
          f"GPU spend ~${spent:.2f}. Fetch with: modal volume get evo2-embeddings embeddings ./evo2_embeddings")


def _fmt_folds(fr: dict) -> str:
    lines = [f"  {'fold':>4} {'held-out phyla':<52} {'n_test':>6} {'prev_train':>10} {'prev_test':>9} {'AUC':>6} {'mean P(motile) yes/no':>22}"]
    for f in fr["folds"]:
        auc = f"{f['auc']:.3f}" if f["auc"] == f["auc"] else "  n/a"
        pm = (f"{f['mean_p_motile_actual_yes']:.2f} / {f['mean_p_motile_actual_no']:.2f}"
              if f["auc"] == f["auc"] else "n/a")
        lines.append(f"  {f['fold']:>4} {', '.join(f['test_groups'])[:52]:<52} {f['n_test']:>6} {f['prev_train']:>10.2f} "
                     f"{f['prev_test']:>9.2f} {auc:>6} {pm:>22}")
    return "\n".join(lines)


@app.local_entrypoint()
def diagnose(panel: str = PANEL, genomes: str = "reports/tables/evo2_sweep_genomes.tsv",
             depths: str = "10,25,50,100", n_perm: int = 50, n_splits: int = 5, seed: int = 20260929,
             reports: str = "reports"):
    """Audit a finished sweep: pairing, positive class, fold structure, label-shuffle nulls, GC baseline.

    CPU only (no GPU, no re-embedding): reads the checkpoints already on the volume.
    """
    import json

    ds = sorted({int(x) for x in depths.split(",")})
    with open(genomes, newline="") as f:
        sweep_rows = list(csv.DictReader(f, delimiter="\t"))
    rows = [{"acc": r["acc"], "motility": r["motility"], "group": r["group"]} for r in sweep_rows]

    # Label provenance, re-derived from the panel TSV independently of the sweep's own table.
    pan = _read_panel(panel)
    grp = dict(zip([r["ncbi_assembly_accession"] for r in pan], core.pool_groups([r["gtdb_phylum"] for r in pan])))
    lab = {r["ncbi_assembly_accession"]: r["motility"] for r in pan}
    bad = [r["acc"] for r in rows if lab.get(r["acc"]) != r["motility"] or grp.get(r["acc"]) != r["group"]]
    print(f"label cross-check against {panel}, by accession: {len(rows) - len(bad)}/{len(rows)} match" + (f"  MISMATCH {bad[:5]}" if bad else ""))

    d = diagnose_sweep.remote(rows, ds, n_splits, n_perm, seed)
    Path(f"{reports}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{reports}/tables/evo2_sweep_diagnostics.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")

    a = d["alignment"]
    print("\n=== 1. LABEL ALIGNMENT (embedding <-> accession <-> label) ===")
    print(f"  {a['n_checked']} genomes: meta.acc == accession: {a['all_meta_acc_match']} | all windows lie on this genome's own contigs: "
          f"{a['all_windows_on_own_genome']} | every depth .npy == mean of its own windows: {a['all_npy_equal_mean_of_own_windows']}")
    print(f"  distinct embedding vectors: {a['n_distinct_vectors']}/{a['n_checked']} | max cosine between two different genomes: "
          f"{a['max_cosine_between_different_genomes']:.4f}")
    for x in a["audit"]:
        print(f"  row {x['row']:>3} {x['acc']}  label={x['label_motility']:<3} group={x['group']:<18} loaded {x['loaded'][0]}..{x['loaded'][-1]} "
              f"(meta_acc {x['meta_acc']}, {x['n_windows']} windows on own contigs: {x['windows_lie_on_this_genomes_contigs']})")
    pc = d["pairing_control"]
    print(f"  pairing control (independent of motility): predict each genome's GC from its depth-{pc['depth']} embedding, "
          f"out-of-phylum Spearman r = {pc['spearman_r']:.3f}  (same test with GC shuffled across genomes: {pc['shuffled_pairing_r']:.3f})")

    top = f"n{max(ds)}"
    fr = d["depths"][top]["fold_report"]
    print("\n=== 2. POSITIVE CLASS ===")
    print(f"  y = (motility == 'yes') as bool; fitted classes_ = {fr['classes']}, so predict_proba[:, 1] = P({fr['classes'][1]}) = P(motile), "
          "the same array is used at fit and at score time.")
    diffs = [abs(f['auc'] - f['auc_manual']) for f in fr['folds'] if f['auc'] == f['auc']]
    print(f"  sklearn roc_auc vs independent rank-statistic AUC (positive = motile): max |difference| over folds = {max(diffs):.2e}")

    print("\n=== 3. FOLD STRUCTURE (GroupKFold by pooled phylum) ===")
    print(f"  overall motility prevalence {d['labels']['prevalence_motile']:.2f}; per phylum: " +
          ", ".join(f"{g} {v['prevalence_motile']:.2f} (n={v['n']})" for g, v in d['labels']['by_group'].items()))
    for n in (min(ds), max(ds)):
        print(f"  depth {n}:")
        print(_fmt_folds(d["depths"][f"n{n}"]["fold_report"]))

    print("\n=== 4. SHUFFLED-LABEL NULLS vs REAL (same folds; mean AUC over folds) ===")
    print(f"  {'depth':>5} {'real':>7} | {'global shuffle':>26} | {'shuffle within phylum':>26} | {'ungrouped CV':>12} | {'C=0.01':>7}")
    for n in ds:
        x = d["depths"][f"n{n}"]
        g, w = x["null_global"], x["null_within_group"]
        print(f"  {n:>5} {x['real']['auc_mean']:>7.3f} | {g['mean']:.3f} ± {g['sd']:.3f} [{g['q05']:.2f},{g['q95']:.2f}] | "
              f"{w['mean']:.3f} ± {w['sd']:.3f} [{w['q05']:.2f},{w['q95']:.2f}] | {x['ungrouped_stratified_cv']['auc_mean']:>12.3f} | "
              f"{x['strong_regularisation_C0.01_auc']:>7.3f}")
    print(f"  ({d['n_perm']} permutations each; brackets are the 5th-95th percentile of the null)")

    g = d["gc_baseline"]
    print("\n=== 5. GC-CONTENT-ONLY BASELINE (one feature, identical CV) ===")
    print(f"  GC range {g['gc_range'][0]:.2f}-{g['gc_range'][1]:.2f}; Spearman(GC, motile) = {g['spearman_gc_vs_motility']:.3f}; "
          f"grouped-CV AUC = {g['cv']['auc_mean']:.3f} ± {g['cv']['auc_std']:.3f}; per-fold AUCs " + ", ".join(f"{v:.2f}" for v in g['fold_aucs']))
    print(f"\nsaved {reports}/tables/evo2_sweep_diagnostics.json  (CPU only, no GPU spend)")


def _trait_report(panel: str, reports: str, traits: list[str]) -> list[dict]:
    """Which trait has the most WITHIN-phylum variance? Local, stdlib only: reads the panel TSV."""
    rows = _read_panel(panel)
    groups = core.pool_groups([r["gtdb_phylum"] for r in rows])
    tab = core.trait_targets(rows, traits, groups)
    tab.sort(key=lambda t: -t["gini_within"])
    print(f"=== WITHIN-PHYLUM VARIANCE BY TRAIT: {len(rows)} panel genomes, {len(set(groups))} pooled groups (as in the CV) ===")
    print("  within share = fraction of the trait's variance that phylum does NOT explain; off-majority = genomes that disagree with")
    print("  their own group's majority (the within-group information a model could learn); testable group = >= 30 genomes with")
    print("  >= 15 of each class, so a held-out AUC inside that group can be scored.")
    print(f"  {'binary target':30s} {'prev':>5} {'Gini within':>11} {'within share':>12} {'off-majority':>14} {'testable groups':>15} {'genomes in them':>15}")
    for t in tab:
        print(f"  {t['target']:30s} {t['prevalence']:>5.0%} {t['gini_within']:>11.3f} {t['within_share']:>12.1%} "
              f"{t['off_majority']:>6} ({t['off_majority_frac']:>4.0%}) {t['testable_groups']:>15} {t['testable_genomes']:>15}")
    for t in tab[:3]:
        print(f"\n  {t['target']}: per-group detail (SE = Hanley-McNeil SE of a within-group AUC of 0.70 using every panel genome of the group)")
        for gname, v in sorted(t["per_group"].items(), key=lambda kv: -kv[1]["n"]):
            se = f"{v['se_at_full_panel']:.3f}" if v["se_at_full_panel"] == v["se_at_full_panel"] else "  n/a"
            print(f"    {gname:20s} n={v['n']:>4} positive={v['n_positive']:>4} ({v['prevalence']:>4.0%})  testable={str(v['testable']):5s} SE={se}")
    _write_tsv(f"{reports}/tables/trait_within_phylum_variance.tsv", tab,
               ["target", "prevalence", "n_positive", "gini_total", "gini_within", "within_share", "off_majority",
                "off_majority_frac", "testable_groups", "testable_genomes"])
    return tab


def _first_at_least(xs, ys, thr):
    return next((x for x, y in zip(xs, ys) if y >= thr), None)


@app.local_entrypoint()
def reliability(panel: str = PANEL, genomes: str = "reports/tables/evo2_sweep_genomes.tsv", seed: int = 20260929,
                reps: int = 10, n_boot: int = 100, reports: str = "reports", traits_only: bool = False,
                within_phylum: bool = False):
    """Sampling-depth reliability curve + which trait has the most within-phylum variance. CPU only, no GPU."""
    import json
    import statistics as st

    _trait_report(panel, reports, ["gram", "shape", "motility", "spore", "oxygen", "temperature"])
    if traits_only:
        return
    with open(genomes, newline="") as f:
        rows = [{"acc": r["acc"], "motility": r["motility"], "group": r["group"]} for r in csv.DictReader(f, delimiter="\t")]
    d = reliability_run.remote(rows, seed, reps, n_boot, within_phylum)
    Path(f"{reports}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{reports}/tables/evo2_depth_reliability.json").write_text(json.dumps(d, indent=2, sort_keys=True, default=str) + "\n")

    c, grid = d["curve"], d["grid"]
    cos = c["cosine"]
    print(f"\n=== SAMPLING-DEPTH RELIABILITY: {d['n_genomes']} genomes, pool of {d['pool_n']} evenly spaced windows each ===")
    print(f"  mean cosine between different genomes:  raw {cos['raw']:.4f}  ->  mean-centred {cos['centred']:.4f}  ->  centred + per-dim scaled {cos['centred_scaled']:.4f}")
    print("  (variances below are computed in that centred, scaled space; the shared component is removed)")
    if c.get("within_groups"):
        print(f"  WITHIN-PHYLUM MODE: genomes are centred and scaled within each of {c['n_groups']} pooled groups, so 'between' is the variation")
        print("  among genomes of the SAME phylum, the differences a within-phylum comparison has to resolve.")
    print(f"  per-window variance {c['per_window_variance']:.0f}, between-genome variance {c['between_full']:.0f} (sum over {c['dims_effective']:.0f} standardised dims)")
    print(f"\n  {'n':>4} | {'within/between':>26} | {'random-subset':>13} | {'reliability':>24} | {'GC Spearman':>11} {'GC R2':>7} | {'motility AUC (ungrouped)':>24}")
    sysm, rnd, mod, an = c["systematic"], c["random"], c["model"], d["anchors"]
    xs, rel, ratio = [], [], []
    for n in grid:
        e = sysm.get(n)
        m = mod[n]
        ratio_v, rel_v = (e["ratio"], e["reliability"]) if e else (m["ratio"], m["reliability"])
        xs.append(n), rel.append(rel_v), ratio.append(ratio_v)
        a = an[n]
        rng_txt = f"{ratio_v:>7.3f} [{e['ratio_ci'][0]:.3f},{e['ratio_ci'][1]:.3f}]" if e else f"{ratio_v:>7.3f} (model, extrapolated)"
        rel_txt = f"{rel_v:>6.3f} [{e['reliability_ci'][0]:.3f},{e['reliability_ci'][1]:.3f}]" if e else f"{rel_v:>6.3f} (model)"
        rnd_txt = f"{rnd[n]['ratio']:>13.3f}" if n in rnd else f"{'-':>13}"
        print(f"  {n:>4} | {rng_txt:>26} | {rnd_txt} | {rel_txt:>24} | {st.mean(a['spearman']):>11.3f} {st.mean(a['r2']):>7.3f} | "
              f"{st.mean(a['motility_auc']):>8.3f} ± {(st.stdev(a['motility_auc']) if len(a['motility_auc']) > 1 else 0):.3f}")
    gains = [(n, sysm[n]["systematic_gain"]) for n in grid if n in sysm and "systematic_gain" in sysm[n]]
    print("  even spacing vs random windows (within-variance ratio, >1 = even spacing wins): " + ", ".join(f"n={n}: {g:.2f}" for n, g in gains))
    print("  n=100 has no disjoint halves in a pool of 100, so it comes from the per-window-variance model (within = per-window variance / n).")

    def noise(key, sdkey=None):
        return max(st.stdev(an[n][key]) if len(an[n][key]) > 1 else 0.0 for n in grid) if key else 0.0

    print("\n  WHERE EACH CURVE FLATTENS (95% of the total gain from n=1 to the best n; 'flat' = total gain within 2x noise):")
    fl = core.flattens_at(xs, rel, True)
    print(f"    split-half reliability : n = {fl['n']}   (reliability >= 0.90 at n = {_first_at_least(xs, rel, .90)}, >= 0.95 at n = {_first_at_least(xs, rel, .95)}, >= 0.99 at n = {_first_at_least(xs, rel, .99)})")
    fr = core.flattens_at(xs, ratio, False)
    print(f"    within/between ratio   : n = {fr['n']}   (ratio <= 0.10 at n = {next((x for x, y in zip(xs, ratio) if y <= .10), None)}, <= 0.05 at n = {next((x for x, y in zip(xs, ratio) if y <= .05), None)})")
    for name, key in (("GC Spearman", "spearman"), ("GC R^2", "r2")):
        ys = [st.mean(an[n][key]) for n in grid]
        f = core.flattens_at(xs, ys, True, noise=noise(key))
        print(f"    {name:23s}: " + ("flat within noise from n=1" if f["flat_within_noise"] else f"n = {f['n']}") + f"   (n=1: {ys[0]:.3f}, best: {max(ys):.3f})")
    ys = [st.mean(an[n]["motility_auc"]) for n in grid]
    f = core.flattens_at(xs, ys, True, noise=max(noise("motility_auc"), max(st.mean(an[n]["motility_cv_sd"]) for n in grid)))
    print(f"    motility AUC (ungrouped): " + ("no depth effect detectable (flat within noise)" if f["flat_within_noise"] else f"n = {f['n']}") + f"   (n=1: {ys[0]:.3f}, best: {max(ys):.3f})")
    print(f"\nsaved {reports}/tables/evo2_depth_reliability.json and trait_within_phylum_variance.tsv  (CPU only, no GPU spend)")


def _ci(x: dict) -> str:
    return f"{x['auc']:.3f} [{x['ci'][0]:.3f},{x['ci'][1]:.3f}]"


@app.local_entrypoint()
def lopo(panel: str = PANEL, phyla: str = "Pseudomonadota,Bacillota,Actinomycetota,Bacteroidota", n_windows: int = 10,
         n_perm: int = 50, n_boot: int = 1000, seed: int = 20260929, batch: int = 40, reports: str = "reports",
         target: str = "motility"):
    """Leave-one-phylum-out, AUC scored WITHIN the held-out phylum: Evo 2 vs composition baselines. CPU only.

    --target is one of motility, oxygen_aerobe, oxygen_facultative, shape_rod (positive class vs the rest).
    """
    import json

    if target not in TARGETS:
        raise SystemExit(f"--target must be one of {sorted(TARGETS)}")
    col, pos = TARGETS[target]
    keep = [x.strip() for x in phyla.split(",") if x.strip()]
    rows = [{"acc": r["ncbi_assembly_accession"], "phylum": r["gtdb_phylum"], "genus": r["gtdb_genus"],
             "motility": r["motility"], "y": r[col] == pos}
            for r in _read_panel(panel) if r["gtdb_phylum"] in keep]
    have = _existing("embeddings")
    emb = [r for r in rows if f"{r['acc']}__n{n_windows}.npy" in have and f"{r['acc']}__meta.json" in have]
    print(f"{len(emb)} of {len(rows)} genomes in {len(keep)} phyla have a depth-{n_windows} embedding on the volume")
    if len(emb) < 0.98 * len(rows):
        raise SystemExit(f"Need >= 98% embedded. Run: modal run -m src.evo2_modal::embed_all --n-windows {n_windows} --phyla {','.join(keep)}")

    feats_have = _existing("features")
    todo = [r["acc"] for r in emb if f"{r['acc']}__seqfeat_n{n_windows}.npy" not in feats_have]
    print(f"sequence features: {len(emb) - len(todo)} cached, {len(todo)} to compute (CPU)")
    errs = []
    for res in seqfeat_batch.map([todo[i:i + batch] for i in range(0, len(todo), batch)], kwargs={"n_windows": n_windows}) if todo else []:
        errs += res["errors"]
    if errs:
        print(f"  {len(errs)} genomes failed feature extraction, e.g. {errs[0][0]}: {errs[0][1].strip().splitlines()[-1]}")

    print(f"evaluating (leave-one-phylum-out, {n_perm} shuffles, {n_boot} genus-cluster bootstraps)...")
    d = lopo_run.remote(emb, n_windows, n_perm, n_boot, seed)
    Path(f"{reports}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{reports}/tables/evo2_lopo_{target}.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")

    names = {"evo2": "Evo 2", "kmer_genome": "k-mer(genome)", "kmer_windows": "k-mer(windows)", "gc": "GC only"}
    print(f"\n=== {target.upper()} ({col} = {pos} vs rest), LEAVE-ONE-PHYLUM-OUT: {d['n_genomes']} of {d['n_requested']} genomes, depth {d['n_windows']} ===")
    print(f"  AUC is scored WITHIN the held-out phylum, trained on the other {len(d['phyla']) - 1}. Brackets: 95% genus-cluster bootstrap CI.")
    print(f"  Evo 2 saw {d['median_fraction_of_genome_seen_by_evo2']:.1%} of a median {d['median_genome_mb']:.1f} Mb genome; "
          "k-mer(genome) sees all of it, k-mer(windows) sees only the windows Evo 2 saw.")
    print("  Floor: a constant score (within-phylum majority class) ties every pair, so its within-phylum AUC is exactly 0.500.")
    print(f"\n  {'held-out phylum':<16} {'n':>4} {'motile':>6} {'genera':>6} | " + " | ".join(f"{v:^21}" for v in names.values()) + " | floor")
    for h in d["phyla"]:
        p = d["per_phylum"][h]
        print(f"  {h:<16} {p['n']:>4} {p['prevalence']:>6.0%} {p['n_genera']:>6} | " + " | ".join(f"{_ci(p['models'][k]):^21}" for k in names) + " | 0.500")
    print(f"  {'MACRO MEAN':<16} {'':>4} {'':>6} {'':>6} | " + " | ".join(f"{_ci(d['macro']['models'][k]):^21}" for k in names) + " | 0.500")

    print("\n  PAIRED DIFFERENCE IN AUC (same genera resampled for both models; 95% CI; share of resamples where Evo 2 is ahead)")
    for pair in ("evo2 - kmer_genome", "evo2 - kmer_windows", "evo2 - gc", "evo2_C1 - kmer_genome"):
        cells = []
        for h in d["phyla"]:
            x = d["per_phylum"][h]["deltas"][pair]
            cells.append(f"{h[:6]} {x['delta']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]")
        m = d["macro"]["deltas"][pair]
        print(f"  {pair:<22} " + " | ".join(cells) + f" || MACRO {m['delta']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}] ahead in {m['share_boot_positive']:.0%}")

    print("\n  SHUFFLE NULL (labels permuted within each phylum, model refit; mean ± sd of the null AUC, and p = P(null >= observed))")
    for k in names:
        cells = []
        for h in d["phyla"]:
            x = d["per_phylum"][h]["models"][k]["null"]
            cells.append(f"{h[:6]} {x['mean']:.3f}±{x['sd']:.3f} p={x['p']:.3f}")
        m = d["macro"]["models"][k]["null"]
        print(f"  {names[k]:<15} " + " | ".join(cells) + f" || MACRO {m['mean']:.3f}±{m['sd']:.3f} p={m['p']:.3f}")

    print("\n  REGULARISATION (C chosen by inner leave-one-phylum-out over the training phyla) and the untuned pre-specified pipeline")
    for k in names:
        print(f"  {names[k]:<15} chosen C: " + ", ".join(f"{h[:6]} {d['per_phylum'][h]['models'][k]['C']:g}" for h in d["phyla"]))
    print("  Evo 2, C=1 fixed:  " + ", ".join(f"{h[:6]} {d['per_phylum'][h]['models']['evo2_C1']['auc']:.3f}" for h in d["phyla"]) +
          f" || macro {d['macro']['models']['evo2_C1']['auc']:.3f}")

    m = d["macro"]["deltas"]["evo2 - kmer_genome"]
    wins = sum(d["per_phylum"][h]["deltas"]["evo2 - kmer_genome"]["ci"][0] > 0 for h in d["phyla"])
    losses = sum(d["per_phylum"][h]["deltas"]["evo2 - kmer_genome"]["ci"][1] < 0 for h in d["phyla"])
    if m["ci"][0] > 0:
        verdict = "Evo 2 beats whole-genome tetranucleotide frequencies"
    elif m["ci"][1] < 0:
        verdict = "Evo 2 does NOT beat tetranucleotide frequencies: they are ahead (this is the headline)"
    else:
        verdict = "no detectable difference between Evo 2 and whole-genome tetranucleotide frequencies (the CI spans zero)"
    print(f"\n  VERDICT vs k-mer(genome): {verdict}. Macro ΔAUC {m['delta']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}]; "
          f"Evo 2 clearly ahead in {wins}/{len(d['phyla'])} phyla, clearly behind in {losses}/{len(d['phyla'])}.")
    print(f"\nsaved {reports}/tables/evo2_lopo_{target}.json  (CPU only, no GPU spend)")


TAXONOMY_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120_taxonomy.tsv.gz"
TAXONOMY_MIN_BYTES = 5_000_000  # the real file is ~10 MB; anything smaller is a truncated download


def _panel_families(pan: list[dict], cache: str = "data/raw/gtdb/bac120_taxonomy.tsv.gz") -> tuple[dict, dict]:
    """{GTDB accession: family} from GTDB's small taxonomy file (cached under data/raw/gtdb like the other GTDB files).

    The panel has genus/order/class/phylum but no family. A genome's family is used only if the taxonomy file agrees with
    the panel on all four of those ranks (guards against the file being a different GTDB release); otherwise it gets a
    unique placeholder, so it can match nothing. Returns (families, report).
    """
    import gzip
    import urllib.request

    path = Path(cache)
    if not path.exists() or path.stat().st_size < TAXONOMY_MIN_BYTES:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(TAXONOMY_URL, timeout=300) as resp, open(tmp, "wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
        if tmp.stat().st_size < TAXONOMY_MIN_BYTES:
            tmp.unlink()
            raise SystemExit(f"GTDB taxonomy download looks truncated (< {TAXONOMY_MIN_BYTES} bytes); not using it")
        tmp.replace(path)
    tax = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            acc, t = line.rstrip("\n").split("\t")
            tax[acc] = dict(x.split("__", 1) for x in t.split(";"))
    fam, mismatch, missing = {}, 0, 0
    for r in pan:
        a = r["gtdb_accession"]
        t = tax.get(a)
        if t is None:
            missing += 1
        elif (t["g"], t["o"], t["c"], t["p"]) == (r["gtdb_genus"], r["gtdb_order"], r["gtdb_class"], r["gtdb_phylum"]) and t["f"]:
            fam[a] = t["f"]
            continue
        else:
            mismatch += 1
        fam[a] = f"?unknown:{a}"
    return fam, {"genomes": len(pan), "missing": missing, "rank_mismatch": mismatch, "file": str(path)}


def _rank_row(d: dict, fmt) -> str:
    return " | ".join(fmt(k) for k in d)


@app.local_entrypoint()
def nearclade(panel: str = PANEL, phyla: str = "Pseudomonadota,Bacillota,Actinomycetota,Bacteroidota", n_windows: int = 10,
              target: str = "motility", n_splits: int = 10, n_perm: int = 50, n_perm_order: int = 30, n_boot: int = 200,
              seed: int = 20260929, batch: int = 40, reports: str = "reports"):
    """Near-clade prediction: leave-genus-out within each phylum, AUC by taxonomic distance to the nearest training genome. CPU only."""
    import json

    if target not in TARGETS:
        raise SystemExit(f"--target must be one of {sorted(TARGETS)}")
    col, pos = TARGETS[target]
    keep = [x.strip() for x in phyla.split(",") if x.strip()]
    pan = [r for r in _read_panel(panel) if r["gtdb_phylum"] in keep]
    fam, frep = _panel_families(pan)
    print(f"GTDB taxonomy (family) for {frep['genomes']} genomes: {frep['missing']} missing, {frep['rank_mismatch']} disagree with the panel on genus/order/class/phylum")
    rows = [{"acc": r["ncbi_assembly_accession"], "phylum": r["gtdb_phylum"], "class": r["gtdb_class"], "order": r["gtdb_order"],
             "family": fam[r["gtdb_accession"]], "genus": r["gtdb_genus"], "y": r[col] == pos} for r in pan]
    have = _existing("embeddings")
    emb = [r for r in rows if f"{r['acc']}__n{n_windows}.npy" in have and f"{r['acc']}__meta.json" in have]
    print(f"{len(emb)} of {len(rows)} genomes have a depth-{n_windows} embedding on the volume")
    if len(emb) < 0.98 * len(rows):
        raise SystemExit(f"Need >= 98% embedded. Run: modal run -m src.evo2_modal::embed_all --n-windows {n_windows} --phyla {','.join(keep)}")
    feats_have = _existing("features")
    todo = [r["acc"] for r in emb if f"{r['acc']}__seqfeat_n{n_windows}.npy" not in feats_have]
    print(f"sequence features: {len(emb) - len(todo)} cached, {len(todo)} to compute (CPU)")
    for res in seqfeat_batch.map([todo[i:i + batch] for i in range(0, len(todo), batch)], kwargs={"n_windows": n_windows}) if todo else []:
        for e in res["errors"]:
            print("  feature extraction failed:", e[0])

    print(f"leave-genus-out within phylum ({n_splits} genus-grouped folds), {n_perm}+{n_perm_order} shuffles, {n_boot} bootstraps ...")
    d = nearclade_run.remote(emb, n_windows, n_splits, n_perm, n_perm_order, n_boot, seed)
    Path(f"{reports}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{reports}/tables/evo2_nearclade_{target}.json").write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    _print_nearclade(d, target, col, pos, f"{reports}/tables/evo2_lopo_{target}.json")
    print(f"\nsaved {reports}/tables/evo2_nearclade_{target}.json  (CPU only, no GPU spend)")


def _a(x: dict, key: str = "auc", ci: str = "ci") -> str:
    return f"{x[key]:.3f} [{x[ci][0]:.2f},{x[ci][1]:.2f}]"


def _print_nearclade(d: dict, target: str, col: str, pos: str, lopo_path: str) -> None:
    import json

    P = d["phyla"]
    M = ["evo2", "kmer_genome", "kmer_windows", "gc", "tax_prior"]
    nm = {"evo2": "Evo 2", "kmer_genome": "k-mer(genome)", "kmer_windows": "k-mer(windows)", "gc": "GC only", "tax_prior": "tax-prior*"}
    print(f"\n=== NEAR-CLADE: {target} ({col} = {pos} vs rest), leave-genus-out within phylum, depth {d['n_windows']}, {d['n_genomes']} of {d['n_requested']} genomes ===")
    print(f"  Genera never span train and test ({d['n_splits']} genus-grouped folds per phylum); each model trains on the OTHER genera of the same phylum.")
    print("  *tax-prior is an extra baseline: trait prevalence among training genomes in the nearest shared taxon (family > order > class > phylum).")
    print("  Floor (constant score) has AUC exactly 0.500. Brackets: 95% genus-cluster bootstrap CI.")

    print("\n  1. HELD-OUT GENUS STRUCTURE (AUC needs both classes inside the group scored)")
    print(f"  {'phylum':<16} {'genomes':>7} {'genera':>6} {'1 genome':>8} {'>=2':>5} {'both classes':>12} {'all +':>6} {'all -':>6} {'genomes in scorable genera':>27}")
    for h in P:
        e = d["per_phylum"][h]; g = e["genus_stats"]
        print(f"  {h:<16} {e['n']:>7} {g['n_genera']:>6} {g['single_genome']:>8} {g['two_or_more']:>5} {g['scorable_both_classes']:>12} {g['all_positive']:>6} {g['all_negative']:>6} "
              f"{g['genomes_in_scorable']:>15} ({g['genomes_in_scorable'] / e['n']:.0%})")
    print("  Within-genus AUC uses only same-genus pairs, so it exists only for the 'both classes' genera. The pooled AUC below uses every held-out genome.")

    print("\n  2. HOW CLOSE IS THE NEAREST TRAINING RELATIVE? (lowest rank shared with any training genome; positives in brackets)")
    print(f"  {'phylum':<16} " + " ".join(f"{NC_LABELS_[r]:>16}" for r in core.NC_RANKS))
    for h in P:
        dc = d["per_phylum"][h]["distance_counts"]
        print(f"  {h:<16} " + " ".join(f"{dc[r]['n']:>9} ({dc[r]['n_positive']:>4})" for r in core.NC_RANKS))

    lopo = None
    if Path(lopo_path).exists():
        lopo = json.load(open(lopo_path))
    print("\n  3. PER-PHYLUM AUC: NEAR-CLADE (leave-genus-out, pooled) vs OUT-OF-CLADE (leave-phylum-out, from the earlier run)")
    print(f"  {'phylum':<16} | " + " | ".join(f"{nm[m]:^19}" for m in M) + " | floor" + ("  ||  out-of-clade: Evo 2 / k-mer(g) / GC" if lopo else ""))
    for h in P:
        m = d["per_phylum"][h]["models"]
        tail = ""
        if lopo and h in lopo["per_phylum"]:
            lm = lopo["per_phylum"][h]["models"]
            tail = f"  ||  {lm['evo2']['auc']:.3f} / {lm['kmer_genome']['auc']:.3f} / {lm['gc']['auc']:.3f}"
        print(f"  {h:<16} | " + " | ".join(f"{_a(m[k], 'pooled_auc', 'pooled_ci'):^19}" for k in M) + " | 0.500" + tail)
    mm = d["macro"]["models"]
    tail = ""
    if lopo:
        lm = lopo["macro"]["models"]
        tail = f"  ||  {lm['evo2']['auc']:.3f} / {lm['kmer_genome']['auc']:.3f} / {lm['gc']['auc']:.3f}"
    print(f"  {'MACRO MEAN':<16} | " + " | ".join(f"{_a(mm[k], 'pooled_auc', 'pooled_ci'):^19}" for k in M) + " | 0.500" + tail)
    print("  within-genus AUC (same-genus pairs only; genera scored in brackets):")
    for h in P:
        e = d["per_phylum"][h]
        if "within_genus_scored" in e:
            ws = e["within_genus_scored"]
            print(f"    {h:<16} ({ws['n_genera']:>3} genera, {ws['n_pairs']:>5} pairs) " + " | ".join(f"{nm[k]} {_a(e['models'][k]['within_genus'])}" for k in M))
        else:
            print(f"    {h:<16} no genus holds both classes")

    print("\n  4. DECAY CURVE: pooled AUC by taxonomic distance to the nearest TRAINING genome")
    br = d["bin_rule"]
    print(f"  Bins are merged until each holds >= {br['min_n']} genomes and >= {br['min_each_class']} of each class (over all phyla); a phylum's cell is scored only if it meets that itself.")
    for b in d["bins"]:
        print(f"\n  [{b['label']}]")
        print(f"  {'phylum':<16} {'n':>5} {'pos':>4} {'genera':>6} | " + " | ".join(f"{nm[k]:^19}" for k in M))
        for h in P:
            c = b["phyla"][h]
            if c["sufficient"]:
                print(f"  {h:<16} {c['n']:>5} {c['n_positive']:>4} {c['n_genera']:>6} | " + " | ".join(f"{_a(c['models'][k]):^19}" for k in M))
            else:
                print(f"  {h:<16} {c['n']:>5} {c['n_positive']:>4} {c['n_genera']:>6} | too few genomes or too few of one class to score")
        if b["macro"]:
            print(f"  {'MACRO (' + str(b['macro']['n_phyla']) + ' phyla)':<16} {'':>5} {'':>4} {'':>6} | " + " | ".join(f"{_a(b['macro']['models'][k]):^19}" for k in M))
            dl = b["macro"]["deltas"]
            print("  Evo 2 minus: " + "; ".join(f"{k.split(' - ')[1]} {v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for k, v in dl.items()))

    print("\n  5. PAIRED DIFFERENCES, pooled leave-genus-out AUC (Evo 2 minus baseline; same genera resampled; 95% CI)")
    for pair in ("evo2 - gc", "evo2 - kmer_genome", "evo2 - kmer_windows", "evo2 - tax_prior"):
        cells = []
        for h in P:
            x = d["per_phylum"][h]["deltas"][pair]
            cells.append(f"{h[:6]} {x['delta']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]")
        x = d["macro"]["deltas"][pair]
        print(f"  {pair:<20} " + " | ".join(cells) + f" || MACRO {x['delta']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]")

    print("\n  6. SHUFFLE NULLS (mean ± sd of the null pooled AUC; p = P(null >= observed))")
    for mode, title in (("phylum", "labels permuted within each phylum"), ("order", "labels permuted within each ORDER (order-level prevalence kept: signal beyond the order?)")):
        if mode not in d["nulls"]:
            continue
        nl = d["nulls"][mode]
        print(f"  {title}  [{nl['n_perm']} permutations]")
        for k in M:
            if k not in nl["models"]:
                continue
            r = nl["models"][k]
            print(f"    {nm[k]:<15} " + " | ".join(f"{h[:6]} {r['phyla'][h]['mean']:.3f}±{r['phyla'][h]['sd']:.3f} p={r['phyla'][h]['p']:.3f}" for h in P) +
                  f" || MACRO {r['macro']['mean']:.3f}±{r['macro']['sd']:.3f} p={r['macro']['p']:.3f}")

    g = d["macro"]["deltas"]["evo2 - gc"]
    tp = d["macro"]["deltas"]["evo2 - tax_prior"]
    kg = d["macro"]["deltas"]["evo2 - kmer_genome"]
    say = lambda x, a, b: (f"{a} is ahead of {b}" if x["ci"][0] > 0 else f"{b} is ahead of {a}" if x["ci"][1] < 0 else f"no detectable difference between {a} and {b}")  # noqa: E731
    print("\n  VERDICT (macro over the four phyla, pooled leave-genus-out AUC)")
    print(f"    Evo 2 vs GC content:        {say(g, 'Evo 2', 'GC')} (Δ {g['delta']:+.3f} [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}])")
    print(f"    Evo 2 vs k-mer(genome):     {say(kg, 'Evo 2', 'k-mer(genome)')} (Δ {kg['delta']:+.3f} [{kg['ci'][0]:+.3f},{kg['ci'][1]:+.3f}])")
    print(f"    Evo 2 vs taxonomy prior:    {say(tp, 'Evo 2', 'the taxonomy prior')} (Δ {tp['delta']:+.3f} [{tp['ci'][0]:+.3f},{tp['ci'][1]:+.3f}])")
    ev = [b for b in d["bins"] if b["macro"]]
    if len(ev) > 1:
        print("    Evo 2 macro AUC from nearest to farthest bin: " + " -> ".join(f"{b['label']}: {b['macro']['models']['evo2']['auc']:.3f}" for b in ev))
        print("    Taxonomy prior, same bins:                    " + " -> ".join(f"{b['label']}: {b['macro']['models']['tax_prior']['auc']:.3f}" for b in ev))


NC_LABELS_ = {"family": "same family", "order": "same order", "class": "same class", "phylum": "phylum only"}
