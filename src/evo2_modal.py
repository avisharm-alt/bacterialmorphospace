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
UA = "bacterialmorphospace/0.1 (research; genome download)"


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------
@app.function(image=gpu_image, gpu=GPU, timeout=900)
def verify_env() -> dict:
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
        "torch": torch.__version__, "cuda": torch.version.cuda, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "flash_attn": flash_attn.__version__, "gpu": torch.cuda.get_device_name(0),
        "transformer_engine_dists": te, "HAS_TE": layers.HAS_TE,
        "evo2": md.version("evo2"), "vtx": md.version("vtx"),
    }


@app.function(image=cpu_image, volumes={HF_DIR: hf_vol}, timeout=3600, cpu=2, memory=8192)
def prime_weights() -> dict:
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
        t0 = time.perf_counter()
        from evo2 import Evo2

        self.model = Evo2(core.MODEL_NAME)
        self.load_s = time.perf_counter() - t0
        self.task_id = os.environ.get("MODAL_TASK_ID", "local")

    def _embed_windows(self, seq: str, starts: list[int], batch_size: int):
        import numpy as np
        import torch

        tok = self.model.tokenizer
        vecs, times = [], []
        for i in range(0, len(starts), batch_size):
            chunk = starts[i:i + batch_size]
            ids = torch.tensor([tok.tokenize(seq[s:s + core.WINDOW]) for s in chunk], dtype=torch.int).to("cuda:0")
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
        import json

        import numpy as np

        t_call = time.perf_counter()
        base = {"acc": acc, "task_id": self.task_id, "load_s": self.load_s}
        try:
            os.makedirs(EMB_DIR, exist_ok=True)
            pool_n = max(depths)
            npy = {n: f"{EMB_DIR}/{acc}__n{n}.npy" for n in depths}
            win_path, meta_path = f"{EMB_DIR}/{acc}__windows.npy", f"{EMB_DIR}/{acc}__meta.json"
            missing = [n for n in depths if not os.path.exists(npy[n])]
            if not missing:
                return {**base, "ok": True, "status": "skipped", "gpu_seconds": 0.0}

            status = "embedded"
            if os.path.exists(win_path) and os.path.exists(meta_path) and np.load(win_path, mmap_mode="r").shape[0] == pool_n:
                pool = np.load(win_path)  # a pool from an earlier run: derive the missing depths for free
                meta = json.loads(open(meta_path).read())
                status = "derived"
            else:
                _, seq = core.longest_contig(f"{GENOME_DIR}/{acc}.fna.gz")
                starts = core.window_starts(len(seq), pool_n)
                pool, w_times = self._embed_windows(seq, starts, batch_size)
                non_acgt = [sum(c not in "ACGT" for c in seq[s:s + core.WINDOW]) / core.WINDOW for s in starts]
                meta = {"acc": acc, "contig_len": len(seq), "pool_n": pool_n, "starts": starts,
                        "window_times": w_times, "max_non_acgt_frac": max(non_acgt), "batch_size": batch_size}
                if save_windows:
                    _save_npy(win_path, pool.astype(np.float32))
                    with open(meta_path, "w") as f:
                        json.dump(meta, f)
            for n in missing:
                idx = core.subset_indices(pool_n, n)
                _save_npy(npy[n], pool[idx].mean(axis=0).astype(np.float32))
            out_vol.commit()
            return {**base, "ok": True, "status": status, "gpu_seconds": time.perf_counter() - t_call,
                    "contig_len": meta["contig_len"], "max_non_acgt_frac": meta["max_non_acgt_frac"]}
        except Exception as e:  # noqa: BLE001  deterministic failures are reported, not retried
            return {**base, "ok": False, "error": f"{type(e).__name__}: {e}", "gpu_seconds": time.perf_counter() - t_call}


@app.function(image=cpu_image, volumes={OUT_DIR: out_vol}, timeout=1800, cpu=2, memory=8192)
def evaluate_sweep(rows: list[dict], depths: list[int], n_splits: int) -> dict:
    """Per-depth GroupKFold AUC on `motility`, plus timings from the per-genome meta files."""
    import json

    import numpy as np

    out_vol.reload()
    y = np.array([r["motility"] == "yes" for r in rows])
    groups = np.array([r["group"] for r in rows])
    bad, results, timing = [], [], {}
    metas = {r["acc"]: json.load(open(f"{EMB_DIR}/{r['acc']}__meta.json")) for r in rows}
    for n in depths:
        X = np.stack([np.load(f"{EMB_DIR}/{r['acc']}__n{n}.npy") for r in rows])
        if X.shape[1] != core.EMB_DIM or not np.isfinite(X).all():
            bad.append(n)
        res = core.evaluate_depth(X, y, groups, n_splits)
        secs = [core.depth_seconds(m["window_times"], m["pool_n"], n) for m in metas.values()]
        s_gen = float(np.mean(secs))
        # windows must overlap when the longest contig is shorter than n non-overlapping windows
        overlap = float(np.mean([m["contig_len"] < n * core.WINDOW for m in metas.values()]))
        results.append({"depth": n, "n_genomes": len(rows), **res, "frac_overlapping": overlap, "s_per_genome": s_gen,
                        "hours_2580": core.project_hours(s_gen), "usd_2580": core.usd(s_gen * core.N_PANEL)})
    ov = [m["max_non_acgt_frac"] for m in metas.values()]
    return {"results": results, "invalid_depths": bad, "n_positive": int(y.sum()), "n_groups": int(len(set(groups))),
            "max_non_acgt_frac": float(max(ov)), "mean_contig_len": float(np.mean([m["contig_len"] for m in metas.values()]))}


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


def _run_embedding(accs: list[str], depths: list[int], batch_size: int, max_usd: float, save_windows: bool):
    """Stream GPU results, print running spend, stop cleanly if the budget is hit. Returns (failures, spent_usd)."""
    gpu_s, loads, failures, done = 0.0, {}, [], 0
    t0 = time.time()
    results = Embedder().embed.map(accs, order_outputs=False, return_exceptions=True,
                                   kwargs={"depths": depths, "batch_size": batch_size, "save_windows": save_windows})
    for res in results:
        done += 1
        if isinstance(res, Exception):
            failures.append({"acc": "?", "error": f"{type(res).__name__}: {res}"})
            continue
        gpu_s += res.get("gpu_seconds", 0.0)
        loads[res["task_id"]] = res["load_s"]
        if not res["ok"]:
            failures.append(res)
        spent = core.usd(gpu_s + sum(loads.values()))
        if done % 10 == 0 or done == len(accs):
            print(f"  [{done}/{len(accs)}] {time.time() - t0:6.0f}s wall | GPU spend ~${spent:5.2f} "
                  f"({len(loads)} containers) | failures {len(failures)}", flush=True)
        if spent > max_usd:
            print(f"\nBUDGET STOP: estimated GPU spend ${spent:.2f} exceeded --max-usd {max_usd}. "
                  "Checkpoints are kept; re-run the same command to resume.")
            raise SystemExit(2)
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
    for f in fails:
        print("EMBED FAILED:", f.get("acc"), f.get("error"))

    failed = {f.get("acc") for f in fails} | {b["acc"] for b in bad}
    final = [{"acc": r["ncbi_assembly_accession"], "motility": r["motility"], "group": r["group"]}
             for r in sample if r["ncbi_assembly_accession"] not in failed]
    print(f"\nevaluating {len(final)} genomes...")
    ev = evaluate_sweep.remote(final, ds, n_splits)
    if ev["invalid_depths"]:
        raise SystemExit(f"invalid embeddings at depths {ev['invalid_depths']}")

    sel = core.select_depth(ev["results"])
    cols = ["depth", "n_genomes", "auc_mean", "auc_std", "folds_scored", "folds_total", "frac_overlapping",
            "s_per_genome", "hours_2580", "usd_2580"]
    _write_tsv(f"{reports}/tables/evo2_sampling_depth_sweep.tsv", ev["results"], cols)
    _write_tsv(f"{reports}/tables/evo2_sweep_genomes.tsv", final, ["acc", "group", "motility"])
    core.dump_json({**sel, "results": ev["results"], "n_genomes": len(final), "seed": seed, "n_splits": n_splits},
                   SELECTED_DEPTH_JSON.replace("reports", reports, 1))

    print(f"\n{'depth':>6} {'AUC':>15} {'folds':>6} {'overlap':>8} {'s/genome':>9} {'h for 2,580':>12} {'$ for 2,580':>12}")
    for r in ev["results"]:
        print(f"{r['depth']:>6} {r['auc_mean']:>8.3f} ± {r['auc_std']:.3f} {r['folds_scored']:>3}/{r['folds_total']} "
              f"{r['frac_overlapping']:>7.0%} {r['s_per_genome']:>9.1f} {r['hours_2580']:>12.1f} {r['usd_2580']:>12.0f}")
    print("overlap = share of genomes whose longest contig is shorter than n non-overlapping windows")
    print(f"\nselected depth (one-SE rule): {sel['selected_depth']}  (best mean AUC at {sel['best_depth']}, SE {sel['one_se']:.3f})")
    print(f"sweep GPU spend ~${spent:.2f} this invocation (Modal dashboard is authoritative)")


@app.local_entrypoint()
def embed_all(panel: str = PANEL, n_windows: int = 0, max_usd: float = 25.0, batch_size: int = 1,
              limit: int = 0, reports: str = "reports", dry_run: bool = False, skip_checks: bool = False):
    """Production run over the whole panel at one depth. Resumable: existing checkpoints are skipped."""
    if n_windows <= 0:
        import json

        p = SELECTED_DEPTH_JSON.replace("reports", reports, 1)
        if not Path(p).exists():
            raise SystemExit(f"pass --n-windows, or run `sweep` first to produce {p}")
        n_windows = json.load(open(p))["selected_depth"]
        print(f"using sweep-selected depth {n_windows}")
    rows = _read_panel(panel)
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
