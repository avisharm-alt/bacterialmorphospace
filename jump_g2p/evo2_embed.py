"""Evo 2 7B DNA embeddings of each JUMP gene's coding sequence, on Modal A100s.

    python -m jump_g2p.evo2_embed <cds.tsv> <out.npz>

Image recipe, model config and layer follow src/evo2_modal.py on the Evo 2 branch (no transformer-engine;
evo2_7b_base; mean-pooled `blocks.28.mlp.l3`). The input is the longest Ensembl CDS per gene, cut to 8,192 bp.
The merged weights are cached on the `evo2-hf-cache` volume.
"""
from __future__ import annotations

import sys

import modal

FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
MODEL_NAME, LAYER, WINDOW = "evo2_7b_base", "blocks.28.mlp.l3", 8192
HF_DIR = "/hf"
hf_vol = modal.Volume.from_name("evo2-hf-cache", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("build-essential", "curl")
         .pip_install("evo2==0.3.0", "vtx==1.1.0", extra_options="--no-deps")
         .pip_install("einops==0.8.1", "biopython", "huggingface_hub", "numpy", "pyyaml", "packaging", "rich",
                      "tqdm", "requests")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install(FLASH_ATTN_WHEEL, extra_options="--no-deps")
         .env({"HF_HOME": HF_DIR}))
app = modal.App("jump-evo2-cds")


@app.cls(image=image, gpu="A100-40GB", volumes={HF_DIR: hf_vol}, timeout=3600, max_containers=4)
class Embedder:
    @modal.enter()
    def load(self):
        from evo2 import Evo2
        self.model = Evo2(MODEL_NAME)

    @modal.method()
    def embed(self, seqs: list[str]) -> list[list[float]]:
        import torch
        out = []
        tok = self.model.tokenizer
        for s in seqs:
            ids = torch.tensor([tok.tokenize(s[:WINDOW])], dtype=torch.int).to("cuda:0")
            with torch.no_grad():
                _, emb = self.model(ids, return_embeddings=True, layer_names=[LAYER])
            x = emb[LAYER]
            if x.shape[-1] != 4096 and x.shape[1] == 4096:
                x = x.transpose(1, 2)
            out.append(x.float().mean(dim=1)[0].cpu().tolist())
        return out


def main(tsv: str, out: str) -> None:
    import numpy as np
    import pandas as pd
    d = pd.read_csv(tsv, sep="\t")
    seqs = d["cds"].tolist()
    chunks = [seqs[i:i + 200] for i in range(0, len(seqs), 200)]
    with modal.enable_output(), app.run():
        res = list(Embedder().embed.map(chunks))
    emb = np.asarray([v for r in res for v in r], dtype=np.float32)
    np.savez_compressed(out, symbol=d["symbol"].to_numpy(), emb=emb)
    print("saved", emb.shape, "to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
