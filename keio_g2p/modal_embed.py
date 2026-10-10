"""ESM-2 protein embeddings on Modal. Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.

    python -m keio_g2p.modal_embed <uniprot_tsv> <out.npz>

Mean-pools the final layer of esm2_t33_650M_UR50D over residues (sequences cut to 1,022 aa).
"""
from __future__ import annotations

import sys

import modal

MODEL = "facebook/esm2_t33_650M_UR50D"
app = modal.App("keio-esm2")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.5.1", "transformers==4.46.3", "numpy>=1.26")


@app.function(image=image, gpu="T4", timeout=3600)
def embed(seqs: list[str]) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).half().cuda().eval()
    out = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res: dict[int, list[float]] = {}
    batch: list[int] = []

    def flush():
        enc = tok([seqs[i][:1022] for i in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].clone()
        m[:, 0] = 0  # drop <cls>
        m[torch.arange(len(batch)), enc["attention_mask"].sum(1) - 1] = 0  # drop <eos>
        e = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        for j, i in enumerate(batch):
            res[i] = e[j].cpu().tolist()

    tokens = 0
    for i in order:
        L = min(len(seqs[i]), 1022) + 2
        if batch and (len(batch) + 1) * max(L, tokens) > 12000:
            flush(); batch.clear()
        batch.append(i); tokens = L
    if batch:
        flush()
    return [res[i] for i in range(len(seqs))]


def main(tsv: str, out: str) -> None:
    import numpy as np
    import pandas as pd
    d = pd.read_csv(tsv, sep="\t")
    with modal.enable_output(), app.run():
        emb = embed.remote(d["Sequence"].tolist())
    np.savez_compressed(out, accession=d["Entry"].to_numpy(), emb=np.asarray(emb, dtype=np.float32))
    print(f"saved {len(emb)} embeddings to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
