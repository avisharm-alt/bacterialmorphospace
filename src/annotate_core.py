"""Pfam presence/absence annotation of a genome: pyrodigal gene calls, pyhmmer against Pfam-A with the curated
gathering (GA) cutoffs (the Traitar-style annotation baseline).

Pure logic, no Modal. `annotate_fasta` takes contigs (name, uppercase DNA) and an open-able HMM path and returns the
set of Pfam accessions present, plus counts for cost accounting.
"""
from __future__ import annotations

import time

MIN_TRAIN_BP = 100_000  # pyrodigal single-genome training needs enough sequence; below this use metagenomic mode


def call_proteins(contigs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """[(protein id, amino-acid sequence without the stop)] for every gene called on every contig."""
    import pyrodigal

    total = sum(len(s) for _, s in contigs)
    if total >= MIN_TRAIN_BP:
        finder = pyrodigal.GeneFinder(meta=False)
        finder.train(*[s.encode() for _, s in contigs])  # train on the whole genome, all contigs
    else:
        finder = pyrodigal.GeneFinder(meta=True)
    out = []
    for name, seq in contigs:
        for i, g in enumerate(finder.find_genes(seq.encode())):
            aa = g.translate().rstrip("*")
            if aa:
                out.append((f"{name}_{i + 1}", aa))
    return out


def search_pfam(proteins: list[tuple[str, str]], hmm_path: str, cpus: int) -> set[str]:
    """Pfam accessions (versionless, e.g. PF00069) with at least one hit above their gathering cutoff."""
    import pyhmmer
    from pyhmmer import easel, plan7

    if not proteins:
        return set()
    abc = easel.Alphabet.amino()
    block = easel.DigitalSequenceBlock(abc, [easel.TextSequence(name=n.encode(), sequence=s).digitize(abc)
                                             for n, s in proteins])
    fams = set()
    with plan7.HMMFile(hmm_path) as hf:
        for top in pyhmmer.hmmsearch(hf, block, cpus=cpus, bit_cutoffs="gathering"):
            if len(top):
                acc = top.query.accession
                acc = acc.decode() if isinstance(acc, bytes) else acc
                fams.add(acc.split(".")[0])
    return fams


def annotate_contigs(contigs: list[tuple[str, str]], hmm_path: str, cpus: int) -> dict:
    """Timed annotation of one genome (or one set of windows). Everything returned is a builtin type."""
    t0 = time.time()
    prots = call_proteins(contigs)
    t1 = time.time()
    fams = search_pfam(prots, hmm_path, cpus)
    t2 = time.time()
    return {"families": sorted(fams), "n_proteins": len(prots), "bp": sum(len(s) for _, s in contigs),
            "gene_call_s": t1 - t0, "hmmsearch_s": t2 - t1}
