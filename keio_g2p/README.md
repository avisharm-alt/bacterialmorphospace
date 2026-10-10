# keio_g2p: knockout → cell shape on unseen genes (E. coli Keio)

Tests whether a gene's protein sequence (ESM-2 650M) or annotation (EcoCyc GO) predicts the cell-shape scores of its
Keio deletion (Campos et al. 2018, 18 shape features), on genes held out in whole chromosomal blocks with homologs
purged from training.

```
python -m keio_g2p.modal_embed raw/uniprot_k12.tsv derived/esm2_650m.npz          # Modal T4, ~5 min
mmseqs easy-search k12.faa k12.faa hits.m8 tmp -e 1e-5 -c 0.5                      # homolog pairs (self hits removed)
python -m keio_g2p.evaluate raw/ derived/esm2_650m.npz results/ derived/k12_homolog_pairs.tsv
python -m keio_g2p.network raw/ derived/esm2_650m.npz results/ derived/k12_homolog_pairs.tsv  # STRING + fitness
python -m keio_g2p.perm_fitness raw/ derived/k12_homolog_pairs.tsv results/fitness_permutation.json
```

Data live in the project folder `data/ecoli_keio/` (sources in its raw/README.md).

**First result (2026-10-10).** Weak but real: shape-mutant AUROC 0.556 on held-out genes (permutation p = 0.01),
shape R² ≈ 0. GO helps slightly more than ESM-2; a random split inflates GO's R² about fivefold (operon/homolog
leakage). Full table: `data/ecoli_keio/results/heldout_gene_report.md`.

**Network inputs (2026-10-10).** STRING links raise shape-mutant AUROC to 0.617 (spectral ridge, no text mining or
curated databases, p = 0.01). RB-TnSeq fitness profiles are the first input with positive shape R² on held-out
genes (+0.028, p = 0.01, positive for all 18 features). Combining every input in one ridge is worse than either.
