# Gene-conditioned Cell Painting image diffusion: held-out genes

Run 2026-10-11 on the avisharm-alt Modal workspace (5 folds × H100, 30k steps each, ~2 h per fold).
300 reproducible JUMP CRISPR knockouts, 5 folds with paralogs purged, 128 px 5-channel crops.
Each fold samples 128 crops per held-out gene four ways from one network: **cond** (its gene code),
**free** (code dropped), **shuffled** (another held-out gene's code), **guided2** (guidance w = 2).
Lower energy distance (ED) / mean error is better. p = one-sided Wilcoxon over the 300 held-out genes.

## Verdict

The model passes the bar on interpretable morphology and on direction of change, and half-passes on DINO:

| | CellProfiler-like features (42) | DINOv2 features |
|---|---|---|
| ED cond / free / shuffled | **2.28** / 2.75 / 2.99 | 1.57 / **1.55** / 1.64 |
| cond beats free | 67% of genes, p = 3e-11 | 45%, p = 0.97 (fails) |
| cond beats shuffled | 62%, p = 4e-10 | 63%, p = 4e-8 |
| mean error cond / free | **3.75** / 4.33 (p ≈ 0) | 3.76 / 3.75 |
| per-gene shift cosine, cond / free / shuffled | **0.47** / 0.01 / 0.04 | **0.19** / 0.01 / 0.01 |
| median per-feature r (real vs generated shift) | **0.59** | 0.10 |

- The gene code moves generated cells the right way for unseen genes: across held-out genes, the real
  shift from the fold mean and the generated shift correlate at a median r = 0.59 per feature
  (cell count, per-channel intensity medians all ≈ 0.63); the no-gene and shuffled versions are ≈ 0.
- In DINO space the conditioned model beats the no-gene version in folds 0–2 (by ~0.06 each), ties fold 3
  and loses badly in fold 4 (1.76 vs 1.50, only 7% of its genes better). Without fold 4 it wins 54% of genes,
  p = 0.01. Fold 4's conditioned samples look less realistic in DINO's view; not yet diagnosed.
- Guidance w = 2 does not help (worse ED than w = 1 in both spaces).
- Realism gap: real crops of *other* held-out genes (a perfectly realistic generator that ignores the gene)
  still score better than any generated set (CP ED 2.01, DINO 0.65). The model gets the direction right
  but its pixels are not yet as close to real as real cells are.
- Profile-level model on the same 300 genes, for reference: ED 7.37 cond vs 8.63 free (p = 2e-17), 66% of genes.

## Figure

`examples.png`: held-out genes where conditioning helped most. Proteasome knockouts (PSMD14, PSMB4, PSMA1)
come out sparse and dying, like the real wells, while the gene-free and other-gene samples are dense healthy
U2OS monolayers.

Files: `summary.json`, `per_gene.csv`, `examples.png`; fold outputs in `../folds/`.

## Fold 4 is a data problem (PLK1)

PLK1, the JUMP positive control, has 569 wells in the image subset (other genes 4-12) and its knockout leaves crops
mostly empty. It is 37% of fold 4's training wells, so 27% of that fold's training crops are near-empty (~5% in the
other folds); the gene-free model and the DINO PCA (fitted on training crops) both inherit that skew. In fold 3 PLK1 is
a test gene and dominates the real "other genes" reference. Fix: cap wells per gene (e.g. 12) and retrain.

## Sequence-code run (stopped)

A run with the chained sequence → predicted DepMap codes (`select --codes=chain`, `run train --run=chain`, same
genes, folds and manifest) was stopped at step 16k of 30k on 2026-10-11 when the Modal budget ran out. Checkpoints:
Modal volume `jump-image-crops` at ckpt/chain/fold<k>.pt, copies in data/jump/images_diffusion_chain/ckpt/chain/.
Training resumes from them. Not yet sampled or scored.
