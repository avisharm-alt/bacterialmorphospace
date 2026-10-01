# Phase 2 spec (DRAFT for approval): Pfam presence/absence as the fifth LOPO model

Status: proposal only. No analysis code has been written or run on labels. The two-stage filter and the
matched-window control were not defined anywhere in the repo, so the definitions below are **my proposal** and every
item marked **[D#]** is a decision for you. If your intended definition differs, say so and I will follow yours.

## 1. Question

Under leave-one-phylum-out (LOPO), scored as AUC *within* the held-out phylum, does the Evo 2 depth-10 embedding predict
motility better than a standard annotation baseline (Pfam presence/absence), and is any gap due to how much sequence each
model sees or to what it extracts from it?

## 2. What is already in hand

- Whole-genome Pfam-A annotation (pyrodigal + pyhmmer, gathering cutoffs) for **2,435 of 2,436** four-phylum genomes, on the
  `evo2-embeddings` volume under `annotations/<acc>.json`. GCF_001583695.1 is not downloadable from NCBI (404 on both paths).
- Binary matrix 2,435 x 16,989 families (of 30,134 models), density 15.4%.
- Motility labels: 1,060 yes / 1,375 no.
- Evo 2 embeddings: **in progress** (GPU `embed_all`, depth 10). The LOPO and the matched-window control cannot run before
  they exist: `lopo_run` needs an embedding and window features for >= 98% of genomes, and the windows come from each
  genome's `embeddings/<acc>__meta.json`.

## 3. Models (all on identical genomes, same CV, same bootstrap, same null)

Existing, unchanged: `evo2`, `kmer_genome`, `kmer_windows`, `gc`, and the constant floor (within-phylum AUC 0.5).

New:

| name | features | role |
|---|---|---|
| `pfam` | whole-genome Pfam presence/absence | the fifth model (annotation baseline) |
| `pfam_windows` | Pfam families found only in the exact 10 x 8,192 bp windows Evo 2 saw | matched-window control |

`pfam` vs `kmer_genome` and `pfam_windows` vs `kmer_windows` pair like with like (whole genome vs windows).

## 4. Two-stage Pfam filter (proposal)

Both stages are fitted on **training rows only**, inside every outer fold and every inner tuning fold. The held-out phylum
and its labels never reach the filter. (A filter fitted once on all genomes would leak labels into the held-out score.)

- **Stage 1, label-free.** Keep families with prevalence in [1%, 99%] among the training genomes. Measured on the real
  matrix: 10,273 of 16,989 kept on all genomes; with training phyla only, 8,925 to 9,895 kept depending on the held-out
  phylum. **[D1]** threshold (1% proposed; 0.5% keeps 11,532, 2% keeps 8,684, 5% keeps 6,313).
- **Stage 2, supervised.** Rank the survivors by association with the trait **stratified by training phylum** (a
  Cochran-Mantel-Haenszel-type statistic, or the mean of within-phylum correlations), so it selects within-phylum signal,
  not phylum identity. Keep the top K. **[D2]** statistic. **[D3]** K grid; proposed {100, 300, 1000, all stage-1
  survivors}, tuned by the same inner leave-one-phylum-out as C (joint K x C grid, ties to the smaller K, then smaller C).
- Then `StandardScaler` + L2 logistic regression, as for every other model, with the intercept-free decision score.

Why two stages: stage 1 removes rare and near-ubiquitous families cheaply without labels; stage 2 stops a 10,000-wide model
being compared with 136-wide composition models on regularisation alone. **[D4]** confirm the filter lives inside CV
(recommended; the alternative leaks labels).

## 5. Matched-window control (proposal)

- Windows: the saved `[contig id, start]` pairs in each genome's embedding meta, 8,192 bp each (10 per genome at depth 10,
  about 82 kb, roughly 2% of a median 4.1 Mb genome).
- Annotation: pyrodigal on those windows (total length is below the 100 kb training minimum, so `annotate_contigs` already
  uses metagenomic mode), then pyhmmer against the same Pfam-A with the same gathering cutoffs. Presence = at least one
  hit above cutoff. Genomes are batched per container so Pfam-A is not reloaded for every genome.
- Same two-stage filter and grid as `pfam`, for comparability. **[D5]** confirm, or use stage 1 only given how few
  families will be present.
- Expectation (unmeasured): tens of families per genome, since windows cut genes at their edges and cover about 2% of the
  genome. If `pfam_windows` is near 0.5, the control shows that equal sequence is not enough for a Pfam model; if it is
  well above 0.5, Evo 2 is being compared against a model that sees the same bases.
- Compute: CPU only, a small fraction of the whole-genome annotation. I will time a batch first and report it.

## 6. Comparisons and outputs

Paired genus-cluster bootstrap (1,000 resamples), within-phylum shuffle null (50 permutations), per held-out phylum and
macro, identical to the existing `lopo` output. Comparison pairs: `evo2 - pfam`, `evo2 - pfam_windows`,
`pfam - kmer_genome`, `pfam_windows - kmer_windows`. Verdict wording reuses the existing rule (macro delta CI vs 0).
**[D6]** primary target is motility only (pre-specified); the other three targets (oxygen_aerobe, oxygen_facultative,
shape_rod) run only if you ask, which adds multiplicity (4 targets x 4 comparisons).

## 7. Implementation plan (after approval)

1. `core.lopo_evaluate`: optional per-model selector applied to training rows only in the inner and outer fits. Unit tests
   on synthetic data: the held-out rows never reach the selector; K is chosen by the inner folds; with no selector the
   output is bit-identical to today.
2. `_lopo_run`: load the annotation matrix aligned to the same genomes, add `pfam` and `pfam_windows`; existing models
   untouched. Regression check: the existing four models' numbers must reproduce exactly on the same embeddings.
3. `annotate_windows` Modal function and entrypoint (CPU), writing `annotations_windows/<acc>.json`.
4. Time one target end to end before any repeat; a 10,000-wide model with an inner K x C grid is heavier than the current
   136- and 4,096-wide ones.

## 8. Not changing

No tuning on held-out labels, no change to existing models or their results, no new GPU work beyond the embeddings.
