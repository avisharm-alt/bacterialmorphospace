# Phase 2 spec: Pfam presence/absence as the fifth LOPO model

Status: **APPROVED decisions D1-D6 (below); one item still open (section 7, the confirmatory contrast).** This file is
committed before any Pfam model is fitted to labels, so the pre-specification below is timestamped by git history.

## 1. Question

Under leave-one-phylum-out (LOPO), scored as AUC *within* the held-out phylum, does the Evo 2 depth-10 embedding predict
motility better than a standard annotation baseline (Pfam presence/absence), and is any gap due to how much sequence each
model sees or to what it extracts from it? The prior result for Evo 2 against composition baselines is a negative; the
annotation arm is run across all four traits so that replication, not a single trait, carries the claim.

## 2. Data in hand

- Whole-genome Pfam-A annotation (pyrodigal + pyhmmer, gathering cutoffs) for **2,435 of 2,436** four-phylum genomes
  (`annotations/<acc>.json` on the `evo2-embeddings` volume). GCF_001583695.1 is a 404 on NCBI (both paths) and is absent
  from every arm. Binary matrix 2,435 x 16,989 families, density 15.4%.
- Evo 2 depth-10 embeddings (2,435 x 4096) and per-genome meta (10 windows of 8,192 bp each) on the same volume.
- Labels: motility 1,060 yes / 1,375 no.

## 3. Models (identical genomes, CV, bootstrap and null)

Existing, unchanged: `evo2`, `kmer_genome`, `kmer_windows`, `gc`, constant floor (within-phylum AUC 0.5).
New: `pfam` (whole-genome presence/absence) and `pfam_windows` (families found only in the exact 10 x 8,192 bp windows
Evo 2 saw: the matched-window control). Pairings: `pfam` vs `kmer_genome`; `pfam_windows` vs `kmer_windows`;
each against `evo2`.

## 4. Two-stage Pfam filter (decisions D1-D5, all APPROVED)

Both stages run **inside every fold**: the outer fold's training phyla, and again inside each inner tuning fold on that
fold's own training rows. The held-out phylum never reaches either stage. In the shuffle null the selection is refit on the
permuted training labels (a selection fitted on the real labels would leak signal into the null).

- **Stage 1, label-free prevalence filter (D1).** Keep families present in at least 0.5% and at most 99.5% of the
  training genomes (about 12 genomes of 2,435; permissive on purpose so mechanistically relevant rare families are not
  dropped). The threshold is reported as a **sensitivity check at the end** (1% and 5%, primary target), not as a finding.
- **Stage 2, supervised ranking (D2).** Cochran-Mantel-Haenszel over the training phyla: for each family, per-stratum 2x2
  table (present/absent x positive/negative), statistic z = sum_s (a_s - E_s) / sqrt(sum_s Var_s), ranked by |z|, so it
  selects within-phylum association and not phylum identity. Strata that are single-class or have one genome contribute
  nothing; families with zero variance rank last.
- **K is a hyperparameter, not a result (D3).** Top-K with K in {100, 300, 1000, all stage-1 survivors}, tuned jointly with
  C by the inner leave-one-phylum-out over the training phyla (mean within-phylum AUC; ties go to the smaller C, then the
  smaller K), exactly as C is tuned today. The K = 100/300/1000/all values are never reported as separate results. The
  **chosen K, C and the stage-1 survivor count are reported per held-out phylum** so stability is visible.
- **Both stages in-fold (D4)** (prevalence is cheap to compute in-fold, so the whole filter is described in one sentence
  in a methods section).
- **`pfam_windows` gets the identical pipeline (D5)**, with K tuned separately per model (each fitted for its own feature
  count; a K above the number of survivors is the same as "all").
- After selection: `StandardScaler` + L2 logistic regression with the intercept-free decision score, as for every model.

## 5. Matched-window control (annotation of the windows)

- Windows: the saved `[contig id, start]` pairs in each genome's embedding meta, 8,192 bp each (about 82 kb, roughly 2%
  of a median 4.1 Mb genome).
- pyrodigal on each genome's windows (82 kb total is below the 100 kb training minimum, so metagenomic mode is used, as
  `annotate_contigs` already does), then one pyhmmer pass per batch of genomes against the same Pfam-A with gathering
  cutoffs, so Pfam-A is read once per batch and not once per genome. Presence = at least one hit above cutoff.
- CPU only; a 100-genome canary was run and timed before the full set. **Measured (label-free, 2,435 genomes, 0 errors,
  about 3.9 CPU-h in total):** every genome has exactly 81,920 windowed bp; median 83 proteins (56-155) and **121 Pfam
  families per genome (64-188)**; the earlier expectation of "tens" was wrong. A genome's window families are in its
  whole-genome set at a median 100% (minimum 95.6%), only 6 families occur in windows alone, and the windows recover a
  median 4.6% (2.2-10.5%) of the genome's families. That last figure is the coverage gap this control isolates.

## 6. Targets and multiplicity (D6, APPROVED)

All four targets: **motility is the primary target**; `shape_rod`, `oxygen_aerobe` and `oxygen_facultative` are
**secondary, Benjamini-Hochberg corrected across the three secondary targets**. Nothing is dropped. The primary/secondary
split and the correction are stated in the output itself. Replication across traits is the point, so multiplicity only
matters if annotation is positive on one trait and that is claimed.

## 7. Open: the confirmatory contrast (must be fixed before the first Pfam model is fitted)

D6 fixes the correction but not the single quantity the p-values come from. Needed: which contrast defines "annotation
comes out positive" for the primary test and for the BH family. See the question put to the project owner; the answer is
recorded here before any Pfam result exists.

## 8. Implementation order

1. Selector hook in `core.lopo_evaluate` (training rows only, in inner, outer and null fits) with leakage tests; with no
   selector the output must be bit-identical to today (regression-checked on the same embeddings).
2. `annotate_windows` Modal function and entrypoint (CPU), grouped search, resumable, canary first.
3. `_lopo_run` / `lopo` entrypoint: add `pfam` and `pfam_windows`, report chosen K per fold, print the pre-specification.
4. Time one target end to end, then run motility (primary), then the three secondary targets, then the threshold
   sensitivity check.

## 9. Not changing

No tuning on held-out labels, no change to existing models or their results, no new GPU work. `nearclade` is not extended.
