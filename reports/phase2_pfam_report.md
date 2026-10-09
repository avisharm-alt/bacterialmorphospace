# Phase 2: Pfam presence/absence in leave-one-phylum-out, results

_Generated 2026-10-09 from the committed result JSONs; every table cell is read from them._

**Status: descriptive. The contrast that governs is pending Avin's decision.** Two candidate definitions of the confirmatory
contrast are recorded in `docs/phase2_pfam_spec.md` (section 7), and section 6 below shows how the results read under each. Nothing in
this report is called confirmatory, and none of it is a headline result.

## 1. What was run

Pfam-A presence/absence (pyrodigal + pyhmmer, gathering cutoffs; 16,995 families) as the fifth model in the leave-one-phylum-out
evaluation, next to Evo 2 (depth 10) and the composition baselines, on the same 2435 genomes of four phyla. Each model is trained on
three phyla and scored by AUC inside the held-out one; the headline number is the equal-weight mean of those four AUCs ("macro").
The two-stage Pfam filter (prevalence 0.5% to 99.5%, then Cochran-Mantel-Haenszel top-K over the training phyla) runs inside every
fold and again on the permuted labels in the shuffle null. `pfam_windows` is the same pipeline on Pfam families found only in the
10 x 8,192 bp windows Evo 2 saw.

- 999 within-phylum label shuffles per run, 1,000 genus-cluster bootstrap resamples, seed 20260929, CPU only.
- The four runs were made in a separate session; the JSONs were copied from `origin/claude/optimistic-newton-6lf55q` at `5fa0250`
  (`reports/tables/evo2_lopo_<target>_pfam.json`, `pfam_threshold_sensitivity.tsv`). Wall time of the null stage per run, in hours
  (motility, `shape_rod`, `oxygen_aerobe`, `oxygen_facultative`): 1.2 0.8 1.0 0.3.
- Intervals are 95% genus-cluster bootstrap intervals (genomes of one genus are near-copies); differences are paired on the same resamples.

## 2. Macro AUC, all models and targets

| target | Pfam | Pfam (Evo 2 windows) | Evo 2 | k-mer (whole genome) | k-mer (Evo 2 windows) | GC only |
|---|---|---|---|---|---|---|
| motility | 0.752 [0.711, 0.791] | 0.590 [0.556, 0.622] | 0.545 [0.502, 0.582] | 0.524 [0.483, 0.566] | 0.543 [0.499, 0.582] | 0.585 [0.536, 0.629] |
| shape_rod | 0.617 [0.559, 0.671] | 0.516 [0.459, 0.570] | 0.476 [0.419, 0.538] | 0.519 [0.448, 0.582] | 0.502 [0.441, 0.560] | 0.492 [0.431, 0.559] |
| oxygen_aerobe | 0.891 [0.861, 0.913] | 0.764 [0.731, 0.792] | 0.753 [0.711, 0.785] | 0.447 [0.410, 0.487] | 0.473 [0.435, 0.512] | 0.590 [0.552, 0.627] |
| oxygen_facultative | 0.597 [0.548, 0.659] | 0.520 [0.481, 0.561] | 0.507 [0.434, 0.561] | 0.641 [0.586, 0.692] | 0.622 [0.556, 0.675] | 0.563 [0.503, 0.610] |

## 3. By held-out phylum

**motility**

| model | Actinomycetota (n = 481, 9% positive) | Bacillota (n = 732, 57% positive) | Bacteroidota (n = 409, 18% positive) | Pseudomonadota (n = 813, 64% positive) | macro mean |
|---|---|---|---|---|---|
| Pfam | 0.748 [0.637, 0.858] | 0.824 [0.768, 0.867] | 0.601 [0.494, 0.701] | 0.836 [0.789, 0.875] | 0.752 [0.711, 0.791] |
| Pfam (Evo 2 windows) | 0.593 [0.487, 0.692] | 0.618 [0.579, 0.660] | 0.548 [0.476, 0.620] | 0.603 [0.558, 0.646] | 0.590 [0.556, 0.622] |
| Evo 2 | 0.407 [0.299, 0.511] | 0.675 [0.620, 0.727] | 0.550 [0.464, 0.640] | 0.546 [0.484, 0.615] | 0.545 [0.502, 0.582] |
| k-mer (whole genome) | 0.666 [0.570, 0.758] | 0.364 [0.300, 0.434] | 0.577 [0.490, 0.667] | 0.491 [0.430, 0.558] | 0.524 [0.483, 0.566] |
| k-mer (Evo 2 windows) | 0.660 [0.572, 0.747] | 0.441 [0.362, 0.516] | 0.576 [0.476, 0.670] | 0.495 [0.432, 0.561] | 0.543 [0.499, 0.582] |
| GC only | 0.666 [0.551, 0.774] | 0.513 [0.442, 0.578] | 0.610 [0.509, 0.701] | 0.552 [0.485, 0.614] | 0.585 [0.536, 0.629] |

**shape_rod**

| model | Actinomycetota (n = 481, 74% positive) | Bacillota (n = 732, 85% positive) | Bacteroidota (n = 409, 98% positive) | Pseudomonadota (n = 813, 91% positive) | macro mean |
|---|---|---|---|---|---|
| Pfam | 0.515 [0.433, 0.602] | 0.707 [0.629, 0.782] | 0.574 [0.388, 0.754] | 0.672 [0.586, 0.735] | 0.617 [0.559, 0.671] |
| Pfam (Evo 2 windows) | 0.523 [0.456, 0.587] | 0.537 [0.482, 0.588] | 0.454 [0.248, 0.646] | 0.548 [0.485, 0.613] | 0.516 [0.459, 0.570] |
| Evo 2 | 0.566 [0.488, 0.645] | 0.445 [0.353, 0.559] | 0.390 [0.201, 0.599] | 0.502 [0.414, 0.585] | 0.476 [0.419, 0.538] |
| k-mer (whole genome) | 0.530 [0.457, 0.611] | 0.563 [0.410, 0.697] | 0.481 [0.290, 0.663] | 0.504 [0.407, 0.584] | 0.519 [0.448, 0.582] |
| k-mer (Evo 2 windows) | 0.522 [0.445, 0.603] | 0.507 [0.429, 0.582] | 0.475 [0.273, 0.670] | 0.503 [0.406, 0.586] | 0.502 [0.441, 0.560] |
| GC only | 0.449 [0.375, 0.527] | 0.481 [0.408, 0.558] | 0.622 [0.394, 0.857] | 0.416 [0.331, 0.526] | 0.492 [0.431, 0.559] |

**oxygen_aerobe**

| model | Actinomycetota (n = 481, 85% positive) | Bacillota (n = 732, 45% positive) | Bacteroidota (n = 409, 79% positive) | Pseudomonadota (n = 813, 84% positive) | macro mean |
|---|---|---|---|---|---|
| Pfam | 0.938 [0.893, 0.968] | 0.907 [0.873, 0.936] | 0.913 [0.810, 0.968] | 0.806 [0.749, 0.850] | 0.891 [0.861, 0.913] |
| Pfam (Evo 2 windows) | 0.853 [0.782, 0.907] | 0.741 [0.703, 0.782] | 0.797 [0.707, 0.856] | 0.664 [0.607, 0.717] | 0.764 [0.731, 0.792] |
| Evo 2 | 0.869 [0.797, 0.920] | 0.712 [0.662, 0.762] | 0.839 [0.716, 0.914] | 0.594 [0.523, 0.660] | 0.753 [0.711, 0.785] |
| k-mer (whole genome) | 0.526 [0.425, 0.624] | 0.353 [0.291, 0.415] | 0.316 [0.233, 0.415] | 0.594 [0.525, 0.657] | 0.447 [0.410, 0.487] |
| k-mer (Evo 2 windows) | 0.625 [0.524, 0.721] | 0.362 [0.301, 0.425] | 0.320 [0.234, 0.422] | 0.584 [0.517, 0.648] | 0.473 [0.435, 0.512] |
| GC only | 0.823 [0.747, 0.882] | 0.574 [0.505, 0.635] | 0.313 [0.231, 0.430] | 0.650 [0.583, 0.714] | 0.590 [0.552, 0.627] |

**oxygen_facultative**

| model | Actinomycetota (n = 481, 8% positive) | Bacillota (n = 732, 30% positive) | Bacteroidota (n = 409, 5% positive) | Pseudomonadota (n = 813, 14% positive) | macro mean |
|---|---|---|---|---|---|
| Pfam | 0.489 [0.388, 0.649] | 0.593 [0.527, 0.654] | 0.518 [0.398, 0.675] | 0.788 [0.724, 0.841] | 0.597 [0.548, 0.659] |
| Pfam (Evo 2 windows) | 0.597 [0.511, 0.699] | 0.518 [0.466, 0.574] | 0.473 [0.374, 0.567] | 0.490 [0.427, 0.551] | 0.520 [0.481, 0.561] |
| Evo 2 | 0.628 [0.368, 0.778] | 0.488 [0.418, 0.554] | 0.283 [0.197, 0.394] | 0.629 [0.549, 0.695] | 0.507 [0.434, 0.561] |
| k-mer (whole genome) | 0.577 [0.415, 0.673] | 0.748 [0.669, 0.811] | 0.593 [0.468, 0.751] | 0.644 [0.563, 0.713] | 0.641 [0.586, 0.692] |
| k-mer (Evo 2 windows) | 0.586 [0.388, 0.705] | 0.726 [0.647, 0.790] | 0.548 [0.420, 0.699] | 0.629 [0.552, 0.698] | 0.622 [0.556, 0.675] |
| GC only | 0.680 [0.507, 0.779] | 0.467 [0.393, 0.543] | 0.439 [0.323, 0.574] | 0.668 [0.594, 0.731] | 0.563 [0.503, 0.610] |

## 4. The four pairings (macro, paired difference in AUC)

| target | Pfam minus k-mer (whole genome) | Pfam (windows) minus k-mer (windows) | Evo 2 minus Pfam | Evo 2 minus Pfam (windows) |
|---|---|---|---|---|
| motility | +0.228 [+0.176, +0.278] | +0.048 [-0.009, +0.102] | -0.208 [-0.263, -0.153] | -0.046 [-0.091, -0.002] |
| shape_rod | +0.098 [+0.030, +0.173] | +0.014 [-0.048, +0.080] | -0.141 [-0.215, -0.058] | -0.040 [-0.111, +0.032] |
| oxygen_aerobe | +0.444 [+0.393, +0.486] | +0.291 [+0.238, +0.337] | -0.138 [-0.171, -0.110] | -0.011 [-0.046, +0.021] |
| oxygen_facultative | -0.044 [-0.117, +0.045] | -0.103 [-0.169, -0.024] | -0.090 [-0.194, -0.011] | -0.013 [-0.101, +0.057] |

- **Pfam minus k-mer (whole genome):** both models see the whole genome, so this compares what annotation extracts with what composition extracts.
- **Pfam (windows) minus k-mer (windows):** the same comparison on only the sequence Evo 2 saw.
- **Evo 2 minus Pfam:** unequal inputs. Pfam sees the whole genome, Evo 2 a median 2% of a 4.1 Mb genome.
- **Evo 2 minus Pfam (windows):** equal inputs. Pfam on the 82 kb of windows recovers a median 4.6% of a genome's families, so this is a comparison of two small views of the genome.

## 5. Chosen K and C per held-out phylum

Each cell: chosen top-K ("all" = every stage-1 survivor), chosen C, then the stage-1 survivors and the number of families selected. K and C are tuned jointly on the training phyla only.

| target | model | Actinomycetota | Bacillota | Bacteroidota | Pseudomonadota |
|---|---|---|---|---|---|
| motility | Pfam | K = 1000, C = 0.001, 11,187 / 1,000 | K = 300, C = 0.001, 10,514 / 300 | K = 100, C = 0.0001, 11,072 / 100 | K = 300, C = 0.0001, 9,947 / 300 |
| motility | Pfam (windows) | K = 100, C = 0.001, 3,781 / 100 | K = 300, C = 1, 3,594 / 300 | K = 300, C = 0.0001, 3,552 / 300 | K = 1000, C = 0.0001, 3,585 / 1,000 |
| shape_rod | Pfam | K = all, C = 0.0001, 11,187 / 11,187 | K = all, C = 0.0001, 10,514 / 10,514 | K = 300, C = 0.0001, 11,072 / 300 | K = all, C = 0.0001, 9,947 / 9,947 |
| shape_rod | Pfam (windows) | K = 300, C = 1, 3,781 / 300 | K = 100, C = 0.0001, 3,594 / 100 | K = 300, C = 0.0001, 3,552 / 300 | K = 100, C = 0.001, 3,585 / 100 |
| oxygen_aerobe | Pfam | K = 300, C = 0.001, 11,187 / 300 | K = all, C = 0.001, 10,514 / 10,514 | K = 300, C = 0.01, 11,072 / 300 | K = all, C = 0.001, 9,947 / 9,947 |
| oxygen_aerobe | Pfam (windows) | K = all, C = 0.0001, 3,781 / 3,781 | K = all, C = 0.0001, 3,594 / 3,594 | K = all, C = 0.0001, 3,552 / 3,552 | K = all, C = 0.0001, 3,585 / 3,585 |
| oxygen_facultative | Pfam | K = 100, C = 0.1, 11,187 / 100 | K = all, C = 0.0001, 10,514 / 10,514 | K = all, C = 0.0001, 11,072 / 11,072 | K = all, C = 0.0001, 9,947 / 9,947 |
| oxygen_facultative | Pfam (windows) | K = 300, C = 0.001, 3,781 / 300 | K = 300, C = 0.001, 3,594 / 300 | K = all, C = 0.0001, 3,552 / 3,552 | K = all, C = 0.1, 3,585 / 3,585 |

**Regularisation at the edge of the grid.** The C grid is 0.0001 to 1. Folds where the chosen C is the smallest value: Pfam 9 of 16, Pfam (windows) 9 of 16, Evo 2 4 of 16, k-mer (whole genome) 2 of 16, k-mer (windows) 4 of 16. When the
choice sits on the edge, a stronger regularisation was not tried and the chosen value may not be the optimum. This applies equally to every number in this report, under either contrast in section 6.

## 6. The shuffle null, and how the conclusion reads under each candidate contrast

Observed macro AUC against the null from 999 within-phylum label shuffles (mean, 95th percentile, and p = (1 + #{null >= observed}) / 1,000; the smallest possible p is 0.001):

| target | Pfam | Pfam (Evo 2 windows) | Evo 2 | k-mer (whole genome) | k-mer (Evo 2 windows) | GC only |
|---|---|---|---|---|---|---|
| motility | 0.752 vs 0.501 (95th pct 0.529), p = 0.001 | 0.590 vs 0.500 (95th pct 0.532), p = 0.001 | 0.545 vs 0.500 (95th pct 0.530), p = 0.011 | 0.524 vs 0.499 (95th pct 0.529), p = 0.075 | 0.543 vs 0.500 (95th pct 0.526), p = 0.002 | 0.585 vs 0.501 (95th pct 0.527), p = 0.001 |
| shape_rod | 0.617 vs 0.500 (95th pct 0.548), p = 0.001 | 0.516 vs 0.500 (95th pct 0.549), p = 0.304 | 0.476 vs 0.500 (95th pct 0.549), p = 0.787 | 0.519 vs 0.499 (95th pct 0.546), p = 0.226 | 0.502 vs 0.499 (95th pct 0.545), p = 0.456 | 0.492 vs 0.501 (95th pct 0.545), p = 0.639 |
| oxygen_aerobe | 0.891 vs 0.501 (95th pct 0.530), p = 0.001 | 0.764 vs 0.500 (95th pct 0.529), p = 0.001 | 0.753 vs 0.501 (95th pct 0.529), p = 0.001 | 0.447 vs 0.500 (95th pct 0.526), p = 0.999 | 0.473 vs 0.499 (95th pct 0.525), p = 0.945 | 0.590 vs 0.500 (95th pct 0.523), p = 0.001 |
| oxygen_facultative | 0.597 vs 0.501 (95th pct 0.541), p = 0.001 | 0.520 vs 0.500 (95th pct 0.540), p = 0.220 | 0.507 vs 0.501 (95th pct 0.538), p = 0.408 | 0.641 vs 0.501 (95th pct 0.541), p = 0.001 | 0.622 vs 0.501 (95th pct 0.542), p = 0.001 | 0.563 vs 0.501 (95th pct 0.538), p = 0.002 |

Two definitions of the confirmatory quantity are on record:

- **S** (spec on `origin/claude/optimistic-newton-6lf55q`, committed 2026-10-07 02:25 UTC, before any real-label fit): the macro AUC of `pfam`
  against the within-phylum shuffle null, one-sided, 999 permutations. Motility alone at 0.05; the three secondary targets
  Benjamini-Hochberg corrected at 0.05.
- **A** (D7 on this branch, recorded 2026-10-07 02:59 UTC, after that branch's first real-label result at 02:55 UTC; see the erratum in the spec): the macro
  `pfam` minus `kmer_genome` difference, with a two-sided p from the paired bootstrap (p = 2 min(s, 1 - s), s the share of resamples with a
  positive difference, floored at 1/1,001), same alpha and BH structure.

| target | S: Pfam macro AUC vs null | S: p | S: BH q | S reading | A: Pfam minus k-mer (whole genome) | A: p | A: BH q | A reading |
|---|---|---|---|---|---|---|---|---|
| motility | 0.752 vs null mean 0.501 | 0.001 | - | above the shuffle null | +0.228 [+0.176, +0.278] | 0.001 | - | ahead of k-mers |
| shape_rod | 0.617 vs null mean 0.500 | 0.001 | 0.001 | above the shuffle null | +0.098 [+0.030, +0.173] | 0.004 | 0.006 | ahead of k-mers |
| oxygen_aerobe | 0.891 vs null mean 0.501 | 0.001 | 0.001 | above the shuffle null | +0.444 [+0.393, +0.486] | 0.001 | 0.003 | ahead of k-mers |
| oxygen_facultative | 0.597 vs null mean 0.501 | 0.001 | 0.001 | above the shuffle null | -0.044 [-0.117, +0.045] | 0.342 | 0.342 | not shown ahead of k-mers |

How each reads, without choosing between them:
- **Under S,** Pfam's macro AUC is above the shuffle null on all four targets, including `oxygen_facultative` (0.597 against a null of 0.501), at the smallest p the 999 permutations allow.
- **Under A,** Pfam is ahead of whole-genome k-mers on motility, `shape_rod` and `oxygen_aerobe`, and **not on `oxygen_facultative`**, where the k-mers are nominally ahead (the difference interval spans zero).
- **The two disagree on one target.** S asks whether Pfam beats chance; A asks whether it beats composition. `oxygen_facultative` is the case where
  composition alone is already above chance (k-mer 0.641, whole genome), so Pfam is above chance but not ahead of it.
- The p-values under A come from the bootstrap and under S from the label shuffle, so they are not on the same footing even where both are at the floor.

## 7. Prevalence-threshold sensitivity (motility)

The stage-1 prevalence threshold was run at 0.5% (the pre-specified value, 999 permutations), 1% and 5% (50 permutations each, so only the AUCs and intervals are comparable, not p-values).

| min prevalence | permutations | model | macro AUC [95% CI] | Evo 2 minus model | stage-1 survivors by held-out phylum (Actino, Bacil, Bacte, Pseud) | chosen K (same order) |
|---|---|---|---|---|---|---|
| 0.5% | 999 | Pfam | 0.752 [0.711, 0.791] | -0.207 | 11,187, 10,514, 11,072, 9,947 | 1000, 300, 100, 300 |
| 0.5% | 999 | Pfam (windows) | 0.590 [0.556, 0.623] | -0.046 | 3,781, 3,594, 3,552, 3,585 | 100, 300, 300, 1000 |
| 1.0% | 50 | Pfam | 0.752 [0.711, 0.791] | -0.209 | 9,895, 9,329, 9,884, 8,925 | 1000, 300, 100, 300 |
| 1.0% | 50 | Pfam (windows) | 0.598 [0.563, 0.632] | -0.054 | 2,495, 2,384, 2,456, 2,436 | 100, 300, 300, 1000 |
| 5.0% | 50 | Pfam | 0.753 [0.711, 0.793] | -0.208 | 6,150, 5,951, 6,126, 5,870 | 1000, 300, 100, 300 |
| 5.0% | 50 | Pfam (windows) | 0.601 [0.568, 0.632] | -0.056 | 531, 501, 541, 531 | 100, 100, 100, 300 |

Pfam's motility AUC does not move with the threshold (0.752 to 0.753) while the stage-1 survivor count falls from about 10,000 to about 6,000 families; the matched-window model moves from 0.590 to 0.601, inside its interval. The chosen K is the same for Pfam at all three thresholds.

## 8. What this does and does not show

- These are within-phylum rankings on four phyla and 2,435 genomes, type strains only. They do not show that annotation would transfer to other phyla or to uncultured organisms.
- **Evo 2 minus Pfam compares unequal inputs** (section 4). The equal-input pairing is the windows one.
- **Pretraining overlap applies to Evo 2 only:** 81% of these genomes were in its GTDB training data (`reports/leakage_audit.md`, section 3). That could inflate Evo 2's AUC; it cannot affect the Pfam or k-mer models. The sensitivity analysis on unseen genomes has not been run.
- Whole-genome Pfam annotation is a different kind of input from a 4,096-number embedding of 82 kb, which is why the windows pairings exist.
- Four targets were run and all are reported, including the ones where Pfam is not ahead.
- Not run here: new Modal jobs of any kind. The `nearclade` results are still not in the repository.
