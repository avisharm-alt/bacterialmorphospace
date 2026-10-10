# drosophila_wings: DGRP wing shape from genotype, step 1 and step 2

These are steps 1 and 2 of the fruit-fly wing plan.
1. **Data in shape.** Wing landmarks are Procrustes-aligned, and each DGRP line gets a mean shape.
2. **A simple baseline.** A genomic ridge model (GBLUP-style) predicts each held-out line's mean shape from its SNPs. It is
   scored against the mean-shape baseline. Related lines never sit on both sides of a split.

The key question for step 3 (the diffusion model) is whether it beats this baseline on the same held-out lines.

**Status (2026-10-10).** The pipeline has been run on the real data. The SNP model does **not** beat the mean shape on
held-out lines once related lines are kept apart (see "Results" below).

## Data to fetch

Put the files in `/mnt/project-files/data/drosophila/raw/` (any path works, since every input is a CLI argument).

| What | Source | Used as |
|---|---|---|
| Wing landmarks: one row per wing with line, sex and x/y for 48 landmarks/semilandmarks (~22,900 wings, 184 lines) | Pitchers et al. 2019, *Genetics* 211:1429, supplemental material on figshare: <https://gsajournals.figshare.com/articles/dataset/Supplemental_Material_for_Pitchers_et_al_2019/6790526> | `--wings` |
| DGRP freeze 2 genotypes (the run used `dgrp2.vcf.gz`, a copy from Zenodo record 155396), plink `dgrp2.bed/.bim/.fam` (fastest), or `dgrp2.vcf` or `dgrp2.tgeno` | <http://dgrp2.gnets.ncsu.edu/data.html> | `--genotypes` |
| Wolbachia status and major inversion karyotypes (`wolbachia.xlsx`, `inversion.xlsx`) | same DGRP2 page | `--covariates` |

The exact file names in the figshare item have not been checked from here. The loader finds the line, sex and landmark
columns by name (`line`/`dgrp`/`ral`…, `sex`, and `x1..x48`/`y1..y48` or similar such as `LM1x`, `X01`, `P1_X`). Use
`--line-col` and `--sex-col` if it guesses wrong. If the table records the lab or imaging session, pass it with
`--cell-cols` so that it is centred out like sex.

## Running it

```bash
pip install -r requirements.txt
D=/mnt/project-files/data/drosophila
python -m drosophila_wings.pipeline --data-dir $D all \
    --wings $D/raw/<wing table> --genotypes $D/raw/dgrp2 \
    --covariates $D/raw/wolbachia.xlsx $D/raw/inversion.xlsx
```

The steps can also be run one at a time (`prepare`, `genotypes`, `baseline`). The `genotypes` step on the full VCF takes
about 5 minutes (about 4.4M variants); plink is faster, and `--thin N` keeps every N-th variant.

Outputs:
- `derived/line_means.csv`: the line × tangent-coordinate mean shapes. These are the targets for step 3 as well.
- `derived/wings_tangent.npz`: every aligned wing, with its centroid size, line and sex. The diffusion model needs the
  within-line spread, not only the means.
- `derived/genotypes_filtered.npz` and `derived/grm.npz`: the filtered genotypes and the genomic relationship matrix.
- `results/baseline_report.md`, `baseline_metrics.json`, `cv_repeats.csv`, `per_line_errors.csv`.

Dry run on simulated data (`--simulated` labels the report):
```bash
python -m drosophila_wings.pipeline simulate --out /tmp/sim/raw
python -m drosophila_wings.pipeline --data-dir /tmp/sim all --wings /tmp/sim/raw/sim_wings.csv \
    --genotypes /tmp/sim/raw/sim_dgrp --covariates /tmp/sim/raw/sim_covariates.csv --simulated
```

## What the code does, and the choices it makes

**Shape (`procrustes.py`, `shape.py`).**
- Generalised Procrustes alignment of all wings, without reflection, followed by orthogonal projection to tangent space.
  Semilandmarks are not slid again.
- Each wing has its sex's mean shape subtracted (and lab's, with `--cell-cols`). The line mean then weights the two sexes
  equally, so a line imaged mostly as females is not pulled toward female shape. `--sex F` analyses one sex instead.
- Lines with fewer than `--min-wings` (10) wings are dropped. `--remove-allometry` removes the common within-sex slope of
  shape on log centroid size. It is off by default.
- **Noise ceiling.** The Spearman-Brown corrected split-half reliability of the line means is the R² that a perfect
  genotype-to-mean predictor could reach against these noisy means.

**Genotypes and relatedness (`io.py`, `relatedness.py`).**
- Biallelic SNPs only (from VCF and tgeno), with MAF ≥ 0.05 and at most 20% missing. Both filters are recomputed on the
  phenotyped lines.
- VanRaden GRM: ZZ'/m.
- **Leakage control.** Lines whose normalised genomic relationship is above `--related-threshold` (0.1) are joined by
  single linkage into groups, and whole groups are held out together, in the outer folds and in the inner tuning folds
  alike. The report lists the most related pairs, so the threshold can be checked against the real distribution.
  On the real panel, 0.1 is just above the 99th percentile of pairs (0.089). At 0.05, single linkage chains 79 of the
  166 lines into one group, larger than a fold.

**Models (`model.py`).** Every tangent coordinate is predicted at once.

| model | what it is |
|---|---|
| `mean` | the training-fold mean shape: the baseline to beat |
| `covariates` | OLS on Wolbachia status and inversion karyotypes only |
| `snp` | kernel ridge on the GRM, which is equivalent to ridge on all SNPs, i.e. GBLUP |
| `snp+cov` | covariates as fixed effects, then GBLUP on their residual (a two-stage approximation of the joint mixed model) |

The ridge penalty is chosen by inner group CV on a 25-point grid. Outer CV is 5-fold and repeated 5 times.

**Metrics.**
- R² against the mean shape: 1 − SSE(model)/SSE(mean). A value above 0 means the model beats the mean shape.
- The mean Procrustes distance to the observed line mean.
- The share of held-out lines that are predicted better than by the mean shape.
- The correlation between predicted and observed scores on PCs 1–3 of the line means.
- A permutation test, which shuffles genotypes across lines within the same folds.
- The same models with random, relatedness-blind folds. These are labelled "leaky" and are shown only to measure how
  much leakage would inflate the score.

## What to expect

With about 180 mostly unrelated inbred lines and a polygenic trait, genomic prediction of line means has a low theoretical
ceiling. Expected accuracy scales roughly as n·h²/(n·h² + Mₑ), and Mₑ, the number of effectively independent segments, is
large in the DGRP because LD decays fast. In simulation (150 lines, h² = 0.5, polygenic), the honest R² was about 0.02,
while relatedness-blind folds reported about 0.09. A small honest R² on the real data would therefore not be a bug. It
would be the bar that the diffusion model has to clear, and the reason why the comparison has to use the same grouped
folds.

## Results on the real data (2026-10-10)

The inputs were `BothLabs_Wings_28Oct.csv` (22,923 wings, centred within sex × lab with `--cell-cols Lab`) and
`dgrp2.vcf.gz` (1.68M SNPs after filtering). Covariates were the inversion dosages (`In2Lpred`, `In2Rpred`, `In3Rpred`)
from `Hetinversionscores205.csv`, plus Wolbachia status. 166 lines are both phenotyped and genotyped. The full report is
in `/mnt/project-files/data/drosophila/results/`.

| folds | model | R² vs mean shape |
|---|---|---|
| related lines kept apart (0.1) | GBLUP | −0.001 ± 0.001 (permutation p = 0.36) |
| related lines kept apart (0.1) | covariates / GBLUP + covariates | −0.025 / −0.028 |
| relatedness-blind (leaky) | GBLUP | +0.037 ± 0.011 |

- **Noise ceiling.** The line means are very reliable (split-half 0.994), so the near-zero score is not noise in the
  targets.
- **The genetic signal is real but only between relatives.** Pairs with normalised relationship above 0.2 have line-mean
  shapes correlated about 0.34, against about 0 for unrelated pairs. GBLUP exploits that only when relatives straddle the
  split. That is the leaky gain.
- **Sensitivity.** The result holds at other thresholds: GBLUP R² is +0.009 at 0.2 and −0.002 at 0.05.
- **Covariates.** OLS on the inversion and Wolbachia covariates overfits slightly (R² below 0).

# Step 3: genotype-conditioned diffusion model

`diffusion.py` is a small conditional denoising diffusion model (MLP denoiser with FiLM conditioning, cosine schedule,
ancestral sampling) over whitened shape PCs of the aligned wings. It is conditioned on a genotype code (the top 10
genomic PCs of the training lines' GRM; other lines are projected onto them) and on sex. The genotype code is dropped for
15% of training wings, so the same network also gives a genotype-free model. `generative.py` trains it per fold, generates
wings for every held-out line and scores them; `diffusion_cv.py` is the entry point; `modal_app.py` runs the folds on Modal
T4 GPUs.

```bash
pip install torch 'modal[api-proxy-support]' matplotlib   # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET must be set
python -m drosophila_wings.diffusion_cv run --backend modal               # about 10 minutes, 16 GPU containers
python -m drosophila_wings.diffusion_cv run --backend local --quick       # CPU smoke test (writes "QUICK CPU TEST")
```

The outer folds are copied from the baseline's `results/per_line_errors.csv`, so step 3 is scored on exactly the step 2
held-out lines. Outputs go to `results/diffusion/`: `diffusion_report.md`, `diffusion_metrics.json`, `per_line_scores.csv`,
`summary_test_lines.csv`, the possibility-space tables and samples, and three figures.

**What is compared on each held-out line** (all sex-matched to its real wings):

| method | what it is |
|---|---|
| `mean+resid` | training mean shape + real within-line scatter from training lines: the baseline to beat |
| `pooled` | random training wings (the population, so far too spread out for one line) |
| `diffusion-geno` / `-nogeno` | the diffusion model with / without the line's genotype code |
| `diffusion-shuffled` | the same model trained with genotype codes shuffled across training lines (control) |
| `diffusion-resid-*` | a model trained on within-line residuals only, added to the training mean: does genotype predict the scatter? |
| `oracle-mean+...` | the same scatter generators added to the line's own observed mean (not achievable; tests spread alone) |

Scores: energy distance between generated and real wings (whole distribution), the step 2 R² on the generated line mean,
and the within-line variance (spread) of generated vs real wings.

## Step 3 results on the real data (2026-10-10)

166 lines, 22,593 wings, 5 related-grouped folds, 512 generated wings per held-out line. Full report:
`/mnt/project-files/data/drosophila/results/diffusion/diffusion_report.md`.

| held-out lines | energy ×10⁻³ | line-mean R² vs mean | spread log ratio | spread r across lines |
|---|---|---|---|---|
| `mean+resid` (baseline) | 16.4 | 0.001 | +0.01 | −0.07 |
| `diffusion-geno` | 15.4 | −0.047 | +0.86 | −0.08 |
| `diffusion-nogeno` | 14.5 | −0.004 | +0.89 | 0.02 |
| `diffusion-shuffled` | 15.6 | −0.071 | +0.89 | 0.00 |
| `diffusion-resid-geno` | 16.8 | 0.000 | −0.17 | 0.04 |
| `diffusion-resid-nogeno` | 16.8 | −0.002 | −0.15 | 0.04 |

- **Genotype does not help on new lines, for the mean or for the spread.** The genotype-conditioned model is no better than
  the same model with genotype dropped or shuffled. Its line means are slightly worse than the training mean (R² −0.047),
  matching the step 2 result.
- **Why: held-out genotype codes are short.** Training lines' codes have mean length 2.47, held-out lines' 0.66. An
  unrelated DGRP line projects near the centre of the training lines' genomic PCs, so the model treats it as "a typical
  line" and generates something close to the whole population.
- **The lower energy distances are hedging, not knowledge.** `pooled` and `diffusion-nogeno` beat the baseline on energy
  distance (about 90% of lines) only because they are 2–3 times too spread out (log ratio about +0.9). When the mean is
  unknown, a broad cloud covers the line's real wings better than a tight cloud in the wrong place. The genotype model does
  not beat its own genotype-free version.
- **Within-line spread is a real, reliable trait** (split-half reliability of log within-line variance 0.90), and the
  residual model learns it for lines it has seen (r = 0.47 between generated and real spread on training lines). It does
  not transfer to new lines (r = 0.04), and related pairs share little of it (similarity 0.03 for 20 pairs above 0.2).
- **The model is somewhat under-dispersed.** Generated residual scatter is about 15% too narrow (log ratio −0.15), which is
  why `diffusion-resid-*` lose to `mean+resid` on energy. On whole wings, generated samples are about 20% too narrow in
  whitened units before sampling-step tuning; this is a known MLP-diffusion limitation and does not change the comparison
  between conditioned, genotype-free and shuffled versions of the same model.
- **Possibility space sketch** (one model on all 166 lines). Generated line means for random genotype codes, crosses and 2×
  extrapolated codes all fall inside the observed 95% region and span 20–45% of the observed between-line variance. The
  model reproduces a contracted version of the observed shape space; it does not open new regions, because the genotype
  code carries no transferable signal.

**Next.** A genotype representation that transfers between unrelated lines is the bottleneck, not the generator: candidate
genes or SNP sets from published wing-shape GWAS (Pitchers et al. 2019's hits), gene-level burden scores, or embeddings
of the variants themselves, all scored on the same grouped folds. Mapping the possibility space without genotype (the
genotype-free model, or conditioning on observed line means) is possible now.

## Step 3b: gene-informed genotype codes (2026-10-10)

`python -m drosophila_wings.diffusion_cv codes` builds per-fold codes (`genecodes.py`); `run --codes wing-genes fold-gwas
pitchers-hits` scores the diffusion model with each, next to genomic PCs. A code is a line's phenotype predicted by kernel
ridge from one SNP set, fitted on training lines only (out-of-fold for the training lines themselves). Gene coordinates
are UCSC dm3 FlyBase tables in `raw/annotation/`.

| held-out lines | ridge alone: shape R² | diffusion: energy ×10⁻³ (geno / no-geno / shuffled) | diffusion: line-mean R² |
|---|---|---|---|
| genomic PCs | −0.001 (step 2) | 15.4 / 14.5 / 15.6 | −0.047 |
| `wing-genes` (82 genes ±5 kb, 45k SNPs) | 0.000 | 16.2 / 14.5 / 16.0 | −0.097 |
| `fold-gwas` (top 2,000, re-selected in fold) | −0.049 | 18.6 / 14.5 / 18.2 | −0.211 |
| `pitchers-hits` (831 published hits; leaky) | +0.045 | 17.1 / 14.5 / 17.8 | −0.109 |

- **No gene-informed code helps.** Every conditioned model is worse than the genotype-free model and no better than its
  own shuffled control, for line means and for within-line spread (spread r across held-out lines between −0.08 and +0.07).
- **The only positive number is leaky.** Ridge on the published hits reaches R² +0.18 in four folds (−0.49 in the fifth),
  because those SNPs were picked using the held-out lines. Re-selecting SNPs inside each fold (`fold-gwas`) gives −0.05.
- **Sharper codes make the diffusion model worse.** Gene codes give each line a distinctive value, so the model uses them
  as a line ID: training lines are reproduced almost exactly (energy 0.03×10⁻³ for `fold-gwas`), and new lines get a
  confident wrong answer instead of a hedge.
