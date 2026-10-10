# drosophila_wings: DGRP wing shape from genotype, step 1 and step 2

These are steps 1 and 2 of the fruit-fly wing plan.
1. **Data in shape.** Wing landmarks are Procrustes-aligned, and each DGRP line gets a mean shape.
2. **A simple baseline.** A genomic ridge model (GBLUP-style) predicts each held-out line's mean shape from its SNPs. It is
   scored against the mean-shape baseline. Related lines never sit on both sides of a split.

The key question for step 3 (the diffusion model) is whether it beats this baseline on the same held-out lines.

**Status.** The code is written and tested on simulated data only. The real data could not be downloaded from the build
sandbox (figshare and the DGRP site are blocked by its network policy), so **no real results exist yet**.

## Data to fetch

Put the files in `/mnt/project-files/data/drosophila/raw/` (any path works, since every input is a CLI argument).

| What | Source | Used as |
|---|---|---|
| Wing landmarks: one row per wing with line, sex and x/y for 48 landmarks/semilandmarks (~22,900 wings, 184 lines) | Pitchers et al. 2019, *Genetics* 211:1429, supplemental material on figshare: <https://gsajournals.figshare.com/articles/dataset/Supplemental_Material_for_Pitchers_et_al_2019/6790526> | `--wings` |
| DGRP freeze 2 genotypes, plink `dgrp2.bed/.bim/.fam` (fastest), or `dgrp2.vcf` or `dgrp2.tgeno` | <http://dgrp2.gnets.ncsu.edu/data.html> | `--genotypes` |
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
- **Leakage control.** Lines whose normalised genomic relationship is above `--related-threshold` (0.05) are joined by
  single linkage into groups, and whole groups are held out together, in the outer folds and in the inner tuning folds
  alike. The report lists the most related pairs, so the threshold can be checked against the real distribution.

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
