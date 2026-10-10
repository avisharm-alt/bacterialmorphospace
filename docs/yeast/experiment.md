# Yeast genotype → morphology: experiment specification

Established 2026-10-10 before model fitting. Exploratory benchmark, not a preregistered confirmatory study.

## Scope

Predict a deletion strain's 501-dimensional CalMorph aggregate profile from the deleted protein's sequence and genetic background. These are culture-level summaries (including some within-culture variation statistics), NOT individual-cell observations. Generated samples express a model's predictive distribution over aggregate profiles, NOT demonstrated cell-to-cell variation. No arbitrary genome, point mutation, viability, or causal inverse-design claims.

Primary question: predict a combination of cell area and elongation absent from the training phenotypes. Primary dataset: SCMD2 4,718 non-essential deletion mutants. Auxiliary: 122 reference cultures; 1,982 deletions in a pdr1Δ pdr3Δ snq2Δ background and 749 background-control cultures. The latter is a second study; background and study are confounded. It is an exploratory transfer test.

## Inputs, targets and controls

- Input: amino-acid frequencies, ordered dipeptide frequencies, log sequence length, and a background indicator. Optional frozen ESM-2 embeddings replace sequence descriptors. Never input gene descriptions, GO phenotype annotations, morphology-derived gene embeddings, test expression, or measured test morphology.
- Scope: the reference protein plus deletion identity describes a defined gene deletion; it is not a full-genome encoder. ESM pretraining exposure is not excluded or a claim of novelty.
- Exact identical sequences share a split group; supplied external homology groups can make gene splits stricter. Default splitting does NOT exclude every homolog or shared pathway.
- CalMorph negative sentinel values and non-finite entries become missing. Fit target imputation, feature filtering, standardization, and PCA on TRAIN only. Never score an imputed target. Median aggregation of replicate rows is allowed within genotype only, before splitting, with counts reported.
- Each study's external WT mean/SD fixes effect coordinates. WT controls are reference information available at prediction time, not held-out mutants. Target values are expressed as deviation in WT SD units, with a numerical variance floor.

## Frozen splits

1. `gene`: deterministic sequence-group-disjoint 70/15/15 train/validation/test on single deletions.
2. `composition`: C11-1_A (unbudded cell area) AND C115_A (unbudded major/minor-axis ratio) above the single-study WT mean + 1 SD. Every such mutant is test; every exact sequence group containing one is excluded from train/validation (non-combination group mates are an excluded buffer). Training/validation contain both constituent high features individually. Validation is 15% of remaining groups. Threshold choice is fixed, not optimized on predictions. This is phenotype-space extrapolation, not a held-out double knockout.
3. `background`: deterministic held-out deletion groups in the triple-deletion-background dataset (15% validation, 15% test); remaining quadruples plus ALL singles form training. Same deletion in the single background is intentionally available: novel deletion × background combinations, not novel genes. Remove the three background genes from this task. Never claim separation of study and background effects.

IDs and source hashes are frozen before training. All runs retain the same split file; no model-specific resplitting. Separate training runs/seeds are reported separately.

## Models and evaluation

Training-mean and empirical-profile distribution; tuned ridge; diagonal Gaussian neural predictor; conditional velocity-prediction DDPM in train-only whitened PCA space. Model selection uses validation only. Baselines operate on the SAME PCA targets to expose the same representation ceiling. Ridge alpha grid: 1, 10, 100, 1000, 10000. Neural epochs chosen by validation loss with fixed seed. DDPM validation noise is fixed. The primary run has 32 latent dimensions, 100 noise steps, 32 predictive samples; larger settings are explicit follow-ups.

Report MSE standardized by TRAIN target SD, skill against training mean, empirical CRPS, multivariate energy score, 90% interval coverage/width, held-out combination probability and centroid error on the two axes, and paired group-bootstrap differences. PCA oracle reconstruction error shows the representation ceiling. Sample variation is not a calibrated epistemic uncertainty estimate. Interval coverage is measured, not assumed.

Negative controls: shuffled-training-label ridge; genotype-shuffled test conditioning for neural models (same sample seed). Compare matched genotypes against wrong genotypes within background. A realistic map without better conditional prediction is a negative result.

Success requires held-out improvement over simple baselines, informative genotype conditioning, and tolerable distributional calibration; attractive generated points alone are not success. Distribution scoring includes all observed target entries, no zero-filled missing-target scoring. No claim of biological impossibility from empty map cells.

## Map and inference

Map axes are physical CalMorph descriptors in WT SD units, with separate observed, held-out predicted, and user-query layers. Predict only supported single-deletion/background inputs. Genotypes not measured are labeled predictions; arbitrary multi-deletion requests fail rather than silently extrapolate. Export numerical profiles, interval bands and provenance. Do not render invented micrographs as observations.

## Compute

Local CPU/MPS supported. Modal uses one T4 GPU container with a finite timeout, no retries, no automatic parameter sweep, and a reusable volume. A run's output includes elapsed time but is not a billing quote. The default pilot is at most 20 minutes per invocation; start with one split, inspect outputs, then run remaining fixed splits if useful. Image data are not needed for this aggregate-profile benchmark.

Implementation note before full fitting: a two-epoch integration smoke test exposed unstable epsilon sampling at the terminal cosine-noise step. The full benchmark uses velocity prediction, avoiding division by the tiny terminal signal. The smoke run is not a scientific result.
