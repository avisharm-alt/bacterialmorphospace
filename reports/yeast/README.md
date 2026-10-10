# Yeast morphology: first completed benchmark

Run date: 2026-10-10. **The pipeline works; the primary unseen-phenotype-combination hypothesis is not demonstrated by these models.** This is a single-seed exploratory result, not a general impossibility claim.

## Open the results

- [Interactive composition benchmark](composition.html): measured profiles, held-out predictions, model selector, gene search and intervals.
- [Candidate atlas](candidates.html): 2,716 eligible unmeasured deletion/background combinations, with a map and searchable table. Rankings are speculative, not validated biological possibilities.
- [Experiment specification](../../docs/yeast/experiment.md) and [reproduction instructions](../../yeast_morphology/README.md).

## What was built

Pinned public SCMD/SGD downloads, exact gene matching and an exclusion audit; 6,665 measured genotype/background profiles with 501 morphology descriptors. Frozen train/validation/test assignments; train-only preprocessing and PCA; protein-composition and frozen ESM-2 inputs; ridge, Gaussian and conditional velocity-DDPM predictors; empirical, shuffled-label and wrong-genotype controls; proper predictive scores, cluster-bootstrap intervals, inference, and a bounded Modal runner.

ESM-2 represents the deleted protein, plus a background indicator. This is a defined deletion-strain predictor, not an arbitrary whole-genome model. Targets describe cultures, not single cells or images.

## Results

All values below use ESM-2 features. Positive MSE skill improves on the training mean. Lower CRPS is better. Nominal coverage is 90%.

| Task | Test profiles | Model | MSE skill | CRPS | 90% coverage |
|---|---:|---|---:|---:|---:|
| composition | 476 | raw_empirical | -0.0082 | 0.9654 | 0.732 |
| composition | 476 | ridge | 0.0011 | 0.9732 | 0.668 |
| composition | 476 | gaussian | 0.0027 | 0.9575 | 0.695 |
| composition | 476 | diffusion | 0.0002 | 0.9626 | 0.681 |
| gene | 703 | raw_empirical | -0.0283 | 0.5459 | 0.849 |
| gene | 703 | ridge | -0.0162 | 0.5502 | 0.790 |
| gene | 703 | gaussian | -0.0184 | 0.5557 | 0.826 |
| gene | 703 | diffusion | -0.0204 | 0.5561 | 0.818 |
| background | 295 | raw_empirical | -0.0214 | 0.6423 | 0.801 |
| background | 295 | ridge | 0.0228 | 0.6302 | 0.794 |
| background | 295 | gaussian | 0.0085 | 0.6431 | 0.825 |
| background | 295 | diffusion | -0.0212 | 0.6495 | 0.742 |

The composition test withholds every single-deletion culture above +1 reference SD in both area and elongation (476 mutants), while retaining each high trait separately in training. Diffusion puts 7.25% of its predictive mass in that region, but has only 0.023% MSE improvement over the mean and 68.1% coverage for nominal 90% intervals. Gaussian has lower CRPS than diffusion. Raw empirical sampling has lower multivariate energy score than diffusion.

For composition, correct-minus-wrong-genotype diffusion CRPS is -0.00099, with a 95% sequence-group bootstrap interval [-0.00256, 0.00054]. That interval includes zero. The model generates some novel combinations, but this run does not show that it assigns them to the correct genotypes.

The gene and background tasks show small correct-versus-wrong conditioning signals. Nevertheless, diffusion has negative MSE skill against the training mean in both; ridge is better on CRPS. Background ridge achieves 2.28% MSE skill. This is insufficient to validate the atlas rankings. Background and study are confounded, and a paired-single-profile transfer baseline remains a useful additional control.

## Candidate atlas and interpretation

The atlas transfers measured single deletions to the pdr1Δ pdr3Δ snq2Δ background where that combination has no measurement. It uses the background diffusion checkpoint without fitting to candidate outcomes. Novelty means predictive mass in unoccupied 1-reference-SD area/elongation bins, excluding samples with nonpositive raw area or axis ratio below one. Only those two axes receive physical checks; all 501 descriptors are not constrained. The table reports invalid-axis sample fractions.

These are experimental hypotheses. Empty regions can reflect sampling bias, study effects or model error. Predictive spread is not established epistemic uncertainty. No inference establishes viability, and the model does not accept arbitrary mutation combinations.

## Reproducibility and validation

- 171 repository tests passed, including 13 yeast tests covering split leakage, train-only transforms, missing-target scoring, analytic CRPS/energy, neural roundtrips, embedding alignment and unsupported inference.
- ESM embeddings and the primary composition run executed on Modal T4. Gene and background runs executed locally on CPU. Same frozen splits, 120-epoch limits, 32 PCA dimensions and 32 predictive samples; validation early stopping. Device and time caps are retained in configurations.
- Compact metrics, run configurations, training summaries, maps and data/split/embedding manifests are committed here. Large raw data, weights, full predictive arrays and the local environment are ignored but reproducible.
- Local complete runs: `reports/yeast/runs/esm-composition-final`, `esm-gene-final`, `esm-background`. Modal retains the embedding and composition artifacts in the dedicated `yeast-morphology` volume.
- A smoke test led to the velocity parameterization before full fitting. The first ESM embedding attempt encountered internal stop markers; noncanonical residues were mapped to X and cached work resumed. An inefficient repeated decompression loader was stopped and corrected before final benchmark runs. Those aborted/pilot runs are excluded from the table.
- Subsequent changes to atlas presentation do not alter fitted model parameters; each run records module hashes. Package versions are recorded in metrics and `requirements-tested.txt`.

## Most useful next investigation

Test whether the genetic representation is the bottleneck before scaling diffusion. Add a paired-single-profile transfer control for the background task, and compare phenotype-free functional/genetic-network features with the current deleted-protein embeddings under stricter homology grouping. Separately measure the PCA ceiling on held-out combinations. Repeat promising comparisons across fixed seeds and an independent study before prioritizing atlas candidates for experiments.
