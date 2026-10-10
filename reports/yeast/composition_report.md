# Yeast genotype-to-morphology benchmark

Task: **composition**. Features: **esm2**.

This run predicts culture-level aggregate profiles, not individual-cell images.

Train / validation / test: 3579 / 632 / 476.
Retained traits: 501; train PCA variance retained: 0.738; oracle test reconstruction MSE: 0.600.

| Model | Skill vs mean | CRPS ↓ | Energy ↓ | 90% coverage | Joint-region probability |
|---|---:|---:|---:|---:|---:|
| mean | 0.000 | 1.210 | 1.664 | 0.000 | 0.000 |
| empirical | -0.007 | 0.978 | 1.344 | 0.664 | 0.022 |
| raw_empirical | -0.008 | 0.965 | 1.319 | 0.732 | 0.000 |
| ridge | 0.001 | 0.973 | 1.337 | 0.668 | 0.031 |
| shuffled_ridge | -0.008 | 0.978 | 1.343 | 0.668 | 0.031 |
| gaussian | 0.003 | 0.958 | 1.319 | 0.695 | 0.076 |
| gaussian_wrong_genotype | 0.001 | 0.960 | 1.321 | 0.694 | 0.073 |
| diffusion | 0.000 | 0.963 | 1.327 | 0.681 | 0.072 |
| diffusion_wrong_genotype | -0.000 | 0.964 | 1.328 | 0.680 | 0.072 |

Lowest CRPS in this run: gaussian. This comparison is exploratory. Examine genotype-shuffled controls and paired intervals in metrics.json before claiming genotype-specific prediction.

## Paired comparisons

- ridge_minus_empirical, CRPS difference -0.0048, 95% group-bootstrap interval [-0.0083, -0.0013].
- gaussian_minus_ridge, CRPS difference -0.0157, 95% group-bootstrap interval [-0.0196, -0.0117].
- gaussian_minus_wrong_genotype, CRPS difference -0.0023, 95% group-bootstrap interval [-0.0055, 0.0004].
- diffusion_minus_ridge, CRPS difference -0.0107, 95% group-bootstrap interval [-0.0141, -0.0067].
- diffusion_minus_wrong_genotype, CRPS difference -0.0010, 95% group-bootstrap interval [-0.0026, 0.0005].

## Limits

- Culture-level profiles, not images or single-cell distributions.
- Exact sequence groups only unless supplied homology groups; homolog transfer is possible.
- Genetic background is confounded with study.
- Predictive sample spread is not established epistemic uncertainty.
- Single-run exploratory results; no viability or causal generalization claim.

Artifacts: `map.html`, `metrics.json`, `per_gene.tsv`, all prediction arrays, training histories and model checkpoints.
