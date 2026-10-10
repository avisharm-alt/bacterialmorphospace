# Yeast genotype-to-morphology benchmark

Task: **gene**. Features: **esm2**.

This run predicts culture-level aggregate profiles, not individual-cell images.

Train / validation / test: 3285 / 703 / 703.
Retained traits: 501; train PCA variance retained: 0.754; oracle test reconstruction MSE: 0.258.

| Model | Skill vs mean | CRPS ↓ | Energy ↓ | 90% coverage | Joint-region probability |
|---|---:|---:|---:|---:|---:|
| mean | 0.000 | 0.737 | 0.953 | 0.000 | 0.000 |
| empirical | -0.022 | 0.550 | 0.711 | 0.787 | 0.098 |
| raw_empirical | -0.028 | 0.546 | 0.709 | 0.849 | 0.098 |
| ridge | -0.016 | 0.550 | 0.711 | 0.790 | 0.123 |
| shuffled_ridge | -0.020 | 0.550 | 0.711 | 0.790 | 0.115 |
| gaussian | -0.018 | 0.556 | 0.710 | 0.826 | 0.253 |
| gaussian_wrong_genotype | -0.032 | 0.560 | 0.716 | 0.824 | 0.254 |
| diffusion | -0.020 | 0.556 | 0.712 | 0.818 | 0.258 |
| diffusion_wrong_genotype | -0.034 | 0.560 | 0.716 | 0.816 | 0.254 |

Lowest CRPS in this run: raw_empirical. This comparison is exploratory. Examine genotype-shuffled controls and paired intervals in metrics.json before claiming genotype-specific prediction.

## Paired comparisons

- ridge_minus_empirical, CRPS difference 0.0005, 95% group-bootstrap interval [-0.0017, 0.0029].
- gaussian_minus_ridge, CRPS difference 0.0056, 95% group-bootstrap interval [0.0027, 0.0083].
- gaussian_minus_wrong_genotype, CRPS difference -0.0042, 95% group-bootstrap interval [-0.0058, -0.0025].
- diffusion_minus_ridge, CRPS difference 0.0059, 95% group-bootstrap interval [0.0034, 0.0083].
- diffusion_minus_wrong_genotype, CRPS difference -0.0036, 95% group-bootstrap interval [-0.0057, -0.0014].

## Limits

- Culture-level profiles, not images or single-cell distributions.
- Exact sequence groups only unless supplied homology groups; homolog transfer is possible.
- Genetic background is confounded with study.
- Predictive sample spread is not established epistemic uncertainty.
- Single-run exploratory results; no viability or causal generalization claim.

Artifacts: `map.html`, `metrics.json`, `per_gene.tsv`, all prediction arrays, training histories and model checkpoints.
