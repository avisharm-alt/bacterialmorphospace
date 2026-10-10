# Yeast genotype-to-morphology benchmark

Task: **background**. Features: **esm2**.

This run predicts culture-level aggregate profiles, not individual-cell images.

Train / validation / test: 6068 / 297 / 295.
Retained traits: 501; train PCA variance retained: 0.760; oracle test reconstruction MSE: 0.268.

| Model | Skill vs mean | CRPS ↓ | Energy ↓ | 90% coverage | Joint-region probability |
|---|---:|---:|---:|---:|---:|
| mean | 0.000 | 0.850 | 1.093 | 0.000 | 0.000 |
| empirical | -0.012 | 0.649 | 0.835 | 0.741 | 0.136 |
| raw_empirical | -0.021 | 0.642 | 0.828 | 0.801 | 0.135 |
| ridge | 0.023 | 0.630 | 0.812 | 0.794 | 0.287 |
| shuffled_ridge | -0.001 | 0.637 | 0.821 | 0.795 | 0.207 |
| gaussian | 0.009 | 0.643 | 0.818 | 0.825 | 0.385 |
| gaussian_wrong_genotype | -0.003 | 0.648 | 0.824 | 0.822 | 0.383 |
| diffusion | -0.021 | 0.650 | 0.837 | 0.742 | 0.261 |
| diffusion_wrong_genotype | -0.059 | 0.664 | 0.854 | 0.733 | 0.267 |

Lowest CRPS in this run: ridge. This comparison is exploratory. Examine genotype-shuffled controls and paired intervals in metrics.json before claiming genotype-specific prediction.

## Paired comparisons

- ridge_minus_empirical, CRPS difference -0.0184, 95% group-bootstrap interval [-0.0265, -0.0108].
- gaussian_minus_ridge, CRPS difference 0.0128, 95% group-bootstrap interval [0.0046, 0.0199].
- gaussian_minus_wrong_genotype, CRPS difference -0.0049, 95% group-bootstrap interval [-0.0127, 0.0033].
- diffusion_minus_ridge, CRPS difference 0.0193, 95% group-bootstrap interval [0.0081, 0.0308].
- diffusion_minus_wrong_genotype, CRPS difference -0.0144, 95% group-bootstrap interval [-0.0287, -0.0007].

## Limits

- Culture-level profiles, not images or single-cell distributions.
- Exact sequence groups only unless supplied homology groups; homolog transfer is possible.
- Genetic background is confounded with study.
- Predictive sample spread is not established epistemic uncertainty.
- Single-run exploratory results; no viability or causal generalization claim.

Artifacts: `map.html`, `metrics.json`, `per_gene.tsv`, all prediction arrays, training histories and model checkpoints.
