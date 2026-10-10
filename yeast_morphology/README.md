# Yeast genotype → morphology

A complete, runnable research pipeline for predicting budding-yeast **aggregate morphology profiles** from a defined deleted gene and genetic background. It tests whether predictions recover a phenotype combination absent from training. This project is independent of the bacterial pipeline and Claude's Drosophila wing branch.

The output is a measured/predicted morphology map, numerical profiles, model checkpoints and honest out-of-sample scores. It does **not** generate microscopy images, predict arbitrary complete genomes, or establish viability of unmeasured organisms.

The [first completed results and interactive maps](../reports/yeast/README.md) include all three ESM benchmarks and 2,716 unmeasured combination predictions. The primary extrapolation hypothesis is **not demonstrated**.

## Quick start

Python 3.12 is recommended.

```bash
python3 -m venv .venv-yeast
source .venv-yeast/bin/activate
pip install -r yeast_morphology/requirements.txt
python -m yeast_morphology.cli fetch
python -m yeast_morphology.cli prepare
python -m yeast_morphology.cli split
python -m pytest -q tests/yeast
python -m yeast_morphology.cli train --task composition \
  --out reports/yeast/runs/composition --epochs 120
```

`prepare` verifies the downloaded hashes. A changed remote file fails instead of silently changing the experiment. The legacy SCMD site publishes HTTP downloads and its HTTPS certificate is mismatched; we use its published HTTP URLs without disabling TLS verification. SHA256 locks reproduce this retrieval, but are not author-signed authenticity guarantees.

Open `reports/yeast/runs/composition/map.html` in a browser. It is a self-contained interactive artifact, with model selection, gene search, click-to-inspect prediction intervals, pan and zoom. `map.png` is the static companion. `report.md` and `metrics.json` contain the scientific results.

Repeat the fixed evaluation with `--task gene` and `--task background`, each in a new `--out` directory. All models use the same split and train-only representation. The exact primary test is in [the experiment specification](../docs/yeast/experiment.md).

## Data and biological meaning

| Source | What it supplies |
|---|---|
| [SCMD single deletions](http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/summary.php?pj=mt4718) | 4,718 mutants, 501 CalMorph descriptors, 122 reference cultures |
| [SCMD altered background](http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/summary.php?pj=quadruple) | 1,982 deletions in pdr1Δ pdr3Δ snq2Δ, 749 reference cultures |
| [SGD proteins](https://downloads.yeastgenome.org/sequence/S288C_reference/orf_protein/orf_trans_all.fasta.gz) | Reference sequences describing which gene is deleted |

Some historical ORFs are absent from the current reference FASTA. `prepared/audit.json` lists every exclusion. No gene-name guessing or phenotype-driven rematching is used. Raw profiles are transformed into deviations from their study's WT mean, in WT SD units; reference controls are external calibration information available at inference time. Background effects cannot be separated from study effects with these data.

The 501 columns are culture summaries, including variation statistics. A sample from the model is a prediction of a **culture's aggregate profile**, not a simulated individual yeast cell. Missing entries are never scored as observed values.

## Three distinct tests

- **gene:** held-out deleted genes, with identical protein sequences grouped. Other homologs may still cross splits. Supply `split --homology-groups groups.tsv` (columns `gene`, `group`, every gene required) for an externally curated stricter grouping; write to a new split directory.
- **composition:** hold out ALL mutants above WT + 1 SD on BOTH unbudded area (`C11-1_A`) and elongation (`C115_A`). Keep examples with each high trait separately in training. Exact-sequence group mates are buffered. No test morphology enters PCA, scalers, model selection or conditioning. Phenotypes are used to DEFINE this challenge, not to predict it.
- **background:** hold out deletion × altered-background pairs; the same gene's single-deletion profile may be in training. This is explicitly a known-gene/new-background combination test, not a novel-gene test. It is also a cross-study test.

`data/yeast/splits` contains immutable per-row assignments and source/split fingerprints. Code rejects a modified split or mismatched dataset.

## Models

- Training-mean predictor and an empirical training-profile distribution.
- Ridge, validation-selected regularization; predictive samples add centered validation residuals.
- A neural diagonal-Gaussian predictor.
- Conditional DDPM on whitened train-only PCA coordinates (default 32). Conditioning is protein-sequence features plus the genetic-background indicator.
- Shuffled-label ridge and wrong-genotype neural conditioning as negative controls.

Default genetic features are log protein length, 20 amino-acid frequencies and 400 ordered dipeptide frequencies. They are a deliberately modest sequence baseline. Optional frozen ESM-2 embeddings offer richer sequence information. Neither uses GO terms, phenotype annotations, target gene descriptions or test morphology. Protein-language-model pretraining may include yeast; this is not a pretraining-novelty benchmark.

Ridge and neural models share the PCA representation. The mean and raw empirical controls also expose performance without PCA compression. Test oracle reconstruction error reports how much that representation discards. Diffusion is not presumed to beat ridge. Predictive spread must be judged by coverage; it is not automatically a measure of epistemic uncertainty.

## Modal GPU

Modal must already be authenticated. Install `modal` in your runner environment. The local raw/prepared/split files must exist before loading the app.

```bash
# One T4, <=20 minutes per invocation, no automatic retries or sweep.
modal run -m yeast_morphology.modal_app --mode embed --out data/yeast/esm2.npz
modal run -m yeast_morphology.modal_app --task composition --use-esm \
  --out reports/yeast/runs/esm-composition --epochs 120
modal run -m yeast_morphology.modal_app --task gene --use-esm \
  --out reports/yeast/runs/esm-gene --epochs 120
modal run -m yeast_morphology.modal_app --task background --use-esm \
  --out reports/yeast/runs/esm-background --epochs 120
```

Jobs use a dedicated `yeast-morphology` volume; no existing Evo 2 volume is touched. Model and result files are returned to the local output directory and retained on the volume. Embedding jobs cache each protein, so a timeout can be resumed. A training invocation allows at most 300 epochs and 300 seconds per neural model, within a 1,200-second function timeout. These are resource/time bounds, not dollar estimates; Modal's dashboard is authoritative for billing. Failed jobs are not automatically retried.

To compute ESM locally, install `fair-esm==2.0.0` and run `python -m yeast_morphology.cli embed`. All residues are encoded in <=1,000-aa nonoverlapping windows and pooled by residue count; long proteins are not silently truncated. The model is `esm2_t12_35M_UR50D`, layer 12.

## Inference

```bash
python -m yeast_morphology.cli predict \
  --run reports/yeast/runs/esm-composition \
  --embeddings data/yeast/esm2.npz \
  --genes YAL002W,YAL004W --background single --model diffusion \
  --out reports/yeast/predictions/example
```

Use genes present in the embedding file for ESM models. Outputs are `profiles.tsv`, predictive `samples.npz`, and a provenance manifest. `training_genotype` flags in-sample queries. Requests for unsupported backgrounds, unknown genes, or arbitrary multi-gene designs fail; the background's already-deleted genes cannot be deleted again. A supported request can still be biologically inviable—no viability classifier is provided. Load only trusted model bundles created by this project: Python pickle/PyTorch checkpoints are not safe untrusted interchange formats.

## Evaluation and artifacts

Report train-standardized MSE and skill against the training mean, CRPS, multivariate energy score, 90% interval coverage and width, and held-out-region probability. Paired group bootstrap intervals compare models and genotype controls. No imputed targets enter scores. Point estimates alone are not evidence of successful extrapolation.

Each run saves configuration and code hashes, environment versions, full training history, selected epochs, transforms, checkpoint weights, predictive samples, per-gene errors, static and interactive maps. Data, environments, and large model artifacts are gitignored. Compact audited result summaries belong in `reports/yeast/`.

## Source layout

`data.py` ingestion and features; `splits.py` frozen splits; `models.py` transforms and predictors; `metrics.py` scores; `experiment.py` training and inference; `embeddings.py` ESM; `report.py` map; `modal_app.py` bounded GPU runner; `cli.py` entry point.

## Explore unmeasured genotype combinations

After fitting the `background` model:

```bash
python -m yeast_morphology.cli atlas \
  --run reports/yeast/runs/esm-background --embeddings data/yeast/esm2.npz \
  --model diffusion --out reports/yeast/atlas
```

This predicts the altered-background profiles of genes with a measured single deletion but no altered-background measurement. `atlas.html` combines a measured/predicted scatter map with a sortable/searchable candidate inventory, `candidates.tsv` contains all rankings, and `candidate_profiles.npz` contains profile means/intervals plus samples on the two map axes. Novelty is the fraction of predictive mass in previously unoccupied 1-reference-SD bins. **It is not confidence**: a poorly calibrated model may rank nonsense as novel. The output is an experimental candidate list, not a catalog of viable organisms. Use `--max-genes` only for an explicitly labeled integration smoke test; the default covers every eligible gene.
