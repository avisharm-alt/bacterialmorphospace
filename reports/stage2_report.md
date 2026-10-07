# Stage 2: initial genome-to-trait benchmark

_2026-10-07 · Stage 1 inputs: BacDive API v2 snapshot (record DOI stamp 20260601), GTDB v232 · code base `e16e68d`_

## Status and question

This is a reproducible **prevalence and taxonomy baseline**, plus a runnable
4-mer sequence baseline awaiting FASTA files. It is not an Evo 2 evaluation.
Adrian's preliminary report that Evo 2 predictions fell toward chance across
clades while taxonomy performed similarly is an **unreproduced observation**.

Stage 1 found 2,580 species in 90 of 216 six-trait cells. The occupancy
shortfall relative to independent traits shrank from 27 to 5 cells under an
order-stratified null and was not significant there (P = 0.067). No core-panel
cell survived that null. This does not show that unobserved combinations are
viable or impossible: traits are coarse, 98.8% of the panel represents type
strains, and an order-level zero requires about seven expected species for
reliable detection. [Stage 1 details](attrition_report.md).

## Input and label audit

The committed Stage 1 species tables are the evaluation units. Five traits
use the 5,634-species no-spore panel; spore uses the 2,580-species core panel.
There are no duplicate GTDB species, GTDB accessions, NCBI assembly accessions,
or GenBank assemblies, including version-stripped GenBank IDs, **within either
panel**. Exact sequence identity between different assemblies was not tested.
All evaluated coarse labels,
assembly IDs, and genus-to-phylum labels are present. GTDB v232 taxonomy
supplies family labels for all 5,666 distinct accessions across both panels.

| Trait | N | Class counts in evaluation panel |
| --- | ---: | --- |
| Gram | 5,634 | negative 3,588; positive 2,046 |
| Shape | 5,634 | rod 4,780; coccus 494; other 360 |
| Motility | 5,634 | no 3,230; yes 2,404 |
| Spore | 2,580 | no 1,961; yes 619 |
| Oxygen | 5,634 | aerobe 3,917; facultative 1,131; anaerobe 586 |
| Temperature | 5,634 | meso 5,392; thermo 182; psychro 60 |

The larger panel covers 1,927 genera, 459 families, 187 orders, 70 classes,
and 27 phyla; the core panel covers 1,235, 357, 141, 59, and 24 respectively.
Of the 5,634 larger-panel matches, 5,461 used BacDive assembly accessions and
173 used designations. Temperature was derived from a reported optimum for
3,753 species, a growth range for 1,426, and one positive growth point for
455. Coarse `shape=other` combines 360 cases, and 150 microaerophile labels
are folded into facultative oxygen. These are label-quality concerns, not
missing values. The five-trait panel has 3,087 missing spore labels; they are
never imputed or used in the spore task.

The two panels share 2,580 species but select different assemblies for 32 of
them. Among shared species, their selected strain labels differ in 1 Gram,
1 shape, 4 motility, and 4 oxygen cases. Tasks use one panel at a time, so
these alternatives cannot cross a train/test split. The committed species
tables omit raw BacDive observations and conflict flags; a full reference and
assay audit requires regenerating the ignored joined Parquet from the original
cache. [Machine-readable audit](stage2_audit.json).

## Evaluation design

Each trait has 20 seeded, approximately 80/20 splits: random species within
the panel, or disjoint held-out genus, family, or order. Every test assembly,
GenBank accession, and GTDB species is absent from its training fold; the
specified held taxon is absent too. The same split indices serve both models.
No target species name, BacDive ID, phenotype, match method, or temperature
source is used as a feature. GTDB taxonomy from genus through phylum is assumed
available for a query genome after genome-based classification; deployment to
a new genome must run a comparable taxonomy assignment first.

- **Prevalence:** predict the training fold's most common class.
- **Taxonomy:** find the finest rank observed in training, falling back toward
  phylum; estimate its class frequencies with five training-fold prior counts,
  then use a class-balanced decision rule. The prior strength, label set, and
  decision rule are fixed, with no hyperparameter search.
- **Optional sequence:** normalized 4-mer frequencies per assembly and cosine
  similarity to training-fold class centroids. The parser excludes ambiguous
  bases and contig joins. It requires a FASTA for every evaluated assembly;
  there are no sequence results in this report.

The metrics are balanced accuracy and macro F1 over the fixed coarse classes.
All 20 test folds contained every class, including the rare temperature bins.
The ranges below are 2.5th–97.5th percentiles over 20 resplits: **split
variability**, not confidence intervals for independent samples. Test species
can recur across repeats. Per-fold sample sizes and metrics are in
[`stage2_split_metrics.tsv`](tables/stage2_split_metrics.tsv); both metric
ranges and median train/test sizes are in
[`stage2_benchmark_summary.tsv`](tables/stage2_benchmark_summary.tsv).

## Initial results

Each cell gives mean **balanced accuracy / macro F1** for the taxonomy baseline.
Prevalence balanced accuracy is 0.500 for the binary traits and 0.333 for the
three-class traits in every split; its macro F1 and all uncertainty ranges are
in the summary table. No genome sequence features were evaluated.

| Trait | Random species | Held genus | Held family | Held order |
| --- | ---: | ---: | ---: | ---: |
| Gram | .978 / .974 | .978 / .975 | .965 / .965 | .950 / .953 |
| Shape | .676 / .584 | .600 / .485 | .570 / .429 | .420 / .305 |
| Motility | .820 / .816 | .782 / .778 | .766 / .746 | .606 / .568 |
| Spore | .945 / .928 | .927 / .911 | .897 / .869 | .728 / .674 |
| Oxygen | .824 / .807 | .777 / .760 | .693 / .707 | .516 / .464 |
| Temperature | .700 / .487 | .614 / .392 | .569 / .358 | .569 / .269 |

For example, motility balanced accuracy is .820 (.800–.833) under random
species splitting and .606 (.521–.754) with orders held out; macro F1 is
.816 (.798–.829) and .568 (.409–.744), respectively. Oxygen changes from
.824 (.808–.848) to .516 (.298–.745) balanced accuracy. The broader splits
also vary more in test composition and size; this is a descriptive benchmark,
not a causal estimate of evolutionary distance. Gram remains high even at
order level, so loss of generalization depends strongly on the trait.

[`stage2_distance_metrics.tsv`](tables/stage2_distance_metrics.tsv) records
the nearest training taxonomic rank for each test subset. It is a rank proxy,
not a phylogenetic branch length. Within the random species split, motility's
mean balanced accuracy is .850 for test species sharing a training genus,
.733 for same family but no training genus, and .650 for same order but no
training family. The last group has only 558 test predictions across 20
overlapping repeats; small, class-incomplete subsets should not be read as
precise estimates. A phylogenetic tree or patristic distances would improve
this analysis.

## Evo 2 pretraining overlap

[OpenGenome2's dataset card](https://huggingface.co/datasets/arcinstitute/opengenome2/blob/main/README.md)
states that its prokaryotic source includes GTDB v220 and that it was used to
train Evo 2. The evaluation table is GTDB v232. Comparison with the official
[v220](https://data.gtdb.ecogenomic.org/releases/release220/220.0/) and
[v232](https://data.gtdb.ecogenomic.org/releases/release232/232.0/) taxonomy
accession lists finds exact v220 GTDB accessions for **2,240/2,580 core**
assemblies (86.8%) and **4,993/5,634 no-spore** assemblies (88.6%). Ignoring
assembly version and GTDB RS/GB prefix raises these to 2,244 and 4,998.
These are source-membership checks, **not confirmed Evo 2 training exposure**:
OpenGenome2 can filter or transform source assemblies; its other sources may
contain nonmatching or homologous sequences; GTDB versions, RefSeq/GenBank
aliases, and sequence-level near duplicates remain untested. Conversely, a
v232 accession absent from v220 is not evidence that its sequence is novel to
Evo 2. We did not download the multi-terabyte corpus. The repository's
[`species_metadata.csv`](https://huggingface.co/datasets/arcinstitute/opengenome2/blob/main/species_metadata.csv)
is species-level metadata and cannot establish assembly or sequence exposure.

## Reproduction, tests, and next experiment

From the repository root, with `requirements.txt` installed:

```bash
mkdir -p data/raw/gtdb
curl -fsSL https://data.gtdb.ecogenomic.org/releases/release232/232.0/bac120_taxonomy_r232.tsv.gz -o data/raw/gtdb/bac120_taxonomy_r232.tsv.gz
curl -fsSL https://data.gtdb.ecogenomic.org/releases/release220/220.0/bac120_taxonomy_r220.tsv.gz -o data/raw/gtdb/bac120_taxonomy_r220.tsv.gz
python -m src.stage2 prepare
python -m src.stage2 benchmark --repeats 20
python -m pytest -q
```

The benchmark command itself runs offline from committed inputs. Source URLs
and SHA256s are pinned in [`stage2_sources.json`](stage2_sources.json).
SHA256s for `core_panel_species.tsv`, `no_spore_panel_species.tsv`, and the
Stage 2 taxonomy map are, respectively,
`7ee69db6e48134e3cecb2e175af38281b3f485bf3e695ea83d56196e1bbfe1f2`,
`c991b33af7d4b0282c3002f8568008e504084fc6720b2f383b18ba74ba06afb9`,
and `32d5460564179557c3c246f88e36fb91a774be1d53426faa70a7e6b8c1a35c78`.
This run used Python 3.12.14, pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1,
and pytest 9.1.1. `python -m pytest -q` passed 165 tests, including split,
duplicate accession, train-only prediction, and contig-boundary checks.

**Next experiment:** acquire assembly FASTAs for a prespecified, balanced
subset, verify accession and sequence hashes, and run the fixed 4-mer baseline
on identical splits. Report missing FASTAs and coverage by trait and clade
before interpreting a sequence score. Then decide whether exact/near-duplicate
screening against a tractable OpenGenome2 manifest and Evo 2 feature extraction
are practical. Evo 2 should be considered only after the taxonomy and 4-mer
comparison, with pretraining exposure explicitly stratified or excluded.

## Separate vertebrate morphology lead

[Cornell's oVert project](https://news.cornell.edu/stories/2024/04/vertebrate-3d-scan-project-opens-collections-all)
provides specimen-linked vertebrate CT scans and catalog metadata, useful for
a separate morphology project if scan-derived traits and specimen/genome links
can be defined. It is **not a bacterial phenotype dataset** and has no role in
the benchmark above.
