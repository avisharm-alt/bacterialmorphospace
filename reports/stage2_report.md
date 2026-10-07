# Stage 2: genome-to-trait benchmark, first reproducible pass

_7 October 2026. Initial results use the committed BacDive × GTDB v232 species tables. These are taxonomy and prevalence baselines, not genome-model results._

## Stage 1 result carried forward

The six-trait panel contains 2,580 species in 90 of 216 cells. Independent traits would fill about 117; the occupancy deficit shrinks from 27 to 5 cells under an order-level phylogenetic null and is not significant there (P = 0.067). No core-panel cell survives that null. This does not establish that empty combinations are viable or impossible. Traits are coarse, 98.8% of core species are represented by type strains, only 24 of 172 GTDB bacterial phyla are covered, and the order-level analysis has reliable power only for a zero whose expected count is about 7 species or more. See [the Stage 1 report](attrition_report.md).

## Data and audit

The committed `data/final/core_panel_species.tsv` (2,580 rows) and `no_spore_panel_species.tsv` (5,634 rows) are the current reproducible inputs. The original joined `strains.parquet`, raw BacDive cache, GTDB metadata, and genome sequences are absent locally. The panel TSVs have normalized labels but no raw label observations or conflict flags, so a full label adjudication audit needs the joined table. Stage 1 provenance is BacDive API v2 records stamped `20260601` and GTDB v232; checksums for the panel inputs and config are in [`stage2/audit.json`](stage2/audit.json), with original fetch details in [`run_manifest.json`](run_manifest.json).

The tables share all 2,580 core species. For 32 shared species the panels chose different assemblies; among shared species there are 1 Gram, 1 shape, 4 motility, 4 oxygen, and 33 spore label differences (none for temperature). These are **different strain rows**, not evidence that one row's label is wrong. The benchmark keeps each core row intact and appends only the 3,054 additional no-spore species. This yields one row per GTDB species and no repeated RefSeq or GenBank assembly accession. Identical sequence content under distinct accessions cannot be excluded until FASTAs are available.

| Trait | Labelled species | Class counts | Missing in 5,634-species union |
|---|---:|---|---:|
| Gram | 5,634 | negative 3,587; positive 2,047 | 0 |
| Shape | 5,634 | rod 4,779; coccus 495; other 360 | 0 |
| Motility | 5,634 | no 3,230; yes 2,404 | 0 |
| Spore | 2,580 | no 1,961; yes 619 | 3,054 |
| Oxygen | 5,634 | aerobe 3,915; facultative 1,133; anaerobe 586 | 0 |
| Temperature | 5,634 | meso 5,392; thermo 182; psychro 60 | 0 |

The union spans 1,927 genera, 187 orders, 70 classes, and 27 phyla. It is selected for complete coverage of the five-trait no-spore panel; these counts do not represent all matched genomes. Family is absent from the committed TSVs. `src/report.py` now writes it on a future Stage 1 regeneration; `src.stage2 --gtdb-metadata PATH` also accepts the *exact* metadata file whose SHA-256 is recorded in `run_manifest.json`. The family split remains unrun until that file is available.

## Design and first results

Each trait is scored on its labelled species. Five folds are repeated with five fixed seeds. The within-clade comparison randomly holds out species; held-out genus and order comparisons assign an entire taxon to one fold. The within-clade name means that most test species share a training clade, not that every species does: in the first repeat, 4,360 of 5,634 test species have a training species in their genus. No genome accession is repeated in the selected species set. A held-out family split runs automatically when family is present.

Prevalence predicts the most common **training** label. Taxonomy uses GTDB genus, family when present, order, class, then phylum, taking the majority training label in the narrowest rank with at least five training species and backing off otherwise. It uses taxonomy obtainable from an evaluation genome at prediction time; it never reads a test label. The thresholds and 4-mer size are fixed in code, with no hyperparameter search. The optional genome arm reads exact accession-named FASTA files, calculates fixed 4-mer frequencies, fits feature mean, standard deviation, and class centroids on training rows only, and uses nearest-centroid prediction. It could not be run because no FASTAs are cached. Evo 2 features are deferred pending data, compute, and the OpenGenome2 overlap audit.

Balanced accuracy averages recall over classes present in a test fold; macro F1 averages F1 over those same classes. Folds with only one test class are unscorable and excluded (all 25 folds per summary row here were scorable). The table shows mean balanced accuracy / mean macro F1 across 25 test folds. [`stage2/summary.tsv`](stage2/summary.tsv) contains the 5th–95th percentiles of fold scores, which describe **split variability**, not population confidence intervals. [`stage2/folds.tsv`](stage2/folds.tsv) preserves every fold and sample size.

| Trait (N) | Prevalence BA | Taxonomy within BA / F1 | Held-out genus BA / F1 | Held-out order BA / F1 |
|---|---:|---:|---:|---:|
| Gram (5,634) | 0.500 | 0.975 / 0.972 | 0.975 / 0.973 | 0.965 / 0.965 |
| Shape (5,634) | 0.333 | 0.532 / 0.587 | 0.413 / 0.439 | 0.351 / 0.341 |
| Motility (5,634) | 0.500 | 0.793 / 0.792 | 0.761 / 0.759 | 0.622 / 0.608 |
| Spore (2,580) | 0.500 | 0.908 / 0.907 | 0.895 / 0.893 | 0.668 / 0.572 |
| Oxygen (5,634) | 0.333 | 0.774 / 0.809 | 0.737 / 0.773 | 0.483 / 0.486 |
| Temperature (5,634) | 0.333 | 0.404 / 0.438 | 0.397 / 0.425 | 0.344 / 0.346 |

The order-split taxonomy BA 5th–95th percentile ranges are 0.920–0.989 (Gram), 0.332–0.374 (shape), 0.513–0.784 (motility), 0.476–0.907 (spore), 0.428–0.525 (oxygen), and 0.333–0.368 (temperature). Spore's wide range reflects uneven class and clade allocation. Temperature has only 60 psychrophiles. The high Gram score persists across orders because broad taxonomic ranks strongly track the label; it is not a genome-sequence benchmark. The preliminary email observation that Evo 2 predictions decline across clades while taxonomy performs similarly remains **unreproduced**.

[`stage2/distance.tsv`](stage2/distance.tsv) stratifies first-repeat out-of-fold predictions by the deepest GTDB rank shared with a training species. This is an **ordinal taxonomy proxy**, not phylogenetic branch length. For example, within-clade motility taxonomy BA is 0.822 for 4,360 species sharing a training genus and 0.719 for 1,192 whose closest shared rank is order. The finer bins have fewer species and different class mixes, so this is descriptive, not a causal distance effect.

## Pretraining overlap and limitations

[OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2), an Evo 2 pretraining corpus, lists GTDB v220 as a source; this project uses v232. Shared source lineage makes overlap plausible, but different GTDB releases neither prove nor rule out shared assemblies. We have **not** obtained a complete OpenGenome2 accession manifest, so exact overlap, sequence-identical duplicates, and train/test contamination cannot yet be quantified. The optional `--pretraining-accessions PATH` screen counts exact and version-stripped GCA/GCF accession intersections (including the panel's RefSeq–GenBank crosswalk), but it only becomes evidence for corpus-wide overlap if the supplied manifest is complete and provenance-verified. No multi-terabyte corpus download is needed or planned for this milestone. Any future Evo 2 score must be labelled with this unresolved pretraining exposure.

Other limits: taxonomy availability at prediction time must be verified for a real deployment; genus labels can change between GTDB releases; panel selection favors well-described cultured species; family is pending; no genome baseline ran; and branch-length phylogenetic distance was not computed. These results do not test organism viability or impossibility from empty trait cells.

## Reproduction and next experiment

From the repository root, using Python 3.12.14, pandas 3.0.6, NumPy 2.5.3 and pytest 9.1.1:

```bash
.venv/bin/python -m pytest -q                   # 167 passed
.venv/bin/python -m src.stage2                 # ~34 s, writes reports/stage2/
.venv/bin/python -m src.stage2 --gtdb-metadata data/raw/gtdb/bac120_metadata.tsv.gz
.venv/bin/python -m src.stage2 --gtdb-metadata data/raw/gtdb/bac120_metadata.tsv.gz --accessions path/to/subset.txt --genomes path/to/accession_fastas
```

The latter two commands are **planned**, not completed. First recover the hash-matched GTDB v232 metadata for family ranks, then obtain a small accession-matched FASTA subset with representation across orders and classes. Run all three baselines on the *same* genome subset and report label/assembly coverage changes. Next obtain a complete OpenGenome2 accession manifest or equivalent release index and run the overlap screen before spending compute on Evo 2. A sequence identity check is needed for different accessions encoding the same genome. Only after those checks should an Evo 2 comparison be interpreted.

## October 9 briefing for Kavita

- **Finding:** Stage 1's apparent six-trait occupancy deficit nearly disappears under the order null. The method is insensitive to rare combinations and the type-strain panel has narrow coverage.
- **Concern:** Species and genome leakage can inflate within-clade prediction; Evo 2 may also have seen evaluation genomes through OpenGenome2. The earlier Evo 2 impression is unreplicated.
- **Design:** Pair every label with its selected assembly; compare prevalence, taxonomy, fixed 4-mers, then possibly Evo 2 on identical accessions. Hold out genus, family, and order; fit every transformation on training folds.
- **Initial evidence:** Taxonomy's order-held-out BA is 0.965 for Gram but only 0.622 for motility, 0.483 for oxygen, 0.351 for shape and 0.344 for temperature. No genome-sequence result is available.
- **Decisions:** Which trait and clade coverage should define the first FASTA subset? Can the team supply the exact v232 metadata and a complete OpenGenome2 accession index? What compute budget and contamination criterion would justify Evo 2 features?

Adrian's [Cornell oVert lead](https://news.cornell.edu/stories/2024/04/vertebrate-3d-scan-project-opens-collections-all) concerns 3D scans of **vertebrate** collections. It may support a separate vertebrate morphology project, including scan access and shape quantification, but it is not a bacterial phenotype source for this benchmark.
