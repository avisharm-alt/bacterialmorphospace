# oVert / MorphoSource: feasibility note

_2026-10-07. Scoping for the team discussion, not a pipeline. Everything below is metadata. No scan and no genome was downloaded. Queries and counts are reproducible with `docs/overt_scoping_query.py`; its output is `reports/tables/overt_genome_coverage.tsv`._

**What is there.** The oVert project on MorphoSource (project 000368762) holds **17,930 media items from 11,304 specimens** today. That is more than the project-page figures of about 16,800 and 10,500. By type: 15,371 volumetric image series (CT stacks), 1,835 meshes, 702 images, 22 other. **Only 52.6% of items are open download (9,419); 47.5% are restricted (8,511).** Restricted items need an approval from the data owner per item, so they are not available to us by default. This differs from the "most items are open" in the notes I was given.

## 1. Does MorphoSource have an API or a bulk metadata export?

- **API: yes, and metadata needs no key.** Unauthenticated GET requests to `https://www.morphosource.org/api/media`, `/api/physical-objects` and `/api/projects/<id>` return JSON (checked today). The project is selected with `f.project=000368762`. Other filters: `f.media_type`, `f.publication_status`, `f.taxonomy_gbif`. Pagination: `per_page` (1,000 works) and `page`; the response carries `total_count`.
- **Documentation is thin.** The official JSON API page (Duke wiki) says details "will be added in the future". The endpoints above come from the official Python client (`morphosource` on PyPI, read as text, not run) and from probing. The website's HTML pages sit behind an anti-bot challenge, so the API is the only practical route.
- **Bulk export: none found.** The practical bulk route is to page the API: all 11,304 specimens took 12 requests, about 30 seconds.
- **Downloads** need an account API key (per the client's documentation; not tested here), and restricted items need approval.

## 2. How many oVert taxa have a genome on NCBI?

I used the whole oVert specimen list, not a sample. Each specimen's `taxonomy_name` was reduced to a species binomial (subspecies collapsed, "sp.", "NULL" and genus-only entries dropped), giving **8,527 species in 5,764 genera** from 10,896 of 11,304 specimens. Species names were matched to the four NCBI vertebrate `assembly_summary.txt` files (RefSeq and GenBank, mammalian and other; 6.5 MB, downloaded 2026-10-07).

| | species | share of 8,527 |
|---|---|---|
| any NCBI assembly | **1,455** | 17.1% |
| a RefSeq assembly | 293 | 3.4% |
| a chromosome- or complete-level assembly | 583 | 6.8% |
| with an **open mesh** (738 species) and any assembly | 134 (37 with RefSeq) | |
| with an **open volume** (4,067 species) and any assembly | 623 (154 with RefSeq) | |

At specimen level, 2,138 of 11,304 specimens (18.9%) belong to a species with an assembly. NCBI's "reference or representative" label sits on 1,436 of the 1,455 and does not separate good assemblies from the rest, so I report assembly level and RefSeq status instead.

**Sampling check.** The only sample is a check on the name matching. NCBI Taxonomy resolves synonyms and GBIF-style names often differ, so a name match could under-count. I re-queried 60 randomly chosen unmatched names (of 7,072; `random.Random(20261007)`) through Entrez, which resolves synonyms. **None of the 60 had an assembly under a synonym**, so the 17% is not mostly a naming artefact (95% upper bound about 5% of unmatched names). The check works: 15 of 15 matched names and a known synonym (`Canis familiaris`) come back positive.

**Pretraining overlap, again.** OpenGenome2's eukaryote list (`species_metadata.csv`, 2,977 chordate species) contains **787 of the 1,455** genome-bearing oVert species by name (54%). This is a lower bound: the list is assumed complete, but the prokaryote list in the same file turned out to cover only the release-220 additions (`reports/leakage_audit.md`). Any Evo 2 test on vertebrates needs the same seen/unseen audit, and here accessions are listed, so it can be done by accession.

## 3. Which morphology features are cheap?

| source | amount | size | cheap features | cost |
|---|---|---|---|---|
| open **meshes** | 1,441 (394 more restricted) | median 77 MB, 90th percentile 217 MB | length, width, height ratios, surface area, volume, convex-hull ratios, curvature statistics, landmark-free shape descriptors | low: CPU only, no segmentation step |
| open **volumes** | 7,537 (7,834 more restricted) | median 1.9 GB, 90th percentile 5.2 GB (first 1,000 returned, not a random sample) | bone and tissue volumes, whole-body proportions | high: needs segmentation first, and 2-5 GB per specimen |
| **metadata only** (API) | all | none | voxel spacing, image count, body part, specimen size scale | free, but not morphology |

Meshes are mostly skulls and skeletons (`skull`, `Skull`, `Cranium`, `Skeleton`, `whole body`, `Humerus` are the top part labels), and part labels are free text, so a harmonised cranium-only set is needed before any comparison across taxa. Contrast-stained soft-tissue scans (diceCT; 111 of the first 1,000 open volumes returned) would need their own segmentation. I would start with meshes of species that also have a genome (134 species).

## 4. How would we block phylogenetic leakage for vertebrates?

Phylum is useless as a holdout, because everything here is Chordata. The analogues of what the bacterial work does:

- **Hold out clades.** Leave-one-class-out for the deep test (Actinopterygii, Aves, Mammalia, Reptilia, Amphibia), and leave-order-out and leave-family-out below it, always scored within the held-out clade. A `nearclade` analogue (leave-genus-out, accuracy by taxonomic distance to the nearest training species) needs a vertebrate taxonomy; a time-calibrated tree (for example VertLife or TimeTree) gives patristic distance directly.
- **Phylogenetic baselines and nulls.** The taxonomy-prior baseline becomes "mean of the nearest relatives"; phylogenetic generalised least squares is the standard regression baseline. For continuous traits the Mk null becomes Brownian-motion or Ornstein-Uhlenbeck simulation along the tree, plus shuffles within order. Report phylogenetic signal (Blomberg's K, Pagel's lambda) for each morphology feature first: a feature with strong signal is mostly predictable from relatives alone.
- **Specimens are not independent.** Cluster the bootstrap by species or genus. Most species here have one specimen, but some have many (82 for `Poecilia reticulata`).
- **Scan and collection batch leakage (new for vertebrates).** Museum, scanner, voxel size and preservation protocol are confounded with taxon (the largest collections are single-group: ichthyology, herpetology, birds). A model can learn taxon from the scan parameters. Mesh-derived features reduce this but do not remove it. Treat the collection or facility as a group or covariate, and test whether the features predict facility.
- **Pretraining leakage.** About half of the genome-bearing species were probably in Evo 2's training data (section 2). Use the `unseen` tier of the audit as a sensitivity set.

**One cost point on the genome side.** The bacterial recipe embeds 10 windows of 8,192 bp per genome, which is about 2% of a 4 Mb genome. On a 1-3 Gb vertebrate genome the same recipe covers about 0.003-0.008%, mostly repeats and non-coding sequence. The sampling design would need rethinking (for example gene-centred windows), so the genome arm is not a straightforward port.

## 5. What I would ask the team

1. Which morphology traits do they care about: shape of one structure (cranium) or whole-body proportions? That decides meshes against volumes.
2. Is it acceptable to start with the 134 open-mesh species that have a genome? Anything larger needs restricted-access requests or volume segmentation.
3. Whether a vertebrate arm should use Evo 2 at all, given the sampling and pretraining points above.
