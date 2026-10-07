# Leakage audit

_2026-10-07. Numbers are taken from committed tables, except the pretraining overlap in section 3, which this audit computed (inputs and hashes below). Nothing was re-run on Modal for it._

**Phylogenetic leakage** here means a model scores well on a trait by looking up nearby relatives, or by exploiting how
traits are distributed over the tree, and not because it has learned anything about trait biology. This project has
three places where it can happen: the occupancy nulls (lineage structure mimicking a constraint), the prediction
tests (relatives, tuning and selection shared between train and test), and Evo 2's own pretraining (the model may have
seen the test genomes).

## 1. Prediction tests: what each control closes

| control | leakage it closes | what it does not close | status | result |
|---|---|---|---|---|
| **LOPO scored within the held-out phylum** (`lopo`) | Relatives of the test genome in training: the whole test phylum is absent from training. Phylum-identity shortcut: AUC is computed inside the held-out phylum only, so telling Bacillota from Pseudomonadota earns nothing. | Anything shared across phyla (convergent traits learned from other phyla are legitimate signal). Pretraining (section 3). | Run, depth 10, four phyla, 2,435 genomes. | Motility: Evo 2 macro AUC 0.545 [0.50, 0.58], per phylum 0.41 to 0.68. Near chance on motility, `shape_rod` and `oxygen_facultative`, **not on `oxygen_aerobe` (0.753)**. Table in 1a. |
| **`nearclade`** (leave-genus-out within phylum, AUC by taxonomic distance to the nearest training genome) | Same-genus near-copies. Reports how AUC decays as the nearest training relative gets more distant, which separates lookup from transfer. | A relative in the same family still lets a model look up. That is the point of the decay curve, not a flaw. | Run. | Evo 2 predicts within clade. **Numbers not in the repo** (see 1a). |
| **Taxonomy-prior baseline** (`tax_prior` in `nearclade`) | Diagnostic, not a fix: trait prevalence among training genomes in the nearest shared taxon (family, then order, class, phylum). Any model that cannot beat it has learned lookup, not biology. | | Run. | Taxonomy alone predicts as well as the genome. **Numbers not in the repo** (see 1a). |
| **Within-phylum and within-order shuffle nulls** | A model that has learned only phylum or order prevalence. Labels are permuted inside each phylum (`lopo`) or inside each phylum and each order (`nearclade`, order prevalence kept) and the model is refitted, so the null AUC includes whatever taxonomic structure a model can exploit. | Structure below order (family, genus) in the order-level null. | `lopo` phylum shuffle: run, 999 permutations. `nearclade` order shuffle: run, number of permutations unknown. | `lopo`: the null mean is 0.499 to 0.501 for every model on all four targets, so the refit pipeline does not manufacture signal. `nearclade` order-level null: **not in the repo**. |
| **Genus-cluster bootstrap** | Pseudo-replication in the uncertainty: genomes of one genus are near-copies, so a genome-level bootstrap is too narrow. Paired differences use the same resampled genera. | Does not change a point estimate. | `lopo`: run, 1,000 resamples. `nearclade`: 200 by default, result not in the repo. | `lopo` intervals are in table 1a. |
| **In-fold feature selection** (Pfam two-stage filter, `docs/phase2_pfam_spec.md` section 4) | Label information from the held-out phylum reaching feature choice. Both stages (prevalence, then Cochran-Mantel-Haenszel top-K) are fit on training phyla only, again inside every inner tuning fold, and refit on the permuted labels in the shuffle null. | The two branches define the confirmatory contrast differently (see 1a), so these results are descriptive until that is settled. | Implemented and unit-tested: the selector never sees the held-out phylum and is refit on permuted labels in the null (`tests/test_pfam_filter.py`); no genus crosses `nearclade` folds (`tests/test_embed_core.py`). Run on real labels in another session (999 permutations, all four targets). | Table 1a. Chosen K and stage-1 survivors per fold are in the result JSONs (motility: K = 1000, 300, 100, 300 and 9,947 to 11,187 survivors per held-out phylum). |
| **Inner leave-one-phylum-out tuning of C (and K)** | Tuning regularisation on the held-out phylum. C and K are chosen on the training phyla only. | | Run. | Chosen C per fold is in the result JSONs. Motility, Evo 2: 1, 0.1, 0.0001, 0.1 for Actinomycetota, Bacillota, Bacteroidota, Pseudomonadota. |
| **Intercept-free pooled scores** (`nearclade`) | A pooled-AUC artefact: fold intercepts encode training prevalence, which biases the pooled AUC of an uninformative feature below 0.5 (0.40 in a simulation, 0.48 after the fix). | | Run. | README, "Other traits and near-clade prediction". |

### 1a. Leave-one-phylum-out results (descriptive)

The `lopo` result JSONs for the four targets (run in a separate session, 999 within-phylum shuffles, 1,000 genus-cluster
resamples, depth 10, 2,435 genomes) were brought in from `origin/claude/optimistic-newton-6lf55q` at `5fa0250`:
`reports/tables/evo2_lopo_<target>_pfam.json` and `pfam_lopo_summary.tsv`. **The `nearclade` JSONs are still not in the
repository**, so the leave-genus-out decay curve and the taxonomy-prior baseline have no numbers here, and "taxonomy alone predicts as well as the genome" is not
checked by this audit.

**These numbers are descriptive.** The two branches record the confirmatory contrast differently: the Pfam spec on that
branch fixes "macro Pfam AUC above the within-phylum shuffle null, 999 permutations", and the spec on this branch records
`pfam` minus `kmer_genome` (D7). That conflict is unresolved, so nothing below is called confirmatory or a headline.

Macro mean over the four held-out phyla of the AUC inside each phylum, with the 95% genus-cluster bootstrap interval:

| target | Pfam | Pfam (Evo 2 windows) | Evo 2 | k-mer (whole genome) | k-mer (Evo 2 windows) | GC only |
|---|---|---|---|---|---|---|
| motility | 0.752 [0.711, 0.791] | 0.590 [0.556, 0.622] | 0.545 [0.502, 0.582] | 0.524 [0.483, 0.566] | 0.543 [0.499, 0.582] | 0.585 [0.536, 0.629] |
| shape_rod | 0.617 [0.559, 0.671] | 0.516 [0.459, 0.570] | 0.476 [0.419, 0.538] | 0.519 [0.448, 0.582] | 0.502 [0.441, 0.560] | 0.492 [0.431, 0.559] |
| oxygen_aerobe | 0.891 [0.861, 0.913] | 0.764 [0.731, 0.792] | 0.753 [0.711, 0.785] | 0.447 [0.410, 0.487] | 0.473 [0.435, 0.512] | 0.590 [0.552, 0.627] |
| oxygen_facultative | 0.597 [0.548, 0.659] | 0.520 [0.481, 0.561] | 0.507 [0.434, 0.561] | 0.641 [0.586, 0.692] | 0.622 [0.556, 0.675] | 0.563 [0.503, 0.610] |

Four plainly labelled comparisons, paired differences in macro AUC with 95% intervals (`pfam_lopo_summary.tsv`):

| target | Pfam vs shuffle null (macro AUC, null mean, p) | Pfam minus k-mer (whole genome) | Evo 2 minus Pfam | Evo 2 minus Pfam (Evo 2 windows) | Pfam (windows) minus k-mer (windows) |
|---|---|---|---|---|---|
| motility | 0.752, 0.501, 0.001 | +0.228 [+0.176, +0.278] | -0.208 [-0.263, -0.153] | -0.046 [-0.091, -0.002] | +0.048 [-0.009, +0.102] |
| shape_rod | 0.617, 0.500, 0.001 | +0.098 [+0.030, +0.173] | -0.141 [-0.215, -0.058] | -0.040 [-0.111, +0.032] | +0.014 [-0.048, +0.080] |
| oxygen_aerobe | 0.891, 0.501, 0.001 | +0.444 [+0.393, +0.486] | -0.138 [-0.171, -0.110] | -0.011 [-0.046, +0.021] | +0.291 [+0.238, +0.337] |
| oxygen_facultative | 0.597, 0.501, 0.001 | -0.044 [-0.117, +0.045] | -0.090 [-0.194, -0.011] | -0.013 [-0.101, +0.057] | -0.103 [-0.169, -0.024] |

p = 0.001 is the smallest value 999 permutations can give.

Two things in these numbers bear on the audit. First, the earlier summary "Evo 2 drops to chance across phyla" holds for motility
(0.545), `shape_rod` and `oxygen_facultative`, but not for `oxygen_aerobe` (0.753), and on motility Evo 2 is above chance in
Bacillota (0.675 [0.62, 0.73]) and below it in Actinomycetota (0.407 [0.30, 0.51]), so a macro mean hides large differences between
phyla. Second, GC content alone (0.585) is ahead of Evo 2 on motility, which is what a simple composition baseline is there to show.
Neither point is a leakage test; both are context for reading the table.

## 2. Occupancy nulls (the same idea, applied to the map)

| control | leakage it closes | result |
|---|---|---|
| **Permutation nulls: global, phylum, class, order** (`reports/attrition_report.md` section 3) | The deficit of occupied cells is lineage structure, not constraint. Traits are fixed in different clades, which a global shuffle cannot see. | Core map, 216 cells: 90 occupied against 117.3 expected under independence (deficit 27.3, P = 0.0001). The expected count falls and the deficit shrinks as the null is restricted: phylum 109.3 (19.3), class 102.6 (13.6), order 87.9 (4.9, P = 0.067). Source: `reports/tables/occupied_vs_expected.tsv`. |
| **Small-strata handling** | Dropping small taxa from a stratified shuffle can manufacture empty cells. They are excluded and counted, and an excluded-only empty cell is labelled as such. | 157 of 2,580 species are excluded at order level (`occupied_vs_expected.tsv`). |
| **Positive control** | Checks the method flags a known constraint and attributes it to lineage. | Gram-negative, endospore-forming cocci: flagged by the global null (expected 11.2, observed 0, q = 0.0005), absorbed by the stratified nulls. |
| **Evolutionary null** (Mk models on the GTDB v232 tree, homogeneous and hidden-rate; `reports/evolutionary_null_report.md`) | Permutation nulls treat species as exchangeable inside a taxon. The Mk null simulates independent trait evolution along the actual tree. | 0 cells flagged under either model, 0 robust. 60 of 72 cells are untestable under the homogeneous null (67 under the hidden-rate null), so most absences cannot be tested at all. 29 cells are flagged by some permutation null, and none survives the evolutionary null. |
| **Calibration of the evolutionary null** | Whether the pipeline produces false flags. | 0 of 48 simulated datasets with independent traits had a false flag (24 homogeneous, 24 heterogeneous rates; 300-tip subtrees). |
| **Power** | What the order-level null could not have detected. | A truly empty cell is guaranteed to reach q < 0.05 only if about 7 or more species are expected there (`reports/tables/minimum_detectable_constraint.tsv`: 6.9 at order level). |

## 3. Pretraining leakage: were the panel genomes in Evo 2's training data?

None of the controls above can close this. If Evo 2 saw a panel genome, or a near-copy of it, during pretraining, its
embedding can carry what it memorised and a leave-one-phylum-out split of our labels does not remove that.

**What the OpenGenome2 metadata does and does not say.** The Hugging Face repo `arcinstitute/opengenome2` (revision
`84d2a7e6`, 2026-09-02) has one small manifest, `species_metadata.csv` (15 MB, 76,070 rows). Its prokaryote rows are
28,177 genomes labelled `GTDB_v220`, one per species. By the Evo 2 methods (as reproduced in the NVIDIA BioNeMo docs),
the GTDB part of the training data is the **release 214.1 representative genomes (85,205) plus 28,174 genomes added from
release 220**: species whose cluster had no 214.1 genome. So `species_metadata.csv` lists only the 220 additions. The
214.1 base set is in the 84 GB `fasta/gtdb_v220/v214` shards and is not in the metadata. Matching the panel against
`species_metadata.csv` alone finds **67 of 2,435** genomes (2.8%), which would wrongly suggest almost nothing was seen.

**Method.** The 214.1 representatives were taken from GTDB's public `sp_clusters_r214.tsv`, not from the OpenGenome2
shards. No FASTA or JSONL shard was downloaded. The training set is the union of the 214.1 representatives and the
metadata's `GTDB_v220` rows (112,003 genomes). Each panel genome (GTDB v232 accession, matched on assembly number and
version, ignoring the RS_/GB_ and GCA/GCF prefixes) is placed in one of three tiers (`src/pretraining_overlap.py`):

- **genome_seen**: the genome is itself in the training set.
- **species_seen**: not itself in the training set, but its v232 species cluster contains a genome that is (a same-species genome, at least 95% ANI, was trained on).
- **unseen**: no genome of its v232 species cluster is in the training set.

**Result (2,435 embedded four-phylum genomes).** Table: `reports/tables/opengenome2_overlap_by_phylum.tsv`.

| phylum | n | genome_seen | species_seen | unseen | unseen share |
|---|---|---|---|---|---|
| Actinomycetota | 481 | 354 | 58 | 69 | 14.3% |
| Bacillota | 732 | 611 | 73 | 48 | 6.6% |
| Bacteroidota | 409 | 337 | 35 | 37 | 9.0% |
| Pseudomonadota | 813 | 681 | 73 | 59 | 7.3% |
| **all** | **2,435** | **1,983 (81.4%)** | **239 (9.8%)** | **213 (8.7%)** | |

So **81% of the genomes Evo 2 embeds for us were in its training data, and 91% have a same-species genome in it.** The
split is similar for motile and non-motile genomes (unseen 8.4% and 9.0%; `reports/tables/opengenome2_overlap_by_trait.tsv`).
By other traits the unseen genomes are not a random sample: they are 5 of 296 anaerobes (1.7%), 1 of 107 thermophiles, 0 of 24
psychrophiles, but 190 of 1,741 aerobes (10.9%). They are mostly species described since release 220.

**Caveats.**
- The 214.1 base set is assumed complete because the Evo 2 methods say all 214.1 representatives were used. It is not
  checked against the `v214` shards.
- The published total is 113,379 genomes (85,205 + 28,174). The union here is 112,003, which is 1,376 fewer. The cause is that
  1,379 of the 28,177 metadata rows are also 214.1 representatives. That is not explained (renamed species account for only 97
  of them), and the published total may simply count those genomes twice. It cannot change a tier, since a genome is
  `genome_seen` if either list holds it. The 28,177 rows also differ by 3 from the 28,174 in the methods.
- Only the GTDB part of the corpus is checked. The metagenome part (about 46% of phase-1 tokens) can contain the same
  species and cannot be checked from metadata, so `unseen` means unseen in the GTDB data, not unseen by the model.
- `unseen` is a species-level statement. 90% of the unseen genomes belong to a genus that was trained on (99% of all panel
  genomes do), so a nearby relative was still seen. That matches the phylogenetic-leakage concern and is not a flaw in the
  tiering.
- Accessions, not sequences, were compared. Two assemblies of the same strain with different accessions are `species_seen`.

**Sensitivity analysis (proposed, not run).** Score Evo 2 on test genomes of the `unseen` tier only, with the same
leave-one-phylum-out training as now, and compare with its score on all test genomes. If Evo 2 relies on memorised
genomes the unseen score should fall. The k-mer baselines never saw the corpus, so the same seen/unseen difference for
k-mers shows how much of any change is just the unseen genomes being a different kind of genome. Whether it is
feasible, for each target, among the unseen genomes (`reports/tables/opengenome2_unseen_feasibility.tsv`):

| target | verdict | why |
|---|---|---|
| motility | feasible, wide intervals | every phylum has at least 10 of each class (motile/non-motile: Actinomycetota 11/58, Bacillota 32/16, Bacteroidota 10/27, Pseudomonadota 23/36), spread over 26-51 genera. With about 10 positives an AUC has a standard error near 0.08, so only a large drop is detectable. |
| `oxygen_aerobe`, `oxygen_facultative` | not feasible | the unseen genomes are almost all aerobes (Bacteroidota 37 of 37) and have 0-8 facultative genomes per phylum. |
| `shape_rod` | not feasible | 1-19 non-rods per phylum (Bacteroidota 1). |

Using "not genome_seen" (unseen plus species_seen, 452 genomes) roughly doubles the motility counts, and is a defensible
secondary definition. The contrast and the definition must be written into a spec and committed before anything is run, as Phase 2 did. The run
needs a CPU job on Modal (the embeddings are already computed), so it needs your go-ahead.

**Erik Hartman's k-mers.** They would help only in one case: if they are per genome, keyed by accession, and cover the
GTDB part of OpenGenome2. Then they would check the tiering directly, including the 214.1 base set this audit infers, and
could also give a k-mer baseline on the exact training composition. If they are aggregated over the whole corpus or over
shards, they would not help with either. Worth asking him: which k, whether the unit is genome, species or shard, whether
GTDB genomes are keyed by accession, and whether the metagenome shards are included.

**Other point to keep in mind.** The OpenGenome2 README says the `jsonl` layout carries phylogenetic tags (used in phase-2
training). This audit does not test whether Evo 2's embeddings of untagged windows encode taxonomy more strongly because of
that.

## Reproducing section 3

```bash
# inputs (about 90 MB; the OpenGenome2 shards are not touched)
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("arcinstitute/opengenome2", "species_metadata.csv", repo_type="dataset", local_dir="og2")
PY
curl -O https://data.gtdb.ecogenomic.org/releases/release214/214.1/auxillary_files/sp_clusters_r214.tsv
curl -o sp_clusters_v232.tsv https://data.gtdb.ecogenomic.org/releases/latest/auxillary_files/sp_clusters.tsv
python -m src.pretraining_overlap --og2-metadata og2/species_metadata.csv \
    --r214-clusters sp_clusters_r214.tsv --clusters sp_clusters_v232.tsv
```

| input | sha256 |
|---|---|
| `species_metadata.csv` | `49ea43a34e567de7b3e98d6aca948f4929489bd4655c9f0c58d7a0d309095f48` |
| `sp_clusters_r214.tsv` (GTDB 214.1) | `b81e1b8661f119b649db6aa5da99e0e16b4a2742722f17f8f2dcfdbdc3523233` |
| `sp_clusters_v232.tsv` (GTDB latest = v232, downloaded 2026-10-07) | `96414cdc04addaac0280593f3d32a7c8c30cbc2aa498a9fee81886e478ec9fc9` |

Outputs: `reports/tables/opengenome2_overlap_genomes.tsv` (one row per core-panel genome, 2,580), `_by_phylum.tsv`, `_by_trait.tsv`,
`opengenome2_unseen_feasibility.tsv`.
