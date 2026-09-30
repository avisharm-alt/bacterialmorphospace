# bacterialmorphospace

Stage 1 of an occupancy map of bacterial trait space. The question here is not modelling but feasibility.
How many bacterial strains have **both** a complete panel of curated phenotype traits (BacDive) **and** a
matched sequenced genome (GTDB/NCBI)? Which trait combinations are empty **beyond what the marginal trait
frequencies predict**?

The deliverable is [`reports/attrition_report.md`](reports/attrition_report.md).

**Result (BacDive API v2 snapshot of 2026-09, GTDB v232).** Bacterial trait space is sparser than independence predicts, and that
sparsity is almost entirely phylogenetic.
- **Core map.** 2,580 species occupy 90 of 216 six-trait cells, where independent traits would fill 117. The deficit shrinks from 27 to
  5 cells as the permutation null is restricted to within phylum, class and order, and it is not significant at order level (P = 0.07).
- **No novel constraints.** No trait combination is empty beyond what lineage structure explains.
- **Power.** The order-level null could have detected a constraint that empties a combination expected to hold about 7 or more species.
- **Scope.** Six coarse traits, and type strains only.

## What it does

```
BacDive API v2  ──fetch (cached, resumable)──►  data/raw/bacdive/*.json.gz
GTDB bac120 metadata (latest) ────────────────►  data/raw/gtdb/
          │
     extract   8 traits → raw + normalised columns, conflicts, bin sources ─► data/interim/bacdive_traits.parquet
          │
       join    BacDive strain ↔ GTDB genome (accession, then normalised designation) ─► data/final/strains.parquet
          │
      report   attrition, coverage, power, occupancy + permutation null ─► reports/
```

| Module | Role |
|---|---|
| `src/fetch.py` | BacDive ID sweep (100 IDs per call, every response cached, restart skips cached batches); GTDB download + version |
| `src/traits.py` | Field paths and extraction for the 8 traits; keeps every raw observation |
| `src/normalise.py` | Strain-designation normalisation (unit-tested in `tests/test_normalise.py`) |
| `src/join.py` | Genome matching, unmatched log, before/after-normalisation match rates |
| `src/analysis.py` | Panel flags, dereplication, occupancy, permutation null, BH correction |
| `src/report.py` | The markdown report, TSV tables, manifest, accession lists |
| `config.toml` | **All** bins, value mappings, panels, matching rules and analysis settings |

## Credentials

**None are needed.** Since February 2026 the BacDive API needs no registration or login. This was
verified against <https://api.bacdive.dsmz.de/> in September 2026. GTDB files are public.

Optionally, copy `.env.example` to `.env` and set `BACDIVE_CONTACT_EMAIL`, or a full `BACDIVE_USER_AGENT`.
It is only used to build a polite User-Agent so DSMZ can contact you if a crawl misbehaves. `.env` is gitignored.

## Re-running from scratch

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                      # ~150 unit tests, no network
python -m src.pipeline all            # fetch → extract → join → report
```

- **First run:** about 10 minutes on a normal connection. That covers roughly 2,500 BacDive calls
  (IDs 1–250,000 in batches of 100) and one 290 MB GTDB download.
- **Re-runs** make no network calls. Every BacDive batch is cached as
  `data/raw/bacdive/fetch_<start>_n<count>.json.gz`, and GTDB is cached in `data/raw/gtdb/`.
- **Interrupted crawl:** run the same command again. Only uncached batches are requested.

The stages can also be run on their own. This is useful when iterating on downstream logic:

```bash
python -m src.pipeline fetch      # --max-new-batches N for a partial test crawl; --refresh-gtdb to re-download
python -m src.pipeline extract    # refuses to run on an incomplete sweep unless --allow-incomplete
python -m src.pipeline join
python -m src.pipeline report
```

To pull a **new BacDive or GTDB snapshot**, delete `data/raw/bacdive/` (or `data/raw/gtdb/`, or pass `--refresh-gtdb`) and
re-run `all`. Versions are recorded in `reports/run_manifest.json` and at the top of the report:
- the BacDive fetch dates and the release stamp in record DOIs (BacDive exposes no release number)
- the GTDB release from `VERSION.txt`, plus the metadata file's sha256
- the git commit and the config sha256

### Gitignored outputs: regeneration and checksums

`data/final/strains.parquet` (the joined table, about 11 MB) and `data/interim/unmatched_strains.tsv` (the unmatched log,
about 9.7 MB) are not committed. Regenerate them from the cache with:

```bash
python -m src.pipeline all      # or, if data/raw is already populated: python -m src.pipeline extract && python -m src.pipeline join
sha256sum data/final/strains.parquet data/interim/unmatched_strains.tsv
```

Snapshot: BacDive API v2 (record DOIs stamped `20260601`), GTDB v232,
code `c3015ed` (unchanged since `b44b2fb`).

| file | rows | sha256 (file bytes) | sha256 (row contents) |
|---|---|---|---|
| `data/final/strains.parquet` | 102,187 | `25dafa183111b9e6fe1fab8422d537e1183337d400a92a94af7db7dc083c4a7f` | `1a57137ec2a8148e658ce90169132e2d73f1a869cda789ff5b16d1b735dfdd2a` |
| `data/interim/unmatched_strains.tsv` | 79,352 | `07b30b67a07bdabff132698a4f96630585d4e4957a100debf9c5f612e0b84c7a` | `ba59b3c4f3a72a287f02b9d162b599beff0503397d4eb59d5802dbc4f84b3c17` |

Regenerating twice from the same cache gave byte-identical files. Parquet bytes can still differ across pandas or pyarrow
versions, so the row-content hash is the version-independent check. It is the sha256 of
`pandas.util.hash_pandas_object(table.astype(str), index=False)`. Every run writes both hashes to
`reports/run_manifest.json` → `outputs`, and that file is authoritative for the run that produced it.

### BacDive crawl notes

- There is no list-all endpoint, so the crawl sweeps the ID space. `/v2/fetch/{id;id;…}` drops non-existent IDs silently.
- The ID space has **large gaps**. In the 2026 snapshot, IDs 24,901–99,900 are empty. For that reason there is no
  "stop after N empty batches" rule, because it would silently truncate the crawl. The sweep always runs to `[bacdive].max_id`.
  A warning fires if strains turn up within `warn_if_max_found_within` of the cap.
- `extract` refuses to run on an incomplete sweep.
- Genome-based predictions are never requested (`?predictions=1` is not used). Extraction also fails loudly if a record
  carries a non-empty predictions section.

## Traits and bins

| Trait | BacDive v2 path (under `results["<id>"]`) | Coarse bins (primary) | Fine bins (sensitivity) |
|---|---|---|---|
| Gram stain | `Morphology › cell morphology › gram stain` | negative / positive | same |
| Cell shape | `Morphology › cell morphology › cell shape` | rod / coccus / other | rod / coccus / spiral / filamentous / other |
| Motility | `Morphology › cell morphology › motility` | no / yes | same |
| Spore formation | `Physiology and metabolism › spore formation › spore formation` | no / yes | same |
| Oxygen tolerance | `Physiology and metabolism › oxygen tolerance › oxygen tolerance` | aerobe / facultative (+ microaerophile, flagged) / anaerobe | aerobe / facultative / microaerophile / anaerobe |
| Temperature | `Culture and growth conditions › culture temp` | psychro < 15 / meso 15–45 / thermo ≥ 45 °C | + hyperthermo ≥ 80 |
| pH (column only) | `Culture and growth conditions › culture pH` | acido < 5.5 / neutro 5.5–8.5 / alkali ≥ 8.5 | same |
| Halophily (column only) | `Physiology and metabolism › halophily › halophily level` | curated category as given | NaCl growth tests kept raw in `nacl_tests_raw` |

Two panels are analysed:
- the **core panel**: the first six traits (216 coarse cells)
- the **no-spore panel**: the same without spore formation (108 cells). Spore formation is the coverage bottleneck, and
  dropping it more than doubles N.

pH (about 7% coverage) and curated halophily (under 0.5%) are kept as columns but excluded from the occupancy analysis.

### Column conventions

Every trait `X` in `data/final/strains.parquet` has these columns:

- `X_raw`: every observation as JSON, including its BacDive reference id. Nothing is dropped in cleaning.
- `X_fine`, `X`: the fine and coarse normalised values.
- `X_fine_conflict`, `X_conflict`: true when references disagree at that level. The normalised value is then
  **null**; no value is chosen or voted.
- `X_n_obs`: the number of observations.
- Temperature and pH also have:
  - `X_bin_source`: `optimum`, `growth_range`, `growth_single_point` or `none`.
  - `X_opt`, `X_growth_min`, `X_growth_max` and `X_point`, the value that was binned.
- Oxygen also has `oxygen_microaerophile_folded`.

Unmapped raw values (for example "Gram-variable" or "aerotolerant") become null for gram and oxygen, and "other" for shape.
All are counted in `data/interim/unmapped_values.tsv`.

## Genome matching

1. **Accession.** BacDive's own `INSDC accession` (GCA, unversioned) is matched to GTDB `ncbi_genbank_assembly_accession`.
   A listed assembly missing from GTDB (failed GTDB QC, or not yet included) is reported as its own attrition line.
2. **Designation.** Exact equality of normalised designations, from BacDive's culture-collection numbers,
   strain designation and DSM number against GTDB `ncbi_strain_identifiers`. Designations are split on `, ; / | =`.
   A match on **any** designation counts. Designations with a known culture-collection code
   (`DSM 20231`) are accepted as they are. Bare ones (`BS 107`) also need **taxon agreement**. That means the BacDive genus
   equals the GTDB or NCBI genus of the genome, or a BacDive NCBI tax ID equals the genome's `ncbi_taxid` or `ncbi_species_taxid`.
   Short bare designations (`LP1`, `B86`) recur across unrelated genera, and this rule rejects those collisions. The tax-ID part
   recovers strains whose genus was renamed.
3. **No fuzzy matching.**
4. **Choosing among candidates.** When several genomes qualify, one is chosen deterministically: genus agreement,
   then GTDB representative, then RefSeq, then assembly level, then CheckM2 quality.
5. **Logging.** Every unmatched strain is written to `data/interim/unmatched_strains.tsv` with the keys tried and a reason.

The normalisation rules are documented in the `src/normalise.py` docstring and pinned by `tests/test_normalise.py`.
In summary: split, fold case and accents, drop type-strain markers (`T`, `^T`, `(T)`), fold aliases (`DSMZ → DSM`,
`NCIB → NCIMB`), drop separators, and strip leading zeros after a collection code.

`ncbi_assembly_accession` in the final table is the accession GTDB uses: RefSeq `GCF_…` where available, otherwise GenBank
`GCA_…`. The GenBank accession is always in `assembly_genbank`. To download genomes:

```bash
datasets download genome accession --inputfile data/final/assembly_accessions_core_panel_species.txt
datasets download genome accession --inputfile data/final/assembly_accessions_no_spore_panel_species.txt
```

## Occupancy analysis

- **Primary unit:** one strain per GTDB species. Type strains are preferred, then the best assembly.
- **Sensitivities:**
  - no dereplication
  - temperature from a reported optimum only
  - fine bins
  - one strain per GTDB genus
- **Nulls.** Four are run on every full map, and their per-cell q-values are reported side by side.
  - **Global:** each trait column is permuted across all species. This preserves every marginal but ignores phylogeny, so it
    flags combinations that are rare only because the traits are fixed in different clades.
  - **Stratified (phylum, class, order):** each trait column is permuted only among species of the same GTDB taxon. The expected
    count per cell is Σ over taxa n·∏ p(level within the taxon).
  - **Order-level survivors are the only candidates.** A cell that fires globally but not at order level is phylogenetic structure.
- **Small strata.** Taxa with fewer than `min_stratum_size` (default 5) species are excluded from that null and counted, never
  silently kept. The exclusion can never create an empty cell: "empty beyond chance" requires the cell to be empty among
  *all* species. Cells emptied only by the exclusion are labelled as such.
- **Tests.** For each cell and null the report gives the observed count, the expected count, the one-sided permutation p-value
  for "emptier than chance", and a BH q-value over *testable* cells (expected ≥ `min_expected_for_test`, default 3). The global
  and order nulls are re-run with an independent seed.
- **Positive control.** A known constraint is configured in `config.toml` (`[[analysis.positive_controls]]`): Gram-negative
  endospore-forming cocci. The global null should flag it and the order null should not. The report checks this explicitly.
- **Survivor diagnostic.** Cells that survive the order shuffle are re-tested with a family-level shuffle
  (`survivor_diagnostic_levels`). This is a diagnostic only; it is not part of the survival criterion.
- **Power.** `detectable_zero()` gives the smallest expected count E at which a truly empty cell must reach q < α. With m testable
  cells, BH guarantees this when P(null count = 0) ≤ α/m, which is about E ≥ ln(m/α) under Poisson. The reported minimum
  detectable constraint is the largest of that bound, the empirical permutation threshold and the testability floor.
- **Also reported:** occupied vs expected cells under every null, per-phylum occupancy, and the expected species per cell
  for every map size.

## Evo 2 genome embeddings on Modal

`src/evo2_modal.py` embeds the core-panel genomes with Evo 2 7B (layer `blocks.28.mlp.l3`, 4096-dim). It lives
beside the pipeline and reads `data/final/core_panel_species.tsv`. Pure logic (window placement, sampling, download
checks, the classifier) is in `src/embed_core.py` and is unit-tested without a GPU (`tests/test_embed_core.py`).
`config.toml` is deliberately untouched, because the run manifest hashes it.

```bash
source ~/venvs/modal/bin/activate
modal run -m src.evo2_modal::sweep --dry-run              # sample + cost estimate, spends nothing
modal run -m src.evo2_modal::sweep                        # sampling-depth experiment
modal run -m src.evo2_modal::embed_all --n-windows 25     # production run; resumable
```

**Recipe.** Evo 2 7B does not need transformer-engine (TE), but a *partially* installed TE raises a `RuntimeError` that
evo2's `except ImportError` misses (evo2 issue #201); a fully *absent* TE gives `HAS_TE = False`. So the image is a plain
`nvidia/cuda:12.8.1-devel` + Python 3.11 base (not NGC, which ships TE), `evo2==0.3.0` and `vtx==1.1.0` go in with
`--no-deps`, torch is 2.8.0 (cu128), and flash-attn is the prebuilt `v2.8.3` wheel for `cu12torch2.8cxx11abiTRUE-cp311`.
The image build fails if any transformer-engine distribution is present, and `verify_env` asserts `HAS_TE is False` on a GPU.
The model must be `evo2_7b_base` (8k context), the only config without FP8 input projections.

**Storage.** Volume `evo2-hf-cache` holds the ~14 GB weights (`HF_HOME`). Volume `evo2-embeddings` holds
`genomes/<acc>.fna.gz` and `embeddings/<acc>__n<N>.npy` (4096 float32), plus `__windows.npy` and `__meta.json`
(per-window vectors and GPU timings) for sweep genomes. A genome is skipped when its `.npy` exists, and each file is
written atomically, so an interrupted run resumes where it stopped. Fetch results with
`modal volume get evo2-embeddings embeddings ./evo2_embeddings`.

**Downloads** run on CPU containers, not the GPU. Every file must exceed 20 kB, match `Content-Length`, match NCBI's
`md5checksums.txt` and decompress to FASTA. Nothing is inferred from an exit status. If the RefSeq accession is gone, the
GenBank accession is tried.

**Windows.** `n` windows of 8,192 bp, drawn across **all** contigs of at least 8,192 bp, not only the longest. Sampling
from the longest contig alone would embed one fragment of a draft assembly, and how fragmented an assembly is tracks how well
studied the organism is, which can correlate with the traits. The usable contigs are laid end to end and `n` points are spaced
evenly along that axis, so each contig gets windows in proportion to its length. Each window sits inside its own contig, and none
spans a junction. Windows are mean-pooled within a window and then across windows. The sweep embeds one pool of `max(depths)`
windows per genome and takes each smaller depth as an evenly spaced subset of it. That is 100 forward passes per genome instead
of 185, and it makes the depths a paired comparison. A subset window sits within 0.5% of the usable length of where an
independent draw would put it. Each genome's meta file records `sampling`, the windows, and contig statistics, and the sweep
refuses to mix checkpoints from different sampling schemes.

**Sweep.** About 200 genomes, stratified across GTDB phyla by cap-and-fill (phyla with <10 species in the panel are pooled
into `other`). It reports `roc_auc` for `motility`
with `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))` under `GroupKFold` grouped by pooled phylum,
alongside s/genome and the projected hours and dollars for 2,580. Outputs: `reports/tables/evo2_sampling_depth_sweep.tsv`,
`evo2_sweep_genomes.tsv` and `evo2_sweep_selected_depth.json` (smallest depth within one SE of the best mean AUC;
`embed_all` reads it when `--n-windows` is omitted). Both entry points take `--max-usd` and stop cleanly, with checkpoints
kept, when the running estimate exceeds it. Caveats to read the result with:
- `frac_overlapping` is the share of genomes whose total usable sequence (all contigs >= 8,192 bp) is shorter than `n`
  windows, so their windows must overlap. On the 100-genome default sample it is 0% at every depth (median 3,975 kb usable
  across 15 contigs). Drawing from the longest contig alone would have given 46% at depth 100.
- Some folds can be single-class; those are skipped and counted in `folds_scored`.
- A few windows contain scaffold gaps (7 of 10,000 windows exceed 5% ambiguous bases in the default sample). They are
  embedded as they are; the worst window per genome is in its meta file.

### Motility, leave-one-phylum-out (depth 10, four testable phyla)

```bash
# 1. embed the 2,436 genomes of Pseudomonadota, Bacillota, Actinomycetota and Bacteroidota (~2,400 still to do, ~$13)
caffeinate -i modal run -m src.evo2_modal::embed_all --n-windows 10 --max-usd 16 \
    --phyla Pseudomonadota,Bacillota,Actinomycetota,Bacteroidota
# 2. CPU only: sequence features, then hold out each phylum in turn and score AUC WITHIN it
modal run -m src.evo2_modal::lopo
```

The evaluation holds out one phylum, trains on the other three, and scores AUC inside the held-out phylum, so it
measures within-phylum signal on an unseen clade (a pooled fold AUC mostly ranks phyla against each other, which is what
made the first sweep uninformative). Models, all on the same genomes: Evo 2 depth-10 embedding; canonical tetranucleotide
frequencies of the whole genome (136 features); the same frequencies over only the windows Evo 2 saw; genome GC; and the
constant-score floor, whose within-phylum AUC is exactly 0.5. Regularisation `C` is tuned for every model by an inner
leave-one-phylum-out over the training phyla, so a 4096-d and a 136-d model are each regularised for their own size; the
untuned `C=1` pipeline is reported too. Intervals are a genus-cluster bootstrap (genomes of one genus are near-copies),
differences are paired on the same resamples, and the shuffle null permutes labels within each phylum and refits.

`reliability --within-phylum` repeats the depth study centred within phylum: ordinary reliability is dominated by
between-phylum differences, but a within-phylum comparison depends on the much smaller differences among genomes of one phylum.

### Other traits and near-clade prediction (CPU only, existing depth-10 embeddings)

```bash
# leave-one-phylum-out for another target: motility (default), oxygen_aerobe, oxygen_facultative, shape_rod
modal run -m src.evo2_modal::lopo --target oxygen_aerobe --n-boot 200

# near-clade: leave-genus-out inside each phylum, AUC by taxonomic distance to the nearest training genome
modal run -m src.evo2_modal::nearclade --target motility --n-boot 200
```

`nearclade` puts genera into 10 folds per phylum (a genus is never in both train and test) and trains each model on
the other genera of the same phylum, with `C` tuned by an inner genus-grouped CV. For each held-out genome it records the
lowest taxonomic rank (family, order, class, phylum) it shares with any training genome, and reports pooled AUC by that
distance. Family comes from GTDB's `bac120_taxonomy.tsv.gz`, cached in `data/raw/gtdb/`; it is used only where the file
agrees with the panel on genus, order, class and phylum. Distance bins are merged until each holds enough genomes of both
classes, and a phylum's cell is scored only if it meets that rule itself. Also reported: how many genera can be scored at
all (both classes present), a within-genus AUC from same-genus pairs only, the trivial floor (0.5), a taxonomy-prior
baseline (trait prevalence in the nearest shared taxon of the training genomes), and shuffle nulls that permute labels
within each phylum and within each order (order-level prevalence kept).

Pooled scores use intercept-free decision values. A fold's intercept just encodes its training prevalence, which shifts
between folds and, once folds are pooled, biases the AUC of an uninformative feature below 0.5 (0.40 in a simulation
against 0.48 after the fix). Within one fold or phylum a constant offset changes no AUC, so the earlier
leave-one-phylum-out numbers are unaffected.

## Layout

```
src/            pipeline modules
tests/          unit tests (normalisation, traits, join, analysis)
config.toml     every tunable choice
data/raw/       cached API responses + GTDB download       (gitignored)
data/interim/   extracted traits, unmatched log (gitignored), unmapped values, snapshot info
data/final/     strains.parquet (joined table, gitignored), per-panel species tables, accession lists
reports/        attrition_report.md, tables/*.tsv, run_manifest.json
notebooks/      exploration only; nothing load-bearing
```
