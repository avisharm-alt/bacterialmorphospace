# BacDive × GTDB: attrition and trait-space occupancy

_Generated 2026-09-29T05:15:44+00:00 · code `b44b2fbc37b09dea1b4c23a26eff523f7a93db59` · config sha256 `96c49f9e4f4a`_

## Snapshot

| source | version | retrieved | detail |
|---|---|---|---|
| BacDive | API v2, record DOIs stamped `20260601` (BacDive exposes no release number) | 2026-09-29 → 2026-09-29 | IDs 1–250,000 swept in full (2,500 batches of 100); highest ID found 179,194; genome-based predictions not requested |
| GTDB | v232 (Released Apr 15, 2026) | 2026-09-29 | bac120_metadata.tsv.gz, 878,998 genomes, sha256 `1650d3164666` |

Empty BacDive ID ranges ≥1,000 IDs: 24,901–99,900, 128,801–130,100, 179,201–250,000.

## Headline

- **Data.** 102,187 BacDive strains. 3,414 have the 6-trait core panel, 2,630 of them with a GTDB genome, giving **2,580 species**. Dropping spore formation gives the 5-trait no-spore panel: **5,634 species** (2.2×) on half the cells (108 vs 216).
- **Positive control passed.** Gram-negative endospore-forming cocci is a known phylogenetic constraint, not a finding. The global null flags it (expected 11.2, observed 0). Under the order-level shuffle it is untestable (expected 0.5), as it should be once the null knows which clades carry which traits (§7).
- **The occupancy deficit is mostly phylogenetic.** Core panel: 90 cells occupied vs 117 expected under the global null (P = 0.0001). The phylum, class and order shuffles give 109, 103 and 88 (order: 83 observed among kept species, P = 0.067). The no-spore panel behaves the same (69 vs 78 global; 66 vs 69 at order level, P = 0.11) (§8).
- **core panel (6 traits): 0 cells survive the order-level shuffle.** 4 of the 4 cells flagged by the global null do not survive at order level: phylogenetic structure (§10).
- **no-spore panel (5 traits): 1 cell survives the order-level shuffle.** The global null flags 1 cell, all of which also survive at order level (§10).
  - `gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso`: expected 5.4 under the order shuffle, observed 0, q = 0.011 (independent seed 0.0098). Fragile: optimum-only: untestable; genus-level: not significant, q = 0.43. Diagnostic beyond the three requested ranks: a family-level shuffle makes it untestable (expected 1.6). 1 of all 102,187 BacDive strains has the combination. **Treat as family-level structure until shown otherwise (§11).**
- **Samples are type strains.** 98.8% of the core-panel species are represented by a type strain (§12).

## 1. Attrition

Counts are strains. "Usable" means a normalised value exists: at least one observation, mappable to a bin, and no conflict between references (conflicting observations null the value rather than picking one).

| stage | strains | % of BacDive | with GTDB genome | % with genome | GTDB species | note |
|---|---|---|---|---|---|---|
| 1. BacDive strains retrieved | 102,187 | 100.0% | 22,835 | 22.3% | 18,248 |  |
| 2. ≥1 of the 8 traits populated (raw) | 51,202 | 50.1% | 18,334 | 35.8% | 14,459 |  |
| 3. gram stain — usable value | 15,514 | 15.2% | 9,514 | 61.3% | 8,522 |  |
| 3. cell shape — usable value | 14,802 | 14.5% | 8,887 | 60.0% | 7,933 |  |
| 3. motility — usable value | 13,809 | 13.5% | 8,736 | 63.3% | 7,821 |  |
| 3. spore formation — usable value | 5,424 | 5.3% | 4,254 | 78.4% | 4,099 |  |
| 3. oxygen tolerance — usable value | 21,855 | 21.4% | 9,510 | 43.5% | 8,400 |  |
| 3. temperature — usable value | 50,310 | 49.2% | 18,117 | 36.0% | 14,334 | bin from optimum, else growth range, else single growth point |
| 3. pH — usable value | 6,740 | 6.6% | 5,298 | 78.6% | 5,201 | outside the analysed panels |
| 3. halophily (curated level) — usable value | 361 | 0.4% | 247 | 68.4% | 245 | curated category only; NaCl tests kept as a column |
| 3. (NaCl growth tests present, raw column only) | 9,677 | 9.5% | 6,947 | 71.8% | 6,594 | not binned; kept in nacl_tests_raw |
| 4. morphology panel (4) complete | 4,047 | 4.0% | 3,126 | 77.2% | 3,053 | gram + shape + motility + spore |
| 5. core panel (6) complete | 3,414 | 3.3% | 2,630 | 77.0% | 2,580 | gram + shape + motility + spore + oxygen + temperature |
| 5a. core panel (6), temperature from optimum only | 2,680 | 2.6% | 2,189 | 81.7% | 2,168 |  |
| 5b. no-spore panel (5) complete | 10,055 | 9.8% | 6,184 | 61.5% | 5,634 | gram + shape + motility + oxygen + temperature |
| 5c. original 8-trait panel (core + pH + curated halophily) | 153 | 0.1% | 115 | 75.2% | 114 | for reference; not analysed |

### 1a. Per-trait losses

| trait | any raw observation | usable (normalised) | lost to conflict | no binnable value (unmapped, unparsed, or only negative/qualitative entries) | usable with genome |
|---|---|---|---|---|---|
| gram stain | 15,606 | 15,514 | 26 | 66 | 9,514 |
| cell shape | 15,181 | 14,802 | 379 | 0 | 8,887 |
| motility | 14,058 | 13,809 | 249 | 0 | 8,736 |
| spore formation | 5,443 | 5,424 | 19 | 0 | 4,254 |
| oxygen tolerance | 23,254 | 21,855 | 1,390 | 9 | 9,510 |
| temperature | 50,393 | 50,310 | 75 | 8 | 18,117 |
| pH | 6,794 | 6,740 | 5 | 49 | 5,298 |
| halophily (curated level) | 364 | 361 | 3 | 0 | 247 |

Unmapped and unparsed raw values (all kept in the `*_raw` columns; full list in `data/interim/unmapped_values.tsv`):

| kind | trait | value | count |
|---|---|---|---|
| unmapped | gram | variable | 135 |
| unmapped | oxygen | aerotolerant | 21 |
| unmapped | shape | spore-shaped | 7 |
| unmapped | oxygen | microaerotolerant | 5 |
| unmapped | shape | ring-shaped | 3 |
| unmapped | shape | dumbbell-shaped | 2 |
| unmapped | shape | crescent-shaped | 1 |
| unparsed | temperature | 25--30 | 1 |

### 1b. Temperature bin sources, core panel × genome × species

| temperature_bin_source | N | % |
|---|---|---|
| optimum | 2,162 | 83.8% |
| growth_range | 296 | 11.5% |
| growth_single_point | 122 | 4.7% |

Oxygen: 30 core-panel species are "facultative" only because microaerophile was folded in (`oxygen_microaerophile_folded`). Across all strains, 663 of the 1,390 coarse oxygen conflicts involve a microaerophile observation.

## 2. Genome matching

Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality of normalised strain designations. A designation with no culture-collection code (e.g. `BS 107`) is accepted only with **taxon agreement**: the BacDive genus equals the GTDB or NCBI genus of the genome, or one of BacDive's NCBI tax IDs equals the genome's `ncbi_taxid` or `ncbi_species_taxid`. There is no fuzzy matching anywhere.

### 2a. What normalisation bought

Strains with ≥1 accepted GTDB genome. Steps A–C use the designation route alone and apply the same acceptance rule, so they differ only in how strings are compared. Step E equals the final match count.

| step | all strains n | all strains % | core panel n | core panel % | no-spore panel n | no-spore panel % |
|---|---|---|---|---|---|---|
| A. designation, raw whole strings (before normalisation) | 8,008 | 7.84 | 613 | 17.96 | 1,376 | 13.68 |
| B. designation, + split into tokens, raw | 19,443 | 19.03 | 2,429 | 71.15 | 5,553 | 55.23 |
| C. designation, + normalised keys (after normalisation) | 21,361 | 20.90 | 2,530 | 74.11 | 5,972 | 59.39 |
| D. accession route alone | 18,765 | 18.36 | 2,543 | 74.49 | 5,833 | 58.01 |
| E. final: accession or normalised designation | 22,835 | 22.35 | 2,630 | 77.04 | 6,184 | 61.50 |
| (info) raw tokens without the taxon check — includes cross-genus collisions, not a match count | 27,177 | 26.60 | 2,673 | 78.30 | 6,610 | 65.74 |

Splitting multi-designation strings does most of the work (A→B). Normalisation proper adds B→C. The last row is what a naive raw-token join would report, and includes cross-genus collisions (`LP1`, `B86`, `M120` recur across unrelated genera).

### 2b. Genome attrition, all strains

|  | strains | % |
|---|---|---|
| strains in set | 102,187 | 100.0% |
| BacDive lists ≥1 GCA assembly | 19,639 | 19.2% |
| … of which ≥1 listed assembly is in GTDB | 18,765 | 18.4% |
| … listed assembly not in GTDB (failed GTDB QC, or absent) | 874 | 0.9% |
| matched via direct accession | 18,765 | 18.4% |
| matched via designation — collection code (e.g. DSM 20231) | 450 | 0.4% |
| matched via designation — bare designation + taxon agreement | 3,620 | 3.5% |
| matched, total | 22,835 | 22.3% |
| unmatched | 79,352 | 77.7% |

### 2c. Genome attrition, core-panel-complete strains

|  | strains | % |
|---|---|---|
| strains in set | 3,414 | 100.0% |
| BacDive lists ≥1 GCA assembly | 2,585 | 75.7% |
| … of which ≥1 listed assembly is in GTDB | 2,543 | 74.5% |
| … listed assembly not in GTDB (failed GTDB QC, or absent) | 42 | 1.2% |
| matched via direct accession | 2,543 | 74.5% |
| matched via designation — collection code (e.g. DSM 20231) | 31 | 0.9% |
| matched via designation — bare designation + taxon agreement | 56 | 1.6% |
| matched, total | 2,630 | 77.0% |
| unmatched | 784 | 23.0% |

### 2d. Match quality checks

| check | value | N |
|---|---|---|
| designation route agrees with accession route (strains where both fire) | 99.94% | 17,291 |
| bare-designation hits agree with accession route (only bare keys hit, accession also present) | 99.91% | 5,466 |
| matched genome's GTDB/NCBI genus equals BacDive genus | 95.2% | 22,835 |
| designation matches accepted via NCBI tax ID only (genus differs) | 3.3% | 134 |
| candidates span >1 GTDB species (ambiguous) | 1.2% | 22,835 |

Genus disagreement on accession-route matches mostly reflects GTDB reclassification (split genera) rather than wrong matches; those strains are kept.

### 2e. Unmatched strains

All 79,352 are listed with the keys tried in `data/interim/unmatched_strains.tsv` (gitignored; regenerate with `python -m src.pipeline join`).

| reason | strains |
|---|---|
| no_bacdive_assembly;designations_not_in_gtdb | 74,182 |
| no_bacdive_assembly;bare_designation_taxon_mismatch | 3,564 |
| no_bacdive_assembly;no_usable_designation | 804 |
| bacdive_assembly_not_in_gtdb;designations_not_in_gtdb | 654 |
| bacdive_assembly_not_in_gtdb;bare_designation_taxon_mismatch | 148 |

## 3. Trait coverage co-occurrence

Strains with usable values for both traits (diagonal = single-trait coverage), over all BacDive strains.

| index | gram stain | cell shape | motility | spore formation | oxygen tolerance | temperature | pH | halophily (curated level) |
|---|---|---|---|---|---|---|---|---|
| gram stain | 15,514 | 14,357 | 13,379 | 5,054 | 12,198 | 14,857 | 6,441 | 349 |
| cell shape | 14,357 | 14,802 | 13,021 | 4,575 | 11,529 | 14,139 | 5,895 | 334 |
| motility | 13,379 | 13,021 | 13,809 | 4,567 | 11,096 | 13,386 | 5,866 | 326 |
| spore formation | 5,054 | 4,575 | 4,567 | 5,424 | 4,426 | 5,325 | 4,070 | 210 |
| oxygen tolerance | 12,198 | 11,529 | 11,096 | 4,426 | 21,855 | 21,388 | 5,869 | 329 |
| temperature | 14,857 | 14,139 | 13,386 | 5,325 | 21,388 | 50,310 | 6,700 | 359 |
| pH | 6,441 | 5,895 | 5,866 | 4,070 | 5,869 | 6,700 | 6,740 | 347 |
| halophily (curated level) | 349 | 334 | 326 | 210 | 329 | 359 | 347 | 361 |

P(column usable | row usable):

| index | gram stain | cell shape | motility | spore formation | oxygen tolerance | temperature |
|---|---|---|---|---|---|---|
| gram stain | 1 | 0.925 | 0.862 | 0.326 | 0.786 | 0.958 |
| cell shape | 0.97 | 1 | 0.88 | 0.309 | 0.779 | 0.955 |
| motility | 0.969 | 0.943 | 1 | 0.331 | 0.804 | 0.969 |
| spore formation | 0.932 | 0.843 | 0.842 | 1 | 0.816 | 0.982 |
| oxygen tolerance | 0.558 | 0.528 | 0.508 | 0.203 | 1 | 0.979 |
| temperature | 0.295 | 0.281 | 0.266 | 0.106 | 0.425 | 1 |

Lift = P(both) / (P(row)·P(column)); 1 = independent coverage:

| index | gram stain | cell shape | motility | spore formation | oxygen tolerance | temperature |
|---|---|---|---|---|---|---|
| gram stain | – | 6.39 | 6.38 | 6.14 | 3.68 | 1.95 |
| cell shape | 6.39 | – | 6.51 | 5.82 | 3.64 | 1.94 |
| motility | 6.38 | 6.51 | – | 6.23 | 3.76 | 1.97 |
| spore formation | 6.14 | 5.82 | 6.23 | – | 3.82 | 1.99 |
| oxygen tolerance | 3.68 | 3.64 | 3.76 | 3.82 | – | 1.99 |
| temperature | 1.95 | 1.94 | 1.97 | 1.99 | 1.99 | – |

### 3a. Panel completeness vs independent coverage

| panel | traits | observed_complete | observed_frac | product_of_marginals | expected_complete_if_independent | ratio |
|---|---|---|---|---|---|---|
| morphology (4) | 4 | 4,047 | 0.0396 | 0.000158 | 16.12 | 251 |
| no-spore (5) | 5 | 10,055 | 0.0984 | 0.000313 | 31.98 | 314 |
| core (6) | 6 | 3,414 | 0.0334 | 1.66e-05 | 1.70 | 2,011 |
| core + pH (7) | 7 | 2,807 | 0.0275 | 1.1e-06 | 0.112 | 25,074 |
| original 8 | 8 | 153 | 0.0015 | 3.87e-09 | 0.000395 | 386,868 |

Coverage is nested: strains described with one morphology trait usually have the others, because the traits come together from species descriptions. Spore formation is the binding constraint.

## 4. Does genome availability track trait coverage?

Base rate over all 102,187 strains: 19.2% list a GCA accession in BacDive. Fisher p-values below 1e-300 underflow and are shown as <1e-300.

| set | N | BacDive lists a GCA n | BacDive lists a GCA % | BacDive lists a GCA base % | BacDive lists a GCA odds ratio vs rest | BacDive lists a GCA Fisher p | GTDB genome matched n | GTDB genome matched % | GTDB genome matched base % | GTDB genome matched odds ratio vs rest | GTDB genome matched Fisher p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all BacDive strains | 102,187 | 19,639 | 19.22 | 19.22 | – | – | 22,835 | 22.35 | 22.35 | – | – |
| morphology panel complete (4) | 4,047 | 3,048 | 75.32 | 19.22 | 15.00 | <1e-300 | 3,126 | 77.24 | 22.35 | 13.51 | <1e-300 |
| core panel complete (6) | 3,414 | 2,585 | 75.72 | 19.22 | 14.94 | <1e-300 | 2,630 | 77.04 | 22.35 | 13.04 | <1e-300 |
| core panel complete, optimum temperature | 2,680 | 2,173 | 81.08 | 19.22 | 20.13 | <1e-300 | 2,189 | 81.68 | 22.35 | 17.03 | <1e-300 |

By number of usable core traits:

| core traits usable | N | BacDive lists a GCA % | GTDB genome matched % |
|---|---|---|---|
| 0 | 51,009 | 4.90 | 8.83 |
| 1 | 25,818 | 21.48 | 25.51 |
| 2 | 9,549 | 21.91 | 20.62 |
| 3 | 1,252 | 37.62 | 40.81 |
| 4 | 3,167 | 62.36 | 62.99 |
| 5 | 7,978 | 56.04 | 58.16 |
| 6 | 3,414 | 75.72 | 77.04 |

## 5. Two panels: with and without spore formation

Spore formation is the coverage bottleneck (5.3% of strains) and drove 3 of the 4 global-null hits in the first run. The no-spore panel drops it. Both are analysed identically: one strain per GTDB species, coarse bins, all temperature sources.

| panel | traits | cells | strains with genome | N species | N genera | species per cell | testable cells (global) | testable cells (phylum) | testable cells (class) | testable cells (order) | species kept by order null | order-level survivors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core panel (6 traits) | gram + shape + motility + spore + oxygen + temperature | 216 | 2,630 | 2,580 | 1,235 | 11.90 | 69 | 67 | 62 | 52 | 2,423 | 0 |
| no-spore panel (5 traits) | gram + shape + motility + oxygen + temperature | 108 | 6,184 | 5,634 | 1,927 | 52.20 | 54 | 59 | 53 | 49 | 5,416 | 1 |

Does dropping spore formation buy power? It more than doubles N (2,580 → 5,634 species) and halves the cells, so species per cell rise 4.4×. The *number* of cells the order-level null can test barely moves (52 → 49), but the *fraction* does (24% → 45%). The core panel's extra cells are mostly unreachable at this N. The no-spore panel produced 1 order-level survivor(s), the core panel 0.


Expected species per cell, per map size (global null, species level). Read this before the results: it says where the null has power.


**core panel (6 traits)**

| traits in map | maps | cells | N (species) range | N per cell, median over maps | median expected count in a cell | smallest expected count | testable cells (global null), median over maps |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 2–3 | 4,099–14,334 | 3,355 | 1,797 | 123 | 100% |
| 2 | 15 | 4–9 | 3,356–8,404 | 959 | 327 | 6.58 | 100% |
| 3 | 20 | 8–27 | 2,920–7,574 | 320 | 51.17 | 0.487 | 100% |
| 4 | 15 | 24–54 | 2,647–6,968 | 107 | 12.50 | 0.0509 | 72% |
| 5 | 6 | 72–108 | 2,595–5,634 | 37.86 | 3.38 | 0.02 | 51% |
| 6 | 1 | 216–216 | 2,580–2,580 | 11.94 | 0.834 | 0.00833 | 32% |

**no-spore panel (5 traits)**

| traits in map | maps | cells | N (species) range | N per cell, median over maps | median expected count in a cell | smallest expected count | testable cells (global null), median over maps |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 2–3 | 7,821–14,334 | 3,910 | 1,545 | 123 | 100% |
| 2 | 10 | 4–9 | 6,297–8,404 | 1,180 | 316 | 6.58 | 100% |
| 3 | 10 | 12–27 | 5,827–7,574 | 390 | 45.28 | 0.525 | 97% |
| 4 | 5 | 36–54 | 5,670–6,968 | 158 | 12.50 | 0.176 | 69% |
| 5 | 1 | 108–108 | 5,634–5,634 | 52.17 | 2.98 | 0.0618 | 50% |

### 5a. GTDB phyla (species level)

| GTDB phylum | core panel (6 traits) | no-spore panel (5 traits) |
|---|---|---|
| Pseudomonadota | 813 | 2,351 |
| Actinomycetota | 481 | 1,064 |
| Bacillota | 733 | 1,063 |
| Bacteroidota | 409 | 868 |
| Desulfobacterota | 25 | 43 |
| Campylobacterota | 13 | 43 |
| Deinococcota | 21 | 36 |
| Verrucomicrobiota | 12 | 28 |
| Acidobacteriota | 16 | 26 |
| Chloroflexota | 16 | 21 |
| Bacillota_I | 8 | 15 |
| Spirochaetota | 1 | 15 |
| Fusobacteriota | 5 | 9 |
| Thermotogota | 5 | 8 |
| Synergistota | 6 | 8 |
| Aquificota | 4 | 8 |
| Planctomycetota | 1 | 6 |
| Chrysiogenota | 3 | 5 |
| Myxococcota | 0 | 4 |
| Myxococcota_A | 0 | 3 |
| Campylobacterota_A | 1 | 2 |
| Nitrospirota | 2 | 2 |
| Gemmatimonadota | 2 | 2 |
| Dictyoglomota | 1 | 1 |
| Armatimonadota | 1 | 1 |
| Thermosulfidibacterota | 1 | 1 |
| Bdellovibrionota_B | 0 | 1 |

## 6. The nulls

- **Global.** Each trait column is permuted across all species, which preserves every marginal and destroys all association. It cannot tell a biological constraint from the fact that traits are each fixed in different clades.
- **Stratified (phylum, class, order).** Each trait column is permuted only among species of the same GTDB taxon, preserving every taxon's own trait frequencies. Expected count per cell = Σ over taxa n·∏ p(trait level within the taxon). A cell that is still emptier than chance under the order-level shuffle is not explained by which orders carry which traits. **That is the only category treated as a candidate.**
- **Small strata.** Taxa with fewer than 5 species in the analysed set cannot be meaningfully shuffled. Their species are excluded from that null and counted below, not silently kept. Exclusion is never allowed to manufacture emptiness: a cell is "empty beyond chance" only if it is empty among *all* species. The excluded species are exactly the phylogenetically unusual ones, so this guard is not cosmetic. In this run it matters: `gram=positive / shape=rod / motility=no / spore=yes / oxygen=anaerobe / temperature=thermo` (core panel (6 traits)) holds 4 species, all in orders below the size threshold, so it is empty among kept species and testable under the order null. It is labelled "emptied by exclusion", not flagged.
- **Testability.** A cell is tested only if its expected count under that null is ≥ 3. BH correction is applied within each map over its testable cells; 10,000 permutations per full map; the global and order nulls are re-run with an independent seed.

**Species kept per null, core panel (6 traits)**

| rank | min species per stratum | strata kept | strata excluded | species kept | species excluded |
|---|---|---|---|---|---|
| phylum | 2 | 18 | 6 | 2,574 | 6 |
| phylum | 5 (used) | 14 | 10 | 2,563 | 17 |
| phylum | 10 | 10 | 14 | 2,539 | 41 |
| class | 2 | 38 | 21 | 2,559 | 21 |
| class | 5 (used) | 24 | 35 | 2,522 | 58 |
| class | 10 | 14 | 45 | 2,460 | 120 |
| order | 2 | 93 | 48 | 2,532 | 48 |
| order | 5 (used) | 50 | 91 | 2,423 | 157 |
| order | 10 | 36 | 105 | 2,336 | 244 |

**Species kept per null, no-spore panel (5 traits)**

| rank | min species per stratum | strata kept | strata excluded | species kept | species excluded |
|---|---|---|---|---|---|
| phylum | 2 | 23 | 4 | 5,630 | 4 |
| phylum | 5 (used) | 18 | 9 | 5,617 | 17 |
| phylum | 10 | 12 | 15 | 5,573 | 61 |
| class | 2 | 49 | 21 | 5,613 | 21 |
| class | 5 (used) | 32 | 38 | 5,568 | 66 |
| class | 10 | 17 | 53 | 5,471 | 163 |
| order | 2 | 122 | 65 | 5,569 | 65 |
| order | 5 (used) | 66 | 121 | 5,416 | 218 |
| order | 10 | 46 | 141 | 5,295 | 339 |

## 7. Positive control

**Gram-negative endospore-forming cocci** — `gram=negative / shape=coccus / motility=no / spore=yes / oxygen=aerobe / temperature=meso`

Endospore formation is essentially restricted to Bacillota, which are overwhelmingly Gram-positive, and coccoid endospore formers are rare even there (Sporosarcina, itself motile and Gram-positive). The cell stacks three phylogenetic facts; an independence null cannot see any of them.

This cell was the strongest hit of the first, global-only run. It is reported here as a check on the method, **not as a finding**. Expected behaviour: the global null flags it, and a phylogeny-aware null does not, because its emptiness follows from which clades carry which traits.

| null | species used | observed | expected | testable | q | empty beyond chance |
|---|---|---|---|---|---|---|
| global | 2,580 | 0 | 11.21 | yes | 0.000493 | yes |
| phylum | 2,563 | 0 | 1.66 | no | – | no |
| class | 2,522 | 0 | 0.845 | no | – | no |
| order | 2,423 | 0 | 0.477 | no | – | no |

**Result: passed.** The global null detects the known constraint, and the stratified nulls absorb it: the expected count falls from 11.2 (global) to 1.7 (phylum), 0.8 (class) and 0.5 (order), because within any one taxon the constituent traits barely co-occur.

## 8. Occupied vs expected cells

Number of occupied cells against the number the null predicts for the same species. Under the global null a deficit means the traits co-occur more tightly than independence allows. Under a stratified null the deficit left over is what the taxon's own trait frequencies cannot explain. Stratified rows use the species kept by that null (small strata excluded), for both observed and expected.

| map | null | N used | strata | species excluded (small strata) | cells | occupied | expected occupied (null mean) | null 95% interval | deficit | P(null ≤ observed) |
|---|---|---|---|---|---|---|---|---|---|---|
| morphology panel (4) | global | 3,053 | 1 | 0 | 24 | 23 | 24 | 24–24 | 1 | 0.0005 |
| morphology panel (4) | phylum | 3,041 | 17 | 12 | 24 | 23 | 23.70 | 23–24 | 0.7 | 0.321 |
| morphology panel (4) | class | 2,996 | 27 | 57 | 24 | 23 | 23.10 | 22–24 | 0.1 | 0.67 |
| morphology panel (4) | order | 2,878 | 53 | 175 | 24 | 23 | 22.60 | 21–24 | -0.4 | 0.833 |
| core panel (6 traits) | global | 2,580 | 1 | 0 | 216 | 90 | 117 | 110–124 | 27.30 | 0.0001 |
| core panel (6 traits) | phylum | 2,563 | 14 | 17 | 216 | 90 | 109 | 103–116 | 19.30 | 0.0001 |
| core panel (6 traits) | class | 2,522 | 24 | 58 | 216 | 89 | 103 | 96–109 | 13.60 | 0.0001 |
| core panel (6 traits) | order | 2,423 | 50 | 157 | 216 | 83 | 87.90 | 82–94 | 4.90 | 0.0667 |
| no-spore panel (5 traits) | global | 5,634 | 1 | 0 | 108 | 69 | 78.30 | 73–84 | 9.30 | 0.0016 |
| no-spore panel (5 traits) | phylum | 5,617 | 18 | 17 | 108 | 69 | 78.30 | 73–83 | 9.30 | 0.0001 |
| no-spore panel (5 traits) | class | 5,568 | 32 | 66 | 108 | 69 | 75.80 | 71–81 | 6.80 | 0.0064 |
| no-spore panel (5 traits) | order | 5,416 | 66 | 218 | 108 | 66 | 69.40 | 65–74 | 3.40 | 0.114 |

All 4- and 5-trait subsets of the core traits (full table: `reports/tables/occupancy_subsets_all_nulls.tsv`):

| map | global: occupied / expected | global: P | phylum: occupied / expected | phylum: P | class: occupied / expected | class: P | order: occupied / expected | order: P |
|---|---|---|---|---|---|---|---|---|
| gram + motility + oxygen + temperature | 34 / 35 | 0.349 | 34 / 33 | 0.936 | 34 / 33 | 0.928 | 34 / 33 | 0.957 |
| gram + motility + spore + oxygen | 24 / 24 | 1 | 24 / 24 | 1 | 24 / 24 | 1 | 24 / 24 | 1 |
| gram + motility + spore + oxygen + temperature | 49 / 58 | 0.0005 | 49 / 52 | 0.0345 | 49 / 51 | 0.196 | 46 / 47 | 0.338 |
| gram + motility + spore + temperature | 21 / 23 | 0.004 | 21 / 21 | 0.712 | 21 / 21 | 0.755 | 21 / 20 | 0.99 |
| gram + shape + motility + oxygen | 35 / 36 | 0.002 | 35 / 36 | 0.006 | 35 / 36 | 0.006 | 34 / 36 | 0.0135 |
| gram + shape + motility + oxygen + temperature | 69 / 78 | 0.0016 | 69 / 78 | 0.0001 | 69 / 76 | 0.0064 | 66 / 69 | 0.114 |
| gram + shape + motility + spore | 23 / 24 | 0.0005 | 23 / 24 | 0.321 | 23 / 23 | 0.67 | 23 / 23 | 0.833 |
| gram + shape + motility + spore + oxygen | 58 / 68 | 0.0005 | 58 / 65 | 0.001 | 58 / 63 | 0.0155 | 58 / 58 | 0.515 |
| gram + shape + motility + spore + oxygen + temperature | 90 / 117 | 0.0001 | 90 / 109 | 0.0001 | 89 / 103 | 0.0001 | 83 / 88 | 0.0667 |
| gram + shape + motility + spore + temperature | 44 / 54 | 0.0005 | 44 / 51 | 0.001 | 44 / 48 | 0.03 | 43 / 44 | 0.483 |
| gram + shape + motility + temperature | 30 / 34 | 0.0095 | 30 / 33 | 0.0415 | 30 / 32 | 0.089 | 30 / 31 | 0.378 |
| gram + shape + oxygen + temperature | 41 / 45 | 0.0415 | 41 / 45 | 0.0235 | 41 / 44 | 0.084 | 40 / 41 | 0.476 |
| gram + shape + spore + oxygen | 32 / 36 | 0.0005 | 32 / 35 | 0.006 | 32 / 34 | 0.094 | 32 / 32 | 0.533 |
| gram + shape + spore + oxygen + temperature | 52 / 70 | 0.0005 | 52 / 65 | 0.0005 | 52 / 61 | 0.001 | 49 / 53 | 0.045 |
| gram + shape + spore + temperature | 26 / 30 | 0.006 | 26 / 29 | 0.0245 | 26 / 28 | 0.221 | 26 / 26 | 0.612 |
| gram + spore + oxygen + temperature | 26 / 32 | 0.0005 | 26 / 28 | 0.036 | 26 / 27 | 0.226 | 25 / 26 | 0.392 |
| motility + spore + oxygen + temperature | 28 / 32 | 0.002 | 28 / 29 | 0.171 | 28 / 30 | 0.108 | 28 / 29 | 0.47 |
| shape + motility + oxygen + temperature | 39 / 44 | 0.0055 | 39 / 45 | 0.001 | 39 / 44 | 0.005 | 39 / 41 | 0.28 |
| shape + motility + spore + oxygen | 32 / 36 | 0.0005 | 32 / 36 | 0.0005 | 32 / 35 | 0.001 | 32 / 34 | 0.0635 |
| shape + motility + spore + oxygen + temperature | 55 / 69 | 0.0005 | 55 / 68 | 0.0005 | 54 / 66 | 0.0005 | 52 / 58 | 0.0075 |
| shape + motility + spore + temperature | 25 / 30 | 0.0015 | 25 / 29 | 0.0015 | 25 / 29 | 0.003 | 24 / 27 | 0.0745 |
| shape + spore + oxygen + temperature | 32 / 40 | 0.0005 | 32 / 39 | 0.001 | 32 / 38 | 0.0005 | 30 / 34 | 0.0125 |

## 9. Per-phylum occupancy

Cells each phylum occupies on its own, cells that only it occupies, and the cumulative share of all occupied cells, adding phyla from largest to smallest. "Within-phylum null" permutes traits among that phylum's species only.


**core panel (6 traits)** — 90 cells occupied in total. The three largest phyla (Pseudomonadota, Bacillota, Actinomycetota; 78.6% of species) occupy 84 of them (93%). Pseudomonadota alone occupies 31 (34%).

Largest within-phylum deficit: Bacillota, 56 cells occupied vs 78 expected from its own trait frequencies. Other phyla sit near their own expectation.

| phylum | N | cells occupied | occupied (within-phylum null mean) | cells only this phylum occupies | cumulative occupied (this + larger phyla) | cumulative share of all occupied |
|---|---|---|---|---|---|---|
| Pseudomonadota | 813 | 31 | 29.40 | 9 | 31 | 0.34 |
| Bacillota | 733 | 56 | 78.30 | 23 | 76 | 0.84 |
| Actinomycetota | 481 | 28 | 32.40 | 6 | 84 | 0.93 |
| Bacteroidota | 409 | 21 | 19.40 | 0 | 85 | 0.94 |
| Desulfobacterota | 25 | 5 | 6.20 | 0 | 86 | 0.96 |
| Deinococcota | 21 | 8 | 8.90 | 0 | 87 | 0.97 |
| Acidobacteriota | 16 | 3 | 3.90 | 0 | 87 | 0.97 |
| Chloroflexota | 16 | 10 | 13.80 | 2 | 89 | 0.99 |
| Campylobacterota | 13 | 8 | 7.70 | 1 | 90 | 1 |
| Verrucomicrobiota | 12 | 3 | 3 | 0 | 90 | 1 |
| Bacillota_I | 8 | 3 | 3 | 0 | 90 | 1 |
| Synergistota | 6 | 3 | 3.50 | 0 | 90 | 1 |
| Fusobacteriota | 5 | 2 | 2 | 0 | 90 | 1 |
| Thermotogota | 5 | 2 | 2 | 0 | 90 | 1 |
| Aquificota | 4 | 3 | – | 0 | 90 | 1 |
| Chrysiogenota | 3 | 3 | – | 0 | 90 | 1 |
| Gemmatimonadota | 2 | 1 | – | 0 | 90 | 1 |
| Nitrospirota | 2 | 1 | – | 0 | 90 | 1 |
| Armatimonadota | 1 | 1 | – | 0 | 90 | 1 |
| Campylobacterota_A | 1 | 1 | – | 0 | 90 | 1 |
| Dictyoglomota | 1 | 1 | – | 0 | 90 | 1 |
| Planctomycetota | 1 | 1 | – | 0 | 90 | 1 |
| Spirochaetota | 1 | 1 | – | 0 | 90 | 1 |
| Thermosulfidibacterota | 1 | 1 | – | 0 | 90 | 1 |

**no-spore panel (5 traits)** — 69 cells occupied in total. The three largest phyla (Pseudomonadota, Actinomycetota, Bacillota; 79.5% of species) occupy 61 of them (88%). Pseudomonadota alone occupies 32 (46%).

Largest within-phylum deficit: Bacillota, 41 cells occupied vs 55 expected from its own trait frequencies. Other phyla sit near their own expectation.

| phylum | N | cells occupied | occupied (within-phylum null mean) | cells only this phylum occupies | cumulative occupied (this + larger phyla) | cumulative share of all occupied |
|---|---|---|---|---|---|---|
| Pseudomonadota | 2,351 | 32 | 31.90 | 8 | 32 | 0.46 |
| Actinomycetota | 1,064 | 26 | 28.70 | 3 | 47 | 0.68 |
| Bacillota | 1,063 | 41 | 55.50 | 8 | 61 | 0.88 |
| Bacteroidota | 868 | 24 | 22.50 | 0 | 61 | 0.88 |
| Campylobacterota | 43 | 11 | 10.90 | 1 | 62 | 0.9 |
| Desulfobacterota | 43 | 6 | 7 | 0 | 63 | 0.91 |
| Deinococcota | 36 | 10 | 11.30 | 1 | 65 | 0.94 |
| Verrucomicrobiota | 28 | 8 | 8.30 | 1 | 66 | 0.96 |
| Acidobacteriota | 26 | 5 | 5.20 | 0 | 66 | 0.96 |
| Chloroflexota | 21 | 12 | 13.80 | 1 | 68 | 0.99 |
| Bacillota_I | 15 | 3 | 3.30 | 0 | 68 | 0.99 |
| Spirochaetota | 15 | 6 | 8.50 | 1 | 69 | 1 |
| Fusobacteriota | 9 | 3 | 2.90 | 0 | 69 | 1 |
| Aquificota | 8 | 3 | 3.90 | 0 | 69 | 1 |
| Synergistota | 8 | 4 | 4.40 | 0 | 69 | 1 |
| Thermotogota | 8 | 2 | 2 | 0 | 69 | 1 |
| Planctomycetota | 6 | 4 | 3.50 | 0 | 69 | 1 |
| Chrysiogenota | 5 | 5 | 4.10 | 0 | 69 | 1 |
| Myxococcota | 4 | 1 | – | 0 | 69 | 1 |
| Myxococcota_A | 3 | 2 | – | 0 | 69 | 1 |
| Campylobacterota_A | 2 | 2 | – | 0 | 69 | 1 |
| Gemmatimonadota | 2 | 1 | – | 0 | 69 | 1 |
| Nitrospirota | 2 | 1 | – | 0 | 69 | 1 |
| Armatimonadota | 1 | 1 | – | 0 | 69 | 1 |
| Bdellovibrionota_B | 1 | 1 | – | 0 | 69 | 1 |
| Dictyoglomota | 1 | 1 | – | 0 | 69 | 1 |
| Thermosulfidibacterota | 1 | 1 | – | 0 | 69 | 1 |

## 10. Cell-level results under all four nulls

Every cell that is empty and testable under at least one null, or flagged under at least one. `q` is BH-adjusted within the map; `–` means the cell is untestable under that null (expected below threshold). Full tables, every cell: `reports/tables/cells_<panel>_all_nulls.tsv`.


### 10a. core panel (6 traits) (216 cells, 2,580 species)

| gram | shape | motility | spore | oxygen | temperature | observed (all species) | global: expected | global: q | phylum: expected | phylum: q | class: expected | class: q | order: expected | order: q | global: q (seed 2) | order: q (seed 2) | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | coccus | no | yes | aerobe | meso | 0 | 11.21 | 0.000493 | 1.66 | – | 0.845 | – | 0.477 | – | 0.000493 | – | phylogenetic structure — fires under global only; untestable at order level |
| negative | other | yes | yes | aerobe | meso | 0 | 5.04 | 0.0246 | 0.556 | – | 0.373 | – | 0.331 | – | 0.0245 | – | phylogenetic structure — fires under global only; untestable at order level |
| positive | coccus | yes | no | anaerobe | meso | 0 | 4.42 | 0.0342 | 4.09 | 0.0491 | 1.56 | – | 1.73 | – | 0.0329 | – | phylogenetic structure — fires under global/phylum only; untestable at order level |
| positive | rod | no | no | facultative | thermo | 0 | 3.91 | 0.0446 | 3.21 | 0.108 | 2.02 | – | 0.146 | – | 0.0434 | – | phylogenetic structure — fires under global only; untestable at order level |
| positive | rod | yes | no | facultative | thermo | 0 | 2.93 | – | 3.91 | 0.0524 | 2.63 | – | 0.335 | – | – | – | empty, testable, not significant under any null |
| positive | coccus | no | yes | facultative | meso | 0 | 2.07 | – | 7.79 | 0.00183 | 8.99 | 0.00155 | 1.59 | – | – | – | phylogenetic structure — fires under phylum/class only; untestable at order level |
| positive | coccus | yes | yes | facultative | meso | 0 | 1.55 | – | 8.50 | 0.00149 | 10.91 | 0.000886 | 0.992 | – | – | – | phylogenetic structure — fires under phylum/class only; untestable at order level |
| positive | coccus | yes | yes | anaerobe | meso | 0 | 1.40 | – | 6.96 | 0.00335 | 2.26 | – | 0.84 | – | – | – | phylogenetic structure — fires under phylum only; untestable at order level |

Cross-tabulation (species counts). **0✗** survives the order-level shuffle · 0† fires under a coarser null only (phylogenetic structure) · 0 empty, testable, not significant · `·` empty and untestable under every null.

| gram / shape / motility / spore ↓   oxygen / temperature → | aerobe / psychro | aerobe / meso | aerobe / thermo | facultative / psychro | facultative / meso | facultative / thermo | anaerobe / psychro | anaerobe / meso | anaerobe / thermo |
|---|---|---|---|---|---|---|---|---|---|
| negative / rod / no / no | 8 | 487 | 8 | · | 40 | 1 | · | 83 | 11 |
| negative / rod / no / yes | · | 6 | 1 | · | 2 | 1 | · | 2 | 2 |
| negative / rod / yes / no | 11 | 445 | 8 | 4 | 84 | 5 | · | 33 | 20 |
| negative / rod / yes / yes | · | 12 | 2 | · | 7 | · | · | 13 | 4 |
| negative / coccus / no / no | · | 27 | 1 | · | 2 | · | · | 15 | · |
| negative / coccus / no / yes | · | 0† | · | · | · | · | · | · | · |
| negative / coccus / yes / no | · | 7 | · | · | 1 | · | · | 2 | · |
| negative / coccus / yes / yes | · | 1 | · | · | · | · | · | · | · |
| negative / other / no / no | · | 22 | 1 | · | 3 | 1 | · | 8 | 1 |
| negative / other / no / yes | · | 1 | · | · | · | · | · | 1 | · |
| negative / other / yes / no | · | 14 | · | · | 8 | · | · | 6 | 3 |
| negative / other / yes / yes | · | 0† | · | · | · | · | · | 3 | · |
| positive / rod / no / no | · | 182 | 2 | · | 95 | 0† | · | 53 | 5 |
| positive / rod / no / yes | · | 139 | 10 | · | 15 | 1 | 1 | 22 | 4 |
| positive / rod / yes / no | · | 33 | 1 | · | 11 | 0 | · | 5 | 4 |
| positive / rod / yes / yes | · | 200 | 23 | · | 61 | 4 | · | 29 | 17 |
| positive / coccus / no / no | · | 82 | 2 | · | 53 | · | · | 13 | · |
| positive / coccus / no / yes | · | 2 | · | · | 0† | · | · | 1 | · |
| positive / coccus / yes / no | · | 15 | · | · | 2 | · | · | 0† | · |
| positive / coccus / yes / yes | · | 1 | · | · | 0† | · | · | 0† | · |
| positive / other / no / no | · | 16 | · | · | 9 | · | · | 4 | · |
| positive / other / no / yes | · | 22 | 2 | · | · | 2 | · | 2 | · |
| positive / other / yes / no | · | 2 | · | · | · | · | · | 2 | · |
| positive / other / yes / yes | · | 2 | · | · | · | · | · | 1 | · |

### 10b. no-spore panel (5 traits) (108 cells, 5,634 species)

| gram | shape | motility | oxygen | temperature | observed (all species) | global: expected | global: q | phylum: expected | phylum: q | class: expected | class: q | order: expected | order: q | global: q (seed 2) | order: q (seed 2) | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| positive | coccus | yes | anaerobe | meso | 0 | 7.62 | 0.0018 | 20.53 | 0.000737 | 7.96 | 0.000589 | 5.44 | 0.0108 | 0.00101 | 0.0098 | SURVIVES order-level shuffle (both seeds) |
| negative | coccus | yes | aerobe | thermo | 0 | 3.01 | 0.0959 | 0.733 | – | 0.401 | – | 0.38 | – | 0.11 | – | empty, testable, not significant under any null |
| positive | coccus | yes | aerobe | thermo | 0 | 1.72 | – | 3.13 | 0.135 | 2.45 | – | 0.564 | – | – | – | empty, testable, not significant under any null |
| positive | coccus | no | facultative | thermo | 0 | 0.667 | – | 3.83 | 0.0649 | 2.85 | – | 0.145 | – | – | – | empty, testable, not significant under any null |
| positive | coccus | yes | facultative | thermo | 0 | 0.496 | – | 3.09 | 0.138 | 2.43 | – | 0.128 | – | – | – | empty, testable, not significant under any null |

Cross-tabulation (species counts). **0✗** survives the order-level shuffle · 0† fires under a coarser null only (phylogenetic structure) · 0 empty, testable, not significant · `·` empty and untestable under every null.

| gram / shape / motility ↓   oxygen / temperature → | aerobe / psychro | aerobe / meso | aerobe / thermo | facultative / psychro | facultative / meso | facultative / thermo | anaerobe / psychro | anaerobe / meso | anaerobe / thermo |
|---|---|---|---|---|---|---|---|---|---|
| negative / rod / no | 19 | 1124 | 11 | 1 | 198 | 3 | · | 129 | 15 |
| negative / rod / yes | 19 | 1204 | 13 | 8 | 368 | 10 | 1 | 83 | 33 |
| negative / coccus / no | 1 | 71 | 1 | · | 8 | · | · | 25 | · |
| negative / coccus / yes | 1 | 17 | 0 | · | 3 | · | · | 2 | · |
| negative / other / no | · | 84 | 1 | · | 12 | 1 | · | 19 | 3 |
| negative / other / yes | · | 52 | · | 1 | 17 | · | · | 25 | 5 |
| positive / rod / no | 2 | 660 | 16 | · | 209 | 2 | 1 | 130 | 10 |
| positive / rod / yes | 3 | 315 | 25 | 1 | 99 | 4 | · | 42 | 22 |
| positive / coccus / no | 1 | 165 | 2 | 1 | 148 | 0 | · | 27 | · |
| positive / coccus / yes | · | 18 | 0 | · | 3 | 0 | · | **0✗** | · |
| positive / other / no | · | 84 | 3 | · | 31 | 2 | · | 10 | · |
| positive / other / yes | · | 5 | · | · | 1 | · | · | 4 | · |

## 11. Cells that survive the order-level shuffle

### `gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso` (no-spore panel (5 traits))

Observed 0 among 5,634 species. Expected 7.6 under the global null and 5.4 under the order-level shuffle (q = 0.011; independent seed q = 0.0098).

Robustness across views:

| view | null | N used | observed | expected | testable | q | empty beyond chance | note |
|---|---|---|---|---|---|---|---|---|
| primary | global | 5,634 | 0 | 7.62 | yes | 0.0018 | yes | – |
| primary | order | 5,416 | 0 | 5.44 | yes | 0.0108 | yes | – |
| strain-level | global | 6,184 | 0 | 8.57 | yes | 0.000868 | yes | – |
| strain-level | order | 5,968 | 0 | 5.70 | yes | 0.00833 | yes | – |
| optimum-only | global | 3,786 | 0 | 3.88 | yes | 0.0595 | no | – |
| optimum-only | order | 3,590 | 0 | 2.26 | no | – | no | – |
| fine bins | global | – | – | – | – | – | – | cell not defined at fine resolution (bins split) |
| fine bins | order | – | – | – | – | – | – | cell not defined at fine resolution (bins split) |
| genus-level | global | 1,927 | 0 | 5.23 | yes | 0.0214 | yes | – |
| genus-level | order | 1,703 | 0 | 3.26 | yes | 0.43 | no | – |
| primary (diagnostic: family shuffle) | family | 5,106 | 0 | 1.56 | no | – | no | – |

Orders the order-level null expects to populate this cell (top 10 of 14 by expected count; each order's share of species with each trait level). The top order carries 45% of the expectation.

| order | phylum | N | expected in cell | share gram=positive | share shape=coccus | share motility=yes | share oxygen=anaerobe | share temperature=meso |
|---|---|---|---|---|---|---|---|---|
| Tissierellales | Bacillota | 20 | 2.44 | 0.9 | 0.5 | 0.3 | 0.95 | 0.95 |
| Actinomycetales | Actinomycetota | 455 | 1.07 | 0.976 | 0.211 | 0.176 | 0.066 | 0.989 |
| Lachnospirales | Bacillota | 43 | 0.444 | 0.814 | 0.047 | 0.279 | 1 | 0.977 |
| Peptostreptococcales | Bacillota | 28 | 0.398 | 0.857 | 0.036 | 0.5 | 1 | 0.929 |
| Lactobacillales | Bacillota | 305 | 0.365 | 0.993 | 0.416 | 0.059 | 0.049 | 0.997 |
| Coriobacteriales | Actinomycetota | 27 | 0.285 | 0.963 | 0.148 | 0.074 | 1 | 1 |
| Oscillospirales | Bacillota | 12 | 0.229 | 0.5 | 0.25 | 0.167 | 0.917 | 1 |
| Propionibacteriales | Actinomycetota | 111 | 0.115 | 0.964 | 0.189 | 0.09 | 0.063 | 1 |
| Bacillales | Bacillota | 283 | 0.08 | 0.982 | 0.028 | 0.788 | 0.014 | 0.912 |
| Bacteroidales | Bacteroidota | 102 | 0.011 | 0.029 | 0.029 | 0.147 | 0.863 | 0.98 |

Inside the contributing orders, do the constituent trait levels ever occur in the same family? 51% of the order-level expectation comes from orders in which **no single family** carries every constituent level; there the combination is only "expected" because the order mixes families. Top 5 orders:

| order | expected in cell | species | families | rarest constituent level in this order | families carrying that level | families carrying every constituent level |
|---|---|---|---|---|---|---|
| Tissierellales | 2.44 | 20 | 8 | motility=yes | Clostridiisalibacteraceae, Gottschalkiaceae, Proteiniboraceae, Tepidimicrobiaceae, VENL01 | none |
| Actinomycetales | 1.07 | 455 | 15 | oxygen=anaerobe | Actinomycetaceae, Bifidobacteriaceae, Cellulomonadaceae, Demequinaceae | Actinomycetaceae, Cellulomonadaceae |
| Lachnospirales | 0.444 | 43 | 6 | shape=coccus | Lachnospiraceae | Lachnospiraceae |
| Peptostreptococcales | 0.398 | 28 | 7 | shape=coccus | Peptostreptococcaceae | Peptostreptococcaceae |
| Lactobacillales | 0.365 | 305 | 9 | oxygen=anaerobe | Aerococcaceae, Lactobacillaceae, Listeriaceae, Streptococcaceae | Lactobacillaceae, Streptococcaceae |

Across **all** 102,187 BacDive strains, with or without a genome, 1 strain has this combination: Tonsilliphilus suis (BacDive 22988, no genome).

**The family-level diagnostic absorbs it:** expected 1.56 under a family shuffle (untestable). Read this as phylogenetic structure at family level unless a finer analysis says otherwise.


What surviving the order shuffle does and does not mean: within orders that contain every constituent trait, the combination is rarer than independence predicts. It can still be phylogenetic structure *below* order (family, genus), and the whole panel is described type strains (§12). It is a candidate, not a result.

### 11a. Order-level survivors in lower-dimensional maps

All 4- and 5-trait subsets of the core traits, order-level null. Lower-dimensional maps have more species per cell, so they can test cells the full panels cannot. Not corrected across maps.

| map | cell | order: expected | order: q | global: q |
|---|---|---|---|---|
| gram + shape + motility + oxygen | gram=positive / shape=coccus / motility=yes / oxygen=anaerobe | 5.64 | 0.0102 | 0.00257 |
| gram + shape + motility + oxygen + temperature | gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso | 5.44 | 0.0108 | 0.0018 |

## 12. Sensitivity views

Global and order-level nulls in each view. Genus-level keeps one strain per GTDB genus, so its order-level strata are small and many species are excluded.

| panel | view | null | N used | occupied | expected occupied | testable cells | empty✗ | emptied by exclusion |
|---|---|---|---|---|---|---|---|---|
| core panel (6 traits) | primary | global | 2,580 | 90 | 117 | 69 | 4 | 0 |
| core panel (6 traits) | primary | phylum | 2,563 | 90 | 109 | 67 | 4 | 0 |
| core panel (6 traits) | primary | class | 2,522 | 89 | 103 | 62 | 2 | 1 |
| core panel (6 traits) | primary | order | 2,423 | 83 | 87.90 | 52 | 0 | 7 |
| core panel (6 traits) | strain-level | global | 2,630 | 90 | 118 | 71 | 4 | 0 |
| core panel (6 traits) | strain-level | order | 2,477 | 84 | 88.40 | 55 | 0 | 6 |
| core panel (6 traits) | optimum-only | global | 2,168 | 87 | 109 | 60 | 3 | 0 |
| core panel (6 traits) | optimum-only | order | 2,006 | 80 | 82.90 | 47 | 0 | 7 |
| core panel (6 traits) | fine bins | global | 2,512 | 112 | 145 | 66 | 3 | 0 |
| core panel (6 traits) | fine bins | order | 2,354 | 99 | 104 | 52 | 0 | 13 |
| core panel (6 traits) | genus-level | global | 1,235 | 86 | 104 | 59 | 3 | 0 |
| core panel (6 traits) | genus-level | order | 1,067 | 77 | 77.90 | 40 | 0 | 9 |
| no-spore panel (5 traits) | primary | global | 5,634 | 69 | 78.30 | 54 | 1 | 0 |
| no-spore panel (5 traits) | primary | phylum | 5,617 | 69 | 78.30 | 59 | 1 | 0 |
| no-spore panel (5 traits) | primary | class | 5,568 | 69 | 75.80 | 53 | 1 | 0 |
| no-spore panel (5 traits) | primary | order | 5,416 | 66 | 69.40 | 49 | 1 | 3 |
| no-spore panel (5 traits) | strain-level | global | 6,184 | 70 | 79.40 | 55 | 1 | 0 |
| no-spore panel (5 traits) | strain-level | order | 5,968 | 68 | 69.60 | 50 | 1 | 2 |
| no-spore panel (5 traits) | optimum-only | global | 3,786 | 65 | 72.20 | 51 | 1 | 0 |
| no-spore panel (5 traits) | optimum-only | order | 3,590 | 63 | 65.90 | 44 | 0 | 2 |
| no-spore panel (5 traits) | fine bins | global | 5,460 | 95 | 116 | 66 | 4 | 0 |
| no-spore panel (5 traits) | fine bins | order | 5,240 | 88 | 94.70 | 57 | 1 | 7 |
| no-spore panel (5 traits) | genus-level | global | 1,927 | 61 | 69.40 | 48 | 1 | 0 |
| no-spore panel (5 traits) | genus-level | order | 1,703 | 55 | 60.10 | 42 | 0 | 6 |

### 12a. Type strains

| N | % |
|---|---|
| 2,539 | 98.4% |
| 2,468 | 95.7% |
| 2,549 | 98.8% |
| 2,458 | 95.3% |
| 81 | 3.1% |

## 13. Caveats

- **Type-strain sampling.** 98.8% of core-panel species are represented by a type strain. The map describes *described, culturable, formally characterised* species: organisms someone isolated in pure culture and wrote up under a standard protocol. Uncultured lineages (most candidate phyla, most of the tree by genome count) are absent. Trait values come from protocol-driven species descriptions: growth tested at standard temperatures, a fixed test battery. The marginals are those of the described world, and a cell can be empty because no one has isolated and described such an organism.
- **Exchangeability below order.** The order-level shuffle removes structure between orders but not within them. Families and genera share traits, so a survivor can still be phylogenetic structure at a finer rank. The genus-level view is a crude check, not a fix.
- **Stratified nulls lose power.** Testable cells fall from global to order level (§5). A cell that stops firing at order level can be untestable rather than explained. §10 says which.
- **Temperature source.** Across BacDive most temperature values are single cultivation temperatures (35,639 single-point vs 7,427 optimum). In the core-panel set 83.8% of bins come from an optimum. The optimum-only view (§12) removes the rest.
- **Multiple testing.** BH is applied within each map over its testable cells only. The subset maps are not jointly corrected.
- **Resolution.** Permutation p-values cannot go below 1/(n_perm+1) = 1.0e-04 for the full maps.

## Files

- `data/final/strains.parquet` (gitignored): every BacDive strain with raw and normalised traits, bin sources, conflicts, genome match and panel flags. Regenerate with `python -m src.pipeline all`; checksums in `reports/run_manifest.json` → `outputs`.
- `data/final/{core,no_spore}_panel_species.tsv`: the analysed sets (one strain per GTDB species).
- `data/final/assembly_accessions_{core,no_spore}_panel_species.txt`, `assembly_accessions_all_matched.txt`: for `datasets download genome accession --inputfile …` (RefSeq GCF where GTDB uses RefSeq, else GenBank GCA).
- `data/interim/unmatched_strains.tsv` (gitignored), `data/interim/unmapped_values.tsv`: audit logs.
- `reports/tables/*.tsv`: every table above, plus per-cell results for every panel × view × null.
- `reports/run_manifest.json`: versions, hashes and counts.
