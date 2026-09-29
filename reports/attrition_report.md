# BacDive × GTDB: attrition and trait-space occupancy

_Generated 2026-09-29T04:48:25+00:00 · code `f4fa8baabee2363dd9b910bb503bc765114109fd` · config sha256 `df8029adf221`_

## Snapshot

| source | version | retrieved | detail |
|---|---|---|---|
| BacDive | API v2, record DOIs stamped `20260601` (BacDive exposes no release number) | 2026-09-29 → 2026-09-29 | IDs 1–250,000 swept in full (2,500 batches of 100); highest ID found 179,194; genome-based predictions not requested |
| GTDB | v232 (Released Apr 15, 2026) | 2026-09-29 | bac120_metadata.tsv.gz, 878,998 genomes, sha256 `1650d3164666` |

Empty BacDive ID ranges ≥1,000 IDs: 24,901–99,900, 128,801–130,100, 179,201–250,000.

## Headline

- **102,187** BacDive strains; **3,414** (3.3%) have the full 6-trait core panel; **2,630** of those have a GTDB genome, collapsing to **2,580 GTDB species** — the primary analysis set.
- The original 8-trait panel (adding pH and curated halophily) is complete for **153** strains, **115** with a genome.
- Primary map: 216 cells, 90 occupied (null expectation 117.3); 69 cells are testable (expected ≥ 3); **4 empty cells are emptier than the marginals predict** (BH q < 0.05); 3 of them are flagged under two independent permutation seeds. 24 occupied cells are significantly depleted.
- Coverage is strongly nested: the 6-trait panel is complete 2,011× more often than independent coverage would give. Genome availability is also enriched among well-phenotyped strains (§6).

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
| 3. pH — usable value | 6,740 | 6.6% | 5,298 | 78.6% | 5,201 | outside the core panel |
| 3. halophily (curated level) — usable value | 361 | 0.4% | 247 | 68.4% | 245 | curated category only; NaCl tests kept as a column |
| 3. (NaCl growth tests present, raw column only) | 9,677 | 9.5% | 6,947 | 71.8% | 6,594 | not binned; kept in nacl_tests_raw |
| 4. morphology panel (4) complete | 4,047 | 4.0% | 3,126 | 77.2% | 3,053 | gram + shape + motility + spore |
| 5. core panel (6) complete | 3,414 | 3.3% | 2,630 | 77.0% | 2,580 | gram + shape + motility + spore + oxygen + temperature |
| 5a. core panel (6), temperature from optimum only | 2,680 | 2.6% | 2,189 | 81.7% | 2,168 |  |
| 5b. original 8-trait panel (core + pH + curated halophily) | 153 | 0.1% | 115 | 75.2% | 114 | for reference; not analysed |

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

### 1b. Temperature bin sources, core panel (6) × genome × species

| temperature_bin_source | N | % |
|---|---|---|
| optimum | 2,162 | 83.8% |
| growth_range | 296 | 11.5% |
| growth_single_point | 122 | 4.7% |

Oxygen: 30 of these species are "facultative" only because microaerophile was folded in (`oxygen_microaerophile_folded`). Across all strains, 663 of the 1,390 coarse oxygen conflicts involve a microaerophile observation (e.g. anaerobe + microaerophile, which conflicts once microaerophile is read as facultative).

## 2. Genome matching

Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality of normalised strain designations. A designation with no culture-collection code (e.g. `BS 107`) is accepted only with **taxon agreement**: the BacDive genus equals the GTDB or NCBI genus of the genome, or one of BacDive's NCBI tax IDs equals the genome's `ncbi_taxid` or `ncbi_species_taxid`. The tax-ID criterion recovers genus renames. There is no fuzzy matching anywhere.

### 2a. What normalisation bought

Strains with ≥1 accepted GTDB genome. Steps A–C use the designation route alone and apply the same acceptance rule, so they differ only in how strings are compared. Step E equals the final match count.

| step | all strains n | all strains % | morph. panel n | morph. panel % | core panel n | core panel % |
|---|---|---|---|---|---|---|
| A. designation, raw whole strings (before normalisation) | 8,008 | 7.84 | 695 | 17.17 | 613 | 17.96 |
| B. designation, + split into tokens, raw | 19,443 | 19.03 | 2,878 | 71.11 | 2,429 | 71.15 |
| C. designation, + normalised keys (after normalisation) | 21,361 | 20.90 | 3,017 | 74.55 | 2,530 | 74.11 |
| D. accession route alone | 18,765 | 18.36 | 3,002 | 74.18 | 2,543 | 74.49 |
| E. final: accession or normalised designation | 22,835 | 22.35 | 3,126 | 77.24 | 2,630 | 77.04 |
| (info) raw tokens without the taxon check — includes cross-genus collisions, not a match count | 27,177 | 26.60 | 3,159 | 78.06 | 2,673 | 78.30 |

The last row shows what a naive raw-token join would report. The gap between it and step B is cross-genus collisions (`LP1`, `B86`, `M120` recur across unrelated genera), which the taxon rule rejects.

### 2b. Genome attrition, all strains

|  | strains | % |
|---|---|---|
| strains in set | 102,187 | 100.0% |
| BacDive lists ≥1 GCA assembly | 19,639 | 19.2% |
| … of which ≥1 listed assembly is in GTDB (others: failed GTDB QC or absent) | 18,765 | 18.4% |
| … listed assembly not in GTDB | 874 | 0.9% |
| matched via direct accession | 18,765 | 18.4% |
| matched via designation — collection code (e.g. DSM 20231) | 450 | 0.4% |
| matched via designation — bare designation + genus agreement | 3,620 | 3.5% |
| matched, total | 22,835 | 22.3% |
| unmatched | 79,352 | 77.7% |

### 2c. Genome attrition, core panel (6)-complete strains

|  | strains | % |
|---|---|---|
| strains in set | 3,414 | 100.0% |
| BacDive lists ≥1 GCA assembly | 2,585 | 75.7% |
| … of which ≥1 listed assembly is in GTDB (others: failed GTDB QC or absent) | 2,543 | 74.5% |
| … listed assembly not in GTDB | 42 | 1.2% |
| matched via direct accession | 2,543 | 74.5% |
| matched via designation — collection code (e.g. DSM 20231) | 31 | 0.9% |
| matched via designation — bare designation + genus agreement | 56 | 1.6% |
| matched, total | 2,630 | 77.0% |
| unmatched | 784 | 23.0% |

### 2d. Match quality checks

| check | value | N |
|---|---|---|
| designation route agrees with accession route (strains where both fire) | 99.94% | 17,291 |
| matched genome's GTDB/NCBI genus equals BacDive genus | 95.2% | 22,835 |
| … accession-route matches | 95.0% | 18,765 |
| bare-designation hits agree with accession route (only bare keys hit, accession also present) | 99.91% | 5,466 |
| designation matches accepted via NCBI tax ID only (genus differs) | 3.3% | 134 |
| candidates span >1 GTDB species (ambiguous) | 1.2% | 22,835 |

Genus disagreement on accession-route matches mostly reflects GTDB reclassification (e.g. split genera) rather than wrong matches; those strains are kept.

### 2e. Unmatched strains

All 79,352 are listed with the keys tried in `data/interim/unmatched_strains.tsv`.

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
| core (6) | 6 | 3,414 | 0.0334 | 1.66e-05 | 1.70 | 2,011 |
| core + pH (7) | 7 | 2,807 | 0.0275 | 1.1e-06 | 0.112 | 25,074 |
| original 8 | 8 | 153 | 0.0015 | 3.87e-09 | 0.000395 | 386,868 |

Coverage is nested rather than independent: strains described with one morphology trait usually have the others, because the traits come together from species descriptions. Spore formation is the binding constraint.

## 4. Power: expected strains per cell, before reading results

Species-level, coarse bins. "Testable" means expected count ≥ 3 under independence. Below that an empty cell carries no information (at expected = 1 the null itself is empty 37% of the time).

| traits in panel | panels | cells | N (species) range | N per cell, median over panels | median expected count in a cell | smallest expected count | testable cells (expected ≥ threshold), median over panels |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 2–3 | 4,099–14,334 | 3,355 | 1,797 | 123 | 100% |
| 2 | 15 | 4–9 | 3,356–8,404 | 959 | 327 | 6.58 | 100% |
| 3 | 20 | 8–27 | 2,920–7,574 | 320 | 51.17 | 0.487 | 100% |
| 4 | 15 | 24–54 | 2,647–6,968 | 107 | 12.50 | 0.0509 | 72% |
| 5 | 6 | 72–108 | 2,595–5,634 | 37.86 | 3.38 | 0.02 | 51% |
| 6 | 1 | 216–216 | 2,580–2,580 | 11.94 | 0.834 | 0.00833 | 32% |

Per-panel detail: `reports/tables/power_expected_per_cell.tsv`.

## 5. The core panel (6) + genome set

2,580 GTDB species (from 2,630 strains).

### 5a. GTDB phyla (species level)

| GTDB phylum | N | % |
|---|---|---|
| Pseudomonadota | 813 | 31.5% |
| Bacillota | 733 | 28.4% |
| Actinomycetota | 481 | 18.6% |
| Bacteroidota | 409 | 15.9% |
| Desulfobacterota | 25 | 1.0% |
| Deinococcota | 21 | 0.8% |
| Acidobacteriota | 16 | 0.6% |
| Chloroflexota | 16 | 0.6% |
| Campylobacterota | 13 | 0.5% |
| Verrucomicrobiota | 12 | 0.5% |
| Bacillota_I | 8 | 0.3% |
| Synergistota | 6 | 0.2% |
| Fusobacteriota | 5 | 0.2% |
| Thermotogota | 5 | 0.2% |
| Aquificota | 4 | 0.2% |
| Chrysiogenota | 3 | 0.1% |
| Nitrospirota | 2 | 0.1% |
| Gemmatimonadota | 2 | 0.1% |
| Dictyoglomota | 1 | 0.0% |
| Planctomycetota | 1 | 0.0% |
| Spirochaetota | 1 | 0.0% |
| Armatimonadota | 1 | 0.0% |
| Thermosulfidibacterota | 1 | 0.0% |
| Campylobacterota_A | 1 | 0.0% |

Strain level: Pseudomonadota 824, Bacillota 761, Actinomycetota 487, Bacteroidota 413, Desulfobacterota 25, Deinococcota 22, Acidobacteriota 16, Chloroflexota 16, Campylobacterota 13, Verrucomicrobiota 12, Bacillota_I 8, Synergistota 6, Fusobacteriota 5, Thermotogota 5, Aquificota 4, Chrysiogenota 3, Nitrospirota 2, Gemmatimonadota 2, Dictyoglomota 1, Planctomycetota 1, Spirochaetota 1, Armatimonadota 1, Thermosulfidibacterota 1, Campylobacterota_A 1.

### 5b. Type strains (species level)

|  | N | % |
|---|---|---|
| type strain per BacDive | 2,539 | 98.4% |
| type strain of species per GTDB/NCBI | 2,468 | 95.7% |
| either | 2,549 | 98.8% |
| both | 2,458 | 95.3% |
| BacDive says type, GTDB genome not flagged as type | 81 | 3.1% |

## 6. Does genome availability track trait coverage?

Base rate over all 102,187 strains: 19.2% list a GCA accession in BacDive. The ~24% quoted during planning came from a 2,011-strain sample. Fisher p-values below 1e-300 underflow and are shown as <1e-300.

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

## 7. Occupancy of trait subsets

Each row is a trait subset, analysed on the species with a genome that are complete for that subset. Every trait column is permuted independently (2,000 permutations; 10,000 for the full panel), which preserves the marginals exactly. "Occupied (null)" is the mean number of occupied cells under that null, so occupancy fractions can be read against sparsity. "Empty beyond chance" counts cells with 0 observed that are testable and have BH q < 0.05.

### 7a. Morphology map (primary view)

| panel | traits | N | cells | N/cell | occupied | occ. frac | occupied (null) | occ. frac (null) | testable | empty & testable | empty beyond chance | occupied, depleted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-trait | gram + shape + motility + spore | 3,053 | 24 | 127 | 23 | 0.96 | 24 | 1 | 24 | 1 | 1 | 13 |

### 7b. All 4- and 5-trait subsets and the full panel (primary view)

| panel | traits | N | cells | N/cell | occupied | occ. frac | occupied (null) | occ. frac (null) | testable | empty & testable | empty beyond chance | occupied, depleted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-trait | gram + shape + motility + spore | 3,053 | 24 | 127 | 23 | 0.96 | 24 | 1 | 24 | 1 | 1 | 13 |
| 4-trait | gram + shape + motility + oxygen | 5,670 | 36 | 158 | 35 | 0.97 | 36 | 1 | 36 | 1 | 1 | 17 |
| 4-trait | gram + shape + motility + temperature | 6,968 | 36 | 194 | 30 | 0.83 | 33.80 | 0.94 | 25 | 1 | 1 | 9 |
| 4-trait | gram + shape + spore + oxygen | 2,860 | 36 | 79.40 | 32 | 0.89 | 35.70 | 0.99 | 33 | 4 | 3 | 12 |
| 4-trait | gram + shape + spore + temperature | 3,333 | 36 | 92.60 | 26 | 0.72 | 30.50 | 0.85 | 24 | 1 | 0 | 10 |
| 4-trait | gram + shape + oxygen + temperature | 6,121 | 54 | 113 | 41 | 0.76 | 44.90 | 0.83 | 33 | 0 | 0 | 7 |
| 4-trait | gram + motility + spore + oxygen | 2,874 | 24 | 120 | 24 | 1 | 24 | 1 | 24 | 0 | 0 | 11 |
| 4-trait | gram + motility + spore + temperature | 3,344 | 24 | 139 | 21 | 0.88 | 23.50 | 0.98 | 20 | 0 | 0 | 7 |
| 4-trait | gram + motility + oxygen + temperature | 6,081 | 36 | 169 | 34 | 0.94 | 34.80 | 0.97 | 30 | 0 | 0 | 11 |
| 4-trait | gram + spore + oxygen + temperature | 3,250 | 36 | 90.30 | 26 | 0.72 | 32.30 | 0.9 | 26 | 2 | 2 | 8 |
| 4-trait | shape + motility + spore + oxygen | 2,647 | 36 | 73.50 | 32 | 0.89 | 35.60 | 0.99 | 33 | 3 | 2 | 11 |
| 4-trait | shape + motility + spore + temperature | 3,097 | 36 | 86 | 25 | 0.69 | 30.20 | 0.84 | 23 | 1 | 1 | 7 |
| 4-trait | shape + motility + oxygen + temperature | 5,789 | 54 | 107 | 39 | 0.72 | 44.50 | 0.82 | 33 | 2 | 1 | 8 |
| 4-trait | shape + spore + oxygen + temperature | 2,896 | 54 | 53.60 | 32 | 0.59 | 39.90 | 0.74 | 28 | 3 | 2 | 7 |
| 4-trait | motility + spore + oxygen + temperature | 2,913 | 36 | 80.90 | 28 | 0.78 | 32.10 | 0.89 | 24 | 0 | 0 | 7 |
| 5-trait | gram + shape + motility + spore + oxygen | 2,595 | 72 | 36 | 58 | 0.81 | 68.30 | 0.95 | 55 | 4 | 3 | 18 |
| 5-trait | gram + shape + motility + spore + temperature | 3,030 | 72 | 42.10 | 44 | 0.61 | 54.10 | 0.75 | 37 | 1 | 1 | 15 |
| 5-trait | gram + shape + motility + oxygen + temperature | 5,634 | 108 | 52.20 | 69 | 0.64 | 78.40 | 0.73 | 54 | 2 | 1 | 19 |
| 5-trait | gram + shape + spore + oxygen + temperature | 2,836 | 108 | 26.30 | 52 | 0.48 | 69.50 | 0.64 | 45 | 6 | 5 | 14 |
| 5-trait | gram + motility + spore + oxygen + temperature | 2,857 | 72 | 39.70 | 49 | 0.68 | 58.10 | 0.81 | 42 | 3 | 1 | 17 |
| 5-trait | shape + motility + spore + oxygen + temperature | 2,632 | 108 | 24.40 | 55 | 0.51 | 69.10 | 0.64 | 44 | 3 | 1 | 14 |
| 6-trait (full) | gram + shape + motility + spore + oxygen + temperature | 2,580 | 216 | 11.90 | 90 | 0.42 | 117 | 0.54 | 69 | 4 | 4 | 24 |

### 7c. Same, other views: empty-beyond-chance cells per subset

| traits | primary N | primary empty✗ | strain-level N | strain-level empty✗ | optimum-only N | optimum-only empty✗ | fine bins N | fine bins empty✗ | genus-level N | genus-level empty✗ |
|---|---|---|---|---|---|---|---|---|---|---|
| gram + shape + motility + spore | 3,053 | 1 | 3,126 | 1 | 3,053 | 1 | 3,051 | 5 | 1,329 | 1 |
| gram + shape + motility + oxygen | 5,670 | 1 | 6,227 | 1 | 5,670 | 1 | 5,495 | 4 | 1,929 | 1 |
| gram + shape + motility + temperature | 6,968 | 1 | 7,756 | 1 | 4,363 | 0 | 6,966 | 2 | 2,121 | 0 |
| gram + shape + spore + oxygen | 2,860 | 3 | 2,925 | 3 | 2,860 | 3 | 2,787 | 4 | 1,297 | 0 |
| gram + shape + spore + temperature | 3,333 | 0 | 3,414 | 1 | 2,752 | 0 | 3,331 | 3 | 1,393 | 0 |
| gram + shape + oxygen + temperature | 6,121 | 0 | 6,715 | 0 | 4,142 | 0 | 5,942 | 0 | 2,031 | 0 |
| gram + motility + spore + oxygen | 2,874 | 0 | 2,931 | 0 | 2,874 | 0 | 2,799 | 0 | 1,316 | 0 |
| gram + motility + spore + temperature | 3,344 | 0 | 3,420 | 0 | 2,753 | 0 | 3,344 | 0 | 1,414 | 0 |
| gram + motility + oxygen + temperature | 6,081 | 0 | 6,657 | 0 | 4,115 | 0 | 5,899 | 0 | 2,026 | 0 |
| gram + spore + oxygen + temperature | 3,250 | 2 | 3,327 | 2 | 2,706 | 2 | 3,169 | 2 | 1,402 | 1 |
| shape + motility + spore + oxygen | 2,647 | 2 | 2,699 | 2 | 2,647 | 2 | 2,579 | 3 | 1,256 | 1 |
| shape + motility + spore + temperature | 3,097 | 1 | 3,164 | 1 | 2,552 | 1 | 3,095 | 2 | 1,350 | 1 |
| shape + motility + oxygen + temperature | 5,789 | 1 | 6,348 | 1 | 3,870 | 1 | 5,614 | 2 | 1,970 | 0 |
| shape + spore + oxygen + temperature | 2,896 | 2 | 2,964 | 3 | 2,426 | 2 | 2,823 | 1 | 1,316 | 0 |
| motility + spore + oxygen + temperature | 2,913 | 0 | 2,972 | 0 | 2,423 | 0 | 2,838 | 1 | 1,334 | 0 |
| gram + shape + motility + spore + oxygen | 2,595 | 3 | 2,645 | 3 | 2,595 | 3 | 2,527 | 4 | 1,236 | 2 |
| gram + shape + motility + spore + temperature | 3,030 | 1 | 3,092 | 1 | 2,502 | 1 | 3,028 | 5 | 1,327 | 1 |
| gram + shape + motility + oxygen + temperature | 5,634 | 1 | 6,184 | 1 | 3,786 | 1 | 5,460 | 4 | 1,927 | 1 |
| gram + shape + spore + oxygen + temperature | 2,836 | 5 | 2,901 | 5 | 2,381 | 1 | 2,763 | 5 | 1,294 | 1 |
| gram + motility + spore + oxygen + temperature | 2,857 | 1 | 2,914 | 1 | 2,383 | 1 | 2,782 | 0 | 1,315 | 1 |
| shape + motility + spore + oxygen + temperature | 2,632 | 1 | 2,684 | 1 | 2,205 | 0 | 2,564 | 2 | 1,255 | 1 |
| gram + shape + motility + spore + oxygen + temperature | 2,580 | 4 | 2,630 | 4 | 2,168 | 3 | 2,512 | 3 | 1,235 | 3 |

## 8. Full 6-trait map

| view | description | N | cells | occupied | occupied (null mean) | testable cells | empty & testable | empty beyond chance (q<α) | occupied but depleted (q<α) |
|---|---|---|---|---|---|---|---|---|---|
| primary | one strain per GTDB species · coarse bins · all temperature sources | 2,580 | 216 | 90 | 117 | 69 | 4 | 4 | 24 |
| strain-level | no dereplication (sensitivity) | 2,630 | 216 | 90 | 118 | 71 | 6 | 4 | 24 |
| optimum-only | temperature bin from a reported optimum only | 2,168 | 216 | 87 | 109 | 60 | 4 | 3 | 18 |
| fine bins | fine binning (shape 5, oxygen 4, temperature 4) | 2,512 | 640 | 112 | 145 | 66 | 5 | 3 | 22 |
| genus-level | one strain per GTDB genus (extra sensitivity for congener clustering) | 1,235 | 216 | 86 | 104 | 59 | 6 | 3 | 13 |

### 8a. Cross-tabulation, primary view (N = 2,580 species, 216 cells)

Cell = species count. **0✗** = empty and emptier than chance (testable, q < α). `0` = empty, testable, not significant. `·` = empty and untestable (expected < threshold, so emptiness is uninformative). ↓ = occupied but significantly depleted.

| gram / shape / motility / spore ↓   oxygen / temperature → | aerobe / psychro | aerobe / meso | aerobe / thermo | facultative / psychro | facultative / meso | facultative / thermo | anaerobe / psychro | anaerobe / meso | anaerobe / thermo |
|---|---|---|---|---|---|---|---|---|---|
| negative / rod / no / no | 8 | 487 | 8 ↓ | · | 40 ↓ | 1 | · | 83 | 11 |
| negative / rod / no / yes | · | 6 ↓ | 1 ↓ | · | 2 ↓ | 1 | · | 2 ↓ | 2 |
| negative / rod / yes / no | 11 | 445 | 8 ↓ | 4 | 84 | 5 | · | 33 ↓ | 20 |
| negative / rod / yes / yes | · | 12 ↓ | 2 | · | 7 ↓ | · | · | 13 | 4 |
| negative / coccus / no / no | · | 27 | 1 | · | 2 ↓ | · | · | 15 | · |
| negative / coccus / no / yes | · | **0✗** | · | · | · | · | · | · | · |
| negative / coccus / yes / no | · | 7 ↓ | · | · | 1 ↓ | · | · | 2 | · |
| negative / coccus / yes / yes | · | 1 ↓ | · | · | · | · | · | · | · |
| negative / other / no / no | · | 22 | 1 | · | 3 | 1 | · | 8 | 1 |
| negative / other / no / yes | · | 1 ↓ | · | · | · | · | · | 1 | · |
| negative / other / yes / no | · | 14 | · | · | 8 | · | · | 6 | 3 |
| negative / other / yes / yes | · | **0✗** | · | · | · | · | · | 3 | · |
| positive / rod / no / no | · | 182 ↓ | 2 ↓ | · | 95 | **0✗** | · | 53 | 5 |
| positive / rod / no / yes | · | 139 | 10 | · | 15 | 1 | 1 | 22 | 4 |
| positive / rod / yes / no | · | 33 ↓ | 1 ↓ | · | 11 ↓ | · | · | 5 ↓ | 4 |
| positive / rod / yes / yes | · | 200 | 23 | · | 61 | 4 | · | 29 | 17 |
| positive / coccus / no / no | · | 82 | 2 | · | 53 | · | · | 13 | · |
| positive / coccus / no / yes | · | 2 ↓ | · | · | · | · | · | 1 | · |
| positive / coccus / yes / no | · | 15 | · | · | 2 | · | · | **0✗** | · |
| positive / coccus / yes / yes | · | 1 ↓ | · | · | · | · | · | · | · |
| positive / other / no / no | · | 16 | · | · | 9 | · | · | 4 | · |
| positive / other / no / yes | · | 22 | 2 | · | · | 2 | · | 2 | · |
| positive / other / yes / no | · | 2 ↓ | · | · | · | · | · | 2 | · |
| positive / other / yes / yes | · | 2 | · | · | · | · | · | 1 | · |

### 8b. Cells empty beyond what the marginals predict (primary view)

| gram | shape | motility | spore | oxygen | temperature | expected | q_low | q_low_seed2 | strain-level: observed | strain-level: empty_beyond_chance | optimum-only: observed | optimum-only: expected | optimum-only: empty_beyond_chance | genus-level: expected | genus-level: empty_beyond_chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | coccus | no | yes | aerobe | meso | 11.21 | 0.000493 | 0.000531 | 0 | yes | 0 | 9.12 | yes | 4.98 | yes |
| negative | other | yes | yes | aerobe | meso | 5.04 | 0.0246 | 0.019 | 0 | yes | 0 | 4.87 | yes | 3.07 | no |
| positive | coccus | yes | no | anaerobe | meso | 4.42 | 0.0342 | 0.0353 | 0 | yes | 0 | 3.03 | no | 3.28 | no |
| positive | rod | no | no | facultative | thermo | 3.91 | 0.0446 | 0.051 | 0 | yes | 0 | 2.83 | no | 2.57 | no |

### 8c. Occupied but significantly depleted cells (primary view)

| gram | shape | motility | spore | oxygen | temperature | observed | expected | q_low |
|---|---|---|---|---|---|---|---|---|
| negative | rod | no | no | facultative | meso | 40 | 79.45 | 0.000493 |
| negative | rod | no | yes | aerobe | meso | 6 | 109 | 0.000493 |
| negative | rod | no | yes | anaerobe | meso | 2 | 22.52 | 0.000493 |
| negative | rod | no | yes | facultative | meso | 2 | 25.08 | 0.000493 |
| negative | coccus | yes | no | aerobe | meso | 7 | 26.64 | 0.000493 |
| positive | rod | no | no | aerobe | meso | 182 | 280 | 0.000493 |
| negative | rod | yes | yes | aerobe | meso | 12 | 82.12 | 0.000493 |
| positive | rod | no | no | aerobe | thermo | 2 | 17.07 | 0.000493 |
| positive | rod | yes | no | facultative | meso | 11 | 48.09 | 0.000493 |
| positive | rod | yes | no | aerobe | thermo | 1 | 12.81 | 0.000493 |
| positive | other | yes | no | aerobe | meso | 2 | 12.88 | 0.000493 |
| positive | rod | yes | no | anaerobe | meso | 5 | 43.19 | 0.000493 |
| positive | rod | yes | no | aerobe | meso | 33 | 210 | 0.000493 |
| negative | rod | no | no | aerobe | thermo | 8 | 21.16 | 0.00138 |
| negative | rod | yes | no | anaerobe | meso | 33 | 53.54 | 0.00302 |
| negative | rod | yes | yes | facultative | meso | 7 | 18.82 | 0.00487 |
| negative | coccus | yes | yes | aerobe | meso | 1 | 8.41 | 0.00652 |
| positive | coccus | no | yes | aerobe | meso | 2 | 9.04 | 0.0171 |
| negative | rod | no | yes | aerobe | thermo | 1 | 6.68 | 0.0246 |
| negative | other | no | yes | aerobe | meso | 1 | 6.71 | 0.027 |
| positive | coccus | yes | yes | aerobe | meso | 1 | 6.78 | 0.0279 |
| negative | coccus | no | no | facultative | meso | 2 | 8.14 | 0.0342 |
| negative | coccus | yes | no | facultative | meso | 1 | 6.10 | 0.0414 |
| negative | rod | yes | no | aerobe | thermo | 8 | 15.88 | 0.0427 |

Because the marginals are fixed, a structural gap forces compensating depletion in partner cells, so depleted cells need not be constraints themselves.


### 8d. Does strictness change the map?

Cells flagged in the primary view: 4; in the optimum-only view: 3; in both: 2. The optimum-only view has N = 2,168 and 60 testable cells (primary: 69), so a cell dropping out of the optimum-only list is as likely to be lost power as a changed answer. `reports/tables/full_map_views_compared.tsv` lists every cell in every view.

| gram | shape | motility | spore | oxygen | temperature | observed | expected | empty_beyond_chance | optimum-only: observed | optimum-only: expected | optimum-only: testable | optimum-only: empty_beyond_chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | rod | yes | yes | aerobe | thermo | 2 | 5.01 | no | 0 | 5.25 | yes | yes |
| negative | coccus | no | yes | aerobe | meso | 0 | 11.21 | yes | 0 | 9.12 | yes | yes |
| negative | other | yes | yes | aerobe | meso | 0 | 5.04 | yes | 0 | 4.87 | yes | yes |
| positive | rod | no | no | facultative | thermo | 0 | 3.91 | yes | 0 | 2.83 | no | no |
| positive | coccus | yes | no | anaerobe | meso | 0 | 4.42 | yes | 0 | 3.03 | yes | no |

## 9. Caveats

- **Exchangeability.** The permutation null treats species as exchangeable. They are not: congeners share traits, and BacDive over-samples culturable, clinically and industrially relevant lineages. A cell emptier than the marginals predict can reflect phylogenetic clustering or sampling as much as a biological constraint. The genus-level view is a crude check, not a fix.
- **Selection.** Complete panels come almost entirely from type-strain species descriptions (§5b). The map describes described type strains, not bacterial diversity at large.
- **Temperature source.** Across all of BacDive most temperature values are single cultivation temperatures, not optima (35,639 single-point vs 7,427 optimum). In the analysed core-panel set, however, 83.8% of temperature bins come from an optimum and only 4.7% from a single point (§1b). A single cultivation temperature is usually a routine 28–37 °C, which bins as mesophile by construction. The optimum-only view (§8d) removes these.
- **Null power.** With 216 cells and N in the low thousands, many cells are untestable (§4). An empty untestable cell is not evidence of anything.
- **Multiple testing.** BH is applied within each map over its testable cells only. The subset grids are not jointly corrected.
- **Resolution.** Permutation p-values cannot go below 1/(n_perm+1) = 1.0e-04 for the full map.

## Files

- `data/final/strains.parquet`: every BacDive strain, with raw and normalised traits, bin sources, conflicts, the genome match and panel flags.
- `data/final/core_panel_species.tsv`: the primary analysis set.
- `data/final/assembly_accessions_core_panel_species.txt`: accessions for `datasets download genome accession --inputfile …` (RefSeq GCF where GTDB uses RefSeq, else GenBank GCA). `assembly_accessions_all_matched.txt` holds all matched strains.
- `data/interim/unmatched_strains.tsv`, `data/interim/unmapped_values.tsv`: audit logs.
- `reports/tables/*.tsv`: every table above, plus full per-cell results for each view.
- `reports/run_manifest.json`: versions, hashes and counts.
