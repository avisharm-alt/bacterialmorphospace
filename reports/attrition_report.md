# Bacterial trait space: occupancy and its phylogenetic structure

_Generated 2026-09-29T05:33:20+00:00 · code `c3015ed02729efb277d4d771dc5a68f4e74324aa` · config sha256 `96c49f9e4f4a` · BacDive API v2 (record DOIs `20260601`) · GTDB v232_

## Summary

**Bacterial trait space is sparser than independence predicts, and that sparsity is almost entirely phylogenetic.** On the six-trait core map, 2,580 species occupy 90 of 216 cells; with traits assigned independently they would fill 117 (P = 0.0001). When the null is made phylogeny-aware, shuffling traits only within a phylum, class or order, the deficit narrows at every step: 27 → 19 → 14 → 5 cells. At order level it is not significant (P = 0.067). The five-trait map without spore formation (5,634 species) behaves the same (9 → 9 → 7 → 3 cells; P = 0.11 at order level).

**No trait combination is empty beyond what lineage structure explains.** In the core panel no cell survives the order-level shuffle. In the no-spore panel one cell, `gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso`, passes the order-level criterion (expected 5.4, q = 0.011), but a family-level shuffle makes it untestable (expected 1.6), and 1 of 102,187 BacDive strains has the combination.

The method does detect known constraints. Gram-negative endospore-forming cocci, a combination absent because endospores are a Bacillota trait and Bacillota are Gram-positive, is flagged by the global null (expected 11.2, observed 0) and correctly absorbed by every stratified null (expected 0.5 at order level).

The argument, section by section:

1. **§1 The occupancy map.** Which trait combinations are occupied, and by how many species.
2. **§2 Global null.** Fewer cells are occupied than independent traits would fill.
3. **§3 Stratified nulls.** The shortfall shrinks as the null respects phylum, class, then order, and is not significant at order level.
4. **§4 Positive control.** The method flags a known constraint globally and attributes it to lineage.
5. **§5 No novel constraints.** No cell survives both the order-level shuffle and the family-level check.
6. **§6 Power.** The smallest constraint the order-level null could have detected.
7. **§7 Methods note.** Why dropping small strata from a stratified permutation test creates false positives.

## Read this first: what the result does and does not show

1. **Six coarse traits are a low-dimensional slice of phenotype.** The maps have 2–3 levels per trait (216 and 108 cells). Absence of detectable constraint at this resolution is not absence of constraint. Constraints can live in finer bins, in continuous values (cell size, optimum temperature), or in traits not measured here (metabolism, envelope chemistry, genome features).
2. **The panel is type strains.** 98.8% of core-panel species are represented by a type strain: culturable, formally described organisms, characterised with standard protocols. The panel spans 24 of the 172 bacterial phyla in GTDB v232. Whole uncultured phyla are absent by construction. The marginals are those of the described world, and a combination can be absent because no one has isolated and described such an organism.
3. **Power is limited.** At order level only 52 of 216 core-panel cells are testable (no-spore: 49 of 108). A true zero is guaranteed to reach q < 0.05 only if the order-level null expects **≥ 6.9 species** in that cell (core panel; 0.29% of the species tested) or **≥ 6.9** (no-spore; 0.13%). Only 37 core-panel cells meet that bar (no-spore: 39). They hold 95% (98%) of the expected species. **We could have detected a constraint that empties a combination the lineage structure predicts to hold about 7 or more species. Constraints on rarer combinations are invisible to this analysis** (§6).
4. **Exchangeability below order.** The order-level shuffle removes structure between orders, not within them. That is why the single order-level survivor was re-tested at family level (§5).

## 1. The occupancy map

102,187 BacDive strains; 3,414 have the six-trait core panel, 2,630 of them with a matched GTDB genome, dereplicated to one strain per GTDB species (type strain preferred). Attrition and matching details are in Appendix A–B.

| panel | traits | cells | strains with genome | N species | N genera | species per cell | testable cells (global) | testable cells (phylum) | testable cells (class) | testable cells (order) | species kept by order null | order-level survivors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core panel (6 traits) | gram + shape + motility + spore + oxygen + temperature | 216 | 2,630 | 2,580 | 1,235 | 11.90 | 69 | 67 | 62 | 52 | 2,423 | 0 |
| no-spore panel (5 traits) | gram + shape + motility + oxygen + temperature | 108 | 6,184 | 5,634 | 1,927 | 52.20 | 54 | 59 | 53 | 49 | 5,416 | 1 |

Dropping spore formation, the coverage bottleneck (5.3% of strains), raises N 2.2× on half the cells. That lifts the testable share at order level from 24% to 45% of cells, though the absolute number barely moves.


**core panel (6 traits): species per cell.** **0✗** survives the order-level shuffle · 0† flagged by a coarser null only (phylogenetic structure) · 0 empty, testable, not flagged · `·` empty and untestable under every null.

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


**no-spore panel (5 traits): species per cell.** **0✗** survives the order-level shuffle · 0† flagged by a coarser null only (phylogenetic structure) · 0 empty, testable, not flagged · `·` empty and untestable under every null.

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


## 2. The global null: fewer occupied cells than independence predicts

Each trait column is permuted across all species, which keeps every trait's frequency and destroys all association between traits. The observed map has fewer occupied cells than the permuted maps: traits co-occur more tightly than independence allows.

| map | null | N used | cells | occupied | expected occupied (null mean) | null 95% interval | deficit | P(null ≤ observed) |
|---|---|---|---|---|---|---|---|---|
| morphology panel (4) | global | 3,053 | 24 | 23 | 24 | 24–24 | 1 | 0.0005 |
| core panel (6 traits) | global | 2,580 | 216 | 90 | 117 | 110–124 | 27.30 | 0.0001 |
| no-spore panel (5 traits) | global | 5,634 | 108 | 69 | 78.30 | 73–84 | 9.30 | 0.0016 |

Cells the global null flags as empty beyond chance (q < 0.05): 4 in the core panel, 1 in the no-spore panel.

| panel | gram | shape | motility | spore | oxygen | temperature | global: expected | global: q | global: q (seed 2) |
|---|---|---|---|---|---|---|---|---|---|
| core panel (6 traits) | negative | coccus | no | yes | aerobe | meso | 11.21 | 0.000493 | 0.000493 |
| core panel (6 traits) | negative | other | yes | yes | aerobe | meso | 5.04 | 0.0246 | 0.0245 |
| core panel (6 traits) | positive | rod | no | no | facultative | thermo | 3.91 | 0.0446 | 0.0434 |
| core panel (6 traits) | positive | coccus | yes | no | anaerobe | meso | 4.42 | 0.0342 | 0.0329 |
| no-spore panel (5 traits) | positive | coccus | yes | · | anaerobe | meso | 7.62 | 0.0018 | 0.00101 |

The global null cannot distinguish a constraint from the fact that traits are each fixed in different clades. §3 separates the two.

## 3. Stratified nulls: the deficit is phylogenetic

Each trait column is permuted only among species of the same GTDB taxon, so every taxon keeps its own trait frequencies. Expected count per cell = Σ over taxa n·∏ p(trait level within the taxon). Whatever deficit remains is what lineage composition cannot explain. Taxa with fewer than 5 species are excluded and counted (§7); stratified rows use the kept species for both observed and expected.

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

- **core panel (6 traits):** deficit 27.3 → 19.3 → 13.6 → 4.9 cells (global → phylum → class → order); it narrows at every step. Order level: 83 occupied vs 87.9 expected, P = 0.067.

- **no-spore panel (5 traits):** deficit 9.3 → 9.3 → 6.8 → 3.4 cells (global → phylum → class → order); it does not narrow monotonically. Order level: 66 occupied vs 69.4 expected, P = 0.11.

**Lower-dimensional maps.** Across all 22 4- and 5-trait subsets of the core traits, the smallest order-level P for an occupancy deficit is 0.0075; after BH correction across maps the smallest q is 0.099. No subset map retains a significant deficit at order level. Full table: `reports/tables/order_level_deficit_across_maps.tsv`.

### 3a. Where the occupancy comes from: per phylum


**core panel (6 traits).** 90 cells occupied. The three largest phyla (Pseudomonadota, Bacillota, Actinomycetota; 78.6% of species) account for 84 of them (93%). The largest within-phylum deficit is Bacillota: 56 cells vs 78 expected from its own trait frequencies.

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

**no-spore panel (5 traits).** 69 cells occupied. The three largest phyla (Pseudomonadota, Actinomycetota, Bacillota; 79.5% of species) account for 61 of them (88%). The largest within-phylum deficit is Bacillota: 41 cells vs 55 expected from its own trait frequencies.

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

### 3b. Cell-level results under all four nulls

Every cell that is empty and testable under at least one null, or flagged under at least one. `–` = untestable under that null. Every cell, every null: `reports/tables/cells_<panel>_all_nulls.tsv`.


**core panel (6 traits)**

| gram | shape | motility | spore | oxygen | temperature | observed (all species) | global: expected | global: q | phylum: expected | phylum: q | class: expected | class: q | order: expected | order: q | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| negative | coccus | no | yes | aerobe | meso | 0 | 11.21 | 0.000493 | 1.66 | – | 0.845 | – | 0.477 | – | phylogenetic structure — fires under global only; untestable at order level |
| negative | other | yes | yes | aerobe | meso | 0 | 5.04 | 0.0246 | 0.556 | – | 0.373 | – | 0.331 | – | phylogenetic structure — fires under global only; untestable at order level |
| positive | coccus | yes | no | anaerobe | meso | 0 | 4.42 | 0.0342 | 4.09 | 0.0491 | 1.56 | – | 1.73 | – | phylogenetic structure — fires under global/phylum only; untestable at order level |
| positive | rod | no | no | facultative | thermo | 0 | 3.91 | 0.0446 | 3.21 | 0.108 | 2.02 | – | 0.146 | – | phylogenetic structure — fires under global only; untestable at order level |
| positive | rod | yes | no | facultative | thermo | 0 | 2.93 | – | 3.91 | 0.0524 | 2.63 | – | 0.335 | – | empty, testable, not significant under any null |
| positive | coccus | no | yes | facultative | meso | 0 | 2.07 | – | 7.79 | 0.00183 | 8.99 | 0.00155 | 1.59 | – | phylogenetic structure — fires under phylum/class only; untestable at order level |
| positive | coccus | yes | yes | facultative | meso | 0 | 1.55 | – | 8.50 | 0.00149 | 10.91 | 0.000886 | 0.992 | – | phylogenetic structure — fires under phylum/class only; untestable at order level |
| positive | coccus | yes | yes | anaerobe | meso | 0 | 1.40 | – | 6.96 | 0.00335 | 2.26 | – | 0.84 | – | phylogenetic structure — fires under phylum only; untestable at order level |

**no-spore panel (5 traits)**

| gram | shape | motility | oxygen | temperature | observed (all species) | global: expected | global: q | phylum: expected | phylum: q | class: expected | class: q | order: expected | order: q | category |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| positive | coccus | yes | anaerobe | meso | 0 | 7.62 | 0.0018 | 20.53 | 0.000737 | 7.96 | 0.000589 | 5.44 | 0.0108 | SURVIVES order-level shuffle (both seeds) |
| negative | coccus | yes | aerobe | thermo | 0 | 3.01 | 0.0959 | 0.733 | – | 0.401 | – | 0.38 | – | empty, testable, not significant under any null |
| positive | coccus | yes | aerobe | thermo | 0 | 1.72 | – | 3.13 | 0.135 | 2.45 | – | 0.564 | – | empty, testable, not significant under any null |
| positive | coccus | no | facultative | thermo | 0 | 0.667 | – | 3.83 | 0.0649 | 2.85 | – | 0.145 | – | empty, testable, not significant under any null |
| positive | coccus | yes | facultative | thermo | 0 | 0.496 | – | 3.09 | 0.138 | 2.43 | – | 0.128 | – | empty, testable, not significant under any null |

Cells flagged under phylum or class but not globally arise when a taxon mixes sub-clades. Within Bacillota, for example, spore formers and cocci sit in different orders, so a phylum-level shuffle predicts spore-forming cocci that an order-level shuffle does not.

## 4. Positive control: the method detects a known constraint

**Gram-negative endospore-forming cocci**: `gram=negative / shape=coccus / motility=no / spore=yes / oxygen=aerobe / temperature=meso`

Endospore formation is essentially restricted to Bacillota, which are overwhelmingly Gram-positive, and coccoid endospore formers are rare even there (Sporosarcina, itself motile and Gram-positive). The cell stacks three phylogenetic facts; an independence null cannot see any of them.

This was the strongest hit of the first, global-only run. It is a check on the method, **not a finding**. Expected behaviour: the global null flags it, and a phylogeny-aware null attributes it to lineage.

| null | species used | observed | expected | testable | q | empty beyond chance |
|---|---|---|---|---|---|---|
| global | 2,580 | 0 | 11.21 | yes | 0.000493 | yes |
| phylum | 2,563 | 0 | 1.66 | no | – | no |
| class | 2,522 | 0 | 0.845 | no | – | no |
| order | 2,423 | 0 | 0.477 | no | – | no |

**Result: passed.** The expected count falls from 11.2 (global) to 1.7 (phylum), 0.8 (class) and 0.5 (order): within any one taxon the constituent traits barely co-occur, so the combination is not expected to be occupied in the first place.

## 5. No novel constraints survive

- **core panel (6 traits):** 0 cells survive the order-level shuffle (4 flagged by the global null).
- **no-spore panel (5 traits):** 1 cell survives the order-level shuffle (1 flagged by the global null).

### The one order-level survivor, and why it is not a finding: `gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso` (no-spore panel (5 traits))

- **Order level.** Observed 0; the order-level null expects 5.4 (q = 0.011; independent seed q = 0.0098).
- **Family level (diagnostic).** Expected falls to 1.56, below the testability floor.
- **Where the expectation comes from.** 45% comes from one order (Tissierellales, 20 species). 51% comes from orders in which no single family carries every constituent trait level, i.e. the combination is "expected" only because the order mixes families.
- **Sensitivity.** optimum-only: untestable; genus-level: not significant (q = 0.43).
- **In BacDive as a whole.** 1 of 102,187 strains, with or without a genome, has the combination: *Tonsilliphilus suis* (BacDive 22988, no genome).

Top contributing orders, and whether any family inside them carries every constituent level:

| order | expected in cell | species | families | rarest constituent level in this order | families carrying that level | families carrying every constituent level |
|---|---|---|---|---|---|---|
| Tissierellales | 2.44 | 20 | 8 | motility=yes | Clostridiisalibacteraceae, Gottschalkiaceae, Proteiniboraceae, Tepidimicrobiaceae, VENL01 | none |
| Actinomycetales | 1.07 | 455 | 15 | oxygen=anaerobe | Actinomycetaceae, Bifidobacteriaceae, Cellulomonadaceae, Demequinaceae | Actinomycetaceae, Cellulomonadaceae |
| Lachnospirales | 0.444 | 43 | 6 | shape=coccus | Lachnospiraceae | Lachnospiraceae |
| Peptostreptococcales | 0.398 | 28 | 7 | shape=coccus | Peptostreptococcaceae | Peptostreptococcaceae |
| Lactobacillales | 0.365 | 305 | 9 | oxygen=anaerobe | Aerococcaceae, Lactobacillaceae, Listeriaceae, Streptococcaceae | Lactobacillaceae, Streptococcaceae |

### Lower-dimensional maps

Order-level survivors in the 4- and 5-trait subsets (not corrected across maps). Projections of the same cell count as one observation, not independent support:

| map | cell | order: expected | order: q | global: q |
|---|---|---|---|---|
| gram + shape + motility + oxygen | gram=positive / shape=coccus / motility=yes / oxygen=anaerobe | 5.64 | 0.0102 | 0.00257 |
| gram + shape + motility + oxygen + temperature | gram=positive / shape=coccus / motility=yes / oxygen=anaerobe / temperature=meso | 5.44 | 0.0108 | 0.0018 |

**No transplant or follow-up experiments are proposed: there is no surviving target.**

## 6. Power: the smallest constraint this analysis could have detected

For a cell that is truly empty, its permutation p-value is p0 = P(null count = 0). With m testable cells, BH gives q ≤ p0·m. So p0 ≤ 0.05/m **guarantees** q < 0.05 whatever the other cells do, and since q ≥ p0, p0 ≤ 0.05 is **necessary**. Under a Poisson approximation p0 ≈ e^(−E), where E is the null's expected count. That gives E ≥ ln(m/0.05) guaranteed and E ≥ ln(1/0.05) ≈ 3.0 at best. The within-taxon permutation is usually less dispersed than Poisson, so the empirical p0 from the 10,000 permutations is also used. The **minimum detectable constraint** is the largest of the Poisson bound, the empirical threshold and the testability floor (3), so it holds conservatively.

| panel | null | species used | cells | testable cells (m) | E needed, guaranteed (Poisson ln(m/α)) | E needed, guaranteed (empirical, this null) | minimum detectable constraint (E) | as share of species | cells with E >= minimum detectable constraint | share of expected species mass in those cells |
|---|---|---|---|---|---|---|---|---|---|---|
| core panel (6 traits) | global | 2,580 | 216 | 69 | 7.23 | 7.31 | 7.31 | 0.28% | 41 | 91% |
| core panel (6 traits) | order | 2,423 | 216 | 52 | 6.95 | 5.51 | 6.95 | 0.29% | 37 | 95% |
| no-spore panel (5 traits) | global | 5,634 | 108 | 54 | 6.98 | 7.37 | 7.37 | 0.13% | 44 | 98% |
| no-spore panel (5 traits) | order | 5,416 | 108 | 49 | 6.89 | 5.70 | 6.89 | 0.13% | 39 | 98% |

**Statement.** At order level we could have detected, with q < 0.05, any constraint that empties a combination the lineage structure predicts to hold ≥ 6.9 species (core panel, 0.29% of species) or ≥ 6.9 species (no-spore, 0.13%). None was found. The 179 core-panel cells below that bar are untested territory, not evidence of occupancy. A constraint that thins rather than empties a cell would need a larger E still.

Expected species per cell by map size (global null, species level):


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

## 7. Methods note: dropping small strata from a stratified permutation test manufactures false positives

A within-stratum permutation cannot shuffle a stratum of one or two species, so the obvious fix is to drop small strata. Done naively, that creates false "forbidden combinations". Small strata are not a random subset of species: they are phylogenetically isolated lineages, deep-branching or sparsely described, and those are exactly the organisms that carry unusual trait combinations, i.e. the ones that populate rare cells. Dropping them removes a rare cell's only occupants. The null, computed from the large strata that remain, still predicts the cell from within-stratum trait frequencies. Observed zero, expected several: a confident, spurious "empty beyond chance".

It happened here. During development, before the guard existed, the first stratified run flagged `gram=negative / shape=rod / motility=no / oxygen=anaerobe / temperature=thermo` (no-spore panel (5 traits)) as empty beyond chance at order level, and it would have been reported as a novel constraint. The cell holds 15 species. Every one of them sits in an order with fewer than 5 species in the panel (order sizes: 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4), so excluding small orders emptied it. In this run it has q = 0.059 against 3.5 expected among kept species. With this run's seed it falls just short of q < 0.05: the false positive is threshold-marginal, which is exactly why it would have been easy to believe.

| panel | cell | species in cell (all) | species in cell (kept) | order-null expected (kept) | q it would have had | would have been flagged | order sizes of its species |
|---|---|---|---|---|---|---|---|
| core panel (6 traits) | gram=positive / shape=rod / motility=no / spore=yes / oxygen=anaerobe / temperature=thermo | 4 | 0 | 3.84 | 0.052 | no | 2, 2, 4, 4 |
| no-spore panel (5 traits) | gram=negative / shape=rod / motility=no / oxygen=anaerobe / temperature=thermo | 15 | 0 | 3.47 | 0.0593 | no | 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4 |

**Fix used here.** A cell counts as "empty beyond chance" under a stratified null only if it is empty among *all* species. Excluded species still count as occupants; they are just not shuffled. Cells emptied only by the exclusion are labelled `emptied_by_exclusion` and reported. An equivalent alternative keeps small strata as fixed, unpermuted blocks that contribute identically to observed and null counts. The general point applies to any stratified or blocked permutation test with a minimum block size: phylogenetically stratified trait tests, stratified enrichment tests, case-control tests blocked by site. The exclusion rule is itself a selection on the covariate that defines the strata, and it must not change the observed statistic.

Other settings: a cell is tested only if its expected count is ≥ 3 under that null; BH correction within each map over its testable cells; 10,000 permutations per full map (2,000 per subset map); the global and order nulls re-run with an independent seed. Species kept per null:


**core panel (6 traits)**

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

**no-spore panel (5 traits)**

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

---

## Appendix A. Snapshot and attrition

| source | version | retrieved | detail |
|---|---|---|---|
| BacDive | API v2, record DOIs stamped `20260601` (BacDive exposes no release number) | 2026-09-29 → 2026-09-29 | IDs 1–250,000 swept in full (2,500 batches of 100); highest ID found 179,194; genome-based predictions not requested |
| GTDB | v232 (Released Apr 15, 2026) | 2026-09-29 | bac120_metadata.tsv.gz, 878,998 genomes, sha256 `1650d3164666` |

Empty BacDive ID ranges ≥1,000 IDs: 24,901–99,900, 128,801–130,100, 179,201–250,000.

Counts are strains. "Usable" = a normalised value exists (≥1 observation, mappable, no conflict between references; conflicts null the value).

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

**Per-trait losses**

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

**Unmapped and unparsed raw values** (kept in the `*_raw` columns; full list `data/interim/unmapped_values.tsv`):

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

**Temperature bin sources, core panel species**

| temperature_bin_source | N | % |
|---|---|---|
| optimum | 2,162 | 83.8% |
| growth_range | 296 | 11.5% |
| growth_single_point | 122 | 4.7% |

Oxygen: 30 core-panel species are "facultative" only because microaerophile was folded in; 663 of the 1,390 coarse oxygen conflicts across all strains involve a microaerophile observation.

## Appendix B. Genome matching

Route 1 is BacDive's own INSDC assembly accession, matched to GTDB unversioned. Route 2, used only when route 1 fails, is exact equality of normalised strain designations. A designation without a culture-collection code needs taxon agreement (same genus, or same NCBI taxon ID). There is no fuzzy matching.

**What normalisation bought.** Steps A–C use the designation route alone, under one acceptance rule; step E equals the final match count.

| step | all strains n | all strains % | core panel n | core panel % | no-spore panel n | no-spore panel % |
|---|---|---|---|---|---|---|
| A. designation, raw whole strings (before normalisation) | 8,008 | 7.84 | 613 | 17.96 | 1,376 | 13.68 |
| B. designation, + split into tokens, raw | 19,443 | 19.03 | 2,429 | 71.15 | 5,553 | 55.23 |
| C. designation, + normalised keys (after normalisation) | 21,361 | 20.90 | 2,530 | 74.11 | 5,972 | 59.39 |
| D. accession route alone | 18,765 | 18.36 | 2,543 | 74.49 | 5,833 | 58.01 |
| E. final: accession or normalised designation | 22,835 | 22.35 | 2,630 | 77.04 | 6,184 | 61.50 |
| (info) raw tokens without the taxon check — includes cross-genus collisions, not a match count | 27,177 | 26.60 | 2,673 | 78.30 | 6,610 | 65.74 |

**Genome attrition, all strains**

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

**Genome attrition, core-panel strains**

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

**Match quality**

| check | value | N |
|---|---|---|
| designation route agrees with accession route (both fire) | 99.94% | 17,291 |
| bare-designation hits agree with accession route | 99.91% | 5,466 |
| matched genome's GTDB/NCBI genus equals BacDive genus | 95.2% | 22,835 |
| designation matches accepted via NCBI tax ID only | 3.3% | 134 |
| candidates span >1 GTDB species | 1.2% | 22,835 |

**Unmatched strains** (79,352; listed in `data/interim/unmatched_strains.tsv`, gitignored, regenerated by `join`):

| reason | strains |
|---|---|
| no_bacdive_assembly;designations_not_in_gtdb | 74,182 |
| no_bacdive_assembly;bare_designation_taxon_mismatch | 3,564 |
| no_bacdive_assembly;no_usable_designation | 804 |
| bacdive_assembly_not_in_gtdb;designations_not_in_gtdb | 654 |
| bacdive_assembly_not_in_gtdb;bare_designation_taxon_mismatch | 148 |

## Appendix C. Trait coverage and genome availability

Strains with usable values for both traits (diagonal = single-trait coverage):

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

Lift = P(both) / (P(row)·P(column)); 1 = independent coverage:

| index | gram stain | cell shape | motility | spore formation | oxygen tolerance | temperature |
|---|---|---|---|---|---|---|
| gram stain | – | 6.39 | 6.38 | 6.14 | 3.68 | 1.95 |
| cell shape | 6.39 | – | 6.51 | 5.82 | 3.64 | 1.94 |
| motility | 6.38 | 6.51 | – | 6.23 | 3.76 | 1.97 |
| spore formation | 6.14 | 5.82 | 6.23 | – | 3.82 | 1.99 |
| oxygen tolerance | 3.68 | 3.64 | 3.76 | 3.82 | – | 1.99 |
| temperature | 1.95 | 1.94 | 1.97 | 1.99 | 1.99 | – |

**Panel completeness vs independent coverage.** Coverage is nested: traits come together from species descriptions.

| panel | traits | observed_complete | observed_frac | product_of_marginals | expected_complete_if_independent | ratio |
|---|---|---|---|---|---|---|
| morphology (4) | 4 | 4,047 | 0.0396 | 0.000158 | 16.12 | 251 |
| no-spore (5) | 5 | 10,055 | 0.0984 | 0.000313 | 31.98 | 314 |
| core (6) | 6 | 3,414 | 0.0334 | 1.66e-05 | 1.70 | 2,011 |
| core + pH (7) | 7 | 2,807 | 0.0275 | 1.1e-06 | 0.112 | 25,074 |
| original 8 | 8 | 153 | 0.0015 | 3.87e-09 | 0.000395 | 386,868 |

**Genome availability.** Base rate: 19.2% of all strains list a GCA accession in BacDive. Fisher p-values below 1e-300 are shown as <1e-300.

| set | N | BacDive lists a GCA n | BacDive lists a GCA % | BacDive lists a GCA base % | BacDive lists a GCA odds ratio vs rest | BacDive lists a GCA Fisher p | GTDB genome matched n | GTDB genome matched % | GTDB genome matched base % | GTDB genome matched odds ratio vs rest | GTDB genome matched Fisher p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all BacDive strains | 102,187 | 19,639 | 19.22 | 19.22 | – | – | 22,835 | 22.35 | 22.35 | – | – |
| morphology panel complete (4) | 4,047 | 3,048 | 75.32 | 19.22 | 15.00 | <1e-300 | 3,126 | 77.24 | 22.35 | 13.51 | <1e-300 |
| core panel complete (6) | 3,414 | 2,585 | 75.72 | 19.22 | 14.94 | <1e-300 | 2,630 | 77.04 | 22.35 | 13.04 | <1e-300 |
| core panel complete, optimum temperature | 2,680 | 2,173 | 81.08 | 19.22 | 20.13 | <1e-300 | 2,189 | 81.68 | 22.35 | 17.03 | <1e-300 |

## Appendix D. Sensitivity views, phyla, type strains

Global and order-level nulls in each view (primary = one strain per GTDB species, coarse bins, all temperature sources).

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

**GTDB phyla (species level)**

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

**Type strains, core panel**

|  | N | % |
|---|---|---|
| type strain per BacDive | 2,539 | 98.4% |
| type strain of species per GTDB/NCBI | 2,468 | 95.7% |
| either | 2,549 | 98.8% |
| both | 2,458 | 95.3% |
| BacDive says type, GTDB genome not flagged as type | 81 | 3.1% |

## Files

- `data/final/strains.parquet` (gitignored): every BacDive strain, with raw and normalised traits, genome match and panel flags. Regenerate with `python -m src.pipeline all`; checksums are in `README.md` and `reports/run_manifest.json` → `outputs`.
- `data/final/{core,no_spore}_panel_species.tsv`: the analysed sets.
- `data/final/assembly_accessions_*.txt`: inputs for `datasets download genome accession --inputfile …`.
- `data/interim/unmatched_strains.tsv` (gitignored), `data/interim/unmapped_values.tsv`: audit logs.
- `reports/tables/*.tsv`: every table above, and per-cell results for every panel × view × null.
- `reports/run_manifest.json`: versions, hashes, counts.
