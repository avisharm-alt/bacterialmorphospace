# bacterialmorphospace

Stage 1 of an occupancy map of bacterial trait space. The question here is not modelling but feasibility.
How many bacterial strains have **both** a complete panel of curated phenotype traits (BacDive) **and** a
matched sequenced genome (GTDB/NCBI)? Which trait combinations are empty **beyond what the marginal trait
frequencies predict**?

The deliverable is [`reports/attrition_report.md`](reports/attrition_report.md).

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

The **core panel** is the first six traits. pH (about 7% coverage) and curated halophily (under 0.5%) are kept as columns
but excluded from the occupancy analysis.

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
```

## Occupancy analysis

- **Primary unit:** one strain per GTDB species. Type strains are preferred, then the best assembly.
- **Sensitivities:**
  - no dereplication
  - temperature from a reported optimum only
  - fine bins
  - one strain per GTDB genus
- **Null:** each trait column is permuted independently, which preserves every marginal. For each cell, the report gives the
  observed count, the expected count, the one-sided permutation p-value for "emptier than chance", and a BH q-value over
  *testable* cells. A cell is testable when its expected count is at least `min_expected_for_test` (default 3).
- **Headline category:** *empty beyond chance*, meaning observed 0, testable and q < α. Occupied cells that are significantly
  depleted are reported separately, because with fixed marginals a real gap forces compensating depletion elsewhere.

## Layout

```
src/            pipeline modules
tests/          unit tests (normalisation, traits, join, analysis)
config.toml     every tunable choice
data/raw/       cached API responses + GTDB download       (gitignored)
data/interim/   extracted traits, unmatched log, unmapped values, snapshot info
data/final/     strains.parquet (joined table), core-panel species table, accession lists
reports/        attrition_report.md, tables/*.tsv, run_manifest.json
notebooks/      exploration only; nothing load-bearing
```
