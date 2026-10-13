# Antscan: does a genome beat the family tree at predicting ant 3D shape?

Antscan (Katzke et al., *Nature Methods* 2026) has synchrotron CT scans of about 2,200 ants. GAGA
(Vizueta et al., *Cell* 2025) has 163 annotated ant genomes. This package asks whether an ant's genome
predicts the 3D shape of a species from a **genus the model has never seen**, beyond what the family tree alone gives.

## Result (2026-10-13, 107 species, 199 worker scans, 79 genera, 10 subfamilies)

Leave-one-genus-out, kernel ridge from a species code to the species' mean worker shape. R² is relative to
predicting the training mean.

| code | R² (all shape blocks) | size block | thickness block |
|---|---|---|---|
| subfamily only | −0.017 | 0.097 | 0.062 |
| **family tree only** (GAGA dated phylogeny) | **0.065** | 0.226 | 0.063 |
| **genome** (16,805 gene-family copy numbers) | **0.057** | 0.214 | 0.069 |
| genome + tree | 0.051 | 0.221 | 0.019 |
| genome shuffled across species (30×) | −0.001 ± 0.002 | | |
| tree shuffled across species (30×) | −0.001 ± 0.002 | | |

- The genome and the tree both beat their shuffled versions (p = 0.03, the floor with 30 shuffles).
- **The genome does not beat the tree.** Genome − tree = −0.008 (95% bootstrap over genera −0.021 to +0.007);
  the genome is closer than the tree for 49% of species.
- **The genome adds nothing to the tree.** Genome + tree (0.051) is no better than tree + a shuffled genome
  (0.064 ± 0.003, p = 1.0). The inner loop mostly picks a 0.2 weight on the genome kernel.
- The gene-family kernel correlates r = 0.69 with the tree kernel: copy-number profiles mostly re-encode relatedness.
- Almost all of the predictable signal is **body size** (R² ≈ 0.22). Scale-free shape is barely predicted by either.

**Caveat, and the main limit of this pass:** two workers of the same species differ almost as much as two
species do (within-species share of variance 0.54; size is the exception, pair r = 0.91). Museum ants are posed
differently and worker castes vary in size, so whole-body descriptors are dominated by pose and polymorphism.
The thickness block (distance transform + granulometry, independent of pose) is the most repeatable
non-size measure (pair r up to 0.69). Part-level measurements (head, mesosoma, gaster) would be the next step.

## Pipeline

```
antscan_specimens_with_genome_flags.csv + GAGA_species_list.txt ──taxa.match──► 107 genomes ↔ 199 worker scans
biomedisa.info STL (100–400 MB each) ──mesh.fetch_and_process──► shapes/<specimen>.npz (deleted after reduction)
GAGA Orthogroup_table_N0_genecount + dated tree ──genome──► gene-family kernel, Brownian-motion tree kernel
run_baselines ──► results/baseline_metrics.json, per_species_errors.csv, predictions.npz
diffusion3d (+ modal_app) ──► genome/tree/none/shuffled-conditioned 3D DDPM on 32³ volumes, held-out genera
```

| Module | Role |
|---|---|
| `taxa.py` | Name normalisation and GAGA ↔ Antscan matching (hand-checked synonyms) and scan choice |
| `mesh.py` | Binary STL reader, largest component, size, PCA frame, shape distributions, appendage-free body core, thickness |
| `collect.py` | Parallel, resumable download → reduce → delete. `--all-taxa` fetches every taxon as a shape prior |
| `genome.py` | Gene-family features, Newick parser, phylogenetic covariance |
| `evaluate.py` | Kernel ridge with nested leave-one-genus-out, block-weighted R² |
| `run_baselines.py` | The held-out-genus comparison with shuffled nulls and a genus bootstrap |
| `diffusion3d.py` | 3D U-Net DDPM with classifier-free guidance; grouped-by-genus folds; pretraining on all other scans minus the test genera |
| `modal_app.py` | GPU entry point (not run: compute is paused) |

Units: Antscan STLs are in voxels, except the stitched 6.11 µm scans of large ants, which are already in µm
(`mesh.stl_unit_mm`).

## Re-running

```
python -m antscan.collect                       # 199 worker scans, ~1 h on 4 CPUs
python -m antscan.run_baselines --perms 30      # ~20 min single-threaded (set OMP_NUM_THREADS=1)
python -m antscan.collect --all-taxa --castes worker queen male --per-taxon 2   # 1,751 scans, many hours
python -m antscan.diffusion3d --config tiny      # CPU smoke test
python -m antscan.diffusion3d --config gpu --backend modal   # 20 runs (5 folds × 4 codes), needs a GPU
```

Data live in `/mnt/project-files/data/antscan/` (`gaga/`, `shapes/`, `results/`).
