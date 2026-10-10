# jump_g2p: gene in, Cell Painting morphology out (JUMP CRISPR, human U2OS)

A gene-conditioned diffusion model that generates Cell Painting morphology profiles for the CRISPR knockout of a
human gene, tested on genes it never saw.

- **Data:** JUMP cpg0016 CRISPR (source_13): 7,968 knocked-out genes, ~5 wells each, 599 Cell Painting features
  (sphered + Harmony), reduced to 32 whitened PCs. Gene information: DepMap 24Q2 dependency profiles, STRING v12
  (no text mining or curated databases), co-essentiality neighbours.
- **Model:** conditional diffusion (keio_g2p/diffusion.py, adapted from the fly-wing model) over well profiles,
  conditioned on a cross-fitted gene code (predicted mean profile from the three gene inputs).
- **Test:** 10 folds holding out whole chromosome arms (CRISPR proximity bias), paralogs purged.

```
python -m jump_g2p.predictability raw/ derived/human_homolog_pairs.tsv results/      # can held-out genes be predicted?
python -m jump_g2p.diffusion_cv  raw/ derived/human_homolog_pairs.tsv results/diffusion  # held-out diffusion test (Modal)
python -m jump_g2p.bundle        raw/ derived/human_homolog_pairs.tsv model/           # final model on all genes (Modal)
python -m jump_g2p.predict NUP93                                                        # gene in, morphology out
```

`predict` works for any of ~18.5k genes in DepMap or JUMP and writes generated profiles, a summary (effect strength,
closest imaged knockouts) and a figure with real Cell Painting fields of the wells closest to the prediction next to a
non-targeting control. The images are retrieved real wells for illustration, not generated pixels.

**Result (2026-10-10).** On 7,968 held-out genes the gene-conditioned model beats the same network without the gene
(energy distance 2.095 vs 2.150, Wilcoxon p = 2e-12) and with shuffled genes (2.281, p = 2e-58); among the most
reproducible quarter of knockouts the gain is 6% (3.38 vs 3.60). Never-imaged examples: NUP93 → closest imaged
knockouts AHCTF1 (ELYS) and NUP153 (nuclear pore); SF3B1 → EFTUD2, SNRNP200 (spliceosome); RPL3 → EIF2S2, DDX10.
