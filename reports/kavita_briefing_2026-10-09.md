# Kavita briefing — October 9, 2026

**Stage 1:** In BacDive type strains matched to GTDB v232, 2,580 species fill
90 of 216 six-trait cells. The 27-cell deficit against independent traits
shrinks to five under the order-level phylogenetic null (P = 0.067), with no
core cell surviving. Coarse traits, type-strain sampling, and power only for
zeros expected to contain about seven or more species limit any claim about
unobserved combinations. Empty cells do not prove viability or impossibility.

**Why the benchmark matters:** Adrian's preliminary Evo 2 result—better trait
prediction within clades, toward chance across distant clades, with taxonomy
about as good—is unreproduced. Similarity of relatives, repeated assemblies,
and possible pretraining exposure can inflate a random held-out score.

**Design and first evidence:** We audited unique assemblies and labels, added
GTDB v232 family labels, and ran 20 random-species and held-genus, family, and
order splits per trait. A prevalence comparator and a training-fold-only
taxonomy predictor were evaluated with balanced accuracy and macro F1. In the
larger five-trait panel (N = 5,634), motility taxonomy balanced accuracy falls
from .820 (.800–.833 across splits) to .606 (.521–.754) with held orders;
oxygen falls from .824 (.808–.848) to .516 (.298–.745). Gram remains .950
at held order. Spore uses N = 2,580. There is **no k-mer or Evo 2 score yet**:
the assembly FASTAs are absent from this checkout. GTDB v220 contains exact
accessions for 4,993 of 5,634 larger-panel genomes, so OpenGenome2 exposure
is plausible, but actual training inclusion is unverified. See the
[Stage 2 report](stage2_report.md) for all traits, macro F1, and limitations.

**Decisions to make:**

1. Approve a prespecified, clade-balanced FASTA subset and acquisition source
   for the 4-mer baseline, with missing-genome accounting.
2. Decide whether to restrict Evo 2 evaluation to accession-negative genomes
   while pursuing a stronger sequence-overlap audit; accession-negative is
   not guaranteed unexposed.
3. Prioritize which traits and taxonomic distances are biologically useful,
   and whether to regenerate BacDive raw/conflict fields for label review.

**Separate lead:** Cornell's
[oVert](https://news.cornell.edu/stories/2024/04/vertebrate-3d-scan-project-opens-collections-all)
offers vertebrate specimen CT scans for a future morphology effort. It is not
a source of bacterial traits.
