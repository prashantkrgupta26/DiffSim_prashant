# Physics P4 — Ternary Cahn–Hilliard and the Gibbs phase diagram

Real cast blends are (at least) three components: two solutes plus a
solvent. With two independent conserved compositions the free energy is
a surface over the **Gibbs triangle**, and demixing follows **tie-lines**
on that triangle. This concept runs the coupled ternary Cahn–Hilliard
system and visualizes the composition trajectory on the ternary phase
diagram — the natural way to read multi-component morphology.

**Read** the course document, Chapter *"Ternary Cahn–Hilliard"* (start
with `ternary.py`).

**Run:**
```bash
python run.py                 # ternary quench, 64x64
python run.py --level 6
```

`run.py` prints how the composition cloud spreads and the coexisting
phases; compare with [`EXPECTED.md`](EXPECTED.md). The signature
Gibbs-triangle figure comes from `python gen_figures.py`.

| file | role |
|------|------|
| `ternary.py` | the core: `run` + the barycentric `bary_to_xy` map |
| `run.py` | driver; prints the spread/tie-line table |
| `gen_figures.py` | regenerates `p4_*.png` + `numbers/p4.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.ternary_ch.TernaryCHStepper` (fields φ₁, μ₁, φ₂, μ₂;
solvent eliminated) with realistic Flory χ parameters.
