# Physics P6 — Allen–Cahn crystallization

Crystallinity is a **non-conserved** order parameter (crystal can be
created), so it obeys Allen–Cahn rather than Cahn–Hilliard. This chapter
isolates crystallization through four experiments:

1. **grow / melt** — a seeded crystal grows below Tm and melts above it,
   measured by the **quadrature** crystallinity ⟨ψ⟩ = ∫ψ dV / ∫dV (the
   thresholded ψ>0.5 area is a secondary check);
2. **interface velocity** v as a function of undercooling ΔT = Tm − T
   (rises monotonically — constant mobility, no thermal maximum);
3. **critical radius** r\* separating growing from redissolving seeds,
   at two undercoolings (r\* ∝ 1/|drive|, shrinks with undercooling);
4. **Avrami/JMAK** kinetics — the crystalline fraction fit to a
   baseline-corrected exponent n ± CI with R².

**Read** the course document, Chapter *"Allen–Cahn crystallization"*
(start with `crystallization.py`).

**Run:**
```bash
python run.py                 # full battery + baseline checks, 32x32
python run.py --level 6       # finer mesh (slower)
```

`run.py` prints the four experiment tables, writes `outputs/results.json`,
and checks it against `baseline.yaml` (tolerance-based, not bit-identical);
compare with [`EXPECTED.md`](EXPECTED.md). Figures + numbers via
`python gen_figures.py`.

| file | role |
|------|------|
| `crystallization.py` | the core: `run_grow_melt`, `run_interface_velocity`, `run_critical_radius`, `run_avrami`, `quad_mean_psi` |
| `run.py` | driver; prints the battery, writes + checks results |
| `gen_figures.py` | regenerates `p6_*.png` + `numbers/p6.tex` |
| `baseline.yaml` | tolerance-based scientific-invariant checks |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=1) with the
r14 crystallization energetics (PCBM-class, materials.yaml) and
`diffsim.diagnostics.conservation` for the quadrature crystallinity.

**Accelerated parameters:** the Avrami run uses T = 250 K and L_psi = 11,
pedagogical values (not the physical PCBM rate) chosen so X sweeps the
full range in a short run. **Orientation θ is a fixed grain label**, not
an evolved field — evolving θ is the coupled concept of P7.
