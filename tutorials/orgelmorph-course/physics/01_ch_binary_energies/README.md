# Physics P1 — Binary Cahn–Hilliard: two free energies

The first concept. A binary blend demixes to lower its Ginzburg–Landau
free energy; you watch that energy decay and split into bulk and
interfacial parts, for two free-energy models (polynomial double-well
and Flory–Huggins).

**Read** the course document, Chapter *"Binary Cahn–Hilliard"* (start
with `spinodal.py` — the importable core it walks through).

**Run:**
```bash
python run.py                 # both free energies, 64x64, ~1-2 min on CPU
python run.py --energy fh     # just Flory-Huggins
python run.py --level 7       # 128x128 (finer; slower on the CPU solver)
```

`run.py` prints an energy-budget table; compare it with
[`EXPECTED.md`](EXPECTED.md). Field/energy arrays are saved to `out/`.

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/ + ../../latex/numbers.tex
```

| file | role |
|------|------|
| `spinodal.py` | the core: `run_spinodal`, `energy_budget` — read this first |
| `run.py` | the driver you run; prints the self-check table |
| `gen_figures.py` | regenerates the figures and `numbers.tex` the document shows |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` directly — no toy solver.

## Learning objectives

By the end of this chapter you can:

- Write the Ginzburg–Landau free energy and derive the conserved
  (Model-B) Cahn–Hilliard gradient flow, splitting `F` into bulk and
  interfacial parts.
- Explain why the interface width scales as `ℓ ~ √(κ/W)` (not `√κ`) and
  count how many mesh cells it spans — i.e. judge resolution.
- Predict the spinodal's fastest-growing wavelength `λ* = 2π/k*` from the
  dispersion relation and **verify** it against the measured `S(q)` probe.
- Distinguish the three statements about "the energy decreases"
  (continuous Lyapunov / discrete energy-stability / one monotone run)
  and measure the last two.
- Explain why Flory–Huggins needs a log-regularization *and* a box
  projection, and report the cost (projected dofs, mass drift).
- Fit a coarsening exponent `L(t) ~ t^n` with an honest uncertainty and
  say why the length definition and fit window matter.

## Prerequisites

- **Concepts:** partial derivatives and the calculus of variations
  (functional derivative `δF/δc`), Fourier modes, basic thermodynamics of
  mixing (entropy vs enthalpy). Python + NumPy.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment green).
  This is the first physics chapter — no earlier physics chapter is
  assumed.

## Expected cost

- **Device:** any CUDA GPU; the meshes are 2-D and use well under 1 GB of
  device memory, so an 8 GB laptop card is ample. Solver `splu` (the CH
  saddle is indefinite — see C4; cuDSS is *not* used here).
- **Quick mode** (`--mode quick`, 32×32 / 60 steps): **≈ 75 s wall**
  measured on an RTX 6000 Ada (most of it Warp kernel compile + Python
  start-up); target < 2 min.
- **Reference mode** (64×64 / 250 steps, the EXPECTED numbers): a few
  minutes; target 5–30 min.
- **Research mode** (128×128 / 600 steps, 5-seed ensemble): tens of
  minutes.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The four figures (`p1_morph_*`, `p1_energy`, `p1_dispersion`,
   `p1_coarsening`) regenerated from your saved run.
4. **Headline:** the coarsening exponent `n ± sd` (interfacial-area
   length) for *both* free energies, with the fit window stated.
5. **Verification:** the dispersion overlay — measured fastest mode vs
   predicted `k*` — and the largest positive stepwise energy increment
   (state whether the scheme is discretely energy-stable here).
6. **Failure:** drive Flory–Huggins into the box projection (coarsen the
   mesh or raise `B`) and report `proj_dofs` and mass drift rising
   together.
7. **Exploration:** answer one "Explore on your own" question with a plot
   (the `√(κ/W)` width scaling is the recommended one).
8. **Research bridge:** one paragraph on what `λ*` and the coarsening
   exponent predict for a real cast blend's domain size.
