# P9 — expected results (self-check)

Running `python run.py` (defaults: **level 5 = 32×32**, M=2/K=1 film mode,
r14 with the crystal-contact solubility, χ_ca = 1.6) derives the r14
solubility, validates it by an embryo composition sweep, runs the
sub/supercritical embryo control, and runs the evaporation-conditioned
arc; then checks `outputs/results.json` against `baseline.yaml`.

Representative measured values (level 5, cuDSS):

| quantity | value |
|---|---|
| derived **homogeneous** solubility φ\* = 1 − \|drive\|/χ_ca (drive −0.527, χ_ca 1.6) | 0.671 |
| embryo composition-sweep crossover (supercritical embryo) | bracketed, **below** φ\* |
| supercritical embryo (r₀ = 0.20, φ_f = 0.85) | grows |
| subcritical embryo (r₀ = 0.05, φ_f = 0.85) | dissolves (Gibbs–Thomson) |
| WET implant: φ_s at implant / terminal area / ψ_max | ~0.84 / ~0 / ~0 |
| DRY implant: φ_s at implant / terminal area / ψ_max | ~0.63 / ~1.0 / ~0.99 |

**What must be true regardless of hardware** (the `baseline.yaml`
invariants):

- **The solubility φ\* is DERIVED from the free energy, then VALIDATED.**
  Comparing the r14 homogeneous free energy at ψ = 1 vs ψ = 0 at fixed
  composition gives φ\* = 1 − \|drive\|/χ_ca: below it the crystal-contact
  penalty beats the undercooling drive and an embryo redissolves; above it
  it grows. The embryo composition sweep (implant into uniform blends of
  varying φ_f, **no drying**) locates the grow/dissolve crossover.
  **Honest finding:** φ\* is the *homogeneous* solubility; a supercritical
  embryo self-enriches φ_f in its neighbourhood (the P7 crystal-bulk
  channel), so its *effective* growth threshold sits **below** φ\* — the
  derivation is an **upper bound**, and the measured crossover confirms a
  composition threshold exists while lying below φ\*.
- **These runs IMPLANT an embryo — they do not spontaneously nucleate.**
  This is *evaporation-conditioned embryo growth*, not nucleation.
  (Genuine FDT nucleation is the advanced `make_stepper(noise_psi=...)`
  mode, cross-referencing P8.)
- **Seeding has a Gibbs–Thomson floor.** At the same super-solubility
  composition a large embryo (r₀ = 0.20) grows but a small one
  (r₀ = 0.05) redissolves — the critical radius, not just the composition.
- **Wet dissolves, dry grows.** The same embryo implanted in the wet film
  (local φ_f below φ\*) dissolves; implanted mid-drying (drying has raised
  φ_f above φ\*) it grows to a crystalline film. The fate is set by the
  drying-**conditioned local composition** relative to φ\* — *not by time
  or evaporation per se*, which is exactly what the no-drying composition
  sweep isolates.
- **Termination is reported honestly.** A stop at the time horizon is a
  `time_horizon` stop, **not** a "drying time"; other statuses are
  `solvent_target` (dryness), `height_floor`, `min_dt` (stiffness), and
  `step_budget`.

**Recorded simplification.** Constant Onsager mobility with cuDSS instead
of the production Vignes composition-singular mobility
(`tests/test_multiphase_s3.py`); the qualitative arc is unchanged, the
drying-front sharpness is softened. Read the **terminal** state, not a
mid-growth transient. Third-significant-figure differences from a different
card are normal.
