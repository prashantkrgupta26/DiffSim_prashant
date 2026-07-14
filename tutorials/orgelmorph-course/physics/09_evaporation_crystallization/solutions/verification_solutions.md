# Selected full solutions — P9 verification exercises

*Full worked solutions for the two **verification** exercises this
chapter's headline draws on: the derived solubility validated by the
embryo composition sweep, and the sub/supercritical critical-radius
control. Hints for all "Explore on your own" questions are in
`hints.md`. Instructor-only — do not distribute before the deadline.*

---

## V1 — The derived solubility φ*, validated by the embryo composition sweep

**Claim.** Comparing the r14 homogeneous free energy at ψ=1 vs ψ=0 at
fixed composition gives `Δf = φ0·drive + φ0·χ_ca·(1−φ0)`; the crystal is
favoured (`Δf<0`) iff `φ0 > φ* = 1 − |drive|/χ_ca`. This is the
*homogeneous* solubility. An independent, no-drying embryo composition
sweep should bracket a real grow/dissolve crossover, and — because a
supercritical embryo self-enriches its local `φ0` as it orders (the P7
crystal-bulk channel) — the measured crossover should lie *below* the
derived `φ*`, confirming `φ*` is an upper bound rather than an exact
threshold.

**Procedure.**
1. Compute `φ* = 1 − |drive|/χ_ca` from `solubility_threshold` using the
   tutorial's `drive` and `χ_ca`.
2. Run `embryo_composition_sweep`: implant the *same* supercritical
   embryo into uniform blends of increasing `φ_f` with **no drying**,
   and classify grow vs dissolve at each composition.
3. Bracket the crossover composition from the sweep.
4. Compare the bracket to `φ*` and confirm the direction (below, not
   above) of any gap.

**Expected result.** From `EXPECTED.md`: `drive = −0.527`, `χ_ca = 1.6`
gives derived `φ* = 0.671` (homogeneous). The composition sweep brackets
a crossover that lies **below** `φ* = 0.671` — a supercritical embryo
implanted into a blend just under the homogeneous threshold still grows,
because as it orders it locally enriches `φ0` above the ambient value
(the crystal-bulk channel), pushing the *effective* local composition at
the embryo above the homogeneous crossover even when the *bulk* ambient
composition sits below it. The derivation is therefore an honest upper
bound: it correctly identifies that a threshold exists and gives its
right order of magnitude, but a real (self-enriching) embryo grows at
somewhat lower ambient compositions than `φ*` alone would predict.

**Common wrong conclusion.** "The sweep crossover (below φ*) means the
derivation is wrong or the code has a bug." No — the gap is the
*expected*, physically-explained signature of the P7 crystal-bulk
self-enrichment channel operating in a spatially-resolved simulation
that the homogeneous (well-mixed) derivation cannot capture. Reporting
`φ*` alone without the validating sweep — or reporting the sweep without
explaining the direction of the gap — is the incomplete version of this
result; the full solution states both numbers and the mechanism
connecting them.

---

## V2 — Sub/supercritical embryo radius control at fixed composition

**Claim.** At a fixed super-solubility composition (`φ_f` above the
composition-sweep crossover), a large embryo (`r0=0.20`) grows while a
small one (`r0=0.05`) redissolves — composition being favorable is
necessary but not sufficient; the embryo must also clear the
Gibbs–Thomson critical radius `r*` (P6's interface-vs-bulk energy
balance, `r* ~ σ/|drive|`).

**Procedure.**
1. Fix a composition `φ_f` known (from V1's sweep) to be super-solubility
   (above the measured crossover).
2. Implant a supercritical-radius embryo (`r0=0.20`) and march; record
   terminal fate.
3. Implant a subcritical-radius embryo (`r0=0.05`) at the *same*
   composition and march; record terminal fate.
4. Confirm the two runs diverge (grow vs dissolve) despite identical
   composition, isolating radius as the controlling variable.

**Expected result.** From `EXPECTED.md`: at `φ_f = 0.85` (comfortably
super-solubility, above both `φ* = 0.671` and the sweep crossover), the
`r0=0.20` embryo **grows**, while the `r0=0.05` embryo **dissolves**
(Gibbs–Thomson). Both runs share the same ambient composition and
undercooling — the only difference is radius — so the divergence in
outcome isolates the critical-radius mechanism cleanly, exactly as P6's
`run_critical_radius` control does for the Allen–Cahn model alone.

**Common wrong conclusion.** "If the composition is above φ*, any embryo
will grow — composition is the only thing that matters." This is
false and the entire point of the control: composition sets whether
growth is thermodynamically *possible* (a large enough embryo would
grow), but a small enough embryo pays too much interfacial energy
relative to its bulk gain (the same `2πrσ` vs `πr²|drive|` balance from
P6) and redissolves regardless of favorable ambient composition. A
report that only checks composition and skips the radius control has
not verified the seeding-is-delicate lesson this chapter states
explicitly.
