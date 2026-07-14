# P8 — expected results (self-check)

Running `python run.py` (defaults: FDT well on a coarse CPU/splu mesh;
nucleation ensemble at **level 5 = 32×32**, p1 bulk, undercooled melt
T = 0.5·Tm, ψ starts at 0 with NO seed, BDF1) runs the central FDT
verification and the noise-amplitude ensemble sweep, then checks
`outputs/results.json` against `baseline.yaml`.

Representative measured values:

**Central FDT verification** (fix the physical noise, measure equilibrium
variance in a stable quadratic well):

| dt | 3.0e−3 | 1.5e−3 | 7.5e−4 |
|---|---|---|---|
| equilibrium Var ψ | 0.0479 | 0.0460 | 0.0461 |

→ dt-CoV of the variance = **0.022** (dt-independent → the discrete FDT
normalization is verified). Across meshes Var·V_cell is constant to
CoV = **0.11** (equipartition Var ∝ 1/V_cell).

**Noise-amplitude ensemble sweep** (4 seeds/amplitude):

| noise_psi | P(nucleate) | X mean ± sd | nuclei density | clip |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.000 | 0 | 0.000 |
| 0.03 | 1.00 | 0.523 ± 0.015 | 16.5 | 0.148 |
| 0.06 | 1.00 | 0.505 ± 0.011 | 19.2 | 0.205 |

**What must be true regardless of hardware** (the `baseline.yaml`
invariants):

- **The discrete FDT normalization is DERIVED then VERIFIED.** The per-GP
  noise std carries √(2L/(dt·wJ)) so the assembled nodal-force variance is
  mesh/dt-independent. The verification fixes kᵦT, puts ψ in a stable
  quadratic well (clip-free r14, two-sided fluctuations), and shows the
  equilibrium variance converges to a **dt-independent** plateau (a wrong
  1/dt normalization would make it scale with dt) and scales as 1/V_cell
  across meshes (equipartition). This is the central result, not a
  footnote.
- **Zero noise stays amorphous.** With `noise_psi = 0` the melt does not
  nucleate (X ≈ 0, P = 0): ψ = 0 is metastable behind the barrier. A
  nonzero X at zero noise means a bug.
- **Larger noise nucleates more.** The nuclei density increases with the
  amplitude (16.5 → 19.2). Nucleation is a random rare-event process, so
  each amplitude is reported as an **ensemble** (nucleation probability,
  crystalline-fraction distribution, nuclei density, induction time) with
  a mean and a 95% interval via `diffsim.diagnostics.stochastic`, not a
  single seed. At these amplitudes nucleation is essentially certain
  (P = 1); the ensemble quantifies the spread. The induction time is a
  first-passage crossing of X = 0.02.
- **Clipping is quantified.** The undercooled runs clip ψ to [0, 1]; the
  saturated (ψ > 0.99) fraction is reported (≈0.15–0.21). The ψ = 0 floor
  rectifies sub-barrier fluctuations — exactly why the quantitative FDT
  test uses the clip-free r14 well.

**This is a noise-amplitude (kᵦT calibration) sweep, NOT a temperature
sweep**: only `noise_psi` varies, at fixed undercooling. A real
temperature change would also move the driving force and the barrier.

**BDF1 only** (the brick asserts noise off under BDF2). Third-significant-
figure differences from a different card are normal.
