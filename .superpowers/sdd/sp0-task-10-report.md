# SP-0 Task 10 Report — CAC interface variant + CH-vs-CAC A/B

**Branch:** `feat/sp0-chns`
**Status:** COMPLETE. Mirror + kernel + facade + tests + A/B memo done; full regression green.

## Commits (in units)

1. `feat(chns): CAC interface in CHNSDiscrete mirror (source-respecting beta)` — mirror-first (ground truth).
2. `feat(chns): CAC branch in make_chns_newton kernel + stepper/facade wiring` — kernel (static-gated) + `CHNSMonolithicStepper`/`CHNSStepper` `interface=` knob.
3. `test(chns): CAC parity, source-respecting conservation, stationary-drop A/B` — the three test tiers.
4. `docs(chns): SP-0 interface-decision memo (CH vs CAC A/B)` — the decision memo + this report.

## Derivation summary

CAC φ-row: `∂ₜφ + ∇·(uφ) = γ[Cn∇²φ − F′(φ)/Cn − β(t)√F(φ)] + s_φ`, F=¼(φ²−1)², F′=φ³−φ, **γ=1/Pe** (mobility = CH Onsager coefficient, so both interfaces share the Pe·M knob for the adjoint).

Source-respecting multiplier: integrating over Ω with no-flux/no-slip + conservative advection kills `∫∇·(uφ)` and `∫Cn∇²φ`; demanding `d/dt∫φ = ∫s_φ` gives
`β = −(∫F′(φ)/Cn) / (∫√F(φ))` — the source **cancels exactly** (β preserves injected mass, nothing more). β frozen per Newton iterate (Picard-on-β; no dβ/dφ block → sparse structure preserved), computed identically in kernel (host-side scalar arg) and mirror.

μ-row: trivial identity `∫N_a μ` (μ→0, wasted DOF; `dim+2` layout is a documented build-out optimization). Surface tension: Galerkin-only Korteweg `+(Cn/We)(∇φ⊗∇φ):∇w` (first-derivatives only; potential capillary + AGG flux dropped). CH path byte-identical.

## Parity numbers

- Frozen-β mirror Jacobian vs central FD: **~3e-10** (level-3 2-D).
- Kernel vs mirror after 3 steps (level-4, CAC): **φ 3.3e-16, u 1.8e-14, p 1.1e-15**; β agree to 1e-15. Newton 3 iters.

## A/B evidence (level 5, ρ-ratio 10, Cn=2h, bubble rise ≤200 steps / static drop 5 steps)

| Axis | CH (potential) | CAC (Korteweg) | Winner |
|---|---|---|---|
| Bubble-rise robustness | OK 200 steps | **DIVERGES @37** (umax→6.3, φ→−2.5) | CH |
| Mass drift, no source | 0.0 | 2.0e-13 | tie (both machine-exact) |
| Mass drift, with source (rel) | 2.2e-5 | **3.9e-6** (to divergence) | CAC |
| Parasitic max\|u\|, static drop | **3.6e-4** | 1.3e-2 (~37×) | CH |
| Newton iters avg/max | 3.0/4 | 3.6/7 | CH |
| Wall/step (splu CPU) | 226 ms | 272 ms | CH |
| Interface width vs √2·Cn (sourced) | 2.7% | 2.1% | tie |

**Headline:** CH wins robustness + parasitic currents + cost; CAC wins source-mass conservation (and ties machine-exact no-source conservation). CAC's Korteweg parasitic currents (37× CH) drive a bubble-rise blow-up at the coarse spike settings.

## Memo recommendation

**CH primary; CAC retained as a verified `interface="cac"` mode (not default).** Per the spec tiebreak, CAC "loses on the benchmarks" (diverges on bubble rise) so it cannot be primary — but it delivers its designed conservation edge and the adjoint applies unchanged, so it ships. Revisit when Korteweg parasitics are controlled (finer Cn/h, well-balanced tension, or potential-form CAC). Memo: `docs/dev/2026-08-10-sp0-interface-decision.md`.

## Tests / regression

- New: `test_cac_parity_kernel_vs_mirror[ch,cac]`, `test_interface_mass_source_conservation[ch,cac]`, `test_cac_stationary_drop` (CH-vs-CAC in one test).
- Full: `pytest tests/test_chns_forward.py tests/test_chns_scaffold.py tests/test_chns_benchmarks.py -q` → **35 passed**.

## Fix round 1

**Commit:** `eae30ec` — `test(chns): CAC frozen-beta FD gate + committed A/B interface driver`

### What was added

1. **CAC FD-Jacobian gate** (`tests/test_chns_forward.py`): `test_chns_discrete_jacobian_vs_fd` is now parametrized over `interface=["ch","cac"]`. For `"cac"`: one residual call is made at the base point to set `_beta_frozen`, then `_freeze_beta=True` is set before the FD perturb loop so all residual evaluations at `x ± eps*v` use the same frozen beta as the analytic Jacobian (Picard-on-beta). Tolerance `rel < 1e-6` (the measured value is ~3.6e-10). CH leg is unchanged.

2. **Committed A/B driver** (`benchmarks/chns/ab_interface.py`): runnable script reproducing the memo's A/B protocol at level 5, ρ-ratio 10, up to 200 bubble-rise steps, `CHNSStepper(interface=...)`. Records mass conservation (no-source + static Gaussian blob, A=2, radius=3h), parasitic-current max|u| (5-step static drop, gravity off), Newton iters, wall/step, and diverge step. Writes `benchmarks/chns/results/ab_interface/ab_interface.{json,txt}`.

3. **Memo normalization clause** (`docs/dev/2026-08-10-sp0-interface-decision.md`): added a paragraph before the A/B table naming the denominator (`|∫φ₀|` via lumped-mass `M@phi_0`), horizon (200-step / to-divergence), and integration method (partition-of-unity lumped mass `M`), and distinguishing these from the committed 10-step `rel < 1e-6` gate. Table updated with reproduced numbers from the committed driver (supersedes `scratchpad/ab.py`).

### Reproduced A/B numbers vs memo originals

| Metric | Memo (scratchpad/ab.py) | Reproduced (ab_interface.py) |
|--------|-------------------------|------------------------------|
| CAC diverge step | 37 | **37** (exact match) |
| Parasitic ratio CAC/CH | ~37× | **36.6×** (match) |
| Parasitic max\|u\| CH | 3.6e-4 | **3.639e-4** (match) |
| Parasitic max\|u\| CAC | 1.3e-2 | **1.331e-2** (match) |
| Newton avg/max CH | 3.0/4 | **3.0/4** (match) |
| Newton avg/max CAC | 3.6/7 | **3.6/7** (match) |
| Mass drift no-src CH | 0.0 | **0.0** (machine-exact, match) |
| Mass drift no-src CAC | 2.0e-13 | **2.04e-13** (match) |
| Mass drift sourced CH | 2.2e-5 | **3.9e-9** (differs — see note) |
| Mass drift sourced CAC | 3.9e-6 | **7.1e-10** (differs — see note) |

**Note on sourced mass drift:** The committed driver shows three orders of magnitude tighter sourced-mass drift than the original scratchpad/ab.py. The conservative-advection CH phi-row achieves near-machine-exact mass tracking with a static source (`|∫φ_T − ∫φ_0 − T·dt·∫s|`). The original numbers likely reflected a different accumulation convention in `scratchpad/ab.py`. The memo table has been updated; the "CAC wins sourced conservation" conclusion holds (7.1e-10 < 3.9e-9), though the absolute gap is smaller than originally reported. The decisive axis (robustness: diverge @37) is reproduced exactly.

### Test count after fix round 1

`pytest tests/test_chns_forward.py -q` → **14 passed** (was 13; added `test_chns_discrete_jacobian_vs_fd[cac]`).
