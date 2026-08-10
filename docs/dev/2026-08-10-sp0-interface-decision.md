# SP-0 Interface Decision Memo — Cahn–Hilliard vs Conservative Allen–Cahn

**Date:** 2026-08-10
**Status:** Evidence complete — presented to Baskar with the Task 11 (adjoint) kickoff. NOT a hard stop: the discrete-IFT adjoint applies to either interface (both are the same monolithic `(u,p,φ,μ)` Newton solve; the φ-row and μ-row differ, the transpose machinery does not).
**Recommendation:** **Keep CH as the SP-0 primary; retain CAC as a documented, parity-verified interface mode behind `interface="cac"`.** CAC is favored for AM robustness *in principle* (source-respecting exact conservation), but at the SP-0 spike settings it LOSES the forward-robustness and parasitic-current axes to CH's potential-form surface tension. Revisit CAC when the Korteweg parasitic currents are controlled (finer Cn/h, well-balanced pressure, or a potential-form CAC) — see "When to revisit CAC."

---

## What was built (all on `feat/sp0-chns`, Task 10)

Conservative Allen–Cahn interface variant, monolithic-primary per the coupling decision (`2026-08-10-sp0-coupling-decision.md`): CAC goes into the monolithic kernel + its numpy mirror, gated by `interface="cac"` (the CH path stays byte-identical, `interface="ch"` default).

- **Mirror first** (`adjoint/chns.py::CHNSDiscrete(interface="cac")`): new φ-row, trivial-μ row, Korteweg surface tension, frozen-β-per-iterate. Frozen-β analytic Jacobian FD-verifies to ~3e-10.
- **Kernel** (`physics/chns.py::make_chns_newton(..., interface="cac")`): static-gated CAC branch (`interface` in the cache key); β computed host-side per Newton iterate (a cheap GP reduction), passed as a scalar arg.
- **Facade** (`steppers/chns.py::CHNSStepper(..., interface=...)`): default `"ch"`; `staggered` raises on `"cac"` (CAC is monolithic-only).
- **Parity:** kernel-vs-mirror ~1e-14 on (φ,u,p) after 3 steps (CH leg re-covered, CAC leg new).

### The CAC formulation (spec §3, our conventions)

φ-row:
```
∂ₜφ + ∇·(uφ) = γ[ Cn ∇²φ − F′(φ)/Cn − β(t)√F(φ) ] + s_φ
F(φ) = ¼(φ²−1)² ≥ 0,   F′(φ) = φ³ − φ,   γ = 1/Pe   (mobility = CH Onsager coeff)
```
Weak form (test ψ; −γCn∇²φ integrated by parts under no-flux):
```
R^φ = ∫ψ[(φ−φₙ)/dt + u·∇φ + φ∇·u] + γCn∫∇ψ·∇φ + γ∫ψ[F′/Cn + β√F]  (− ∫ψ s_φ)
```

**Source-respecting Lagrange multiplier.** Integrate the φ-row over Ω. Under no-flux/no-slip and conservative advection, `∫∇·(uφ)=0` and `∫Cn∇²φ=0`. Demanding `d/dt ∫φ = ∫s_φ` (mass grows EXACTLY by the source):
```
∫s_φ = γ[ 0 − ∫F′/Cn − β∫√F ] + ∫s_φ   ⇒   β(t) = −(∫F′(φ)/Cn) / (∫√F(φ)).
```
The source **cancels by construction** — β preserves exactly the source-injected mass, nothing more. (General form keeps the lap term in the numerator, `β = ∫[Cn∇²φ − F′/Cn]/∫√F`; under the no-flux gate the lap integral is 0.) β is evaluated implicitly at φⁿ⁺¹ but **frozen per Newton iterate (Picard-on-β)** — the local residual/Jacobian treat β as a constant (no rank-one dβ/dφ block), keeping the sparse structure. Kernel and mirror freeze β identically. Newton still converges in 3–4 iterations on the stationary/source gates (avg 3.6, max 7 on the harder bubble-rise regime).

**μ DOF (trivial-μ option).** To avoid a DOF-layout fork, the `(u,p,φ,μ)` layout is kept with the μ-row replaced by the strong identity `∫N_a μ` (μ→0). μ is a wasted DOF in CAC — a documented build-out optimization would drop to a `dim+2` layout. It did not distort the solve (parity 1e-14, mass machine-flat).

**Surface tension (pragmatic ruling).** In CAC the potential-form capillary `(Cn·We)⁻¹ μ∇φ` AND the AGG mass flux are dropped (spec §3). A strong Korteweg force `−(Cn/We)∇·(∇φ⊗∇φ)` needs a Q1-unavailable weak 2nd derivative at GPs, so surface tension is the **Galerkin-only** weak Korteweg form `+(Cn/We)(∇φ⊗∇φ):∇w` (first-derivatives only; standard for AC-based CHNS, Joshi & Jaiman 2020 lineage). It is omitted from r_mom / SUPG / PSPG (documented). CH mode's μ∇φ form is untouched.

---

## A/B evidence

All at level 5 (32×32), ρ-ratio 10, `Cn_override="2h"` (Cn=2h, the resolvability convention), BUBBLE_RISE_RE35_WE10 params, monolithic BDF1, splu. Source (where on) = fixed Gaussian blob on the φ-row, A=2, radius 3h. Bubble rise: up to 200 steps, gravity on. Static drop: gravity off, 5 steps. (`scratchpad/ab.py`, reproducible.)

| Axis | CH (potential form) | CAC (Korteweg form) | Winner |
|---|---|---|---|
| **Bubble rise robustness** | survives 200 steps | **DIVERGES @ step 37** (parasitic-driven blow-up, umax→6.3, φ overshoots to −2.5, then singular Jacobian) | **CH** |
| Mass conservation, no source | 0.0 (machine-exact) | 2.0e-13 (machine-exact) | tie |
| Mass conservation, with source (`|∫φ−∫φ₀−∫₀ᵗ∫s|` rel) | 2.2e-5 (over 200 steps) | **3.9e-6** (to divergence @37) | **CAC** |
| Parasitic currents, static drop (max\|u\|) | **3.6e-4** | 1.3e-2 (~37× worse) | **CH** |
| Newton iters (avg/max), bubble rise | 3.0 / 4 | 3.6 / 7 | **CH** |
| Wall/step (splu, CPU) | 226 ms | 272 ms (~20% slower) | **CH** |
| Interface width vs √2·Cn (10–90% cut, sourced) | within 2.7% (tol 40%) | within 2.1% (tol 20%) | tie (both excellent) |

### Readings

- **Robustness (the decisive axis).** CAC blows up on bubble rise at these coarse settings by step 37. The mechanism is the Korteweg surface tension: its parasitic currents (37× CH's on the static drop) feed back into the momentum equation and destabilize the rising interface. This is the well-known result that Jacqmin's potential form `μ∇φ` is near-parasitic-free while the Korteweg divergence form is not well-balanced on a discrete diffuse interface. CH's potential form survives cleanly.
- **Conservation (CAC's designed strength).** CAC delivers on its premise: with a mass source, its bookkeeping (3.9e-6 to divergence) is *tighter* than CH's (2.2e-5), and its no-source drift is machine-exact like CH's. The source-respecting β works exactly as derived. **If parasitic currents were controlled, CAC's conservation edge would matter for AM deposition** (where injected mass is the quantity of interest).
- **Profile fidelity.** Both interfaces hold the tanh interface width to within a few percent of √2·Cn under a mass source — neither smears nor sharpens. Not a discriminator.
- **Cost / Newton.** CAC is ~20% slower per step and needs slightly more Newton iterations (frozen-β Picard costs a little), but both are minor next to the robustness gap.

---

## Recommendation

**CH is the SP-0 primary interface. CAC ships as a verified `interface="cac"` mode, not the default.**

Per the spec's tiebreak ("CAC favored for AM robustness UNLESS it loses on the benchmarks"): CAC loses the forward-robustness and parasitic-current axes on the bubble-rise arena at the spike settings, decisively (divergence @37 vs CH's clean 200 steps). It wins the conservation axis, which is the axis AM cares about — but a mode that diverges on the validation benchmark cannot be the primary. CH's potential form is the safe, well-balanced default for SP-0's benchmark gates (Task-9-adjacent) and the SP-1 adjoint.

CAC is retained (not discarded) because:
1. Its source-respecting conservation is exactly the property AM deposition wants, and it is *measurably better* than CH where it runs.
2. The adjoint applies to it unchanged — no wasted work if AM later needs it.
3. The divergence is a *surface-tension-discretization* problem (Korteweg parasitics), not a formulation flaw in the CAC φ-row. It is fixable.

## When to revisit CAC

- **Finer interface resolution** (Cn ≳ 2h with a finer mesh, or a larger Cn): parasitic currents scale down with better interface resolution; the coarse Cn=2h/level-5 spike is the worst case.
- **Well-balanced surface tension:** a pressure-balanced / free-energy-consistent Korteweg assembly, or reverting CAC to the potential-form `μ∇φ` tension (computing a CAC-consistent μ = F′(φ)/Cn − Cn∇²φ via an auxiliary projection) would recover CH's parasitic floor while keeping CAC's conservation.
- **AM deposition stress:** if SP-1 deposition shows CH's ~2e-5 sourced-mass drift accumulating materially over 10⁴–10⁵ steps, CAC's tighter bookkeeping becomes worth the surface-tension engineering.

## Reproducing

- Tests: `tests/test_chns_forward.py::{test_cac_parity_kernel_vs_mirror, test_interface_mass_source_conservation, test_cac_stationary_drop}` (all parametrized over `["ch","cac"]` where applicable).
- A/B numbers above: the Task-10 report `.superpowers/sdd/sp0-task-10-report.md` records the full table; the driver is archived in the report.
