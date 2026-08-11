# SP-0 · CHNS Coupling — Design Spec

**Date:** 2026-08-10
**Status:** Approved design, pre-plan
**Parent:** AM program roadmap (`2026-08-09-am-3dprint-roadmap-design.md`, §5 SP-0)
**Goal:** Build DiffSim's coupled two-phase Cahn–Hilliard–Navier–Stokes brick — variable-density/viscosity, surface-tension-coupled, adjoint-gated — validated against the legacy proteus benchmark set. Everything in the AM ladder (SP-1+) rides on this brick.

## 0. Context

Code survey (2026-08-10) established that DiffSim's NS stack (VMS + SBM + adjoint) and M-component CH stack (adjoint, GPU) are individually mature but have never been coupled: no ρ(φ)/η(φ) in momentum, no μ∇φ force, no flow advection in CH. The legacy `chns_nonnewtonian`/`proteus` C++ code (surveyed same date) provides the reference formulation (Khanwale lineage) and validated benchmark configs; it is reference material only, never a dependency.

**Approved approach (Approach 1):** shared scaffolding first, then twin coupling spikes (monolithic vs staggered) decided on evidence; the CAC-vs-CH interface spike piggybacks on the winning coupling; density-ratio ambition gated at ~10³ (the AM "numerical air" operating point — not true 2000:1 air–water hardening).

## 1. Formulation

Non-dimensional CHNS in the Khanwale form, matching the legacy configs' parameter conventions (Re, We, Cn, Pe, Fr) so their cases translate directly.

- **Phase:** ∂ₜφ + ∇·(**u**φ) = (1/Pe) ∇·(M∇μ), μ = f′(φ) − Cn²∇²φ, f = ¼(φ²−1)² — the existing M=1 CH residual plus one new advection term. Conservative Allen–Cahn (CAC) variant spiked later behind the same interface (§3).
- **Momentum:** ρ(φ)(∂ₜ**u** + **u**·∇**u**) + **J**·∇**u** = −∇p + (1/Re)∇·(2η(φ)**D**) + (Cn·We)⁻¹ μ∇φ + ρ(φ)**g**/Fr², ∇·**u** = 0. **J** = AGG mass-flux correction (active for CH at unmatched densities; dropped for CAC). Surface tension in potential form μ∇φ (auditable by energy diagnostics; minimizes parasitic currents per Jacqmin).
- **Closures:** linear ρ(φ), η(φ) interpolation with positivity pullback (legacy pattern, CHNSIntegrandsGenForm lines 132–202); ratios gated to 10³.
- **Discretization:** VMS/PSPG per existing `ns_bricks` conventions (skew-symmetric convection s=1/2, τ_m h-form), with **τ_m computed per quadrature point from the local η(φ)** — at contrast 10³ a constant-viscosity τ_m mis-stabilizes one phase (research doc §3); Q1 elements; BDF1 first, BDF2 after gates pass (mirrors `MultiPhaseStepper`'s `tstep` ladder).

## 2. Components (all inside DiffSim; no app-repo component in SP-0)

| Component | Path | Contents |
|---|---|---|
| Coupled kernel | `src/diffsim/physics/chns.py` | `make_chns_newton(dim, ...)` residual/Jacobian factory, dim-generic, node-major DOF layout (u₁..u_dim, p, φ, μ) = dim+3/node — or, if staggered wins, the hand-off glue (ρ/η/μ∇φ Gauss-point evaluation shared between steppers). Built to serve both prototypes during the spike. |
| Stepper | `src/diffsim/steppers/chns.py` | `CHNSStepper`, API mirroring `MultiPhaseStepper` (`linsolver=`, `assembly=`, `tstep=`, `src_fns=` passthrough — the deposition hook stays live for SP-1). |
| Adjoint | `src/diffsim/adjoint/chns.py` | Numpy discrete mirror `CHNSDiscrete` + discrete-IFT adjoint (crystallization pattern). |
| Benchmarks | `benchmarks/chns/` | Bubble-rise / dam-break / RT / MMS case definitions (Python dataclasses mirroring legacy `config.txt` parameters), literature reference data, metric extractors. |
| Viz | existing `viz/export.py` (VTU + PVD `time_series`) + a small interface-movie helper (φ=0 contour frames → gif/mp4) | Seed of the AM program's viz module. |

## 3. Spike protocol (the coupling decision)

**Arena:** 2-D bubble rise (Re35/We10), density ratio 10 → 100, short horizon; stress-probe at 10³. Both prototypes share §2 scaffolding.

**Decision axes** (recorded in a committed decision memo in `docs/dev/`, like prior solver decisions):
1. **Forward robustness:** Newton behavior, dt tolerance, mass/energy drift at ratio 10³.
2. **Adjoint tractability:** a written-out transpose plan for each — monolithic: JᵀΛ on the coupled Jacobian; staggered: reverse chain through CH-solve → momentum → pressure-Poisson → correction including the currently-untaped Leray pieces — judged on concreteness.
3. **GPU path:** monolithic — does the blockch Schur factorization extend to the (dim+3)-block saddle; staggered — do existing device solvers cover each sub-solve.

**Timebox:** one task-cluster, not a mini-project. Tiebreak: if evidence is mixed, **monolithic wins by default** (adjoint-first standard). Admissible outcome: *both* survive as modes — the research doc (§3) anticipates monolithic during active deposition/yield transitions and projection for quasi-steady relaxation; if the spike shows complementary strengths, SP-0 builds out one as primary (with the adjoint) and keeps the other's scaffolding as a documented forward-only mode.

**Interface spike (after coupling decision):** CAC-vs-CH A/B on the winner, run *with a `src_fns` mass source on* (fixed Gaussian blob), checking interface-profile fidelity and conservation bookkeeping ("mass matches ∫source dt"). Also memo'd. CAC form: ∂ₜφ + ∇·(**u**φ) = γ[ε∇²φ − F′(φ)/ε − β(t)√F(φ)], with the Lagrange multiplier β(t) modified to respect the source.

## 4. Validation gates & data flow

**2-D quantitative gates** (each = case config + reference data + tolerance; CI-sized and full-sized forms):

| Benchmark | Metrics | Reference |
|---|---|---|
| Bubble rise Re35/We10 & Re35/We125 | centroid height, rise velocity, circularity vs t | Khanwale 2020/2021 (legacy configs' cases); Hysing-style metric definitions |
| Dam break 2-D | surge-front position x(t), column height h(t) | Martin–Moyce experiment + legacy proteus curves |
| RT instability 2-D | spike/bubble penetration vs t | Khanwale plots / Tryggvason lineage |
| MMS | convergence orders for (u, p, φ) | analytic — coupled-system order verification |

**Structural diagnostics on every run:** mass drift (∫φ machine-flat without source), discrete energy budget (kinetic + interface + bulk; monotone in unforced settings — legacy `EnergyAndMassCalculator` pattern via DiffSim diagnostics), parasitic-current magnitude on a static drop.

**3-D gate:** dam break 3-D (legacy config exists) on gpubox — device assembly + blockch-family solver if monolithic won; per-sub-solve device solvers if staggered. Deliberately the smallest 3-D case exercising the full coupling; 3-D performance tuning is not an SP-0 goal.

**Data flow:** case config → `CHNSStepper` march → snapshot store (npz) → VTU/PVD series + interface movie → metric extractors → overlay plot vs reference (the exit artifact). Metric extractors live in `benchmarks/chns/`, not in the solver.

## 5. Adjoint & testing

**Parameters (SP-0 scope):** density ratio, viscosity ratio, We, Pe·M (mobility), Fr — scalars only; field/source parameters arrive in SP-1.
**Objectives:** terminal interface mismatch ∫(φ(T) − φ*)² and centroid height — one distributed, one derived-scalar, exercising both adjoint seeding patterns.

**Three-way gate (crystallization discipline):** hand discrete adjoint = torch twin (extend `torch_twin.py` with the coupled step) = central FD, on a coarse 2-D bubble-rise config, few steps; tolerances matching existing multiphase gates (rel ~1e-6 adjoint/twin; FD to its plateau). **Forward parity first:** the numpy `CHNSDiscrete` mirror is ground truth for the Warp kernel before any adjoint work.

**Test tiers:**
1. Fast pytest — kernel parity vs mirror, single-step residual/Jacobian checks, mass/energy diagnostics, MMS orders at two levels.
2. `@pytest.mark.ad` — three-way gradient gates.
3. `@pytest.mark.gpu` — device parity + 3-D dam-break smoke.
4. Benchmark regressions — metric-vs-reference within tolerance, full-size, on demand (not per-commit).

## 6. Risks & error handling

- **Newton non-convergence / stiffness:** dt halving with a floor raising `dt_underflow` — fail loudly at t=0, don't grind (daisy lesson). Mitigations ready: tanh interface-consistent ICs, brief mobility ramp.
- **ρ/η positivity:** pullback clamps as in legacy; a clamp-activation counter — frequent clamping fails the run rather than silently distorting physics.
- **Parasitic currents / benchmark misses:** first suspects Cn (ε) and mobility scaling per the research doc's tripwire logic — a documented tuning protocol in the benchmark harness, not ad-hoc knob turning.
- **10³-ratio instability:** AGG term is the designed answer (CH path); consistent-and-conservative CAC reformulation is the fallback if CAC wins — both in the spike memo.
- **Monolithic saddle at 3-D:** if blockch doesn't extend cleanly, the 3-D gate falls back to cuDSS at reduced size (inside its ~1M-DOF wall) and the preconditioner extension becomes SP-1-adjacent engine work — SP-0 does not block on it.
- **Scope guards:** no thermal, no non-Newtonian rheology, no toolpath machinery (the spike source is a fixed Gaussian blob, not a moving nozzle), no contact angle. Those are SP-1+.

## 7. Exit gate (from the roadmap, restated)

3-way-verified gradients through the coupled march (density ratio, viscosity ratio, We, mobility, Fr); 2-D benchmark set matched within stated tolerances; ≥1 3-D benchmark on GPU; VTU/PVD series + interface movie produced by the viz path; coupling-decision and interface-decision memos committed.

## 8. Delivered status (2026-08-10 audit)

Audited line-by-line against committed code and artifacts on branch `feat/sp0-chns`.

| Gate item | Status | Evidence |
|---|---|---|
| 3-way-verified gradients — density ratio | MET | `tests/test_chns_adjoint.py`: `test_twin_vs_fd`, `test_hand_vs_twin`, `test_hand_vs_fd` (all `@pytest.mark.ad`); rel ≤ 1e-4 (hand/FD), ≤ 1e-6 (hand/twin); both objectives |
| 3-way-verified gradients — viscosity ratio | MET | Same tests; `eta_ratio` in `PARAMS = ("rho_ratio","eta_ratio","We","mobility","Fr")` |
| 3-way-verified gradients — We | MET | Same tests + `test_descent_smoke_We` (monotone J reduction in 3 GD steps) |
| 3-way-verified gradients — mobility | MET | Same tests; `mobility` param |
| 3-way-verified gradients — Fr | MET | Same tests; `Fr` param |
| 2-D benchmark set — bubble rise Re35/We10 & Re35/We125 | PARTIAL | `tests/test_chns_benchmarks.py::test_bubble_rise` gates against a COMMITTED PROVISIONAL BASELINE (not literature-quantitative Hysing 2009 numbers — see GAP note below); internal reproducibility: centroid 2%, rise-velocity/circularity 10% tol. Full run: 150 BDF2 steps to t=1.5, mass drift 4.0e-15, artifacts in `benchmarks/chns/results/bubble_rise_re35_we10_full/` |
| 2-D benchmark set — dam break | PARTIAL | `tests/test_chns_benchmarks.py::test_dam_break` gates against provisional internal baseline (see GAP note) |
| 2-D benchmark set — RT instability | PARTIAL | `tests/test_chns_benchmarks.py::test_rt` gates against provisional internal baseline (see GAP note) |
| 2-D benchmark set — MMS convergence | MET | `tests/test_chns_benchmarks.py::test_mms_convergence_order`: L2 order ≥ 1.8 for u and φ over levels 4→5→6 (genuine design-order gate, source-forced, sympy-generated) |
| ≥1 3-D benchmark on GPU | MET | `benchmarks/chns/results/dam3d/`: level-5 3-D dam break, 20 BDF1 steps, cuda:0 + cuDSS, 215 622 DOF, 3 Newton iters/step, no NaN, mass machine-flat (≤3e-16), `dam3d_final.vtu` written. Metrics: `dam3d_metrics.json` |
| VTU/PVD series via viz path | PARTIAL | VTU path wired in `benchmarks/chns/run_case.py` (lines ~274–288) and verified working (`_write_pvd` Path-fix commit `186705a`); however no VTU/PVD series committed for 2-D runs (full-run artifacts = metrics JSON + GIF + PNG only, by brief design). Dam3d has one committed VTU (`dam3d_final.vtu`). VTU time-series is functional but requires a `--full` run (uncommitted) to verify the complete PVD series end-to-end |
| Interface movie via viz path | MET | `benchmarks/chns/results/bubble_rise_re35_we10_full/interface.gif` committed (150-frame GIF from full bubble-rise run via `movie.write_interface_movie`) |
| Coupling-decision memo committed | MET | `docs/dev/2026-08-10-sp0-coupling-decision.md` — monolithic primary, staggered forward-only fast mode; ratified by Baskar (Task 8 human gate) |
| Interface-decision memo committed | MET | `docs/dev/2026-08-10-sp0-interface-decision.md` — CH primary, CAC verified `interface="cac"` mode |

**Summary: 9 MET / 4 PARTIAL / 0 GAP**

### Caveats and honest gaps

**Dam-break/RT provisional-baseline gap (spec §4 vs delivered).** Spec §4 lists "Dam break 2-D — surge-front Z(T) vs Martin–Moyce experiment" and "RT instability 2-D — spike/bubble penetration vs Khanwale plots" as quantitative gates. The delivered tests gate against a *committed internal baseline* (`benchmarks/chns/results/ci_baselines.json`) generated by this same code, not against the dimensional literature numbers. This is because the DiffSim formulation is non-dimensional on the unit square at coarse CI resolution (Cn=2h), whereas the Martin–Moyce numbers are from a dimensional experiment (specific ρ/μ/g/σ) and Khanwale/Tryggvason curves are from production-resolution long-time runs. A truly comparable number is not obtainable without a separate non-dimensionalization study and production-scale runs. This gap is documented in `tests/test_chns_benchmarks.py` (SOURCING DECISION block, lines ~36–52) and the Task 9 report. Bubble-rise is the one case with a genuinely nondim literature reference (Hysing 2009 coarse resolution), but the CI test still gates on the internal baseline rather than the Hysing numbers (a separate study is needed to map DiffSim's unit-square nondim to Hysing's [1×2] domain and parameters). All four 2-D cases are STRUCTURALLY verified (mass machine-flat, energy diagnostic, anti-freeze guards, MMS design-order).

**Adjoint is CPU-only (by plan).** The three-way gradient gates run on the numpy `CHNSDiscrete` mirror and torch twin — both CPU. The Warp kernel adjoint (GPU-resident backprop through the coupled assembly) is explicitly deferred to SP-1+ per the spec's tier-3 marker (§5). This is not a gap against SP-0's stated scope.

**Warp adjoint deferred.** The `CHNSAdjoint` uses the numpy mirror's Jacobian (IFT pattern), not the Warp device Jacobian. GPU-resident adjoint assembly is SP-1+.

**VTU/PVD time-series not committed for 2-D runs.** The export path is wired and `_write_pvd` is fixed (commit `186705a`). A future `--full` run with `want_vtu=True` will produce the full PVD series; the code is exercised but the artifact is not checked into the repo (large binary, per design).
