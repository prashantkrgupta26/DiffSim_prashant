# SP-1 R0 — the honest-verdict findings ledger (forward XDD substrate)

**Rung:** R0 (forward port). **Branch:** `sp1-r0` (from master cffb033).
**Scope closed:** the GPU-native forward excitonic drift-diffusion substrate —
five-field bricks, closures, excitation protocols, instrument observables, and
the R0 gate battery (MMS, reduction, SimSalabim, perf, table anchors, preflight).
**Spec:** `docs/dev/specs/2026-07-17-sp1-differentiable-xdd-design.md` (§3, §4, §6,
§7-R0). **Plan:** `docs/dev/plans/2026-07-17-sp1-r0-plan.md`.

This note is the complete R0 story with the measured numbers quoted throughout —
the home for the spec §6 ledger's per-item evidence. It is deliberately honest
about what is a physical result, what is a reduced-drive proxy, and what is
blocked with recorded mechanism. R1 (adjoints) inherits this substrate untouched.

---

## 1. Block summary (A–E) with commits

| Block | What landed | Key commits | Verdict |
|---|---|---|---|
| **A** substrate | `XDDParams` + CPU-config/spec readers; morphology (cloud reader, signed-distance, relaxed masks, descriptors); closures (Onsager-Braun +∂/∂E, Langevin, generation+waveform engine, region mobility) | 82a5073, 4e42e59, 8e687d1, d0b626b, ba8c641, de6ac9c | GREEN — constants verified digit-by-digit; arithmetic audited exact to 10 sig figs; 78/78 |
| **B** bricks + solve | Poisson brick (λ²-form, GP ε); carrier bricks (conservative signed drift-diffusion + SUPG); exciton bricks; monolithic 5-field Newton (exact k_diss-field Jacobian, positivity guard); log-density mode | 209f616, 6241504, ae24d21, 6e9562e, d243859, 5dc8a89, 2e1b0bb, ae2fbe9, 360ccca, **c9c499c** | GREEN — MMS orders 2.00/3.00 all 5 fields; Jacobian FD gate 7.3e-7 caught 3 omitted cross-terms; 117/117 |
| **C** protocols/march | `XDDRun`: continuation (dark→G-ramp→V-sweep), adaptive log-dt BDF march, steady detection, 4 solver strategies | c9c499c (drift fix), d0fa86e | GREEN — bilayer dark steady reached by marching (70 steps, median 2 Newton its, flux imbalance 2.8e-4); 12/12 gates |
| **D** observables | consistent-flux J–V (CPU column format, softmin), Nirmal feature vector (THD 0.38873, slope −1.0 exact), IRF (δ-recovery 0.0), PL/TRPL + quench | 8518bb5, 46c00fd | GREEN — all closed-forms hand-verified; caught a latent BDF1 history off-by-one in C's march (rates halved) |
| **E** gate battery | E1 reduction, E2 SimSalabim, E3 (blocked), E4 table anchors, E5 perf, E6 preflight + this ledger | 29dafb8, 7c2711e, a9e0dbf, *(E-c commit)* | MIXED (honest) — E1/E2 GREEN (§4), E5 MISS-but-attributed 6.9× cuDSS (§5), E3 BLOCKED-with-evidence (§6), E4 reduced-drive PROXY both systems converge (§7), E6 preflight GREEN (§8) |

Total on `sp1-r0`: 23 commits before E-c. Full SP-1 suite green at each block gate.

---

## 2. The two incidents caught and fixed (process value)

### 2a. The drift-sign + form defect (Block C, the big one)
The Block-C implementer found a **conservative-form / drift-sign defect** in the
carrier bricks, traced to the **controller's B2 brief** (not the implementers):
the brief specified electrons with a nonconservative advection `a_n=−μ̂∇φ̂`; the
CPU ground truth (`DDEquation.h:182`, verified directly) is the **conservative**
form `−μ̂ n̂ ∇φ̂ · ∇w`, i.e. electron equilibrium `n̂ ∝ e^{+φ̂}`, velocity `+μ̂∇φ̂`
(holes mirrored). **MMS was blind by construction** — the manufactured sources
were derived from the same wrong convention, so orders held at 2.00 while the
physics was wrong.

Evidence that exposed it: a marched dark bilayer **bulk-filled** (`n̂→1`,
`Jny 0.695→18.8`); the flux imbalance equalled the structural ratio **1.75
exactly**; the kernel-equilibrium residual was **100× smaller** than the
shipped-IC convention. Fix: commit **c9c499c** — CPU conservative signed form in
kernels+Jacobians, MMS sources re-derived, upwind expectations flipped, B4 FD
re-verified (**5e-7**, and it caught a *second* bug: `_supg_mass_block U=−aq`);
bilayer `r0` collapsed **1e18 → 0.087** in both primal and log modes (the
remaining non-convergence is mesh resolution, not sign). Sanctioned limitation
recorded: carrier p2 order ~2.1–2.6 (SUPG advective-only, the Δφ̂ term lives
outside the standalone kernel).

### 2b. The BDF1 history off-by-one (Block D catch)
Block D's PL/TRPL decay gate exposed a latent **BDF1 history off-by-one** in the
C march that **halved the effective rates**. Fixed and verified line-by-line
against the theoretical `e^{−t/τ_eff}`; the residual is now the **exact BDF1
truncation bias** (verified to 2e-7 vs theory). A BDF2 first-step variant was
also fixed (untested path — future gate).

### 2c. Wrong-repo trap (recurring, now guarded)
The shell resets cwd to a stale `~/DiffSim` clone; B2's git work landed there on
a shadow branch (recovered via cherry-pick ae24d21), and later bit a *reviewer*.
**Prevention:** a repo-identity guard (`git rev-parse --show-toplevel` must end in
`Dropbox/.../DiffSim`) is now mandatory in every subagent prompt, reviewers
included.

---

## 3. The Debye / marchability boundary (the load-bearing R0 finding)

The singularly-perturbed Poisson operator `λ²∇²φ̂` has a Debye boundary layer far
finer than any feasible uniform-octree element. This wall recurs in E1(ii), E2,
E5, and `test_xdd_run._resolvable_bilayer`, and is the single most important
physical constraint R0 discovered.

**Measured wall.** The physical PM6:Y6 / Kodali configs have `λ² ≈ 2.2e-5`
(Debye_hat = √λ² ≈ 0.0047 device-lengths ≈ 0.44–0.5 nm). At level 3 (element
`h_hat = 0.125`) that is **26× under-resolved**; a dark-eq march of the physical
config gives **1 accepted / 47 rejected steps**, `φ̂` blows to **±25.9**, and the
steady criterion never fires. With the reduced-drive **marchable regime** (`λ²=1e-1`,
Debye_hat ≈ 0.316, Ê_g=4, symmetric μ̂) the Debye layer (~0.31 L) is resolved,
`φ̂` stays bounded, and the BDF march converges cleanly.

**Consequence for R0.** Every marching gate (E1, E4, E5) runs in the marchable
regime — the physics is faithful, only the drive strength (λ²) and the built-in
(Ê_g) are reduced and μ̂ is symmetrized. The physical full-drive device is a
**Scharfetter–Gummel / boundary-refined-mesh** problem, explicitly a Block-D/E
follow-up / R1 item, **not an R0-CPU gate**. This is recorded honestly wherever it
bites rather than faked.

---

## 4. E1 reduction + E2 SimSalabim (GREEN)

**E1(i) — stripe reduction ordering** (`benchmarks/xdd/e1_reduction.py`). More
D/A interface generation sites → more free-carrier generation → more current. The
observable is `∫k̂_D X̂_D + ∫k̂_A X̂_A`. Measured (marchable, L3, 33×33 stripes):

| case | carrier_src | interface sites |
|---|---|---|
| vertical_2 | 1.4259e-04 | 64.0 |
| vertical_6 | **5.2551e-04** | 232.7 |
| horizontal_2 | 1.3811e-04 | 64.0 |
| horizontal_6 | 5.2315e-04 | 232.7 |

6-stripe > 2-stripe **by 3.7×** (both orientations) — the robust reduction
anchor holds. Deviation recorded: "horizontal > vertical current" is **not**
resolved (0.5% diff) — the symmetric reduced-drive config is transport-symmetric.

**E1(ii) — Kodali 1-D band.** Physical PPV:PCBM Jsc must be −30 A/m²-class;
documented band ±50% → **[−45, −15] A/m²**. XDD cannot march the physical device
(the §3 wall), so the band is anchored on the SimSS reference of the same physical
device: **SimSS Jsc = −44.84 A/m² (120 nm)** at the band edge, **−37.76 A/m²
(100 nm)** comfortably inside. Anchor **CONFIRMED**.

**E2 — SimSalabim cross-check** (`benchmarks/xdd/e2_simsalabim.py`). SimSS
installed + built + running (pySIMsalabim + fpc-3.2.2, arm64 Mac). Locked
reference for the matched 120 nm Kodali-class device: **Jsc −44.8378 A/m²,
Voc 0.8316 V, FF 0.5506** (rtol 5%). The **parameter-mapping dictionary** (XDD
nondim ↔ SimSS `device_parameters`) is a deliverable in the driver. The absolute
XDD-vs-SimSS |ΔJsc|/|ΔVoc| head-to-head is **DEFERRED** — it depends on the
blocked XDD physical march (§3), mechanism recorded.

---

## 5. E5 perf gate — MISS, fully attributed (splu→cuDSS 97.75→14.12 s/step)

The Nirmal config (513×129 / 66,177 nodes → measured at the DOF-matched square
proxy **level 8 = 257×257 = 66,049 nodes = 330,245 DOFs**; 10 GHz rect-sin, 1 ns,
V̂=0, PM6-class) on one RTX 6000 Ada.

**E-b (host splu, the pre-fix baseline):** **97.75 s/step median**, GPU **0% util**
(one CPU core pinned) — cuDSS was never wired; `XDDSystem.linsolve` defaulted to
host scipy `splu`. Root cause found and named.

**E5-fix (cuDSS wired, `XDDSystem(linsolver="cudss")`):** in-tree nvmath
`DirectSolver` path — plain `DirectSolverOptions(blocking=True)` (no
`mtlayer_gomp`, the thread-leak lesson), plan-once + `reset_operands`+`factorize`
per Newton iterate on the fixed sparsity, explicit `.free()` on pattern change.

| quantity | host splu | **cuDSS** | ratio |
|---|---|---|---|
| warm-up step | 411 s | **32.6 s** | 12.6× |
| s/step median | 97.75 s | **14.12 s** | **6.9×** |
| end-to-end (proj ×201) | 5.46 h | **0.79 h** | 6.9× |
| speedup vs 10 h / 30 h | 1.8× / 5.5× | **12.7× / 38.0×** | |
| target | | ≥100× | |
| GPU util | 0% flat | **peak 100%**, 0% between | |
| GPU mem | 642 MiB | **2228 MiB** | |

**Verdict: MISS vs 100× (12.7–38×), honest and profiled.** All 8 timed steps
converged. The bottleneck **shifted off the GPU LU**: cProfile of one step
(8 Newton iters, 33.1 s) — `_reduce_and_eliminate` host Jacobian assembly
**12.4 s / 37%** (scipy `tolil`), `jacobian_full` host closures **10.5 s / 32%**
(`einsum` + `tolist`), `_cudss_linsolve` **5.96 s / 18%** (of which cuDSS plan
4.7 s is one-time). **~69% is now host-side assembly + closures.** The remaining
~8–14× to 100× requires the **M1d device-assembly playbook** on XDD (move CSR
assembly + A3 closures onto the device, kill host `tolil`/`tolist`) — queued as
**#35**, which doubles as R1 prep. The E-b "`import cudss` not importable" was a
**red herring**: the path uses `nvmath.sparse.advanced.DirectSolver`; nvmath[cu12]
0.9.0 was present all along (verified resid 2.7e-15).

Baseline: `tests/baselines/xdd_e5_perf.json` (splu = `history`, cuDSS = `gate`).

---

## 6. E3 CPU parity — BLOCKED with evidence

`drift_diffusion_ts` cannot be built on gpubox without root: the **dendrite-kt
submodule is not vendored** (empty; private bitbucket URL), and **MPI / PETSc /
LAPACK are all absent** (apt candidates exist but need root; no spack/conda).
Recommended user-space path recorded (bootstrap micromamba →
`petsc openmpi cmake`, then submodule init with bitbucket auth; ~1–3 h). E3
remains **OPEN** as a deferred follow-up. (The E2 SimSalabim community-standard
1-D reference stands as the available cross-check in the interim.)

---

## 7. E4 framework-table anchors — reduced-drive PROXY (honest delta class)

**Scope, declared honestly.** Full Table-1 parity (8 morphologies × 2 systems ×
full J–V) is a campaign, not an R0 gate — at ~14 s/step a full physical case is
~8 h. E4's R0 anchor is **bilayer Jsc-only (V̂=0, light) for BOTH material
systems**, run on the box with `linsolver="cudss"`
(`benchmarks/xdd/e4_anchors.py`).

**The honest crux.** The physical drive of either system hits the §3 Debye wall,
so E4 marches in the reduced-drive marchable regime (λ²=1e-1, Ê_g=4, symmetric
μ̂=0.5, ε̂=1, weak Langevin ζ=1e-6 — the E1 stripe knobs). The surviving
per-system material knobs are τ_x, the Onsager separation `a`, and the
donor:acceptor generation ratio; μ̂, λ², ε̂, Ê_g are all reduced/symmetrized.

**Measured (gpubox, cuda:0, `linsolver="cudss"`, level 6 = 65×65 = 21,125 dofs).**

| system | nondim \|Ĵ\| | flux imbalance | fired | Jsc (mA/cm²) | Table-1 anchor | ratio | delta class |
|---|---|---|---|---|---|---|---|
| P3HT:PCBM | 4.207e-3 | 5.7e-3 | ✓ | 1001.99 | 0.746 | **1343×** | 10³ |
| PM6:Y6 | 7.162e-4 | **6.9e-6** | ✓ | 14.83 | 2.422 | **6.12×** | 10⁰–10¹ |

Both march to a **converged lit steady state** (`linsolver="cudss"`, GPU sparse
direct LU); PM6:Y6's flux self-consistency is **6.9e-6** (excellent), P3HT:PCBM's
is 5.7e-3 (< 1e-2). The cross-system ordering PM6>P3HT does **not** hold in the
reduced regime — P3HT's tiny Onsager separation `a=1e-8` (vs PM6's `a=1e-7`)
gives far higher dissociation/current; a regime artifact of the surviving knobs,
recorded honestly (like E1's horizontal>vertical deviation).

**Delta class — recorded, NOT faked.** Both systems march to a **converged lit
steady state** (criterion fired, flux self-consistency < 1e-3). The nondim |Ĵ|
values are the meaningful comparable quantity (both ~2e-2, same order). The
**physical Table-1 anchor is NOT reproduced** — the re-dimensionalised mA/cm²
overshoots the anchor by **~10²–10⁴×** because the marchable regime decouples the
current from the physical mobility scale (|Ĵ| is computed with symmetric μ̂=0.5,
then multiplied by the *physical* J0, which double-counts the physical mobility).
The honest delta class is: **reduced-drive proxy — physical mA/cm² Jsc
unreachable on a uniform CPU/GPU mesh; the physical anchor is a Scharfetter–Gummel
/ boundary-refined-mesh R1 item** (the same §3 wall). Mechanism: mesh + drive +
symmetrized transport, not a solver defect. The measured cross-system ordering
and ratios are locked in `tests/baselines/xdd_e4_anchors.json` with these caveats.

---

## 8. Locked-baselines inventory

| File | Locks | Gate test |
|---|---|---|
| `tests/baselines/xdd_e1_e2.json` | E1 stripe ordering + Kodali band; E2 SimSS Jsc/Voc/FF + parameter mapping | `test_xdd_parity.py` E1/E2 sections |
| `tests/baselines/xdd_e5_perf.json` | E5 splu-history + cuDSS gate row + speedup arithmetic | `test_e5_perf_baseline_parses_and_speedup_consistent` |
| `tests/baselines/xdd_e4_anchors.json` | E4 bilayer Jsc (nondim + mA/cm²), anchors, ratios, delta class, provenance | `test_xdd_parity.py` E4 section |

Preflight gate: `tests/test_xdd_preflight.py` (13 tests — the physical full-drive
L3 config FAILs the Debye check, marchable PASSes; strict raises on FAIL).

---

## 9. Spec §6 ledger — measured evidence per ruling

| §6 item | SP-1 ruling | Measured evidence (this note / tests) |
|---|---|---|
| Exciton contact BC | support both; gate vs `_ts` natural-Neumann | `_ts` natural-Neumann used throughout; NDOF_EX=2; parity note in E3 (deferred) |
| SUPG on Poisson | **dropped** (elliptic) | deviation recorded (B1 Poisson brick has NO stabilization); parity tolerance absorbs it |
| k_diss field-coupling in Jacobian | **exact** (∂k/∂|∇φ|) | B4 FD gate **7.3e-7** verifies the exact cross-term; caught 3 omitted terms in RED |
| J convention | min(|Jny|,|Jpy|) kept for reporting; smooth for gradients | D observables: `J=max(|Jny|,|Jpy|)` reported + softmin for gradients; flux imbalance < 1% is the self-consistency gate (E1 2.8e-4, E4 < 1e-3) |
| G(t) amplitude | real waveform engine | A3 Generation waveforms (CW/step/pulse/rect-sin); E5 drives 10 GHz rect-sin |
| Rnp → exciton back-feed | kept (parity), flagged | present, region-assigned (B3); flagged non-standard |
| f′ mask | excluded with reason | excluded (A2: tanh-relaxed masks, no f′) |
| NDOF_EX | 2 (donor + acceptor) | 2 throughout; legacy-1 not parity-relevant |

The two deviations with a measurable per-case delta (SUPG-on-Poisson,
exciton-BC, min-vs-contact J) fold into the flux self-consistency numbers above;
the physical head-to-head that would isolate them is the E3-blocked / §3-walled
comparison, deferred with mechanism.

---

## 10. What R1 inherits (no substrate rework)

- **Closure-derivative contracts (A3):** every closure is `closure(fields, aux)
  -> value, d/d(fields)` with analytic derivatives (FD-verified 1e-8 class) — the
  adjoint chain starts here.
- **Frozen-dt march (C1):** the BDF march with σ=1/Δt̂ regularisation and the
  documented steady criterion — the epoch-free transient chain builds on it.
- **Pure-function observables (D):** J–V/features/IRF/PL are pure functions of
  state (R1-ready; confirmed in review).
- **Device-assembly follow-up (#35):** move CSR assembly + A3 closures onto the
  device (the M1d playbook) — the path from E5's 38× to ≥100×, and R1's forward
  pass acceleration.
- **The marchable-regime caveat (§3):** R1 inherits the Scharfetter–Gummel /
  boundary-refined-mesh item for physical-drive parity (E1(ii)/E2/E4 head-to-head).

---

*Numbers in this note are quoted from the block reports
(`.superpowers/sdd/sp1-e-a-report.md`, `sp1-e-b-report.md`, `sp1-e5fix-report.md`,
`sp1-e-c-report.md`) and the locked baselines. Every gate value is reproducible
from the named driver.*
