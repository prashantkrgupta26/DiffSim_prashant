# Roadmap reconciliation — strawman (2026-07-20)

**Status: STRAWMAN for Baskar's review.** Reconciles the two forked
milestone numberings into a clean forward spine, split — at Baskar's
direction — into a **Physics-development track (P)** and a
**Software/infrastructure track (S)** that run in parallel and gate each
other at named points.

## Why two tracks

The single linear M-numbering fractured because two *kinds* of work were
sharing one counter: new physics capabilities (what the code can
simulate) and cross-cutting infrastructure (how fast/big/differentiable
any physics runs). They have different reviewers, different gates, and
different funders (physics → papers/campaigns; software → Horizon/scaling).
Splitting them lets each advance without renumbering the other, and makes
the gating explicit: a physics hero run *pulls* the software capability it
needs, and software capabilities *land* against a physics consumer.

---

## What's DONE (both tracks, historical — frozen)

Physics: M0/M0.5 octree foundations · M1a SBM · M1b incompressible NS ·
M1c adjoints + neural-SDF · M2 heat/mass + closures + p2-NS · M3
differentiable adaptivity · M4 phase-field + learned thermodynamics · M5
OrgElMorph (Wodo films, first learned free energy) · SP-1 R0 (forward
XDD substrate).

Software: M1d device migration (cuDSS, device CSR) · #33 mixed-width CSR ·
#34 GH200 coherent-capacity probe · #35 XDD device assembly · #37 film
host-fraction port · #38 ChunkedCSR (2³¹ wall broken) · #39 gpubox-test
gate · remote two-brain toolkit · E-track pedagogy.

*(The old spec §10 numbering — M3=mixed-precision, M6=capstone-inverses,
M7=THB, M8=space-time — is RETIRED as a linear spine; its items are
re-homed into the two tracks below with their original intent intact.)*

---

## TRACK P — Physics development

Ordered by readiness and campaign pull. Each Pn is a milestone with its
own spec → plan → gated rungs.

### P1 — SP-1 R1: the XDD adjoint + inverse rung  **[RATIFIED 2026-07-20]**
Make "differentiable" real for the excitonic DD stack: implicit
steady-state adjoint + taped-transient (frozen-dt) adjoint; first inverse
demos (recover a closure parameter / a morphology descriptor from a J–V
or J(t) trace); instrument digital twins (J–V light/dark/intensity, J(t)
light-modulation, TRPL/PL gated). *Pulls from S:* nothing new — #35 made
the forward solve 37× cheaper, R1 is unblocked today. *Gates:* three-way
gradient ≤1e-6 per design variable, dot-product test, one end-to-end
inverse recovering a planted parameter.

### P2 — NSHT-SBM-Shell: flow over a slender object  **[RATIFIED 2026-07-20 — the next HERO; absorbs RB1–RB3]**
The first scaling hero: Navier–Stokes + heat transfer with shifted-boundary
(SBM) **shell** treatment of a slender/thin object, targeting **100M+ dofs**.
This is the headline physics milestone after P1 — the single-object
precursor that makes P4 (maize canopy) tractable, and the first genuine
exercise of the 100M-DOF scaling pathway end to end.
- *Reference:* `local_code_old/chenghauy-nshtsbm_shell-*.zip` (Dendrite-kt
  / PETSc / MPI). The parity anchor, exactly as XDD gated against DDFields.
  Salient source: `NSEquation.h` (VMS-NS weak form), `HTEquation.h`
  (heat), `SBMCalc.h`/`SBMMarker.h` + `ShellSurrogate2True.h` (the shell
  surrogate-boundary core), the IMGA loop, `CalcVorticity`/`CalcError`.
  2-D/3-D via build flags (`-DENABLE_3D -DSHBM -DIBM`). Read into the P2
  spec at kickoff.
- *Core numerics (declared up front):* pressure projection (fractional-step
  / Chorin-class or projection-VMS), VMS stabilization, **BDF2** time
  integration, and **preconditioner design** for the pressure-Poisson /
  velocity blocks at scale — this is where the block-iterative + FGMRES +
  preconditioner machinery (M1b, blockch) gets pushed to 100M dofs.
- *Pulls from S:* multi-GPU (S4) is the hard dependency for the 100M hero;
  ChunkedCSR (done) already unlocks the single-GPU >20M rung; mixed
  precision (S2) for factor bandwidth at hero scale. **Not** dependent on
  THB.
- *Gates:* NS + thermal MMS (±0.10 per field), **CPU/C++ parity vs the
  Dendro-shell reference** on a slender-object benchmark, projection
  divergence-free tolerance, BDF2 temporal order, a 100M-DOF hero run on
  the Horizon/GH200 tier with a preconditioner-scaling table.
- *Rungs:* R0 2-D NS-SBM-shell verified vs reference → R1 3-D single
  slender object → R2 device-ported + preconditioner study → R3 the
  100M-DOF hero.

### P3 — Thin-shell SBM Navier–Stokes over maize plants
The Horizon maize-**canopy** CFD track — the multi-object continuation of
P2's single slender object (a canopy is many slender shells). *Pulls from
S:* multi-GPU (S4) for canopy scale; inherits P2's shell/projection/VMS/BDF2
machinery directly. **THB (S3) is NOT a dependency** — thin-feature
resolution is handled by the SBM shell treatment + octree refinement P2
establishes; THB stays exploratory and is adopted only if it later proves
itself a lever (see S3). *Gates:* single maize-leaf validation vs reference
before canopy; canopy-scale weak-scaling.

### P4 — NS-PNP for electroconvective hero simulations
Suriya's track (his onboarding Phase 5). VMS-NS + PNP + block-iterative,
device-resident from day one, differentiable from day one, CPU-parity
gated against his existing code. Electroconvective instability as the
hero. Shares P2's projection/VMS/BDF2/preconditioner spine directly.
*Pulls from S:* multi-GPU (S4) for the hero scale; ChunkedCSR (done)
already unlocks the single-GPU rung. *Gates:* MMS per field, CPU parity,
three-way gradients, electroconvective onset benchmark.

### P5 — SP-2: structure→property beyond-FH continuation
The `structure_property_p3.md` track — composition-diverse protocols,
instrument-space observables (S(q,t), height, PSF), genuinely
beyond-Flory–Huggins recovery. Partly prototyped in M5 Forward; P5
formalizes it as a milestone. *Pulls from S:* mixed precision (S2) if the
3-D RVE rung is heavy. (Differentiable-science lineage of P1, schedulable
in parallel with the NS hero tracks.)

### P6 — Additive manufacturing simulation
The AM process twin (melt pool / thermal / process-parameter inverse).
- *Reference:* `local_code_old/baskargroup-admanufacturing-*.zip`
  (Dendrite-kt / PETSc / MPI, first version). Salient source:
  `HTEquation.h` (thermal), `ADMSolver`, `ADMRefine`/`ADMCoarsen`,
  `ADMSubDomainBoundary`, `DiffusivityAssociator`, `VoxelOrder`. Upgrades
  AM from "evidence seeds" to a real parity target.
- *Pulls from S:* mixed precision (S2), moving-body/topology-epoch
  machinery (already in M4). *Gates:* melt-pool benchmark, C++-parity vs
  the reference, process-parameter gradient demo.

**Suggested P order:** P1 (differentiable-science, now) ‖ **P2 the NS hero
(next hero, pulls S4)** → P4 (Suriya-paced, parallel, shares P2's spine) →
P3 (canopy, downstream of P2) ; P5/P6 scheduled against their references
and campaign pull. P1 and P2 run in parallel — different reviewers,
different machinery, no shared critical path.

---

## TRACK S — Software / infrastructure development

Cross-cutting capabilities. Each Sn lands against a named P consumer so
it's never speculative.

### S1 — Perf-lever queue completion  **[in flight]**
#40 Krylov fusion+graph-capture (MERGED — but launch overhead REFUTED as
a wall lever: −2.6%; the machinery is the foundation for the real lever)
→ #41 batch-local GP fallback → #36 mixed-precision fp32-storage+FP64-IR
→ #42 device-resident preconditioner apply (the sync-point lever #40
revealed). Serialized (all touch linsolve.py/krylov_dev.py). *Consumer:*
every P track's step cost. Standing lesson: perf gates are
measurement-only for a reason — #40 proved a plausible hypothesis wrong
cheaply, and the honest negative rerouted the queue to the real floor
(per-inner-solve sync latency).

### S2 — Mixed precision, promoted from #36 to a standing capability
#36 delivers the fp32-storage+FP64-IR hook + measured decision data. S2
is the follow-on: auto-selection policy per device tier (fp64 default on
Ada, fp32 on GH200/B200), the (p, precision) binning and GMRES-IR from
spec §10 M3, and the Tier-MP verification battery (conditioning-cliff
reproduction, conservation sentinels). *Consumer:* P2 3-D RVE, P5 AM at
scale.

### S3 — THB / hierarchical B-spline refinement  **[EXPLORATORY — not load-bearing]**
Spec §10 M7 content (HB/THB extraction on incomplete octrees, SBM pairing,
Poisson→NS, GMG; C¹ primal CH stretch) is preserved as a research
direction, **but no physics milestone depends on it.** Per Baskar
(2026-07-20): the THB use-case isn't settled yet, so THB must not gate P3
(maize) or any hero — thin-feature resolution rides the SBM shell + octree
refinement instead. THB is adopted into the load-bearing spine only if a
future study shows it a decisive lever (DOF-savings vs Lagrange at the
shell/canopy scale). *Consumer:* none required; opportunistic. *Gates
(when explored):* PoU invariants, O(h³) MMS at p=2, DOF-savings vs
Lagrange, Shadkhah-bar conditioning.

### S4 — Multi-GPU (NVLink → multi-node)  **[hard dependency of the NS hero]**
Spec §10 M5 intact, now with the #38 chunk table as its ready-made row
decomposition. Partition + halo overlap + NCCL + multi-device adjoint;
≥80% scaling to 4 GPUs; the documented multi-node attachment point. This
is the gating capability for **P2's 100M-DOF NS hero** and every hero
below it. *Consumer:* P2 NS-shell hero, P3 maize canopy, P4
electroconvective hero, the 100M-DOF Horizon tier. *Gates:* ≥80% to 4
GPUs, deterministic mode, adjoint correctness across the partition.

### S5 — Space-time bricks
Spec §10 M8 intact: GLS heat (k=3), space-time VMS-NS (k=4), periodic
limit cycles, space-time SBM (moving body as static 4-D surface),
adjoint-in-one-solve. Requires M0.5's k=4 foundation (have it).
*Consumer:* P2/P3/P4 periodic/shedding regimes; a differentiability
multiplier (one adjoint solve for a whole trajectory). Lower priority —
schedule after S4 unless a P track pulls it early.

**Suggested S order:** S1 (now) → S2 → **S4 (unblocks the NS hero — the
priority raise)** → S5 ; S3 (THB) exploratory, off the critical path.

---

## Horizon P-items (deadline-driven, cut across both tracks)

- **P0 — COMPLETE:** aarch64/GH200 lane (#29), index widening (#33),
  ChunkedCSR (#38). The single-GPU ceiling is now HBM-bound (~40M dofs),
  not the int32 wall.
- **Fall-2026 campaigns:** OrgElMorph 3-D morphology collection (needs P5
  + S2 + S4), maize-canopy CFD (needs P2 NS hero → P3, + S4). The NS hero
  (P2) is now the pacing physics item for the CFD campaign, and S4
  (multi-GPU) is the pacing software item for both.

## Open rulings still gating things

- **NS-hero reference codes** — RECEIVED (`local_code_old/*.zip`):
  `chenghauy-nshtsbm_shell` (P2), `baskargroup-admanufacturing` (P6).
  Both Dendrite-kt/PETSc/MPI; unzip + read into the respective spec at
  kickoff. (Not committed — gitignored; they're large binaries.)
- **RB1–RB3** (`flow_film_formulation_p2.md`) — RESOLVED 2026-07-20:
  FOLDED INTO P2. The old M6 flow/film Track B is absorbed by the
  NSHT-SBM-Shell hero; RB1–RB3 become P2 spec design questions
  (stabilization/projection/BC choices), settled during the P2 brainstorm
  rather than as standalone rulings.
- **8j auto-select promotion** — decided by #36's measured data (feeds
  S2); needs the #43 G4 GH200 numbers first.
- **THB use-case** (Baskar, open) — what, if anything, THB is decisively
  better at; until answered, S3 stays exploratory.

---

## The one-screen spine

```
DONE:  M0..M5 (physics) · M1d,#33,#34,#35,#36,#37,#38,#39,#40,#41 (software)

TRACK P (physics)                    TRACK S (software)
  P1 SP-1 R1 adjoint  ────────────── S1 perf queue (done: #40/#41/#36;
     (differentiable-science, now)      open: #42 sync, #43 G4)
  P2 NS-SBM-SHELL HERO ◀──── (hard) ─ S4 multi-GPU  ★priority raise★
     flow/slender obj, 100M+ dof        (chunk table ready)
     [Dendro C++ ref] proj/VMS/BDF2/precond
  P3 maize canopy ◀──(downstream of P2)
  P4 NS-PNP hero  ◀──(shares P2 spine) S2 mixed precision (from #36)
  P5 SP-2 beyond-FH                    S5 space-time (later)
  P6 additive mfg [Dendro C++ ref]     S3 THB — EXPLORATORY, not gating
Horizon P0 ✅ → fall-2026 CFD campaign paced by P2 hero + S4
```

**Recommended immediate action (revised):**
1. Ratify **P1 (SP-1 R1)** as the differentiable-science next rung — runs now, unblocked.
2. Ratify **P2 (NSHT-SBM-Shell) as the next hero** and **raise S4 (multi-GPU) to the priority software track** — it's P2's hard dependency for 100M dofs. Send the two Dendro C++ reference-code paths so the P2/P6 parity gates can be scoped.
3. THB (S3) explicitly set exploratory / non-gating, per your call.
4. RB1–RB3: decide whether the old flow/film ruling folds into P2 or stands alone.

P1 and P2 run in parallel (no shared critical path); everything else
schedules against S4's progress and the reference codes.
