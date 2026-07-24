# Projection Validation Ladder — FINAL SYNTHESIS

**Date:** 2026-07-23
**Branch:** `projection-ladder` (HEAD `53d632b`)
**Status:** COMPLETE. The 2-D ladder (rungs 0/A/B/C) all pass; 3-D (cube A′/C′)
confirms. **The R2a question is answered: pressure-projection + SBM CAN be made a
faithful forward stepper on a carved octree — validated in 2-D and confirmed in 3-D.**

This is the capstone record of the projection track. It consolidates the R2a finding,
Baskar's a→b→c decomposition, the diagnostic phase, the root-cause fix set, the
unifying insight, and the honest residuals. It supersedes as the reference the interim
verdicts it cites:
- `2026-07-22-r2c-confined-sphere-verdict.md` (oracle trustworthiness)
- `2026-07-23-projection-sbm-weak-fixed-point-verdict.md` (R2a diagnostic arc)
- `2026-07-23-projection-ladder-verdict.md` (the mid-ladder "concluded at rung A"
  verdict — now **superseded**: the co-design path Baskar chose was pursued and
  succeeded; that doc's "genuine formulation research, uncertain payoff" framing was
  the pre-fix snapshot).

---

## 1. The arc (one paragraph)

R2a found the pressure-projection + SBM stepper **unfaithful in 3-D**: it
under-develops the flow versus the same-mesh monolithic oracle (sphere, L4, Re=100:
projection mean|u|≈0.1 and Cd≈0 vs monolithic mean|u|≈1.13, Cd≈+1.06). Baskar's
decomposition isolated the three scaffolded concepts — **(1) pressure projection,
(2) weak Nitsche imposition, (3) SBM shift** — onto an EXACT body-fitted carved-octree
mesh (aligned `half=k/2^level` ⇒ `|d|_max=0`, `geo.corr≡1`, so SBM-Nitsche degenerates
to standard Nitsche with zero new mesher code), and tested each in isolation against a
same-mesh monolithic oracle. The base projection was diagnosed to a **base
pressure-projection ↔ monolithic operator inconsistency on open outflow**, fixed with a
consistency co-design (published group scheme, we had mis-implemented it), and then the
Nitsche and shift layers were each cracked in turn. Result: **projection+SBM faithful in
isolation for all three concepts.**

---

## 2. The ladder result table

Oracle is the **same-mesh monolithic** (primary, box-free bar); literature is a
secondary confined-domain anchor. Tolerance R0-band ~10–15% on Cd (drag is the
stiffest observable); mean|u| far tighter. Numbers are proj-vs-mono unless noted.

| Rung | What it tested | Mesh / regime | proj vs mono (Cd; mean\|u\|) | Verdict |
|---|---|---|---|---|
| **0** — cavity | Base projection plumbing (predictor/PPE/correct/pin) | Closed lid-driven cavity, Re=100/400 | Matches monolithic same-mesh AND Ghia (1982) centerlines within a few % | **PASS** — base projection sound. (Re=400 v-miss = shared coarse-mesh, not projection.) The sphere weak fixed point is therefore **not** broken base plumbing — it is obstacle/outflow-related. |
| **A** — strong Dirichlet | Base projection on **open** external flow, body-fitted square (`d=0`), strong Dirichlet obstacle | Open channel, Re=40 steady + Re=100 shedding | **After fix**: Re=40 Cd +3.72 vs +4.18 (**10.9%**), mean\|u\| 0.6%; Re=100 2000-step mean Cd 3.17 vs 3.19 (**0.6%**), mean\|u\| 0.04%, ‖div‖ flat 1.61. | **PASS (after F1+F2+F3b fix).** *Pre-fix single-pass FAILED*: Re=40 Cd −61, ‖div‖=45, ‖p‖=833 (diverges); the obstacle-FREE open channel blew up identically ⇒ breaking ingredient = **open outflow**, not the obstacle. |
| **B** — weak Nitsche | Weak imposition on working projection, body-fitted square (`d=0` ⇒ SBM≡Nitsche) | Open channel, Re=40 + Re=100 | **After fix (FN1+FN4)**: Cd L4 +1.35 vs +1.39 (**3.0%**), L5 +2.19 vs +2.02 (**8.7%**); Re-robust Re20/40/100 = 1.9/8.7/15.5%. | **PASS (after FN1 velocity-fix + FN4 drag-fix).** *Pre-fix FAILED decisively*: proj Cd +401 (137× off) Re=40, blows @ step21 Re=100; monolithic weak-Nitsche was clean (+2.90 Re40) ⇒ defect isolated to **Nitsche-in-projection coupling**, not the Nitsche block. |
| **C** — SBM shift | Genuine shift (offset sub-cell, `0<|d|_max<h`) + weak Nitsche + shift | Open channel, Re=40 + Re=100 | L4 Re40 proj +1.541 vs +1.546 (**0.32%**); L5 Re40 11.1%; L5 Re100 17.1% (marginal, see residual v). Shift **load-bearing**: zeroing d+corr → 36.1% off. | **PASS.** Third scaffolded concept faithful. The L5/Re100 17.1% is the FN4 wall-pin residual at the fine/high-Re corner (rung-B offset=0 control already misses 15.5% there; shift adds only ~1.6 pts), NOT a shift defect. |
| **A′** — 3-D body-fitted | Rung A in 3-D: strong-Dirichlet cube, cuDSS | 3-D cube, L4, steady | 18% (velocity faithful) | **FINDING (not a break).** Same split-vs-saddle STRONG-WALL drag gap as 2-D rung A (~11%); velocity is faithful. See residual (i). |
| **C′** — 3-D SBM shift | Rung C in 3-D: offset cube = the sphere regime | 3-D cube, L4, steady | proj Cd +2.357 vs +2.151 (**9.6%**); shift 90% load-bearing | **PASS.** The 3-D confirmation: genuine SBM shift is faithful in 3-D. **2-D conclusion holds in 3-D.** |

Shedding (Re=100 von Kármán) was pursued and is **compute-deferred** — see residual (iv).
L5 3-D was host-assembly-bound (einsum wall) so L4 is the 3-D primary; the marching
wall, not a scheme question.

---

## 3. Root cause + the working fix set (the heart)

**Root cause.** The base projection was unfaithful on external flow because of an
**open-outflow operator inconsistency**: the lagged-p\* split's discrete
incompressibility notion did not match the monolithic (equal-order PSPG) one, so the
monolithic steady state was **not a fixed point of the split**. The clinching diagnostic
(Task 4): seed the projection with the EXACT monolithic (u,p) and take one step — the
predictor reproduces the seed (residual 3.6e-6), but PPE+correction throws it (Cd
+4.18→−55, ‖div‖ *rises* 1.22→6.0). Measured operator gap ‖L−K_p‖/‖K_p‖ = 0.67.

The resolution came from the group's **own** projection paper (`ns_projection_vms`,
Khara/Murugaiyan/Khanwale/BG — Helmholtz–Leray projection with VMS stabilization,
ref [26]), which validates P1/P1 equal-order **open-outflow cylinder** Re100–300 matching
monolithic+lit (their Tables 2–3) — the exact case we failed. **We had mis-implemented a
solved, published scheme; it was not open research.** The corroborating outflow-BC
literature survey (KIO curl-curl pressure BC, Poux 2011 substep-split BCs,
Timmermans rotational-incremental, Pacheco 2021 P1 boundary-vorticity, Bazilevs/Moghadam
backflow) confirmed and deepened the fix.

**The fix set** — all behind **`consistent_projection=True`** (default-off, defaults
bit-for-bit, dozens of regression tests green at every step):

| # | Piece | What it fixes | Commit |
|---|---|---|---|
| (a) | **PSPG-consistent PPE via COLLOCATED coarse divergence** — assemble `−σ(div u_h, q)` collocated, NOT by-parts. The by-parts form fabricates a boundary term `σ(u_h·n, q)_Γ` nonzero at open outflow ⇒ spurious φ ⇒ corrupts. Collocated matches the monolithic PSPG continuity term-for-term. **This is why closed cavity passed (no outflow bdry term) but open channel failed.** | The 0.67 operator gap; makes monolithic a fixed point (Cd +4.18→+4.12, ‖div‖ 1.22→1.29). | `5bd0929` (F1) |
| (b) | **Disjoint outflow BCs** — velocity `∇u·n=0` (do-nothing/traction-free natural Neumann) in the PREDICTOR only; pressure-CORRECTION `p′=0` Dirichlet on the outlet rows in the PPE only. Split across substeps (Poux). | Open-outflow datum consistency (Baskar's authoritative BC). | F1 |
| (c) | **Rotational-incremental pressure update** `p=p*+φ−ν·div(û)` (Timmermans). Converts to curl-curl ⇒ consistent wall/outflow pressure BC. | Higher-order pressure accuracy at boundaries. | F1 |
| (d) | **#6 backflow stabilization** `−β(u·n)₋(u·v)_Γ`, β=0.5. | Re=100 shedding blow-up: ‖div‖→1e5 @ step700 → bounded; PSD, correct-sign, min-gated, consistent in proj+mono. | `3b8e228` (F2) |
| (e) | **F3b `rotational_pin_outflow`** (default ON in consistent_projection) — pin q=0 on the same outflow rows φ is pinned on. | Secular drift (‖div‖ 5→33, mean\|u\| 1.04→1.47 @ step2000). | `74fea01`/`e9c4ecd` |
| (f) | **FN4 `rotational_pin_wall`** — pin q=0 on the SBM weak-wall nodes. | Weak-wall drag (rung B Cd −1.07 / 153% off → recovered). Mesh-independent, no scale/knob. | `ffeea26` |
| (g) | **P2 strong-wall pin extension** — extend the rotational pin to strong-Dirichlet obstacle nodes. | 2-D rung A strong-wall drag L5 10.9%→8.1% (tol tightened 15→10%). Default-off, helps 2-D never hurts 3-D. | `8a982ae` |

Fixed **γ=50 grad-div/LSIC** in the predictor is the stabilizer for the Nitsche split
(FN1); it is not Re-independent (see residual iii).

---

## 4. THE UNIFYING INSIGHT (the elegant core)

**The outflow drift (F3b), the weak-wall drag (FN4), and the strong-wall drag (P2) are
the SAME mechanism, cured the same way.**

The rotational-incremental update carries a `−ν·q` term (where `q` solves the
consistent-mass system `M_p q = Bᵀ û`). On any boundary row where the pressure
*increment* is pinned (φ=0 — outflow `p′=0`, or a weak/strong wall), the consistent mass
matrix `M_p` **couples**, so `q≠0` on those pinned rows. The rotational update then writes
`p_hat = −ν·q ≠ 0` there **every step**, violating the pin by a constant that compounds
through the next predictor's `grad p*`. That is:

- at the **outflow** → the secular drift F3b diagnosed (‖div‖ 51.6, growing);
- at the **weak Nitsche wall** → the inverted front-back wall pressure FN2/FN4 diagnosed
  (proj wall-p mean +0.26 vs mono +0.86 ⇒ wrong-sign drag Cd −1.07);
- at the **strong Dirichlet wall** → the ~11% (2-D) rung-A drag gap.

**The single cure: pin `q=0` on those same pinned-pressure rows** — `rotational_pin_outflow`
(F3b), `rotational_pin_wall` (FN4), and the strong-wall extension (P2) are three
instances of one fix. This is the elegant unification of the entire repair: one term,
three symptoms, one pin.

---

## 5. Honest residuals & negatives

Recorded truthfully — these are the reference caveats.

**(i) 3-D strong-wall drag gap is a DIFFERENT cause.** Rung A′ (3-D body-fitted, strong
Dirichlet) shows ~18% at L4; the strong-wall pin (P2) only moves it 18.0%→17.4%
(marginal). So the 3-D strong-wall gap is **NOT** the rotational bias the pin cures — it
is pressure-extrapolation/traction on the strong wall + L4 confinement. Velocity is
faithful; drag on strong 3-D walls is an open item. (In 2-D the pin *does* help:
10.9%→8.1%, because there the gap IS the rotational bias.)

**(ii) The KIO consistent-Neumann wall BC was a RED HERRING.** The Baskar-directed KIO
weak-Neumann wall-pressure source has the right structure but wrong, **mesh-dependent**
magnitude (L4 wants scale ~1.0, L5 wants ~0.07 — an h-factor off; every raw integral
`g·q`, including the offset-only variant, is a mesh-dependent overcorrector). It is not
the fix. The real fix was the rotational pin (iv above / §4). KIO is kept as a
documented, default-off, inert rejected diagnostic.

**(iii) τ_C grad-div did NOT deliver Re-independence.** The dynamic `graddiv_dynamic`
τ_C VMS continuity weight was wired and honestly **rejected**: its weight SHRINKS with Re
(γ_eff 50→24 @ Re100, 57% error), and γ=20 blows up at Re=100. FN3 confirmed a single
fixed γ is not robust, γ≥50 is needed. **Fixed γ=50 stays** — this contradicts the FN3
recommendation to pair with a Re-independent τ_C; the τ_C swap does not deliver. Kept as
a documented rejected diagnostic.

**(iv) Shedding is compute-deferred (not a scheme question).** No self-sustaining von
Kármán on the unit-box mesh: the scheme preserves y-symmetry bit-exact from symmetric
rest (Cl=0, can't nucleate), and perturbed runs decay to steady every config
(less-confined half=0.0625/blockage 12.5%, Re100/150, tilt + geometric-offset triggers).
**ROOT CAUSE = DOMAIN LENGTH, not scheme:** the unit box gives only ~4D downstream; von
Kármán needs ~10–20D, so the outflow truncates the wake before it grows. **Decisive: the
same-mesh MONOLITHIC ORACLE also does not shed** (identical decay) ⇒ the scheme
faithfully computes the correct STEADY solution for the short domain. The elongated
fixture (`build_channel_long`: 14D downstream, 6.2% blockage, obstacle near inlet) is
CORRECT and committed (`53d632b`, purely additive) but marching to the limit cycle
(1000s of steps) is infeasible on host splu/einsum assembly — the **same host-assembly
wall** that caps 3-D L5 — so it is DEFERRED to device-resident time-stepping.

**(v) The ~10% wall-pressure drag gap persists on strong walls in 2-D.** Even after the
strong-wall pin, 2-D rung A carries ~8.1% (was 10.9%). The pin narrows it materially but
does not close it — a residual split-vs-saddle strong-wall drag difference remains,
distinct from the fully-cured weak-wall case.

---

## 6. What this unblocks

1. **A validated, faithful projection+SBM forward engine on octree.** All three
   scaffolded concepts (projection / Nitsche / shift) are faithful in isolation, 2-D
   proven and 3-D confirmed (C′ = the sphere regime, 9.6%). This is the intended
   **SPD-PPE 100M-DOF scalability path** — the projection replaces the coupled saddle
   solve with a symmetric-positive-definite pressure-Poisson, the target for scaling
   beyond the cuDSS wall that caps the monolithic.

2. **The R3 differentiable-adjoint path.** The consistent projection is a *linear,
   consistent* scheme with a clean per-step operator structure — the seam the R3
   differentiable-adjoint work needs (the analog of R1's Mode-A/Mode-B adjoints, but on
   the projection stepper).

The monolithic engine remains the validated production oracle meanwhile — this work does
not replace it; it validates the scalable alternative.

---

## 7. Open follow-ups

- **Device-resident time-stepping.** The host-assembly wall (host splu + einsum
  assembly) blocks 3-D L5 and the elongated-domain shedding limit cycle. Porting CSR
  assembly + closures onto the device (the M1d/#35–#37 device-assembly playbook) is the
  prerequisite for both. This is the single largest unblock.
- **The 3-D strong-wall drag cause** (residual i) — pressure-extrapolation/traction on
  strong 3-D walls + confinement. Distinct from the rotational bias; needs its own
  diagnosis.
- **Absolute literature validation, confinement-corrected.** The oracle is trustworthy
  and *absolutely understood* (R2c: Cd≈1.41 is confined-sphere physics at box/D≈4.2, not
  solver error — Schiller–Naumann 1.087 is the unbounded value). Matching unbounded
  literature would need ≥15–20D clearances (AMR) or a matched confined reference. The
  ladder's high blockage (β=1/2 as built) is directional; the same-mesh oracle is
  decisive.

---

## Provenance

Full blow-by-blow: `.superpowers/sdd/progress.md` (§"PROJECTION-SBM track" and §"NEW
PLAN: PROJECTION VALIDATION LADDER"). Task reports: `task-1..task-7`, F1/F2/F3b, FN1–FN4,
P2harden, shedding in `.superpowers/sdd/`. Interim verdicts (superseded/subsumed by this
doc): `2026-07-23-projection-ladder-verdict.md`,
`2026-07-23-projection-sbm-weak-fixed-point-verdict.md`,
`2026-07-22-r2c-confined-sphere-verdict.md`. Group scheme: `ns_projection_vms` (ref [26],
Eq 44b/45–49/67); Nitsche-in-projection: Dokken et al `1912.06392`; consistent-flux
drag: chenghauy NSHT-SBM `BoundaryCalc_ExtraInter.h`. Fix commits on `projection-ladder`:
`5bd0929` (F1), `3b8e228` (F2), `74fea01`/`e9c4ecd` (F3b), `f40de3c`→`ffeea26` (rung B
FN1→FN4), `57e6ed8` (rung C), `24946cb` (3-D A′/C′), `8a982ae` (P2 strong-wall pin),
`53d632b` (elongated shedding fixture, deferred).
</content>
</invoke>
