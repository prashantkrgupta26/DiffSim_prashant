# M2 Milestone Report — Heat/Mass, Closures, p2-NS, and the Bunny Frontier

Date: 2026-07-07. Status: **A/B/C COMPLETE; D closed as a measured
frontier** (Baskar rulings: Heat leads; in-loop closures; both Nu
benchmarks; D at p1 + pure-p2 in parallel).

## A — Heat/Mass bricks + coupler (COMPLETE)
Scalar advection-diffusion brick: MMS orders EXACT (p1 2.00/2.00, p2
3.00/3.00 with the VMS-COMPLETE residual — the missing -kappa*lap term
capped p2 at 2 until restored; NSPNP Eq. 28 provenance). Species =
same factory (genericity). Coupler BDF2 (temporal 2.44/2.25,
reference-isolated). SBM-thermal composition: Peclet-aware Nitsche
(the L7 blowup finding), consistent-flux extraction (mesh-converged
0.5556/0.5558/0.5516; direct gradients = noise), coupled Nu(Re=20)=3.21
(confined, honest). **de Vahl Davis: Nu 1.1174 / 2.2426 vs 1.118 /
2.243 (0.05% / 0.02%)** — baselines locked.

## B — Neural closures in-the-loop (COMPLETE; the S2 demo delivered)
dQ/dkappa EXACT (3e-9) through the full composition; three adjoint
rules documented (chi placement, lam-zeroing at replaced rows,
alpha-freezing). field_kappa_gradient API + per-GP gates. MLP-through-
adjoint core (J 57x). **S2 retrain demo: hidden conductivity closure
recovered on de Vahl Davis Ra=1e4 — J 274x, closure RMSE 0.049 on
±0.3 amplitude, from 25 probes** (frozen-flow one-way documented;
fully-coupled = recorded stretch).

## C — p2-NS (COMPLETE; the M1 amendment repaid)
Volume brick p-generic (rolled loops: 1.8 s vs 75-min unrolled).
Cavity gate monotone-to-Ghia (p2-L4 beats p1-L5 at 4x fewer elements).
**Cylinder Cd = 1.334 vs lit 1.33** with the p^2-scaled penalty.
**UNIFIED PENALTY LAW (new, measured): Nitsche alpha scales with the
physics AND the discretization — alpha ~ (1 + Pe/4) x p^2.** The p2
TAPE compiles in 75 s (residual form) and is FD-gated. **PURE-P2
FRAMEWORK fully gated** (volume/faces/tape/assembler/stepper/scalar).
Recorded research item: the p1-p2 BAND for NS (interface pinning;
Cd 4.146 at band(3) — scalar band rules don't transfer verbatim).

## D — the bunny finale (CLOSED AS A MEASURED FRONTIER)
The drag-history recipe executed at L5 with cuDSS-accelerated
transient chains. **The observability law, completed:** (i) sub-cell
edits (0.012 = 0.38 cells) are invisible to EVERY observable type —
drag J0 1.5e-4 matches the velocity-probe class (the QoI type never
mattered; edit-scale/h does); (ii) cell-scale edits (0.03 ~ 1 cell)
ARE observable — J0 2.14e-2, 140x — but naive per-alpha re-carving
makes the landscape carve-jump-dominated and GN stalls. **The cure is
M3 rung 2's actual design** (epoch-CONTINUATION along the optimization
path, not independent carves) — field-tested early, requirements now
measured. M2's converged inverse demos stand: H1 (sphere steady,
1.77e-3), H2 (sphere transient, 2.56e-4), B3 (closure retrain, 0.049).

## Findings index (this milestone)
Penalty law (Pe x p^2) · VMS-complete residual at p2 · consistent-flux
extraction + chi placement · lam-zeroing at replaced rows ·
alpha-freezing contract · rolled-loops rule (nbf>4) · signal ladder
made observable-independent · epoch-homotopy requirements (for M3).

## Open items
1. Baskar: cuFEM memo review; Nova submissions (+ band L6/L7 solver
   diagnostics — commands in the session log); warp filing; w0.
2. M3 rung 2 (epoch continuation) is now DE-RISKED with measured
   requirements — recommend it as M3's opening build.
3. Deferred: NS band-design study; fully-coupled closure retrain;
   taped-load tau(kappa) kernel; higher-Re bunny attempt.

## ADDENDUM (2026-07-08): THE FRONTIER CROSSED
D's measured frontier is now a converged demo: the cell-scale bunny
edit recovered to err 1.23e-3 (7x inside the bar) via M3 epoch
continuation + dense wake observation + mode-sparsity guarding
(baseline m3_bunny_headline.json; the six-run ladder = the methods
story). M2-D and M3 rung 2 close together.
