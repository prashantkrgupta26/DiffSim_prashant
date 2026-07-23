# Projection Validation Ladder — design spec

**Goal.** Determine which of the three scaffolded concepts in "pressure-projection +
SBM" breaks the 3-D flow development, by validating each **in isolation** on an exact
body-fitted mesh, and recover a working, scalable projection scheme. The three
concepts, per Baskar's decomposition:

1. **Pressure-projection** on a body-fitted mesh (the split itself: predictor → PPE → correct).
2. **Weak imposition** of the no-slip Dirichlet condition (Nitsche).
3. **Shifting** the weak BC to a surrogate boundary (SBM).

## Premise — why this ladder, now

The R2a→R2c diagnostic phase (`2026-07-23-projection-sbm-weak-fixed-point-verdict.md`)
root-caused the projection+SBM under-development to a **weak fixed point of the
lagged-pressure split** (the predictor cannot build the driving stagnation pressure
from rest; u_hat pins at ~2% of monolithic). Pressure *instability* was separately
solved (`ppe_fine_scale`+rotational+outflow). But that diagnosis conflated all three
concepts. **This ladder separates them** so we learn exactly where development breaks —
and it is a direct test of the "structurally incapable" claim, which is suspect because
classical projection works for external flow in the literature.

## The enabler — confirmed available, zero new mesher code

The octree mesher already produces an **exact body-fitted carve** (verified 2026-07-23):
- `build_uniform(level, dim=2)` = quadtree; `dim=3` = octree (dim is first-class).
- `Box(center, half)` oracle (`geometry/csg.py`) with **cell-aligned half-width**
  `half = k/2^level` ⇒ `classify_lambda(tree, oracle, lam=0.0, domain="outside")`
  yields **no cut cells** ⇒ `extract_surrogate` faces coincide with the true box
  boundary ⇒ `GeometryData.evaluate` gives **`d=0`, `corr=1.0`** ⇒ the SBM Nitsche form
  reduces to **standard Nitsche exactly** (`S N_a = N_a` when `d=0`).
- Fixture delta from the sphere test is ~2 lines (`Sphere(...)`→`Box(...)`, `lam=0.0`).
- **Rung C** (shift) is obtained by choosing a **non-aligned** half-width
  (e.g. `half=0.20` at level 3 ⇒ `|d|_max≈0.05`) → genuine SBM shift, SAME code path.

So the whole ladder is a *parameter progression* on the existing stepper stack:
**box-alignment** (`d=0` vs `d≠0`) × **BC treatment** (strong Dirichlet vs weak Nitsche).

## The ladder

Each rung compares the **projection stepper** against the **monolithic stepper** on the
**same mesh** (the already-validated same-mesh relative oracle), and against literature.
Geometry is held fixed within the 2-D ladder (square obstacle in a channel); only the
BC treatment / shift changes.

| Rung | Mesh / BC | Isolates | Prediction |
|---|---|---|---|
| **0** | Lid-driven cavity, 2-D (no obstacle) | base projection soundness (Ghia ref) | should pass |
| **A** | Square-in-channel, exact body-fitted (`d=0`), **strong** Dirichlet no-slip | **projection alone** | decisive: predicted to break |
| **B** | Same mesh (`d=0`), **weak Nitsche** no-slip | projection + Nitsche | — |
| **C** | Square **offset** sub-cell (`d≠0`) → **SBM** shift, weak Nitsche | projection + Nitsche + shift | — |
| **A′** | 3-D **cube**-in-channel, body-fitted, strong Dirichlet | 3-D projection confirm | — |
| **C′** | 3-D cube, offset → SBM | 3-D full-stack confirm | — |

**Within every rung, two solver variants are run:**
- **single-pass** (lagged-p*, `inner_iterate=False`) — the current default; expected to
  expose the weak fixed point.
- **stabilized inner iteration** — a damped / Anderson-accelerated predictor↔PPE loop
  (the plain ν-loop is an unstable accelerant; this is its stabilized form). Rung A on a
  clean mesh is its cleanest possible trial and the direct test of the from-rest cure.

## Benchmarks & references

- **Rung 0 — lid-driven cavity:** unit square, moving lid U=1, Re=100 and 400.
  Reference: Ghia, Ghia & Shin (1982) centerline u(y), v(x) profiles. Pass = profiles
  match Ghia within a few % AND projection≈monolithic on the same mesh.
- **Rungs A/B/C — square cylinder in channel (2-D):** standard confined square-cylinder
  benchmark (Breuer et al. 2000 / Okajima). Two Re:
  - **Re=40 (steady)** — clean steady Cd; the primary "does the flow develop from rest"
    test (no shedding confound).
  - **Re=100 (unsteady, shedding)** — mean Cd and Strouhal St for literature anchoring.
  Exact reference values (Cd, St, blockage β) to be pinned in the plan from the cited
  benchmark; the **same-mesh monolithic oracle is the primary, box-free bar**.
- **Rungs A′/C′ — cube in channel (3-D):** Re=40 (steady) primary; monolithic same-mesh
  is the oracle, literature (e.g. Saha / Klotz) secondary.

## Pass/fail bar (per rung)

A rung PASSES iff, for the developed/steady (or mean) state:
1. **Same-mesh oracle:** projection Cd == monolithic Cd within R0 tolerance (box-free,
   the decisive check);
2. **Literature:** within the benchmark's tolerance (Cd, and St for Re=100);
3. **Health:** weak-div decaying, ‖p‖ bounded, |u.n| small (for Nitsche/SBM rungs);
4. **Development:** mean|u| reaches the monolithic magnitude (no weak plateau).
Report which solver variant (single-pass vs stabilized-inner) achieves it. A FAIL at a
rung localizes the breaking concept; the ladder stops advancing and we fix that concept.

## Anti-vacuity / controls

- The monolithic same-mesh comparison is the box-free oracle (already R0-validated;
  cuDSS==splu). Literature is the absolute anchor.
- Rung A is the load-bearing control: if projection matches monolithic with **strong
  Dirichlet on a body-fitted mesh**, the projection scheme is sound and the defect lives
  in Nitsche (B) or shift (C). If A fails, the projection implementation itself is the
  bug (most likely the p* lag/update) or requires the stabilized inner iteration — and
  the "structural" claim is refuted or confirmed accordingly.
- For unsteady Re=100: report mean Cd over ≥1 shedding period + St from the lift signal.

## Compute

Both gpubox GPUs, two lanes (the proven pattern): main repo `DiffSim`→GPU0,
`DiffSim-proj`→GPU1, separate checkouts, no run-lock contention. The projection march is
CPU-splu-bound; the **monolithic oracle** runs cuDSS on GPU — so the two naturally split
(projection CPU / monolithic GPU) and many rung×variant×Re runs parallelize across both
lanes. 2-D rungs are cheap (quadtree, CPU seconds–minutes); 3-D uses cuDSS on GPU.

## Process

Subagent-driven (`subagent-driven-development`), fresh branch `projection-ladder` off
master (keeps it clean from the `projection-sbm` diagnostic branch). Spec → plan →
implement rung-by-rung with review. Standing contracts: same-mesh oracle + literature as
correctness anchors; anti-vacuity (each rung's BC toggle must be load-bearing); honest
measured results; REPO-IDENTITY guard + two-lane isolation.

## What this unblocks

A validated, scalable pressure-projection + SBM forward engine on octree (the 100M path),
built on a scheme we understand rung-by-rung — and, via the clean linear/skew-symmetric
adjoint, the R3 differentiable-design engine. If a rung proves projection cannot be made
faithful, that too is a definitive result that fully validates the monolithic engine.
