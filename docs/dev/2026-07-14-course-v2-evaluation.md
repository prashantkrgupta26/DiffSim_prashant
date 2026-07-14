# Course v2 — scientific-workflow overhaul (spec of record)

Source: Baskar-supplied evaluation, 2026-07-14. Goal: turn each Physics
and Computational tutorial from a *demonstration* into a *repeatable
scientific workflow* that trains a new grad student to derive → implement
→ verify → profile → interpret → perturb → reproduce.

Already landed (critical-eval merge, 2026-07-14) — do NOT recreate:
CI `.github/workflows/ci.yml` (cpu job live, gpu-smoke token-gated);
runtime asserts → typed exceptions (`diffsim.errors`); `SolveResult`
types; honest README GPU claims; PEP 621 packaging + `[cudss]`/`[dev]`
extras. Extend these, don't duplicate.

## Phase 0 — shared foundation (build FIRST; everything depends on it)

- `tutorials/orgelmorph-course/common/`:
  - `config.py` — YAML load + schema validate + save `config.resolved.yaml`.
  - `provenance.py` — write `metadata.json` (diffsim commit, git_dirty,
    timestamp_utc, host, gpu + memory, cuda driver/runtime, python, warp,
    torch, nvmath, precision, device, solver, mesh, time_integrator,
    tolerances, seeds, wall_time, peak_device_memory, exit_reason).
  - `check_results.py` — tolerance/range baseline checker, NONZERO exit
    on failure. Scientific invariants + documented tolerances, NOT bitwise
    GPU identity.
  - `run_base.py` — standardized CLI every run.py uses: `--config
    --device --solver {auto,cudss,splu,blockch,amgx,matrix_free} --output
    --seed --mode {quick,reference,research} --overwrite --resume
    --log-level`. YAML is the canonical run record. `--solver auto` picks
    cuDSS when available, documented fallback for small runs.
  - Output schema `outputs/<run>/{config.resolved.yaml, metadata.json,
    results.json, history.npz, run.log, checkpoints/, figures/}`.
- `00_setup_and_smoke_test/` with `doctor.py` printing the full env probe
  (python, diffsim commit+dirty, CUDA, GPU+mem, driver/runtime, warp,
  torch, FP64, cuDSS/nvmath, AMGX, scipy fallback, a tiny Warp-kernel
  compile+run, an 8x8/16x16 assemble+solve) ending in exactly one of:
  `READY: full CUDA` / `READY: reduced (no cuDSS)` / `NOT READY: <fix>`.
  Document install variants (`pip install -e .`, `.[cudss]`, `.[dev,cudss]`)
  using the extras actually in pyproject.
- Mode runtime targets: quick <2 min, reference 5-30 min, research
  documented.
- `src/diffsim/diagnostics/` reusable library, each fn with units +
  unit test: conservation (quadrature mass, component content, boundary
  flux, moving-domain balance, transfer mass-change), energy (bulk/grad/
  wall/cryst/coupling/total/stepwise-increment), morphology (structure
  factor, peak+first-moment wavelength, two-point correlation, interfacial
  area, anisotropy, phase fractions, percolation), admissibility (min
  fields, simplex residual, projected/clipped fraction), convergence
  (observed order, Richardson), stochastic (ensemble aggregation,
  first-passage, bootstrap CI), profiling, provenance.
- Makefile: `course-doctor tutorials-quick tutorials-reference
  course-figures course-pdf`.
- Extend CI: `unit-tests.yml` (have), add `tutorial-smoke.yml` (cpu/
  reference micro + scheduled self-hosted GPU) + `docs-build.yml` (fail
  when doc numbers are stale vs generated results). Archive metadata/
  results/logs as artifacts.

## Phase 1 — P0 scientific corrections (per chapter; use Phase-0 infra)

- P1: interface width ℓ ~ √(κ/W) not √κ; document FH regularization +
  projection (report phi_min/max, projected_dofs, max_correction);
  separate continuous Lyapunov vs discrete energy-stability vs one
  monotone run (report stepwise increments + largest positive); ADD
  linear-stability (dispersion → fastest k → predicted wavelength vs
  measured S(q)) and coarsening L(t) with ≥2 length defs + fitted
  exponent + interval/uncertainty; 5-seed research ensemble (mean±sd);
  keep one fixed seed for figures but drop "bit-reproducible morphology".
- P2: DO NOT equate lower energy with accuracy. Build a tight-tol/small-dt
  reference; compare each method by relative L2(c(T)) + energy/S(q)/
  domain/phase-fraction error; add a smooth deterministic temporal-order
  problem first. Report REAL cost (accepted/rejected steps, full+half
  solves, Newton+linear its, wall, peak mem) — not accepted-step count.
  Replace "fixed needs min-dt for whole horizon" with a labeled worst-case
  bound OR a matched-accuracy fixed sweep. Add ≥4-tol convergence (error
  vs tol, wall vs error, steps, dt(t), Newton(t)); conclusion = accuracy-
  cost knee. Document variable-step BDF2 coeffs/startup/step-ratio/reject-
  history/error/Newton-failure.
- P3: teach BOTH CH BCs (∇μ·n=0 and κ∇φ·n+f_w'(φ)=0), map both boundary
  integrals to code; QUADRATURE mass (∫φ dV, assembly quadrature) not
  nodal-mean-vs-nominal; wall-energy case must conserve to solver tol —
  fix the 1e-3 drift (projection) with a conservative correction or report
  it honestly and stop calling it exact; full energy budget F=bulk+grad+
  wall; 6 cases (neutral, attract, repel, opposing, periodic-lateral,
  demixing+wetting); g,h ↔ surface-energy asymmetry (contact-angle only if
  implemented); boundary-layer thickness vs κ/curvature.
- P4: replace median-split tie-line with GMM/KDE/gradient-excluded
  clustering (report means, cov, population, sensitivity); plot free-energy
  contours + simplex + Hessian eigenvalues + spinodal + IC + predicted
  binodal + sim cloud; track admissibility; quadrature conservation + lever
  rule; introduce N_i factors; materials loader + schema (provenance/
  status), no manual copying.
- P5: MATCHED terminal state (domain scale at phi_s=0.30/0.20/0.10 for
  every rate; or vs mean solvent fraction continuously) — don't confound
  rate with final state; fix domain-scale UNITS (define metric; length not
  "cells"; a resolved length can't be <1 cell); per-solute moving-frame
  balance h(t)∫φ_i + solvent loss + height; fixed vs physical (×h(t))
  coordinate plots; Biot/Peclet regime map; multi-metric morphology;
  exit_reason honesty.
- P6: Avrami with X*=(X-X0)/(1-X0) or generalized JMAK w/ X0/incubation;
  report fit interval/n/R²/CI + seed-radius + threshold sensitivity;
  crystallinity by quadrature integral (thresholded = secondary);
  interface-velocity vs undercooling + critical-radius experiment (sub/
  supercritical); label accelerated params (e.g. 250 K); state orientation
  is a fixed grain label not evolved field (no implied GB physics).
- P7: RESOLVE p1-vs-r14 — for p1 present the mutually-weighted χ_eff (four
  aa/ac/ca/cc terms), for r14 state whether non-amorphous χ are absolute or
  increments vs χ_aa; unit-test all four pure limits; 3 commensurate
  controls (CH-only / CH+AC no χ-coupling / full) same mask+metric, no
  divide-by-floor amplification; coupled energy budget; causality
  experiments (delayed onset, frozen mask, kinetics@fixed-coupling,
  coupling@fixed-geometry).
- P8: DERIVE the discrete FDT normalization (dt, wJ dependence) then
  VERIFY numerically (fix physical noise, refine dt+mesh, measure
  equilibrium variance in a quadratic well → physical statistic
  converges); ENSEMBLES (nucleation prob, induction-time dist, nuclei
  density, X dist, mean+95%CI) not single seed; rename to noise-amplitude
  sweep (not temperature) unless all T-terms update; quantify clipping;
  first-passage event detection.
- P9: RENAME "evaporation-conditioned embryo growth" (seeds implanted, not
  spontaneous nucleation); add FDT-noise advanced mode for real
  nucleation; 6 controls; DERIVE solubility/growth threshold from the
  implemented p1/r14 free energy (sign conventions) + validate by embryo
  composition sweep; termination status enum; fix docstrings claiming splu
  while cudss selected + unused args + asserts (mostly done in
  critical-eval) + document constant-mobility simplification; temper
  "entirely solvent-fraction" causal language.

- C1: separate error sources (steady MMS or dt small enough for spatial;
  over-resolved mesh for temporal); report L2/H1 for BOTH c and μ + mass +
  residual; ≥4 levels p1 / 3-4 p2 / 4 dt; verify reference (halve ref dt,
  Richardson); control algebraic error (report tols, show tightening
  doesn't move discretization error); add deliberate-failure examples.
- C2: full mixed-system BCs; manufactured vs physical Dirichlet; constraint
  implementation (row replacement, symmetry, Jacobian, hanging nodes,
  matrix viz); flux-balance dm/dt = -∮J·n; BC test matrix; one weak BC.
- C3: RENAME the static-circle demo "Octree refinement + hanging-node
  constraints" OR implement true dynamic AMR (estimate→mark→refine/coarsen
  →2:1→rebuild→conservative transfer→BDF-history transfer→continue) with
  mass/energy/transfer checks; real temporal-adaptivity accounting (not
  horizon/min-dt ratio); tutorial-local wrong-BDF2 impl for the comparison
  (not an inaccessible dev note); space-time interaction (remesh during
  var-step BDF2).
- C4: replace choose_solver(dim,dofs) with measured decision support
  (dofs, nnz, block struct, precision, mem, reuse, tols, dt, hardware,
  library availability) returning rationale; versioned benchmark artifacts
  (commit/machine/gpu/cpu/cuda/versions/mesh/nnz/tols/warmup/repeats/sync);
  timing components (assembly/transfer/symbolic/factor/solve/setup/newton)
  cold+warm, CUDA-synced, median+IQR; correctness (residual/iters/flag/
  diff-from-trusted) alongside speed; qualify "cuDSS always/AMGX no/only
  matrix-free" claims; REMOVE global monkeypatch → supported
  set_linear_solver API; graceful no-cuDSS fallback.
- C5: correct sparsity explanation (FE connectivity/order/fields/blocks/
  constraints/dim, not 3^d-1); COMPLETE memory accounting (row ptrs,
  dup buffers, vectors, block meta, l2g maps, quadrature/geo, history,
  workspaces, fill-in/precond, output) + `estimate_capacity.py`; qualify
  int32 as impl-specific + discuss int64/distributed; compare PHYSICS at
  equal resolution (topology/coarsening/interfacial-area/S(q)/
  stratification/percolation); domain + aspect-ratio effects.

## Phase 2 — pedagogical completeness

Mandatory tutorial template (why-matters, objectives, prereqs, cost,
model+assumptions, dim+nondim params w/ source+uncertainty, numerical
formulation, where-in-diffsim, baseline, verification, failure modes,
guided exercises, research bridge, required deliverable, references) +
INSTRUCTOR.md/solutions/rubric per chapter. P00 model-hierarchy +
thermodynamics + nondimensionalization (symbol table, p1-vs-r14
conventions, dimensional/nondim/numerical/accelerated separation). C00
weak-form → CUDA (strong→IBP→element residual/Jac→quadrature→l2g→CSR→
constraints→Newton→linear→device, mapped to exact files; hands-on: modify
a weak-form coefficient + add its Jacobian + regression test). materials.
yaml schema (value/units/definition/nondim/source{type,citation,location}/
uncertainty/validity/notes) + validated loader; remove MyPapers private
paths; resolvable bibliography keys.

## Phase 3 — research-readiness

Dynamic AMR (if not done in C3); C6 nonlinear diagnostics (+ decision
tree, failing cases, failure checkpoints); C7 CUDA profiling (Nsight,
NVTX, bandwidth vs intensity, plan reuse, HWM); C8 extending-diffsim
(term → schema → residual → analytic Jac → deriv test → limiting case →
MMS → conservation → profile → docs); C9 reproducible campaigns (immutable
configs, sweeps, seed ensembles, SLURM arrays, checkpoint/restart, no
silent failure drops, CI); C10 validation/uncertainty (code vs solution
verification vs model validation vs param uncertainty vs stochastic
variability). P10 material case-study capstone + P11 model-extension
capstone. Assessment model + rubric (§11); definition-of-done checklist
(§14) per tutorial.

## Cross-cutting principles

Every deliverable: resolved config + metadata.json + tolerance checks +
figures generated from saved data (not hand-copied) + a failure-diagnosis
+ a research bridge. Never bit-reproducibility across GPUs — scientific
invariants + documented tolerances. Every chapter's equations must match
the actual production code branch used.
