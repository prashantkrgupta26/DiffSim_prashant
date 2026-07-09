# Response to the Codebase Critical Evaluation

*Triage completed 2026-07-06 (overnight session). Method: for every claim,
first write the test that would expose it, then verdict — CONFIRMED
claims got fixes gated by that test; disagreements are argued with tests,
not prose. All fixes pushed (commits `3c6d1d6`, `918a8db`, `dec21e6`).*

**Headline: the evaluation was right where it hurt.** All five critical
findings CONFIRMED. The deepest one (finding 1) was worse than stated:
not only did the taped Neumann residual omit the β terms — the *entire
Neumann branch of `shape_gradient` had no gradient test anywhere*. Our
verification culture measures what it gates; the lesson is that coverage
*maps* beat test *counts*, and the evaluation earned its 9/10-culture
score by finding the unmapped branch.

---

## Critical findings

| # | Claim | Verdict | Action |
|---|---|---|---|
| 1 | Neumann shape gradients don't match the forward (β terms missing) | **CONFIRMED — and worse** (no Neumann gradient gate existed at all) | β bilinear+load terms added to the taped residual; new gate `test_neumann_shape_gradient.py` parametrized over β∈{0, 0.5}, adjoint≡FD. The gate immediately caught an a-vs-a² slip in our own first fix — working as intended. |
| 2 | `g_fn=None` (valid homogeneous forward) crashes differentiation | **CONFIRMED** | Zeros + skipped g-chain (d(0)/dθ=0); gated adjoint≡FD on a homogeneous problem. |
| 3 | Kernel cache keyed on `__name__` can return the wrong kernel | **CONFIRMED** (by design review) | Keys now use class identity; regression test with two same-name bricks asserts distinct kernels. |
| 4 | TriMesh optimization can use stale BVH geometry | **CONFIRMED** | New `update_vertices()` contract (verts copy + warp mesh point refresh + BVH refit); gate: in-place-mutated oracle ≡ freshly built oracle at 1e-12 after a 1.35× inflate. |
| 5 | Wheel omits `amgx_configs/*.json` | **CONFIRMED** (as the evaluation itself verified) | `package-data` added; verified by building the wheel and listing both JSONs inside it. |

## Solver review

| Claim | Verdict | Action |
|---|---|---|
| AMGX singleton bakes first-call `tol`; `maxiter` not forwarded | **CONFIRMED** | Singleton keyed by (sym, tol, maxiter); `maxiter` forwarded through `solve_linear`. |
| Cached operators can silently use stale matrices | **HAZARD MITIGATED** | The cache is *deliberately* for constant matrices (PPE/mass across steps). Added a (shape, nnz, dtype) fingerprint per `cache_key` that raises on mismatch — catches the wrong-matrix class at zero cost. Full value-hashing would defeat the cache's purpose; the constant-matrix contract is now enforced and documented at the raise site. |
| Convergence guarded by `assert` (stripped under `-O`) | **CONFIRMED** | Both convergence sites now raise `RuntimeError`. |

## Geometry robustness

| Claim | Verdict | Action |
|---|---|---|
| Rotated `Box._R()` assumes 3-D beyond dim 2 | **CONFIRMED** (we run dim=4) | Explicit `NotImplementedError` fence for rotated dim∉{2,3}; unrotated boxes unaffected. SO(4) parametrization queued if ever needed. |
| TriMesh input validation insufficient | **CONFIRMED (partial fix)** | Shape/index-range/non-empty checks added. Watertightness/degeneracy checks are heavier and queued — note the admissibility contract (Newton projection ok-masks) already rejects most degenerate-geometry *consequences* at query time. |
| Batched Newton projection brittle (`linalg.solve` on the whole batch) | **ACKNOWLEDGED, deferred** | Real. The ok-mask machinery bounds the blast radius today (failures are reported per-point, and oracle admissibility is a documented contract). Planned fix: per-point `lstsq` fallback on singular batches. Not rushed overnight — this code is gradient-gated and we change it with the three-way gate armed. |
| Projection helpers hardcoded CPU float64 | **BY DESIGN (prototype epoch)** | The geometry layer is deliberately host/torch-float64 in this epoch (differentiability + closest-point robustness beat throughput at current problem sizes; measured: geometry is not a bottleneck). GPU-resident geometry is the M2 NeuralSDF path. |

## Architecture ("GPU-native overstates")

**ACKNOWLEDGED — with the trajectory.** Fair reading of today's tree:
assembly is host-side scipy; classification/geometry are host/torch. What
is device-resident and measured: element kernels (warp), all tapes, the
fused Krylov solvers, cuDSS direct solves (7.1× at 200k DOFs), AMGX. The
prototype's contract has always been *correct physics + exact adjoints
first, then migrate the hot loop* — this week's 656× constraints win and
the measured solver table are that migration in progress. The README
wording will be tightened from "GPU-native" to "GPU-first, host-verified
prototype" in the next docs pass. (Also noted: the review could not run
the suite in its environment — 195 tests were static-inspected; all
200+ are green post-fixes.)

## Packaging & reproducibility

| Claim | Verdict | Action |
|---|---|---|
| Dependency constraints too permissive | **CONFIRMED** | Major-version caps on all four core deps. |
| GPU backends not represented as extras | **CONFIRMED (partial)** | `cudss` extra added. AMGX has no PyPI package (source build + `pyamgx --no-build-isolation`); documented instead. |
| README commands stale | **CONFIRMED** | Paths updated to the current tutorial ladder. |
| Missing license / CITATION / contributing | **DEFERRED TO PI** | License choice is a project-owner decision (flagged to Baskar); CITATION.cff and CONTRIBUTING will follow it. |

## Meta

The review's strongest service was structural: it found the one
differentiable branch our gate map missed, and it did so from *static
inspection alone*. Two process changes adopted in response:

1. **Gate-map audit**: every `enable_backward` kernel and every
   `shape_gradient`/cotangent branch must appear in a test that compares
   against FD — a checklist now, not a habit.
2. **Parametrize over the OFF-default**: finding 1 survived because every
   test used `beta_neumann=0` (the default). New rule: gradient gates
   parametrize optional physics terms over on *and* off.
