# Rung-2 GPU validation on gpubox — 2026-08-07

OrgElMorph adjoint rung-2 (GPU cuDSS backend) validated on **gpubox**
(2× RTX 6000 Ada, WSL2), diff-orgelmorph @ `7fd50d7`, daisy-morph feat/gradients
source (rsynced). Run via a git worktree + `PYTHONPATH` override (diffsim
editable `.pth` is a plain path, so PYTHONPATH wins); gpubox's own
`track-a2-strengthen` working tree was never touched.

## Results

| Gate | Result |
|---|---|
| `test_cudss_backend_matches_scipy_on_gpu` (DiffSim, cuDSS==splu incl. reuse) | ✅ pass |
| Full `test_multiphase_adjoint.py` on GPU (cuDSS + mean-φ three-way + CPU) | ✅ 24 passed |
| `test_gradients_gpu_uses_cudss_and_matches_cpu` (daisy, GPU grads == CPU grads, rtol 1e-8) | ✅ pass |
| **256×128 net-new run** (level=8, aspect=2, ternary) | ✅ 132,612 DOF, `gpu_cudss_fixed_step`, 20 steps, 79.3 s; χ/mobility/κ grads finite (χ₀₁=+1.02e-8) |
| `test_gpu_integration.py` (6 production-forward tests) | ⚠️ 5 pass / 1 fail (see below) |

## Bug found + fixed (hardware-only) — commit `7fd50d7`

The daisy GPU gradient test initially FAILED:
`RuntimeError: Factorization cannot be performed before plan() has been called`,
with nvmath warning "the specified LHS may have different buffers ... requires
calling plan() and factorize() again."

Root cause in `CudssBackend._solve_csr`: it reused one cuDSS `DirectSolver`
across solves via `reset_operands` when nnz was unchanged. But (1)
`scipy_to_torch_csr` allocates a **new device buffer every call**, while
`reset_operands` assumes the planned buffers are updated **in place**; and (2) a
single backend instance solves both the forward Jacobian `J` (`solve`) and the
adjoint `Jᵀ` (`solve_T`) during one gradient — **same nnz, different sparsity
pattern**, so an nnz-only reuse guard reuses a plan across incompatible patterns.
The standalone DiffSim test missed it because it never reused an instance across
repeated same-nnz solves.

Fix: **plan + factorize a fresh DirectSolver on every solve** (freeing the prior
one). The GPU-gated test was extended to reuse one instance across repeated
same-pattern solves + a `solve_T` after `solve`, pinning the regression.

## Known unrelated failure (NOT rung-2)

`test_gpu_integration.py::test_image_conventions_on_real_run` — a daisy-morph
**image color-convention** assertion: a component-0-rich pixel renders
red-dominant (R=0.74, B=0.47) where `COMPONENT_COLORS[0]` expects blue-dominant
(`|Δ|=0.85 > 0.35` threshold). This exercises the production forward + `r.save()`
compositing + `labels()` — none of which rung-2 touches. It is GPU-gated (skipped
on the CPU-only Mac) so it went unexercised; it is a pre-existing daisy-morph viz
issue to fix separately, and does **not** gate the rung-2 backend.

## Pin bump — deferred

`daisy-morph/pyproject.toml:13` pins diffsim to a GitHub sha. The bump waits on
`diff-orgelmorph` being pushed to origin (GitHub) as part of the merge decision;
it can't point at `7fd50d7` until that commit is on origin.
