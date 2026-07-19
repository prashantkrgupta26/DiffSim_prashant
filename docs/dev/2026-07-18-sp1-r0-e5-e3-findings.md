# SP-1 R0 Block E-b — E5 perf gate + E3 CPU-parity build attempt (gpubox)

Measured on **gpubox** (WSL2 host, 2× NVIDIA RTX 6000 Ada 48 GB, sm_89,
driver 13.2 / Warp 1.14 CUDA Toolkit 12.9), venv `.venv/bin/python`.
Local repo `sp1-r0` @ HEAD; driver `benchmarks/xdd/e5_perf_nirmal.py`.

## Box-state findings (recorded, load-bearing)

1. **WSL libcuda shadowing (blocked GPU until fixed).** On this WSL box
   `libcuda.so.1` resolves via ldconfig to a stale
   `/lib/x86_64-linux-gnu/libcuda.so.580.159.03` that gives Warp
   `CUDA error 100: no CUDA-capable device is detected`. The correct passthrough
   lib is `/usr/lib/wsl/lib/libcuda.so.1`. **All GPU runs must prefix
   `LD_LIBRARY_PATH=/usr/lib/wsl/lib`** — then Warp sees `cuda:0`, `cuda:1`.
   (Raw `ctypes` `cuInit` succeeds regardless because ldconfig also lists the
   WSL path; only Warp's own driver init picks the stale stub.)
2. **`CUDA_VISIBLE_DEVICES=` is empty** in the non-interactive ssh env; harmless
   once the LD_LIBRARY_PATH fix is applied (Warp then enumerates both GPUs).
3. Box was dirty with stale gpu-smoke `logs/*.log` (15-byte CI leftovers) —
   `git stash push -u` (`stash@{0}: sp1-eb: stash stale CI logs/ before sync`),
   then synced. The box git HEAD is master-era; `gpubox-sync.sh` rsyncs the
   sp1-r0 WORKING TREE over it (so the box shows the sp1-r0 files as "modified"
   vs its stale HEAD — expected; used `FORCE=1` for the second push).

## E5 — the Nirmal ≥100× perf gate

**Configuration.** Nirmal device = 513×129 over 400 nm×100 nm = **66,177 nodes**,
10 GHz rectified-sinusoid J(t), 1 ns window (10 optical cycles), short-circuit
V̂=0, PM6:Y6 A1 defaults, analytic bilayer morphology.

**Substrate caveat (honest).** `build_uniform` produces only SQUARE 2^level
grids on the unit square — there is NO anisotropic (513×129) octree in this
codebase; every Block-A/B brick builds on it. Per-step cost is governed by the
linear-system SIZE (DOF count) and sparsity, both fixed by node count, not by
the physical aspect ratio (nondim uses ĥ∈[0,1]). We therefore measured at
**level 8 = 257×257 = 66,049 nodes = 330,245 DOFs** — a **0.19% DOF match** to
the Nirmal 66,177-node mesh — and label the anisotropy proxy explicitly.

**The measured architecture (the headline finding).** Warp assembles element
matrices on `cuda:0`, BUT the global 5-field Jacobian is a **scipy CSR built on
HOST** and the Newton linear solve is **host scipy `splu`** (nonsymmetric sparse
LU); the A3 closures are host numpy. **cuDSS is NOT wired** into
`XDDSystem.linsolve` (default `splu(A.tocsc()).solve`); no cudss python binding
is importable on the box (`from cudss import CudssSolver` → ImportError; only
`nvmath` present). During the 330k-DOF step, `nvidia-smi` shows **GPU util 0 %**
while one python core is pinned at **100 % CPU** — the per-step cost is entirely
host-side sparse LU + host closures. The GPU is used only for element assembly,
a negligible fraction of a Newton iteration at this size.

**Measured numbers (level 8, cuda:0, marchable regime — see below).**

| quantity | value |
|---|---|
| nodes / DOFs | 66,049 / 330,245 |
| warm-up step (JIT compile + 4 Newton iters + host LU) | **411 s** |
| timed steps (walls, iters) | 97.8 s (1it), 288.5 s (3it), 95.2 s (1it) |
| s/step (median / mean, 3 timed steps) | **97.75 s / 160.5 s** |
| steps for 1 ns window (dt=5 ps → 200 steps/1ns) | 201 |
| end-to-end (PROJECTION: median × steps) | **19,648 s / 5.46 h** |
| speedup vs 10 h / 36-core baseline | **1.8×** |
| speedup vs 30 h baseline | **5.5×** |
| target | ≥100× |
| **verdict** | **MISS (off by ~18–55×; host-splu bound, GPU 0 % util)** |
| GPU util / mem mid-run | **0 % / 642 MiB** (both GPUs) |

**Regime note (why "marchable").** The physical PM6:Y6 config hits the
documented CPU-mesh drive wall (Ê_g≈42.5, Debye ≪ h — the same wall E1(ii)/E2
record): the BDF Newton fails at iter 1 on any feasible uniform mesh, so its
per-step timing is NOT representative (Newton bails early → undercounts cost).
We measure the **marchable regime** (E1's documented reduced-drive: λ²=1e-1,
symmetric μ̂, Ê_g=4) where the Newton runs its FULL iteration count → the
per-step cost (assembly + full host LU + closures) is representative of a
converging device step. The LINEAR-SYSTEM size/sparsity — hence the dominant LU
cost — is IDENTICAL to the physical regime (same mesh, same 5-field coupling);
only the physical drive strength differs. This is the honest s/step for the
gate; the physical mesh-wall is a separate recorded caveat.

**Deferred-brainstorm input (the A100-speedup / cuDSS path).** The gate outcome
is dominated by the host `splu` LU, NOT by GPU compute. To reach ≥100× at 330k
DOF the solver must move the LU onto the GPU: **cuDSS** (NVIDIA's GPU sparse
direct solver — the natural fit for this 2-D nonsymmetric Jacobian) wired into
`XDDSystem.linsolve`, keeping the assembled CSR resident on device. A `_try_cudss`
hook already exists in the driver (returns None today → splu). Secondary wins:
(a) reuse the symbolic factorization across the fixed-sparsity Newton iterations;
(b) move closures onto the GPU to kill the host round-trips; (c) an iterative
Krylov+GPU-preconditioner alternative if a direct factor is too memory-heavy.

## E3 — CPU-parity build attempt: **BLOCKED-E3** (evidence)

Source `drift_diffusion_ts` (Dendrite-KT + TalyFEM + PETSc, CMake) rsync'd to
`gpubox:/home/bglab/Baskar/xdd-cpu-ref/`. Canonical inputs present
(`drift_diffusion/test/config.txt`, `morph_bilayer_2D.txt`). Build is BLOCKED on
**every** hard `REQUIRED` dependency:

- **`external/dendrite-kt` submodule NOT vendored** — the local `.gitmodules`
  points at private `bitbucket.org/baskargroup/dendrite-kt.git`; the dir is
  empty. (`external/talylite` cmake modules also referenced, absent.)
- **MPI absent** — no `mpicc`/`mpicxx`/`mpirun`; no openmpi/mpich packages.
  CMakeLists: `find_package(MPI REQUIRED)`.
- **PETSc absent** — no `PETSC_DIR`, no `/usr/lib/petscdir`, no installed
  `petsc-dev`/`libpetsc-real-dev` (apt candidate 3.15.5 exists but installing
  needs **root**, which we do not have). CMakeLists: `find_package(PETSc REQUIRED)`
  + `BUILD_WITH_PETSC ON`.
- LAPACK/BLAS/(MKL) also `REQUIRED`.
- No `spack`/`conda`/`mamba` in user space.

`cmake` 3.22.1 and `gcc`/`g++` ARE present, but they are insufficient alone.

**Recommended path (named, not executed — beyond the 45-min timebox).** Install
the dependency stack into the box **user space without root** via
**`spack`** (bootstrap `spack` under `/home/bglab`, then
`spack install petsc+mpi ^openmpi`), or **conda-forge**
(`micromamba create -n xdd-cpu petsc openmpi cmake` — pulls MPI+PETSc+BLAS with
no root). Then `git submodule update --init --recursive` in the ref (needs
bitbucket auth), and CMake with `-DPETSC_DIR=$CONDA_PREFIX`. Estimated 1–3 h of
setup + build; deferred. The E3 parity (canonical `test/config.txt` Jsc vs XDD
`from_cpu_config`) remains OPEN with this precise blocker recorded.
