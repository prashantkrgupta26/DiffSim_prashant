# 06 — Toward 100M DOF

> ## ⚠️ PLANNED — not yet written
>
> This module is a **placeholder**. It has no `run.py`, `EXPECTED.md`, or
> `baseline.yaml` yet — only this outline and the pointers below. A future
> session builds it following the template of Modules 00–05 and the
> instructions in the course-root `HANDOFF.md`. **Do not** treat the
> numbers below as measured; they are targets.

The pressure-projection engine's reason to exist: its SPD pressure-Poisson
admits algebraic multigrid, so it — unlike the indefinite saddle — reaches
hundred-million-DOF meshes.

## What this chapter will teach

- **The scaling verdict.** The cuDSS direct solver **cannot** factorize the
  film hero (15.15M DOF, 1.6B nnz) on a GH200 — an HBM-saturation capacity
  wall, identical in fp64 and fp32. So the 100M-DOF pathway is **iterative,
  not direct**, which is exactly why the projection engine (SPD-PPE →
  AMG/AMGX) is the load-bearing choice (Chapters 02, 05 set this up).
- **The AMGX PPE study.** The Leray stepper's SPD PPE driven through AMGX
  (PCG + classical-AMG V-cycle) on 3-D uniform meshes: per-solve time, AMG
  iteration count, residual, memory, and the single-GPU DOF ceiling.
  Textbook AMG scaling; 100M fits in roughly 33 GB/GPU. Parity check vs
  `splu` at small levels.
- **Device $K_p$ assembly (the mesh-build wall).** The A100/GH200 forensics
  found a ~61 s/step host-side floor (assembly / scatter / GP fields) common
  to all three GPU generations — a silent Amdahl ceiling. Measure it, then
  show the device-assembled $K_p$ path that removes it, and the *one-time*
  mesh-build cost that remains.
- **The L8→GH200 path.** The PPE scalar DOF map for a 3-D uniform octree at
  level $L$ is $n=(2^L+1)^3$: L7 ~2.1M, L8 ~16.6M, L9 ~133M. 100M sits
  between L8 and L9. The study reveals where the single-GPU ceiling bites
  (the Warp $2^{31}$ array-shape limit vs host COO→CSR assembly) and the
  deployment tier (single big node: GH200/NVL4) it targets.

## Source material (read these to build the module)

- **`tests/ppe_amgx_scaling.py`** — the weak/strong-scaling harness (the real
  brick this module drives). Run on the box in a proj lane with
  `LD_LIBRARY_PATH` set for AMGX:
  ```bash
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build \
    python tests/ppe_amgx_scaling.py --levels 5 6 7 --solver amgx --steps 5
  python tests/ppe_amgx_scaling.py --parity --levels 4 5     # amgx vs splu parity
  ```
- **`docs/dev/production-code-conventions.md`** (scaling-pathway section) —
  the stage-residency table + 100M-DOF budget-line format every scaling
  chapter must carry (bytes/dof, nnz/dof, index width, multi-GPU comms).
- **`docs/dev/2026-07-23-projection-ladder-FINAL-synthesis.md`** — the
  projection engine's validation state; the "iterative-not-direct" scaling
  conclusion (the 15.15M-DOF cuDSS capacity wall).
- The device $K_p$ assembler (merged; the `#2` device-assembly work) — the
  brick that removes the host-side assembly floor.

## Target numbers (to be MEASURED on the box and locked)

- Per-level PPE solve time and AMG iteration count at L5–L8.
- The `amgx`-vs-`splu` parity residual at L4–L5 (must match to solver tol).
- Measured bytes/dof and the projected 100M-DOF single-GPU memory (~33 GB
  target).

## To build this module (checklist)

1. Read the sources above; run `ppe_amgx_scaling.py` on the box (AMGX lane).
2. Write `ppe_scaling.py` (core, curating the harness), `run.py` (harness +
   self-check table), `gen_figures.py` (scaling plot + `numbers/c6.tex`),
   `EXPECTED.md`, `baseline.yaml` — the Module-00–05 template.
3. Validate on the box; lock the measured numbers into `EXPECTED.md` +
   `baseline.yaml`.
4. Fill the LaTeX chapter `latex/chapters/c6.tex` (currently the PLANNED
   stub) with the measured scaling story and remove the PLANNED banner.
5. Add `INSTRUCTOR.md`, `grading_rubric.md`, `solutions/`.

See the course-root `HANDOFF.md` for the full, step-by-step instructions.
