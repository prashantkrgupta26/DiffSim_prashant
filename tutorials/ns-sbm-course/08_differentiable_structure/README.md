# 08 — Differentiable structure

> ## ⚠️ PLANNED — not yet written
>
> This module is a **placeholder** (README stub + outline only). A future
> session builds it following the Module-00–05 template and the course-root
> `HANDOFF.md`. Best built after Modules 00–05 (the forward engines) and 06
> (scaling) are the students' shared baseline.

The NS-SBM engine as an instrument for design: gradients through the solve,
and shape optimization of the immersed body.

## What this chapter will teach

- **The adjoint of the NS-SBM engine.** A gradient of a flow QoI (drag, a
  wake observable) with respect to a control — the SBM shift / the body
  shape, the Reynolds number, a boundary trace. The *steady* adjoint is a
  **single transposed solve** (no trajectory), which is exactly what makes
  it tractable at 100M DOF.
- **Three-way-verified gradients.** Following the sibling differentiable
  track's discipline: custom adjoint $==$ autograd twin $==$ finite
  difference, on the *same* operator the forward stepper marches. A gradient
  is not trusted until all three agree.
- **Shape optimization on the shifted boundary.** Because the SBM shift is
  fixture *data* ($d$, $\text{corr}$; Chapter 04), differentiating with
  respect to the body geometry flows through the distance field — a natural,
  mesh-free shape-design surface.

## Source material (read these to build the module)

- The **SBM adjoint / shape bricks already in the tree**:
  `src/diffsim/sbm/ns_adjoint.py`, `leray_adjoint.py`, `ns_shape.py`,
  `transient_adjoint.py`, `adjoint.py`.
- The **SP-1 R1 XDD adjoint work** (MERGED `fbfeed5`): the
  three-way-verification pattern (adjoint vs torch-twin vs FD), the unified
  `Control` interface, the exposed sensitivity rows — the template this
  chapter mirrors for the NS-SBM operator (`src/diffsim/xdd/adjoint.py`;
  spec `docs/dev/specs/2026-07-20-sp1-r1-adjoint-design.md`).
- The **P2 milestone's** "steady adjoint = 1 transposed solve, tractable at
  100M" argument (deliverable C, the maize steady-differentiable design
  engine).
- The sibling **`orgelmorph-course/differentiable/`** track for the module
  template of a gradient chapter (core module, `run.py` printing the
  three-way agreement, `EXPECTED.md`).

## Target (to be MEASURED and locked)

- A three-way-verified drag (or wake-QoI) gradient with respect to a control
  (shift/shape, Re, or a boundary trace): custom adjoint vs torch-twin vs FD
  agreeing to the FD floor.
- An inverse-recovery or a small shape-optimization step demonstrating the
  gradient drives the QoI.

## Status

**Future.** Ties into the E-differentiable track and the SP-1 R1 adjoint
work. Build it against the real SBM adjoint bricks, three-way-verify, lock
the numbers on the box, fill `latex/chapters/c8.tex`, and remove this
banner.

See the course-root `HANDOFF.md` for the full instructions.
