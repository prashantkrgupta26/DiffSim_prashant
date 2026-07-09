# DiffSim

**GPU-native, differentiable finite-element multiphysics on adaptive octrees.**

DiffSim solves PDEs on complex, moving, *optimizable* geometry — and computes
exact gradients of the results with respect to that geometry, the material
fields, the boundary data, and learned closures. Classical simulation answers
*"what happens?"*; differentiable simulation answers *"what should I change?"*.

![Wodo morphology replication across six regimes](assets/img/morphology_gallery.png)

*Six drying regimes from Wodo & Ganapathysubramanian (CMS 2012), replicated
device-bound on one A100 — from these fields DiffSim recovers the Flory–Huggins
free-energy parameters to 1e-7.*

## Start here

- **[Solver project pages](projects/index.md)** — the front door to each physics
  family (Poisson/SBM, Navier–Stokes, heat & mass, phase field, differentiable
  design, adaptivity): what it solves, the weak form, a quickstart, the
  validation numbers, and the available gradients.
- **Curriculum** (see the nav) — tracks A–F + P, one runnable chapter each, from
  a 2-D manufactured solution to recovering a shape from sensor data.
- **Theory** (see the nav) — the SBM and phase-field formulations, the cuFEM
  adjoint memo, and student briefs.

## What makes it different

**Immersed geometry, never remeshed.** The Shifted Boundary Method imposes
conditions on a grid-aligned surrogate boundary with a Taylor-shift correction,
so geometry changes never remesh a body-fitted grid — and gradients never
differentiate through meshing.

**Differentiable by architecture.** Within an epoch the mesh is frozen and the
solution is a smooth implicit function of the parameters; gradients flow through
an adjoint solve, a Warp-taped sweep, and a Torch geometry chain, verified three
ways (adjoint, autograd twin, finite differences).

**GPU-resident, and taught to be read.** Warp generates the FP64 kernels, the
solve stays on the device, and every numerical decision is documented where it
lives with a test that measures it.

![A solvent front descends through a drying film](assets/img/drying_strip.png)

*An inverse problem in the making — recover the hidden physics of a drying film
by differentiating through the solver.*

The source lives on [GitHub](https://github.com/BaskarGS/diffsim). Milestones M0
through M4 are complete and gated.
