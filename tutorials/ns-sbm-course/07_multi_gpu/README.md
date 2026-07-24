# 07 — Multi-GPU

> ## ⚠️ PLANNED — not yet written
>
> This module is a **placeholder** (README stub + outline only). A future
> session builds it following the Module-00–05 template and the course-root
> `HANDOFF.md`. This is a **future / roadmap** chapter — it depends on
> Module 06 (the single-GPU AMGX baseline) landing first.

Beyond a single GPU: the distributed solve for problems whose DOF count
exceeds one card's memory — the multi-node deployment tier.

## What this chapter will teach

- **When one GPU is not enough.** Module 06 reaches ~100M DOF on a single
  big node (GH200/NVL4). The next tier — the film hero and finer meshes —
  needs a *distributed* solve. This chapter is the roadmap (Track S / the P2
  milestone).
- **Device-FGMRES + distributed AMG.** The mandated iterative path (cuDSS
  cannot factorize the hero) makes a device-resident FGMRES with a
  distributed AMG/AMGX preconditioner the load-bearing kernel. The
  multi-GPU comms pattern (halo exchange for the SpMV, the coarse-grid
  gather) is the design surface.
- **The partitioning + comms budget.** The 100M-DOF budget line (bytes/dof,
  nnz/dof, index width per the mixed-width rule, comms volume) extended to
  $N$ GPUs; the deployment-tier declaration (Horizon gb-large / AWS
  on-demand clusters).

## Source material (read these to build the module)

- The **roadmap reconciliation** (Track P / Track S) and the **P2
  NSHT-SBM-Shell hero spec** (the multi-node hero this chapter serves) —
  `docs/dev/2026-07-20-roadmap-reconciliation-strawman.md` and the P2
  milestone spec.
- The **device-FGMRES design note** (mandatory for the film hero per the
  scaling conclusion; referenced in the SP-1/P2 milestone notes as `#49
  device-FGMRES`).
- **Module 06's single-GPU AMGX study** as the per-GPU baseline the
  multi-GPU path must weak-scale from.
- The **remote-deploy toolkit** (`scripts/remote/`) for the gpubox/Nova
  multi-GPU wiring.

## Status

**Future.** This is a roadmap chapter: the single-GPU AMGX result
(Module 06) must land first — it defines the per-GPU baseline the
distributed solve weak-scales from. When a real multi-GPU distributed solve
exists in the tree, build this module against it (core → `run.py` →
`gen_figures.py` → `EXPECTED.md` → `baseline.yaml`, all validated on the
box), fill `latex/chapters/c7.tex`, and remove this banner.

See the course-root `HANDOFF.md` for the full instructions.
