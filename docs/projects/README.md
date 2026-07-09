# DiffSim project pages

Each page is the front door to one solver family: what it solves, the weak
form, a runnable quickstart, the validation numbers, and — the part unique to
DiffSim — **what gradients are available and to what tolerance**.

| Project | Physics | Signature result |
|---|---|---|
| [Poisson / SBM](poisson-sbm.md) | shifted-boundary diffusion on immersed geometry | asymptotic 2nd order; the dyadic-halo rule |
| [Navier–Stokes](navier-stokes.md) | incompressible flow, p1 & p2, immersed bodies | cavity vs Ghia; cylinder C_d 1.334 (p2) |
| [Heat & mass transfer](heat-mass.md) | buoyant/thermal transport, Boussinesq coupling | de Vahl Davis Nu to 0.05 % |
| [Phase field](phase-field.md) | Cahn–Hilliard / Allen–Cahn / evaporating films | Wodo CMS-2012, 14 cases, ~11 min on one A100 |
| [Differentiable design](differentiable.md) | end-to-end adjoints, INR inverse design | bunny-INR recovery, headline 1.23e-3 |
| [Adaptivity](adaptivity.md) | spatial octree re-mesh + BDF1/2 temporal LTE | zero mass drift across re-carve; 178× dt growth |

All pages share the same shape, so you can skim one and know how to read the
rest. The math links out to `docs/theory/`; the runnable chapters live in
`tutorials/`; the full/parametric validation drivers live in `benchmarks/`; the
provenance of every number is in the findings logs under `docs/dev/`.
