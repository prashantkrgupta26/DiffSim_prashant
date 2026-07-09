# Heat & Mass Transfer

**Scalar transport coupled to flow.** Advection–diffusion of temperature (or a
species) with VMS/SUPG stabilization, coupled back into the momentum equation
through Boussinesq buoyancy, plus consistent surface-flux (Nusselt) functionals
on immersed boundaries.

## The math

Solve ∂T/∂t + u·∇T = κΔT + q with a stabilized weak form; buoyancy enters
momentum as a Boussinesq body force ρβ(T − T₀)g. The VMS scalar residual is
carried **complete through the −κΔT strong term** (the second-derivative `lapN`
tables) — this is what lifts p2 to observed convergence order 3.00 rather than
stalling at ~2.1.

Tutorial chapters: `tutorials/C_time/C2_advection_diffusion.py` (transport),
`tutorials/D_flow/` (the coupled context).

## Quickstart

```bash
python benchmarks/heat-mass/devahl_davis.py         # buoyant cavity, the Nu table
python benchmarks/heat-mass/coupled_nu.py --solver cudss   # heated immersed cylinder
```

## Validation

| Case | Reference | Measured |
|---|---|---|
| Buoyancy cavity, Ra 10³–10⁶ | de Vahl Davis (1983) | Nu_mid **1.1174** / Nu_max **2.2426** — 0.05 % / 0.02 % |
| Heated cylinder, coupled | literature band | Nu_D 3.21 |

de Vahl Davis is the tightest thermal gate in the suite — two digits on both
Nusselt measures. Baseline: `tests/baselines/m2_devahl_davis.json`.

## Differentiability

The scalar adjoint (`sbm/scalar_adjoint.py`) supports field-diffusivity
gradients (`dQ/dκ` to 3e-9) and in-loop learned closures — the
`b2_closure_recovery` / `b3_retrain_demo` drivers train a closure field through
the coupled solve (S2 retrain RMSE 0.049). This is the bridge from the flow
solvers to the [learned-thermodynamics work](phase-field.md).

## Learn more

- Tutorials: C2, D-track
- Benchmarks: [`benchmarks/heat-mass/`](../../benchmarks/heat-mass/README.md)
- Provenance: `docs/dev/m2-milestone-report.md`
