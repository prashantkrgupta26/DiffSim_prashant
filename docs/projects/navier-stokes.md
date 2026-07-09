# Navier–Stokes

**Incompressible flow around immersed bodies**, steady and transient, at p1 and
p2. The same shifted-boundary machinery that carries scalar BCs carries the
vector velocity Dirichlet condition (no-slip on the immersed surface), and the
pressure is stabilized so equal-order (u, p) works.

## The math

Solve ρ(u·∇)u = −∇p + μΔu + f, ∇·u = 0. Two time-discretizations run the same
physics as a standing cross-check: a **monolithic linearized** (Picard/Newton)
stepper and a **Leray projection** stepper. Stabilization is VMS/SUPG-PSPG; the
immersed no-slip is Nitsche with the unified penalty **α ≈ (1 + Pe/4)·p²**.

Tutorial track: `tutorials/D_flow/` (D1 MMS, D2 cavity, D3 cylinder).

## Quickstart

```bash
python benchmarks/navier-stokes/cavity_ghia.py            # lid-driven cavity ladder
python benchmarks/navier-stokes/cylinder_p2.py --solver cudss   # confined cylinder, p2
```

## Validation

| Case | Reference | Measured |
|---|---|---|
| Lid-driven cavity, Re 100 | Ghia, Ghia & Shin (1982) | max profile diff 0.035 (monolithic), 0.0035 (projection) |
| Confined cylinder, Re 20 | Schäfer–Turek band | C_d = 1.352 (p1) / **1.334 (p2, α = p²·10)** |

The p2 story is a cautionary one: with a fixed under-scaled penalty the p2
cylinder gave C_d 3.087 (and the cavity blew up to T = −15.9 at L7). The
order-aware α is what recovers the correct 1.334 — see
`docs/dev/m2-milestone-report.md`.

## Differentiability

Adjoints run through the full steady and transient NS solve. The traction /
drag functional is differentiable w.r.t. shape (the `ns_shape` +
`ns_adjoint` path), verified against finite differences and used directly in
the bunny-drag hero. This is the machinery the [differentiable-design
project](differentiable.md) is built on.

## Learn more

- Tutorials: D1–D3, E1 (shape optimization)
- Benchmarks: [`benchmarks/navier-stokes/`](../../benchmarks/navier-stokes/README.md)
- Provenance: `docs/dev/m1b-deferred-findings.md`, `docs/dev/m2-milestone-report.md`
