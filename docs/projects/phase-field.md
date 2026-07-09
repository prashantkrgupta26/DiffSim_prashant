# Phase Field & Learned Thermodynamics

**Conserved and non-conserved phase-field dynamics** — Cahn–Hilliard (CH),
Allen–Cahn (AC), and ternary **solvent-evaporation films** — plus the milestone
punchline: *learning* the free-energy functional from simulated morphology data.

![Wodo morphology replication](../assets/img/morphology_gallery.png)

*Six regimes from the Wodo & Ganapathysubramanian (CMS 2012) organic-solar-cell
fabrication study, replicated device-bound on one A100 at the paper's 250×100
mesh — slow vs fast drying, off-critical stratification, chain-length effects,
solvent selectivity.*

## The math

- **Allen–Cahn** (non-conserved): ∂φ/∂t = −M(f′(φ) − κΔφ), monolithic Newton on
  the c³ term. MMS orders 2.00 / 3.00 exact; shrinking-circle rate 0.4 % vs
  theory.
- **Cahn–Hilliard** (conserved): mixed (φ, μ) form, ∂φ/∂t = ∇·(M∇μ),
  μ = f′(φ) − κΔφ. MMS orders 2.01 / 3.00; mass conserved to 1.9e-15.
- **Ternary evaporating film**: a Landau-frame moving-boundary formulation with
  a top-surface evaporative flux dh/dt = −k_e·⟨φ_s⟩_top, Onsager mobility
  coupling, and a C¹-regularized Flory–Huggins log to keep fields on the simplex.

Formulation: [`docs/theory/phasefield_formulation_p0.md`](../theory/phasefield_formulation_p0.md).
Course doc: `docs/course/m4_phasefield_tutorial.tex`. Tutorials:
`tutorials/F_phasefield/F1–F3`.

## Quickstart

```bash
python benchmarks/phase-field/wodo_nova.py --list     # the 14-case table (no compute)
python benchmarks/phase-field/wodo_fig67.py           # local 2-D morphology replication
python benchmarks/phase-field/m4_learn_fmix.py        # learn Flory–Huggins from a trajectory
```

## Validation

| Case | Result |
|---|---|
| AC / CH MMS (p1/p2) | orders 2.00/3.00 and 2.01/3.00; mass 1.9e-15 |
| Wodo Figs 3–7, all 14 cases | replicated at 250×100 in **~11 min on one A100** (≈100× the 2012 cost) |
| Learned Flory–Huggins (χ_pf, χ_ps, χ_fs, k_e) | recovered to **1e-7** in 147 s |

## Differentiability — the punchline

This is where DiffSim's gradients stop being a convenience and become the
science. Differentiating through the film solve, we recover four Flory–Huggins
parameters from a single synthetic morphology trajectory to 1e-7. The learning
arc also *mapped its own limits*: the Chebyshev T1 mode of δf′ is an exact gauge
invariance of the dynamics (verified 5e-15), and the higher modes alias
Flory–Huggins over one trajectory's composition range — so recovering a
genuinely beyond-FH functional **requires instrument-space observables** (S(q,t),
film height h(t)) or composition-diverse data. That requirements map is the next
rung.

## Learn more

- Tutorials: F1–F3
- Benchmarks: [`benchmarks/phase-field/`](../../benchmarks/phase-field/README.md)
- Cluster kit: `cluster/wodo_campaign/`
- Provenance: `docs/dev/m4-milestone-report.md`
