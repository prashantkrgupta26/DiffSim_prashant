# Physics P3 — Substrate / surface energy and boundary conditions

Real films are cast on a **substrate** and dry against **air**; those
interfaces are not neutral — one component prefers a wall. This chapter
adds a **wall free energy** to binary Cahn–Hilliard (the `A2` term in the
multiphase brick) and maps **both** Cahn–Hilliard boundary integrals to
the production assembly:

- **(a) mass-flux no-flux**, `grad(mu)·n = 0` — the natural condition on
  the φ (mass-balance) equation; realised as the *absence* of a φ-row
  boundary term, so mass is conserved to machine precision.
- **(b) wall condition**, `kappa grad(phi)·n + f_w'(phi) = 0` — the
  natural condition on the μ equation from the surface energy
  `F_w = ∫ f_w(φ) dS`, `f_w = g φ + h φ²`. It appears in
  `MultiPhaseStepper._assemble_host` as `-∫ N_a (g + 2h φ) dS` on the μ
  rows.

Neither exchanges mass, so the wall re-arranges material without a
reservoir. (Dirichlet pinning — which *does* exchange mass — is the
Computational-track BC chapter.)

## What it measures (all by quadrature)

Six boundary-condition cases — **neutral, attracting, repelling,
opposing walls, confined-lateral, demixing+wetting** — reporting
substrate φ, film-mean φ, enrichment Δφ, boundary-layer thickness, the
**true quadrature mass** `∫φ dV` drift, `projected_dofs`/`max_correction`,
and the full energy budget `F = F_bulk + F_grad + F_wall`. Plus a
**boundary-layer study** `δ ~ √κ` across ≥3 κ, and an honest
**projection/clipping demonstration**.

## Run

The **harness** (canonical workflow — config, provenance, tolerance gate):

```bash
PYTHONPATH=<repo>/src python run_harness.py --config configs/p3.yaml \
    --mode reference --output outputs/p3 --overwrite
PYTHONPATH=<repo>/src python gen_figures.py --run outputs/p3
```

The **student driver** (quick self-check, prints the enrichment table):

```bash
python run.py                 # neutral vs attracting vs repelling wall
```

Compare with [`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `substrate.py` | core: mesh, the wall energy, the two-wall subclass, quadrature diagnostics |
| `run_harness.py` | the harness workflow: 6 cases + δ(κ) study + clip demo → `results.json` |
| `run.py` | student driver; prints the enrichment table |
| `gen_figures.py` | renders `p3_*.png` + `numbers/p3.tex` **from a saved run dir** |
| `configs/p3.yaml` | quick / reference / research modes |
| `baseline.yaml` | tolerance-based reference gate |
| `doc_numbers.yaml` | document-macro ↔ `results.json` map (staleness gate) |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=0) — binary
Cahn–Hilliard plus the real substrate/air wall energy — and the shared
`diffsim.diagnostics` (conservation / energy / admissibility) library for
every number.

## Learning objectives

By the end of this chapter you can:

- Derive both natural boundary integrals of mixed Cahn–Hilliard from the
  weak form — the mass-flux condition on the φ (mass-balance) equation
  and the wall condition on the μ equation — and explain why only the
  wall energy `f_w(φ) = gφ + hφ²` needs a face term in the assembly.
- Read off the preferred surface composition `φ* = −g/(2h)` from the wall
  coefficients and predict whether a wall enriches or depletes the
  substrate before running anything.
- Explain why the wall energy conserves mass exactly (a condition on
  `φ`, not on the flux) while a Dirichlet condition does not, and why the
  *quadrature* mass `∫φ dV` — not the nodal mean — is the correct
  conserved quantity to check near a boundary.
- Predict the boundary-layer scaling `δ ~ √κ` from linearizing the
  Cahn–Hilliard equation about the bulk and **verify** it by fitting the
  measured decay length across several κ.
- Distinguish "the projection never fires for an interior φ*" from
  "conservation is guaranteed" — and force the projection to fire
  honestly (a bounds-violating initial condition) without breaking the
  converged-state mass.
- Read the three-part energy budget `F = F_bulk + F_grad + F_wall` and
  connect the sign/magnitude of `F_wall` to which component wets the
  substrate.

## Prerequisites

- **Concepts:** the calculus of variations applied to a *boundary* term
  (natural boundary conditions from integration by parts), the
  Cahn–Hilliard mixed `(φ, μ)` weak form, and quadrature vs nodal
  quantities. Python + NumPy.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment green)
  and **Chapter P1** (`01_ch_binary_energies`) — P3 adds a boundary term
  to exactly the bulk Cahn–Hilliard flow P1 derives, and reuses P1's
  free-energy/mass-conservation vocabulary (bulk `f(φ)`, gradient energy,
  quadrature mass) without re-deriving it.

## Expected cost

- **Device:** any CUDA GPU; the meshes are 2-D (64×64 at reference) and
  use well under 1 GB of device memory — comparable footprint to P1.
  Solver `splu` (the CH saddle is indefinite, same reason as P1; cuDSS's
  no-pivoting factorization is not used here — see `run_harness.py`).
- **Quick mode** (`--mode quick`, 32×32, the CI smoke tier): comparable to
  P1's measured **≈ 75 s wall** on an RTX 6000 Ada for a similarly-sized
  2-D mesh (most of it Warp kernel compile + Python start-up); target
  < 2 min.
- **Reference mode** (64×64, χ = 2.2, quench to t = 0.4, the `EXPECTED.md`
  numbers): a few minutes; target 5–30 min.
- **Research mode** (128×128, finer quench + denser κ sweep): tens of
  minutes.
- First run of a session pays a one-time Warp kernel-compile cost; later
  runs in the same environment are faster.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The harness `baseline.yaml` check green (or a documented deviation).
3. The four figures (`p3_fields`, `p3_profiles`, `p3_energy`,
   `p3_bdlayer`) regenerated from your saved run.
4. **Headline:** the enrichment `Δφ` for the attracting and repelling
   walls (substrate φ tracking `φ* = −g/(2h)`), the boundary-layer width
   scaling `δ ~ κ^s` with `s` and its fit window, and the true
   (quadrature) mass drift for the wall BC contrasted with a Dirichlet
   BC — all reported as measurements, not assertions.
5. **Verification:** the quadrature mass `∫φ dV` tracked over the run for
   every boundary condition (drift `< 10⁻¹³` in every case) and the
   boundary-layer `δ(κ)` log–log fit against the predicted slope ½.
6. **Failure:** quench from an admissibility-violating initial condition
   (raised fluctuation amplitude) and report `projected_dofs > 0` firing
   while the *converged*-state quadrature mass still holds — then explain
   why a linear wall (`h = 0`, no interior minimum) collapses the time
   step instead.
7. **Exploration:** answer one "Explore on your own" question with
   evidence (the boundary-layer κ-sweep or the opposing-wall
   top/bottom-inversion check are the recommended ones).
8. **Research bridge:** one paragraph on what the enrichment `Δφ` and the
   `δ ~ √κ` boundary layer predict for which component ends up at a real
   device's substrate vs top electrode in a cast film.
