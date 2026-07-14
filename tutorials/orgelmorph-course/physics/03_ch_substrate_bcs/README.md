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
