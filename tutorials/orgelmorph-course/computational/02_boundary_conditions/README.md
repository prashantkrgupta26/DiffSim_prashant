# Computational C2 — Boundary conditions, numerically

The **full** boundary structure of the mixed Cahn–Hilliard system,
measured on one blend. The mixed system has **two** boundary conditions
(a mass-flux term ∇μ·n and a wetting term κ∇c·n), and *how* each is
imposed decides what is conserved and whether the matrix stays symmetric:

- **Natural (no-flux + no-wetting)** — the default; omit both surface
  integrals. Costs nothing, conserves mass exactly.
- **Wall free energy (wetting/Robin)** — `κ∇c·n + g_s'(c)=0`; real thin-
  film physics, in the multiphase brick (see
  `docs/dev/2026-07-13-m5-device-assembly.md`, A2 wall term).
- **Prescribed composition / potential (Dirichlet)** — pins a value by
  editing rows; a reservoir, does not conserve.
- **Manufactured (MMS)** — pins *both* c and μ (what C1 did); a device,
  not a physical reservoir.

Plus: the **flux balance** `d/dt ∫c = −∮J·n` (mass changes at exactly the
wall-flux rate), a **BC test matrix** over the taxonomy, and **weak vs
strong** imposition (row replacement / symmetric elimination / penalty)
with a matrix visualization.

**Read** the course document, Computational Chapter *"Boundary
conditions"* (start with `bc.py`).

**Run:**
```bash
python run.py                 # both treatments, compared, ~30-60 s
python run.py --wall 0.5      # different Dirichlet wall value
```
`run.py` prints the mass-conservation and boundary-layer contrast with a
`PASS/FAIL` check; compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c2_*.png + numbers/c2.tex
```

| file | role |
|------|------|
| `bc.py` | the core: `run_bc`, `compare`, `boundary_free_nodes` — read this first |
| `run.py` | the driver you run; prints the contrast table |
| `gen_figures.py` | regenerates the field + mass figures and `numbers/c2.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` and its unchanged boundary
machinery (`dirichlet=`, `gc_fn`/`gm_fn`, and the natural default).

## Learning objectives

By the end you can:

- Derive the **two** boundary terms of the mixed $(c,\mu)$ weak form
  (the mass flux $M\,\nabla\mu\cdot\mathbf n$ in the transport equation
  and the interfacial term $\kappa\,\nabla c\cdot\mathbf n$ in the
  $\mu$-equation) and say what "natural" means for each.
- Compare the boundary-condition taxonomy — natural (no-flux), wall
  free energy (wetting/Robin), prescribed composition/potential
  (Dirichlet), and manufactured (MMS, pinning both fields) — and state
  which of them conserve mass and which do not.
- State and verify the flux balance
  `d/dt Int c dV = -Int_dOmega J.n dS`, and read off from a run when the
  boundary integral is exactly zero (no-flux) versus a decaying
  reservoir influx (Dirichlet).
- Explain why pinning $c=g_c$ **and** $\mu=g_\mu$ (as C1's MMS study
  did) is a manufactured-solution device, not a physical boundary
  condition, and what a physical composition-pinned contact should
  leave free.
- Contrast strong imposition (row replacement vs symmetric elimination)
  with weak/penalty imposition of a Dirichlet value: which preserves
  matrix symmetry, which is exact, and the `beta` conditioning trade-off
  of the penalty method.
- Run the BC test matrix and diagnose why a `c=0` Dirichlet wall nearly
  conserves mass — and why that is a coincidence of the initial
  condition, not a property of the boundary condition.

## Prerequisites

- **Concepts:** integration by parts / the weak form, symmetric vs
  non-symmetric linear systems, and Chapter 00's environment checks.
- **Chapters:** Computational Chapter 1 (`01_convergence`, this track's
  convergence study) and Physics Chapter P3 (`Substrate and boundary
  conditions`, the physical wall energy this chapter names but does not
  run). Chapter 00 (`00_setup_and_smoke_test`, environment green) is
  always assumed.

## Expected cost

- **Device:** any CUDA GPU, 2-D mesh, well under 1 GB device memory for
  the taught runs; an 8 GB card is ample.
- **Solver:** the CH brick's `(c,mu)` saddle uses `splu` (the same
  indefinite-saddle reason as Physics P1); the tiny weak-vs-strong demo
  is a pure-NumPy dense solve, no GPU needed.
- **Quick/run target:** `python run.py` runs both boundary treatments,
  the flux balance, the BC test matrix, and the weak-vs-strong demo in
  **~30-60 s**. That is comparable to physics P1's quick mode, which
  measured **≈ 75 s wall on an RTX 6000 Ada** (mostly Warp kernel
  compile + Python start-up) — expect the same first-run Warp-compile
  overhead here; subsequent runs in the same session are faster.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, with the
chapter-specific content:

1. The `config.resolved.yaml` + `metadata.json` from a reference run
   (or the run's printed provenance if the harness is not used directly).
2. The nine `[PASS]`/`ALL CHECKS: PASS` lines from `run.py` green, or a
   documented deviation.
3. The four figures (`c2_fields`, `c2_mass`, `c2_matrix`, `c2_flux`, plus
   `c2_penalty`) regenerated from your saved run via `gen_figures.py`.
4. **Headline:** the BC test-matrix mass drifts across the boundary
   taxonomy (natural, `c=+0.9`, `c=-0.9`, `c=0.0`, mixed L/R), the flux
   balance `d/dt Int c = -Int J.n` check (net flux ~0 for no-flux, a
   decaying influx for Dirichlet), and the penalty (`beta`) conditioning
   crossover from the weak-vs-strong demo.
5. **Verification:** the flux-balance overlay — measured `dm/dt` against
   the mass-history derivative it must equal by construction — and the
   penalty error's `1/beta` convergence against the strong (row-replace)
   reference error.
6. **Failure:** push the penalty `beta` past `1e6` until the boundary
   error stops improving / the solve degrades, and report where the
   exactness-vs-conditioning trade breaks down.
7. **Exploration:** answer one "Explore on your own" question with
   evidence (the `g=0` mass-drift coincidence and the mixed L/R case are
   both recommended).
8. **Research bridge:** one paragraph on how the boundary taxonomy here
   maps to a real device (a sealed box vs a composition-controlled
   contact vs a wetting substrate), and what changes when you swap in the
   wall-energy BC of Physics P3.
