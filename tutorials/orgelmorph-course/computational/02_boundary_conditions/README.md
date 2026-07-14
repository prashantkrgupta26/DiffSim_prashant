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
