# Computational C2 — Boundary conditions, numerically

How the two boundary treatments the Cahn–Hilliard brick supports are
imposed, and what each costs and means — measured on the same blend:

- **Natural (no-flux)** — the default. The weak form drops its boundary
  integral when $\nabla\mu\cdot n = 0$, so it costs nothing to impose and
  conserves mass exactly (a sealed box).
- **Dirichlet (fixed value)** — imposed by replacing the boundary rows
  with the identity. It pins $(c,\mu)$ at the wall and *deliberately*
  breaks conservation: the wall becomes a reservoir. This is the
  treatment the manufactured-solution study (C1) needed.

A third kind — a **wall free energy** (a surface term making the natural
flux nonzero, i.e. a wetting/Robin condition) — is real physics for thin
films but lives in the multiphase film brick, not this binary CH brick.
The course text points to where it is implemented
(`wodo_film` / the multiphase A2 wall term); see
`docs/dev/2026-07-13-m5-device-assembly.md`.

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
