# Physics P3 — Substrate / surface energy and boundary conditions

Real films are cast on a **substrate** and dry against **air**; those
interfaces are not neutral. This concept adds a substrate **wall free
energy** to binary Cahn–Hilliard (the `A2` term in the multiphase
brick) and lays out the boundary-condition taxonomy
(natural/no-flux · wall-energy · Dirichlet). You see the wall enrich (or
deplete) one component and compare the resulting stratified morphology
against the neutral no-flux case.

**Read** the course document, Chapter *"Substrate and boundary
conditions"* (start with `substrate.py`).

**Run:**
```bash
python run.py                 # no-flux vs attracting vs repelling wall
python run.py --level 6
```

`run.py` prints the substrate-enrichment self-check; compare with
[`EXPECTED.md`](EXPECTED.md). Regenerate figures/numbers with
`python gen_figures.py`.

| file | role |
|------|------|
| `substrate.py` | the core: mesh with a substrate + `run` with wall energy |
| `run.py` | driver; prints the enrichment table |
| `gen_figures.py` | regenerates `p3_*.png` + `numbers/p3.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=1, K=0) — binary
Cahn–Hilliard plus the real substrate wall-energy interface. The
Dirichlet path is covered in the Computational track's BC concept.
