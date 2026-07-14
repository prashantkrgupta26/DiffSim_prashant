# Physics P1 — Binary Cahn–Hilliard: two free energies

The first concept. A binary blend demixes to lower its Ginzburg–Landau
free energy; you watch that energy decay and split into bulk and
interfacial parts, for two free-energy models (polynomial double-well
and Flory–Huggins).

**Read** the course document, Chapter *"Binary Cahn–Hilliard"* (start
with `spinodal.py` — the importable core it walks through).

**Run:**
```bash
python run.py                 # both free energies, 64x64, ~1-2 min on CPU
python run.py --energy fh     # just Flory-Huggins
python run.py --level 7       # 128x128 (finer; slower on the CPU solver)
```

`run.py` prints an energy-budget table; compare it with
[`EXPECTED.md`](EXPECTED.md). Field/energy arrays are saved to `out/`.

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/ + ../../latex/numbers.tex
```

| file | role |
|------|------|
| `spinodal.py` | the core: `run_spinodal`, `energy_budget` — read this first |
| `run.py` | the driver you run; prints the self-check table |
| `gen_figures.py` | regenerates the figures and `numbers.tex` the document shows |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` directly — no toy solver.
