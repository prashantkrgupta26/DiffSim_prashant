# Physics P5 — Solvent evaporation (the drying film)

An organic film is cast from solution and **dries**: solvent leaves the
top surface, the film thins, and the solutes concentrate until they
demix. Evaporation is the **clock** of morphology formation. This
concept adds the moving, evaporating top surface (film mode) and shows
how the drying rate (Biot number) reshapes the final morphology.

**Read** the course document, Chapter *"Evaporation"* (start with
`evaporation.py`).

**Run:**
```bash
python run.py                 # evaporation-rate sweep, 64x64
python run.py --level 6
```

`run.py` prints drying time, final height, and morphology domain scale
for several evaporation rates; compare with [`EXPECTED.md`](EXPECTED.md).
Drying curves + morphology from `python gen_figures.py`.

| file | role |
|------|------|
| `evaporation.py` | the core: film-mode `run` + domain-scale readout |
| `run.py` | driver; prints the rate sweep |
| `gen_figures.py` | regenerates `p5_*.png` + `numbers/p5.tex` |
| `EXPECTED.md` | reference numbers |

Uses `diffsim.physics.multiphase.MultiPhaseStepper` (M=2, K=0) in film
mode (the Landau-mapped moving frame of the wodo_film formulation).
