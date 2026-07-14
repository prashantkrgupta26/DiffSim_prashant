# Physics P0 — The model hierarchy, order parameters, and scales

The foundation of the Physics track. A mostly-**written** chapter that fixes
the vocabulary the rest of the course assumes:

- **conserved vs non-conserved** order parameters (composition φ vs
  crystallinity ψ), and why that distinction picks the dynamics;
- both **Cahn–Hilliard** (Model B) and **Allen–Cahn** (Model A) as
  **gradient flows** of one Ginzburg–Landau free energy F, and why each
  decreases F;
- why CH is **fourth order** and why the solver uses the **mixed (φ, μ)**
  formulation;
- the **model hierarchy** (binary CH → substrate → ternary → moving-frame
  evaporation → Allen–Cahn crystallization → coupled → stochastic nucleation
  → evaporation-conditioned);
- a **full symbol table** and the **four kinds of parameter** kept apart:
  dimensional material data, nondimensional groups, numerical knobs, and
  deliberately **accelerated** tutorial settings;
- **one complete nondimensionalization** (binary CH → the Cahn number
  κ̃ = (λ/L)²), worked through and reproducing the P1 interface-cell counts;
- the **p1 vs r14** crystallization χ conventions (absolute vs increment;
  cite the P7 finding);
- the **three** distinct "energy decreases" claims (thermodynamically
  motivated / discretely energy-stable / one monotone run).

**Read** the course document, Chapter *"The model hierarchy"* (Chapter
`ch:p0`).

## Deliverable

Complete [`deliverable.md`](deliverable.md): a one-page model map + a
completed dimensional/nondimensional/numerical/accelerated parameter table
for one real system from `materials/materials.yaml`.

## Run it (reproduce the nondimensionalization numbers)

```bash
python run.py --config configs/p0.yaml --device cpu \
    --mode reference --output outputs/p0 --overwrite
python gen_figures.py --run-dir outputs/p0
```

`run.py` computes the worked nondimensionalization (Cahn number, interface
widths in cells, minimum mesh level, accelerated-vs-physical crystallization
drive) from the dimensional inputs in `configs/p0.yaml`, with provenance
(`metadata.json`) and a tolerance baseline (`baseline.yaml`). The
interface-cell counts **match the P1 tutorial's measured values** (2.02 poly
/ 1.43 FH at level 6) — a cross-tutorial consistency check.

| file | role |
|------|------|
| `nondim.py` | the nondimensionalization + resolution calculator (read first) |
| `run.py` | harness driver; prints the groups, checks the baseline |
| `gen_figures.py` | renders the hierarchy diagram, the nondim figure, `numbers/p0.tex` |
| `deliverable.md` | the student deliverable template |
| `baseline.yaml` | tolerance baseline (interface cells must match P1) |
