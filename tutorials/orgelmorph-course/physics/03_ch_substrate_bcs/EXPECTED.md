# P3 — expected results (self-check)

Running `python run.py` (defaults: 64×64, χ = 2.2, quench to *t* = 0.4,
fixed seed) should reproduce the following. The wall free energy is the
bounded quadratic well *f*_w = *g* φ + *h* φ², whose minimum sets the
preferred surface composition φ* = −*g*/(2*h*).

| case | (*g*, *h*) | φ* | substrate φ | enrichment Δφ |
|---|---|---|---|---|
| no-flux (neutral) | (0, 0) | — | 0.493 | −0.007 |
| attracting wall | (−1.5, 1.0) | 0.75 | 0.750 | +0.248 |
| repelling wall | (−0.5, 1.0) | 0.25 | 0.250 | −0.249 |

Mass drift ≤ 2×10⁻³ in all cases.

**What must be true regardless of hardware:**

- **The neutral wall does not stratify** (enrichment ≈ 0): with no wall
  energy the boundary is the ordinary no-flux condition.
- **The attracting wall enriches the substrate** (substrate φ rises to
  ≈ φ* = 0.75, positive enrichment); the **repelling wall depletes it**
  (substrate φ falls to ≈ 0.25, negative enrichment). The substrate
  composition tracks the preferred φ* = −*g*/(2*h*).
- **Mass is (essentially) conserved** — the wall energy is a boundary
  condition on φ, not a mass flux, so it rearranges material without a
  reservoir. The small residual drift (≤ 2×10⁻³) comes from the physical
  projection clip near the wall, not from a leak.
- **φ* must lie inside (0, 1).** If you push the preference outside the
  physical range, φ drives into the Flory–Huggins wall, the projection
  clips (injecting mass), and the time step collapses — a stiffness you
  can trigger deliberately in the exercises.

Third-significant-figure differences from a different card are normal;
the qualitative story (attract enriches, repel depletes, neutral flat,
mass conserved) must hold.
