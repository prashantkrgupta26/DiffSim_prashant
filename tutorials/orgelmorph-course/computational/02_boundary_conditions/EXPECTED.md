# C2 — expected results (self-check)

`python run.py` (fixed seed) should reproduce the following. Field values
depend on the RNG seed fixed in `bc.py`; the *qualitative* contrasts
(no-flux conserves, Dirichlet does not and pins the wall; row-replace
breaks symmetry; penalty converges) must hold exactly. Nine `[PASS]`
lines and `ALL CHECKS: PASS` must print.

## 1. Natural vs Dirichlet (33×33, 60 steps)

| quantity | natural (no-flux) | Dirichlet (wall c=0.9) |
|---|---|---|
| mass drift \|Δm\| | 1.3×10⁻¹⁶ (machine) | 0.95 (material drawn in) |
| edge composition | −0.13 (free) | +0.90 (pinned exactly) |
| field difference max\|c_nf−c_dir\| | — | 2.00 (full range) |

## 2. Flux balance  d/dt ∫c = −∮J·n

| mode | net flux dm/dt |
|---|---|
| no-flux | \|flux\|max = 5.5×10⁻¹⁵ (zero → mass conserved) |
| Dirichlet | +4.15 early → +0.54 late (reservoir shuts off) |

The mixed system has **two** BCs (∇μ·n=0 for mass flux, κ∇c·n for
wetting); natural imposes both by omitting both surface integrals.

## 3. BC test matrix (one blend, several boundaries)

| configuration | #pinned | mass drift | edge |
|---|---|---|---|
| natural (no-flux) | 128 | 1.3e-16 | −0.128 |
| prescribed c=+0.9 | 128 | 0.954 | +0.900 |
| prescribed c=−0.9 | 128 | 0.943 | −0.900 |
| prescribed c=0.0 | 128 | 0.094 | +0.000 |
| mixed: c=+0.9 L/R, no-flux T/B | 66 | 0.290 | +0.628 |

`c=0.0` nearly conserves only because the IC mean is ≈0 — a coincidence of
the initial condition, not the boundary.

## 4. Weak vs strong Dirichlet (tiny −u″=0 system)

- Row replacement: matrix **asymmetric**, solution exact (0.0). Strong but
  breaks SPD.
- Symmetric elimination: matrix **symmetric**, solution exact. Strong,
  keeps SPD.
- Penalty (weakly imposed): boundary error ∝ 1/β — 6.7e-1, 1.7e-1, …,
  2.0e-6 as β = 1e0 … 1e6. Trades exactness for a symmetric matrix and no
  destroyed row (useful on hanging-node / immersed boundaries).

**What must be true regardless of hardware:**

- **No-flux conserves mass to machine precision** and its net boundary
  flux is zero — the natural BC omits both surface integrals.
- **Dirichlet does *not* conserve** (a reservoir), and its influx **decays
  to zero** as the interior reaches the wall value — the flux balance in
  action.
- **The imposition method changes the matrix**: row-replace is asymmetric,
  symmetric-elimination is SPD, penalty is weak but SPD.
- **Manufactured vs physical**: pinning *both* c and μ (as C1 did) is an
  MMS device; a physical composition contact pins only c.
