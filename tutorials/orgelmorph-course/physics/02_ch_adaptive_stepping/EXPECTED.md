# P2 — expected results (self-check)

Running `python run.py` (defaults: 64×64, BDF2, quench to *t* = 0.8,
fixed *dt* = 4×10⁻³, adaptive tol = 2×10⁻³, fixed seed) should reproduce
the following. The Cahn–Hilliard assembly is deterministic (each element
writes its own Jacobian block — no cross-thread atomics), so the step
counts and energies are **bit-reproducible** run to run on the same card.

| quantity | naive fixed | adaptive |
|---|---|---|
| time steps taken | 200 | 97 |
| final free energy *F* | 0.1156 | 0.08047 |
| *dt* range | 4×10⁻³ (held) | 1.0×10⁻⁶ → 0.050 |
| domain scale (cells) | 25.8 | 25.1 |
| phase values [*c*min, *c*max] | — | [−1.00, 1.00] |

- **Step-count saving vs the naive fixed step: 2.1×.**
- A fixed step *safe for the whole quench* (dt ≤ 1×10⁻⁶) would need on
  the order of **800 000** steps — the cost the controller avoids.
- Energy difference |Δ*F*| = 3.5×10⁻² — this is **not** a match: the
  naive fixed step *lags* (higher energy) because it under-resolves the
  quench; the adaptive run reaches a more relaxed state.

**What must be true regardless of hardware:**

- **The adaptive step spans orders of magnitude**: it collapses to
  ≈10⁻⁶ through the spinodal quench (largest truncation error) and grows
  to ≈5×10⁻² through coarsening. This shrinking-then-growing profile is
  the whole point.
- **The naive fixed step is both more expensive and less accurate.** If
  your fixed run took fewer steps than the adaptive run, or reached a
  *lower* energy, re-check the parameters.
- **Both energy histories decrease monotonically** (the Lyapunov
  property of Chapter 1 survives any consistent integrator).
- **The robust statistics agree** (domain scale ≈25 cells, phase values
  ±1) even though the pixelwise morphologies differ — the coarsened
  pattern is step-sequence dependent, exactly as it is seed dependent in
  Chapter 1.

Small differences from a different card or BLAS are normal in the third
significant figure; the qualitative story (dt spans ~10⁴, adaptive
cheaper *and* more accurate) must hold.
