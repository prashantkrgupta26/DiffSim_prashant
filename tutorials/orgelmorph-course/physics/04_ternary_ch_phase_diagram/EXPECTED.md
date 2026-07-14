# P4 — expected results (self-check)

Running `python run.py` (defaults: 64×64, χ = (3.5, 1.0, 0.6), initial
blend (φ₁, φ₂) = (0.35, 0.35), quench to *t* = 0.6, fixed seed) should
reproduce the following.

| quantity | value |
|---|---|
| composition spread (std φ₁): start → end | 0.0201 → 0.2080 |
| coexisting phase A (φ₁, φ₂) | (0.165, 0.519) |
| coexisting phase B (φ₁, φ₂) | (0.535, 0.181) |
| mean drift \|Δφ\| | 5.5×10⁻⁴ |

**What must be true regardless of hardware:**

- **The composition cloud opens.** The spread of φ₁ grows by an order of
  magnitude (≈0.02 → ≈0.2) as the blend demixes — on the Gibbs triangle
  the tight initial blob spreads along a tie-line.
- **Two distinct phases form.** The tie-line endpoints are well
  separated: one phase is φ₁-poor/φ₂-rich, the other φ₁-rich/φ₂-poor
  (a solvent-mediated separation of the two solutes).
- **The mean is conserved** (drift ≈ 5×10⁻⁴): conservation pins the
  *centroid* of the cloud while it opens.
- **Demixing requires an unstable quench.** The blend must sit inside the
  ternary spinodal — the 2×2 exchange Hessian must have negative
  determinant. At (0.35, 0.35) with χ₁₂ = 3.5 it does; lower χ₁₂ (or a
  more dilute blend) is stable and the noise simply decays (spread → 0).
  This is a good thing to verify by experiment (χ sweep).

Third-significant-figure differences from a different card are normal;
the qualitative story (cloud opens along a tie-line, mean conserved)
must hold.
