# P4 — expected results (self-check)

The full workflow is the harness run

```bash
PYTHONPATH=<repo>/src python run_harness.py \
    --config configs/p4.yaml --mode reference --output outputs/p4 --overwrite
```

(blend `synthetic_demix` from `materials/ternary_p4.yaml`: χ = (3.5, 1.0,
0.6), N = (1, 1, 1), initial blend (φ₁, φ₂) = (0.35, 0.35), 65×65, quench
to *t* = 0.6, seed 6). It reproduces the following, and `baseline.yaml`
gates them automatically.

| quantity | value |
|---|---|
| spinodal det *H* @ IC | −6.475 (< 0 ⇒ unstable) |
| critical χ₁₂\* @ IC | 2.840 |
| coexisting phase A (φ₁, φ₂) / population | (0.162, 0.523) / 0.49 |
| coexisting phase B (φ₁, φ₂) / population | (0.529, 0.186) / 0.51 |
| clustering sensitivity (GMM vs. interface-excluded) | 0.130 |
| predicted binodal A / B | (0.122, 0.546) / (0.653, 0.089) |
| composition spread (std φ₁): start → end | 0.0201 → 0.208 |
| simplex residual max\|φ₁+φ₂+φₛ−1\| | 1.1×10⁻¹⁶ |
| per-solute quadrature content drift \|ΔC₁\|, \|ΔC₂\| | 0, 5.6×10⁻¹⁷ |
| lever-rule residual | 6.1×10⁻⁴ |
| N-shift Δχ₁₂\* (N₁: 1 → 2) | −0.860 (2.840 → 1.979; det → −13.605) |
| asymmetric-N phases A / B | (0.096, 0.902) / (0.954, 0.045) |

**What must be true regardless of hardware:**

- **The quench is spinodal-unstable.** The 2×2 exchange Hessian *H* of the
  Flory–Huggins free energy (`spinodal_hessian`) has det *H* < 0 at the
  initial blend. Below the critical χ₁₂\* = 2.84 the noise decays instead
  (a good χ-sweep experiment).
- **Two distinct phases form**, read off by a **2-component Gaussian
  mixture** on the composition cloud (not a median split): one φ₁-poor/
  φ₂-rich, one the mirror, with well-separated means and ~50/50 populations.
  The interface-excluded refit shifts the endpoints by only ≈0.13, so the
  readout is robust.
- **The simulated endpoints sit inside the predicted binodal.** The
  common-tangent tie-line (through the conserved mean) predicts more
  extreme endpoints than a short quench has reached — the theory-vs-sim
  comparison in `p4_landscape.png`.
- **Admissibility + conservation hold.** The Gibbs simplex is satisfied to
  machine precision (exact solvent elimination); each solute's *quadrature*
  content is conserved to round-off; the lever rule closes (residual
  ≈ 6×10⁻⁴).
- **Unequal N enlarge the spinodal.** Raising N₁ from 1 to 2 drops the
  critical χ₁₂\* from 2.84 to 1.98 and deepens det *H*; the asymmetric blend
  demixes almost completely (endpoints near the triangle edges).

Third-significant-figure differences from a different card are normal; the
analytic quantities (det, χ₁₂\*, the N-shift) are hardware-independent.
