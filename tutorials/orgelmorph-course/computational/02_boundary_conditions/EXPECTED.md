# C2 — expected results (self-check)

Running `python run.py` (fixed seed) should reproduce the following. The
field values depend on the RNG seed fixed in `bc.py`; the *qualitative*
contrast (no-flux conserves, Dirichlet does not and pins the wall) must
hold exactly.

| quantity | natural (no-flux) | Dirichlet (wall $c=0.9$) |
|---|---|---|
| mass drift $\lvert\Delta m\rvert$ | 1.3×10⁻¹⁶ (machine) | 0.95 (material drawn in) |
| edge composition | −0.13 (free) | +0.90 (pinned exactly) |
| field range $[c_{\min}, c_{\max}]$ | [−1.01, +1.02] | [+0.68, +1.00] |
| field difference max$\lvert c_{\text{nf}}-c_{\text{dir}}\rvert$ | — | 2.00 |

**What must be true regardless of hardware:**

- **No-flux conserves mass to solver tolerance** (drift ≈ machine
  epsilon). The natural boundary adds no boundary integral to the weak
  form; nothing crosses the wall. This is the same conservation you saw
  in Physics P1.
- **Dirichlet does *not* conserve mass** — and that is correct, not a
  bug. Pinning the boundary makes it a reservoir; here it pulls the
  $c=+0.9$ phase in, so total mass *rises* by ≈ 0.95. A boundary
  condition that fixes a value must let flux adjust.
- **The Dirichlet edge sits exactly at the wall value** (+0.90 to
  rounding): the row-replacement imposes it strongly, node by node.
- **The two fields differ substantially** (max difference ≈ 2, the full
  order-parameter range): the wall reorganizes the morphology near it,
  drawing one phase to the boundary. The boundary condition is not a
  cosmetic detail — it changes the answer.
- **The `ALL CHECKS: PASS` line prints** (no-flux drift < 1e-10;
  Dirichlet drift > 1e-2; edge pinned to wall; fields differ > 0.5).

If no-flux drifts, or Dirichlet's edge is not pinned, the boundary
machinery is being applied wrong — re-read the walkthrough.
