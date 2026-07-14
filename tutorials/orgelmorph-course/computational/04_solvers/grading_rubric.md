# Grading rubric — Computational C4 The solver ecosystem

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "C4 specifics"
column says what to look for.*

| Component | Weight | C4 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly explains the `(c,μ)` saddle is *indefinite* once `f″=3c²−1<0` in the spinodal band; states the exact divergence numbers (`poly+cudss` → `[−552, 542]`, ~500× blow-up; `poly+splu` → `[−1.03, 1.01]`); does **not** claim cuDSS is generally safe just because it survives Flory–Huggins (`[0.001, 0.999]`, energy-specific). |
| **Numerical verification** | 20% | Reports the one-iterate residuals (`splu` ~1e-12, cuDSS ~5e-13, `blockch` ~4e-10) *and* the marched solution together, explicitly stating the tiny cuDSS residual does not predict the diverging march; does **not** use the one-iterate residual alone as the verification. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all three figures (`c4_divergence`, `c4_crossover`, `c4_cited3d`) and `numbers/c4.tex` regenerated from `gen_figures.py`; the harness's own PASS/FAIL block is included and green (or a documented deviation); the 3-D table is clearly marked cited, not re-run. |
| **Software & CUDA** | 10% | `splu` (or `blockch`/`blockch_dev`) used for the actual CH solve, never raw-CH cuDSS; capture done via `capture_system=True` (never a `solve_linear` monkeypatch); clean run with sane verdicts; sensible level/mode choice. |
| **Failure diagnosis** | 10% | The polynomial + cuDSS march is shown diverging (`diverged=True`, field escaping the physical band) and the mechanism explained (no partial pivoting on an indefinite block, error compounding over Newton/time iterations) — not dismissed as a random crash. |
| **Exploration & research bridge** | 10% | One "Explore on your own" question answered with evidence (a sweep or plot, e.g. `max|c|` vs step count for cuDSS); `recommend_solver`'s rationale reported for at least two regimes and connected to a real research-scale choice. |
| **Communication** | 5% | Labeled axes/units on the figures; the divergence table and residual table both reported with their exact numbers (not paraphrased as "big" or "small"); honest about which numbers are live (2-D) vs. cited (3-D). |

## Automatic zero-credit triggers (flag, don't fail silently)

- **Claims cuDSS is safe on the raw CH block** (whether phrased as "cuDSS
  is fine here" or by simply recommending it for the polynomial saddle
  without qualification).
- **Reports the one-iterate residual as evidence of correctness** (e.g.
  "cuDSS's residual was 5e-13, so the solve was correct") without also
  reporting that the marched solution diverges.
- Presents the cited 3-D table (`CITED_3D`/`CITED_FACTS`) as a number this
  run measured, rather than citing `docs/dev/2026-07-13-m5-device-assembly.md`
  / `docs/dev/2026-07-13-blockch-mpf.md`.
- Hand-copied divergence or residual numbers that do not match the
  submitted run's own output / `results.json`.
- Tolerances or the PASS/FAIL self-check loosened to make the gate pass,
  without a documented justification.

## Partial-credit guidance

- Correct divergence table but no mechanism (why cuDSS fails on an
  indefinite, unpivoted saddle) → cap Scientific correctness at half.
- Residual table reported but not paired with the marched-solution
  divergence, or vice versa → half of the Numerical verification weight
  (one without the other is exactly the trap this chapter is built to
  catch).
- Correct headline numbers but the cited 3-D table presented without
  attribution to the dev notes → half of Reproducibility credit.
- A crossover/timing figure with no mention that cuDSS is disqualified
  for CH by correctness → Communication credit only; the science credit
  is in stating the disqualification, not the speed number.
