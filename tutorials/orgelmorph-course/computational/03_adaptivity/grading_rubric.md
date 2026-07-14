# Grading rubric — C3 Octree refinement and temporal adaptivity

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "C3 specifics"
column says what to look for.*

| Component | Weight | C3 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Clearly states the octree refinement is to a *static* geometric criterion, **not** solution-adaptive AMR, and names the additional steps (estimate/mark/refine-coarsen/2:1 balance/rebuild/transfer state+history/continue) dynamic AMR needs; correctly explains why conservative (averaging) transfer is exact and naive (injection) transfer is not; does **not** call the adaptive time-stepping saving "huge" — it is real but modest. |
| **Numerical verification** | 20% | Conservative-transfer radius sweep reproduced (injection error → 0 as feature resolves, averaging exact throughout); BDF2 order comparison reproduced (variable-coefficient ≈2.01 vs forced constant-coefficient ≈0.93 on the same alternating-`dt` sequence); observed vs expected stated explicitly for both. |
| **Reproducibility** | 15% | `run.py`'s eight `[PASS]` checks green (`ALL CHECKS: PASS`) or a documented deviation; all four figures (`c3_octree`, `c3_transfer`, `c3_ladder`, `c3_cost`) regenerated from the student's own run via `gen_figures.py`; node counts reproduce exactly (deterministic), step counts within tolerance (seed/card-dependent). |
| **Software & CUDA** | 10% | `splu` used (not raw-CH cuDSS on the indefinite saddle); hanging-node constraint operator `T` wired into the `DeviceMesh` so the CH brick steps unchanged on the adaptive mesh (`step_ok = True`); sensible octree level / LTE tolerance choices. |
| **Failure diagnosis** | 10% | The forced constant-coefficient BDF2 (`_const_march`, `r≡1`) is run and its collapsed order (~0.93, not ~2) is correctly attributed to the coefficient bookkeeping, not to the mesh or the solver; or an under-resolved octree refinement predicate is diagnosed via degraded node savings / `step_ok`. |
| **Exploration & research bridge** | 10% | One "Explore on your own" question from `c3.tex` answered with a plot (the transfer-error radius sweep or the tightened-LTE-tolerance cost comparison are the recommended ones); a paragraph connecting the conservative-transfer result and the measured cost saving to a real dynamic-AMR implementation. |
| **Communication** | 5% | Labeled axes/units; the ≈1.85×/≈1.79× adaptivity saving reported as *modest* with the horizon it was measured at, never as the disavowed fictional horizon/min-dt ratio; BDF2 orders quoted with the `dt` sequence they were measured on. |

## Automatic zero-credit triggers

- Calling the static octree refinement in §1 "solution-adaptive AMR" or
  "dynamic AMR" without the honesty caveat.
- Quoting the old "horizon / smallest-step" ratio (or any hand-computed
  analogue of it) as the adaptivity speed-up instead of the measured
  Newton/wall savings from the matched-accuracy sweep.
- Reporting the BDF2 constant-coefficient order as "broken" or "0" rather
  than the measured ≈0.93 (collapsed toward 1, not toward 0).
- Hand-copied numbers that do not match the submitted run's printed
  `[PASS]`/`[FAIL]` output or `EXPECTED.md`.

## Partial-credit guidance

- Correct node-savings number but no mention that this is *static*
  refinement, not solution-adaptive AMR → cap Scientific correctness at
  half.
- Injection-error number reported but no radius sweep showing it shrinks
  toward 0 as the feature resolves → half of the Numerical verification
  weight.
- Real cost accounting reported (accepted/rejected, solves, Newton,
  wall) but no matched-accuracy fixed-`dt` comparison → half credit;
  a bare speed-up number with no accounting behind it → no credit for
  this component.
- Both BDF2 orders measured correctly but no explanation of *why* a
  growing `dt` forces the variable-coefficient form → Numerical
  verification credit only, no Scientific correctness credit for that
  half of the headline.
