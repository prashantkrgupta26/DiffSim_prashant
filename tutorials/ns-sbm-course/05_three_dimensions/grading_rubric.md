# Grading rubric — 05 Three dimensions

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter.*

| Component | Weight | 05 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | The honest two-engine verdict stated: monolithic is the working drag path (+0.381); projection composition de-risked but long-time-drag-unstable; "finite ≠ stable"; the confinement caveat on absolute $C_d$; understands why projection is still the 100M path. |
| **Numerical verification** | 20% | Monolithic $C_d$ positive and matching the M1b lock; `--pipeline` invariants (finite, BDF2, axisymmetric) reported. |
| **Reproducibility** | 15% | Resolved config + provenance; verdict card regenerated; fixture invariants ($n_{\text{free}}$, faces, $D/h$) reproduce. |
| **Software & CUDA** | 10% | Correct solver for the scale (`splu` at level 4; `cudss`/`blockamgx` past the wall); clean `exit_reason`. |
| **Failure diagnosis** | 10% | The projection long-time drag transient shown to diverge (the R2 item); "finite" distinguished from "stable". |
| **Exploration & research bridge** | 10% | A GPU level-5 `cudss` run (or a reasoned estimate) with the step-time/DOF change; a paragraph on projection-as-100M-path vs monolithic-as-drag-path. |
| **Communication** | 5% | Fixture invariants and the verdict stated plainly; honest about the open R2 item and confinement. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming the projection composition is a working 3-D drag path (it is not
  yet — long-time transient unstable).
- Reporting the absolute $C_d$ as "wrong" against the unconfined literature
  value without noting confinement.
- Reporting a negative reference $C_d$ as the steady drag (it is the
  transient).
- Hand-copied numbers that do not match `results.json`.

## Partial-credit guidance

- Monolithic $C_d$ correct but the two-engine verdict oversimplified → cap
  Scientific correctness at half.
- `--pipeline` invariants omitted → half of Numerical verification.
