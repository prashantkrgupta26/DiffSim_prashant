# Student assessment model

*The single deliverable and rubric every chapter is graded against. Each
chapter's `grading_rubric.md` is this table, specialised to that
chapter's headline result; this file is the contract they all inherit.*

The course teaches a **repeatable scientific workflow** — derive →
implement → verify → interpret → perturb → reproduce — so the assessment
grades the *workflow*, not a single right answer. The central verification
is **faithfulness**: does the scalable pressure-projection split reproduce
the same-mesh monolithic oracle? A transient flow field is not
bit-reproducible across meshes/GPUs; what is graded is whether the student
produced a reproducible, verified, honestly-interpreted result.

---

## The standard deliverable (every chapter)

A student's submission for any chapter is a short report (a few pages or a
notebook) plus the run artifacts. It contains exactly these eight items:

1. **Resolved run record** — the `config.resolved.yaml` and
   `metadata.json` (diffsim commit + git-dirty flag, GPU, CUDA
   driver/runtime, solver, precision, seeds, wall time, peak device
   memory) from an actual run, not a hand-edited config.
2. **Passing tolerance checks** — the harness `check_results` gate green
   against `baseline.yaml`, *or* a documented, physically-argued
   explanation of any deviation (never a silently loosened tolerance).
3. **Figures / table generated from saved data** — every plot or the
   self-check table rendered from the run's `results.json` /
   `history.npz`, never hand-copied numbers.
4. **The headline result, with its comparison** — the chapter's central
   measurement (the Ghia centerline agreement, the drag $C_d$, the
   projection-vs-monolithic relative difference) reported *with* the
   relative difference and the definition/window it was measured on.
5. **A verification** — the faithfulness check: projection vs same-mesh
   monolithic (drag and/or centerline), *and* where it applies the
   PPE-space solenoidality identity or the anti-vacuity break.
6. **A failure diagnosis** — deliberately break one thing (wrong outflow
   BC / drop `consistent_projection` / zero the SBM shift / set
   $\alpha=0$) and *diagnose* it from the diagnostics: which number went
   wrong, and why.
7. **A perturbation / exploration** — answer at least one "Explore on your
   own" question with evidence (a sweep, a plot, a fitted trend), not
   prose alone.
8. **A research bridge** — one paragraph connecting the chapter's result
   to a real research question (which engine to use at scale, what the
   next chapter or the 100M-DOF path adds).

---

## The standard rubric (weights)

| Component | Weight | What earns full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | The headline result is right, reported with its relative difference and measurement definition; conclusions follow from the data (e.g. does *not* claim the pointwise $\|\nabla\!\cdot u\|$ must be zero for the equal-order VMS scheme; understands the confinement caveat on absolute drag). |
| **Numerical verification** | 20% | The faithfulness check passes: projection matches the same-mesh monolithic within the stated band; where relevant the PPE solenoidality identity is machine-zero and the anti-vacuity break is shown. Observed vs expected stated explicitly. |
| **Reproducibility** | 15% | Resolved config + provenance present; figures/table regenerate from saved data; a second run reproduces the *invariants* within tolerance. No bit-identity claims across GPUs. |
| **Software & CUDA fluency** | 10% | Correct solver / precision / device for the problem (splu for small 2-D; cuDSS/AMGX for scale; the SPD PPE is the AMG target, not the saddle); a clean run with a sane `exit_reason`. |
| **Failure diagnosis** | 10% | A deliberate failure is induced and correctly diagnosed from the diagnostics, with the mechanism explained. |
| **Exploration & research bridge** | 10% | An exploratory question answered with evidence, and a credible connection to a real research question. |
| **Communication** | 5% | Labeled axes/units, honest relative differences, a readable report; claims are qualified where the comparison is coarse-mesh-limited. |

**Grading philosophy.** A wrong headline number honestly reported *with a
correct verification that catches it* scores higher than a right number
with no verification. Bit-reproducibility across GPUs is never required —
scientific invariants (drag, centerline, faithfulness) within documented
tolerances are the standard.

See `DEFINITION_OF_DONE.md` for the instructor-side completion checklist a
chapter (and a submission) must satisfy.
