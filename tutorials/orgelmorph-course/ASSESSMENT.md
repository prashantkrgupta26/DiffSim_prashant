# Student assessment model

*The single deliverable and rubric every chapter is graded against. Each
chapter's `grading_rubric.md` is this table, specialised to that
chapter's headline result; this file is the contract they all inherit.*

The course teaches a **repeatable scientific workflow** — derive →
implement → verify → profile → interpret → perturb → reproduce — so the
assessment grades the *workflow*, not a single right answer. A morphology
is chaotically seed-sensitive; what is graded is whether the student
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
3. **Figures generated from saved data** — every plot rendered by
   `gen_figures.py` from the run's `results.json` / `history.npz`, never
   hand-copied numbers or a screenshot.
4. **The headline result, with uncertainty** — the chapter's central
   measurement (an exponent, a threshold, a coexistence pair, an observed
   order, a timing) reported *with* an interval or standard deviation and
   the window/definition it was measured on.
5. **A verification** — an independent numerical check: an observed
   convergence order vs the predicted rate, a conservation drift vs
   solver tolerance, a comparison to a tight-tolerance reference, or a
   prediction-vs-measurement (e.g. dispersion) overlay.
6. **A failure diagnosis** — deliberately break one thing (wrong BC,
   wrong solver, too-coarse mesh, too-large step) and *diagnose* it from
   the diagnostics: which number went wrong, and why.
7. **A perturbation / exploration** — answer at least one "Explore on your
   own" question with evidence (a sweep, a plot, a fitted trend), not
   prose alone.
8. **A research bridge** — one paragraph connecting the chapter's result
   to a real research question (which knob to turn, what it would predict
   for a real material, what the next chapter or track adds).

---

## The standard rubric (weights)

| Component | Weight | What earns full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | The headline result is right, reported with uncertainty and its measurement window/definition; conclusions follow from the data, not from wishful reading (e.g. does *not* equate lower energy with accuracy). |
| **Numerical verification** | 20% | An independent check passes at the expected rate/tolerance: convergence order, conservation drift, reference comparison, or prediction-vs-measurement. Observed vs expected stated explicitly. |
| **Reproducibility** | 15% | Resolved config + provenance present; figures regenerate from saved data; a second run (or a stated seed) reproduces the *invariants* within tolerance. No bit-identity claims across GPUs. |
| **Software & CUDA fluency** | 10% | Correct solver / precision / device for the problem (e.g. not raw-CH cuDSS on the indefinite saddle); a clean run with a sane `exit_reason`; sensible mesh/mode choice. |
| **Failure diagnosis** | 10% | A deliberate failure is induced and correctly diagnosed from the diagnostics, with the mechanism explained. |
| **Exploration & research bridge** | 10% | An exploratory question answered with evidence, and a credible connection to a real research question. |
| **Communication** | 5% | Labeled axes/units, honest uncertainty, a readable report; claims are qualified where the data is noisy. |

**Grading philosophy.** A wrong headline number honestly reported *with a
correct verification that catches it* scores higher than a right number
with no verification. Bit-reproducibility across GPUs is never required —
scientific invariants within documented tolerances are the standard.

See `DEFINITION_OF_DONE.md` for the instructor-side completion checklist a
chapter (and a submission) must satisfy.
