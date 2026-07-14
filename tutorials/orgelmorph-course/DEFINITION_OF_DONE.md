# Definition of done

*The completion checklist a chapter must satisfy before it ships, and the
same checklist a student submission is measured against (see
`ASSESSMENT.md` for weights). A chapter is "done" only when every box in
all five categories is checked or its absence is explicitly justified.*

A chapter turns a demonstration into a **repeatable scientific workflow**.
It is done when a new grad student can, from the chapter alone, derive the
model, run it reproducibly, verify it, profile it, interpret it, perturb
it, and reproduce it.

---

## 1. Scientific specification

- [ ] The model, its assumptions, and its regime of validity are stated.
- [ ] Every coefficient is given with **units**, a **physical range**, and
      a **source** (`materials/materials.yaml` provenance where real),
      with dimensional / nondimensional / numerical / *accelerated*
      (pedagogical) values kept separate and labeled.
- [ ] The equations in the chapter match the **actual production code
      branch** used (same free energy, same BCs, same mobility model, or
      the simplification is named).
- [ ] The headline claim is a falsifiable, measurable statement — not
      "the energy went down" but "the largest positive stepwise increment
      is X".

## 2. Numerical verification

- [ ] An independent verification is present and passes: convergence
      order vs predicted rate, conservation drift vs solver tolerance, a
      tight-tolerance reference comparison, or prediction-vs-measurement.
- [ ] Error sources are **separated** (spatial vs temporal vs algebraic)
      where relevant; the reported error is the one being studied.
- [ ] Uncertainty is reported: an interval, a standard deviation, or an
      ensemble — never a single asserted slope/threshold.
- [ ] Tolerances in `baseline.yaml` are **scientific invariants**, not
      bitwise GPU identity, and are documented.

## 3. Software & CUDA

- [ ] The correct solver / precision / device is used and justified (e.g.
      not raw-CH cuDSS on the indefinite saddle; documented fallback for
      small runs without cuDSS).
- [ ] The run ends with an honest `exit_reason`; no silently swallowed
      failures, no asserts standing in for typed exceptions.
- [ ] The chapter uses the **production bricks** (`diffsim.physics.*`,
      `diffsim.diagnostics.*`) — no toy re-implementations — or the
      tutorial-local exception (e.g. a forced-`r=1` BDF2 for a comparison)
      is explicit and reproducible from the folder.
- [ ] A deliberate-failure / common-error path is documented (what breaks,
      the symptom, the fix).

## 4. Reproducibility

- [ ] Output follows the standard schema: `config.resolved.yaml`,
      `metadata.json`, `results.json`, `history.npz`, `run.log`,
      `figures/`.
- [ ] Every figure and every quoted number regenerates from saved data
      via `gen_figures.py` — the document's numbers *are* the run's
      output (staleness-gated by `doc_numbers.yaml`).
- [ ] Provenance records diffsim commit + git-dirty, GPU + memory, CUDA
      driver/runtime, solver, precision, seeds, wall time, peak device
      memory.
- [ ] The invariants reproduce on a second run / another card within
      tolerance; morphology is *not* claimed bit-reproducible.

## 5. Pedagogy

- [ ] The mandatory template is complete: why-it-matters, **learning
      objectives**, **prerequisites**, **expected cost**, model +
      assumptions, numerical formulation, where-in-diffsim, baseline,
      verification, failure modes, guided exercises, research bridge,
      **required deliverable**, references.
- [ ] Instructor materials exist: `INSTRUCTOR.md`, `grading_rubric.md`,
      and `solutions/` (hints for all questions, full solutions for a
      couple of verification exercises, typical incorrect conclusions,
      expected runtime ranges, common CUDA/solver errors).
- [ ] Exploratory questions cross-reference earlier/later chapters and
      the other tracks.
- [ ] The required student deliverable is stated exactly (which
      plots/table/code-change/interpretation).

---

**One-line test.** *Could a new student, given only this chapter, produce
the eight-item deliverable of `ASSESSMENT.md` and have it pass the
tolerance gate?* If yes, the chapter is done.
