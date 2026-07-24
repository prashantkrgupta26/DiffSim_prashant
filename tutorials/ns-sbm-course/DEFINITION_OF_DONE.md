# Definition of done

*The completion checklist a chapter must satisfy before it ships, and the
same checklist a student submission is measured against (see
`ASSESSMENT.md` for weights). A chapter is "done" only when every box in
all five categories is checked or its absence is explicitly justified.*

A chapter turns a demonstration into a **repeatable scientific workflow**.
It is done when a new grad student can, from the chapter alone, derive the
weak form, run both engines reproducibly, verify the split against the
oracle, read the right diagnostics, interpret the drag, perturb a knob,
and reproduce it.

---

## 1. Scientific specification

- [ ] The model, its assumptions, and its regime of validity are stated
      (incompressible NS, equal-order $P_1/P_1$ VMS, the Reynolds number,
      the confinement caveat where the domain is a unit box).
- [ ] Every parameter is given with **units / meaning**, a **range**, and
      a **source** ($\nu = UD/\mathrm{Re}$; the Nitsche penalty $\alpha$;
      the SBM offset; the drag reference $q_{\text{ref}}$), with the
      dimensional / nondimensional values kept separate and labeled.
- [ ] The equations in the chapter match the **actual production code
      branch** used (same $\tau_M/\tau_C$ form, same Nitsche sign
      convention, same `consistent_projection` sub-knobs, or the
      simplification is named).
- [ ] The headline claim is a falsifiable, measurable statement — not
      "the flow looks right" but "the projection $C_d$ matches the
      same-mesh monolithic $C_d$ to X%".

## 2. Numerical verification

- [ ] An independent verification is present and passes: **faithfulness**
      of the split to the same-mesh monolithic oracle (the course's
      central check), or a benchmark comparison (Ghia centerline), or the
      PPE-space solenoidality identity.
- [ ] Error sources are **separated** where relevant (coarse-mesh
      discretization vs split-vs-monolithic vs solver tolerance); the
      reported difference is the one being studied.
- [ ] The comparison is reported as a **relative difference / interval**,
      not a single asserted number; the anti-vacuity check (e.g.
      zero-shift, $\alpha=0$) is used where it applies.
- [ ] Tolerances in `baseline.yaml` are **scientific invariants**, not
      bitwise GPU identity, and are documented (measured-then-locked with
      headroom).

## 3. Software & CUDA

- [ ] The correct solver / precision / device is used and justified (e.g.
      **not** cuDSS-without-pivoting on a system that needs it; `splu` for
      the small 2-D meshes; cuDSS/AMGX for scale; the SPD PPE is the
      AMG-friendly one, the indefinite saddle is not).
- [ ] The run ends with an honest `exit_reason`; no silently swallowed
      failures. A known instability (e.g. the base "strong" split on an
      open outflow, or the 3-D projection long-time drag transient) is
      **reported**, not hidden.
- [ ] The chapter uses the **production bricks**
      (`diffsim.steppers.*`, `diffsim.sbm.*`, the validated `tests/`
      fixtures) — no toy re-implementations.
- [ ] A deliberate-failure / common-error path is documented (what breaks,
      the symptom, the fix).

## 4. Reproducibility

- [ ] Output follows the standard schema: `config.resolved.yaml`,
      `metadata.json`, `results.json`, `history.npz`, `run.log`,
      `figures/`.
- [ ] Every figure and every quoted number regenerates from saved data
      via `gen_figures.py` — the document's numbers *are* the run's
      output.
- [ ] Provenance records diffsim commit + git-dirty, GPU + memory, CUDA
      driver/runtime, solver, precision, seeds, wall time, peak device
      memory.
- [ ] The invariants reproduce on a second run / another card within
      tolerance; the *drag / centerline invariants* are what reproduce
      (the transient flow field is not claimed bit-reproducible).

## 5. Pedagogy

- [ ] The mandatory template is complete: why-it-matters, **learning
      objectives**, **prerequisites**, **expected cost**, model +
      assumptions, weak-form / numerical formulation, where-in-diffsim,
      baseline, verification, failure modes, guided exercises, research
      bridge, **required deliverable**, references.
- [ ] Instructor materials exist: `INSTRUCTOR.md`, `grading_rubric.md`,
      and `solutions/` (hints for all questions, full solutions for a
      couple of verification exercises, typical incorrect conclusions,
      expected runtime ranges, common CUDA/solver errors).
- [ ] Exploratory questions cross-reference earlier/later chapters (the
      two engines, the $d=0 \to d\neq0$ progression, the 2-D → 3-D step).
- [ ] The required student deliverable is stated exactly (which
      table/plot/code-change/interpretation).

---

**One-line test.** *Could a new student, given only this chapter, produce
the eight-item deliverable of `ASSESSMENT.md` and have it pass the
tolerance gate?* If yes, the chapter is done.
