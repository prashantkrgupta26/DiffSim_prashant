# Course v2 — Phase 0 shared foundation (build notes + decisions)

Implements the "Phase 0 — shared foundation" section of
`2026-07-14-course-v2-evaluation.md`. Everything downstream (the P0 scientific
corrections in Phase 1, the pedagogical template in Phase 2) builds on this.

## What landed

- `src/diffsim/diagnostics/` — a Warp-free (numpy-only) diagnostic library
  shared by the tutorials and the research code: `conservation`, `energy`,
  `morphology`, `admissibility`, `convergence`, `stochastic`, `profiling`,
  `provenance`. 22 analytic unit tests (`tests/test_diagnostics.py`).
- `tutorials/orgelmorph-course/common/` — `config.py` (YAML load + schema
  validate + resolved-config record), `provenance.py` (metadata.json context
  manager, thin over the diagnostics provenance), `check_results.py`
  (tolerance/range baseline checker, nonzero exit), `run_base.py` (the
  standard CLI + output layout + `run_tutorial` glue), `check_doc_staleness.py`
  (doc-vs-data gate). 13 Warp-free tests (`tests/test_course_common.py`).
- `00_setup_and_smoke_test/` — `doctor.py` (full env probe, live Warp kernel +
  8x8 assemble/solve, ends in exactly one READY/NOT READY line), `run.py`
  (harness smoke), configs, `baseline.yaml`, `EXPECTED.md`, `README.md`.
- P1 ported to the harness as the reference example + regression
  (`physics/01_ch_binary_energies/run_harness.py` + `configs/p1.yaml` +
  `baseline.yaml` + `doc_numbers.yaml`).
- `Makefile` (`course-doctor`, `tutorials-quick`, `tutorials-reference`,
  `course-figures`, `course-pdf`, `course-tests`).
- CI: `tutorial-smoke.yml` + `docs-build.yml` (extend, don't replace,
  `ci.yml`). Same token-gate (`ENABLE_GPU_CI`) as the existing gpu-smoke.

## Decisions beyond the spec (sensible defaults)

1. **`run_harness.py` alongside `run.py`, not replacing it.** The task says do
   NOT rewrite the existing per-chapter `run.py` files (a later phase) but DO
   port one chapter as the reference. So P1 gets a *new* harness entry point;
   the original `run.py` is untouched and still reproduces its numbers. The
   only edit to an existing tutorial file is an **additive** `linsolver="splu"`
   passthrough on `spinodal.run_spinodal` (default preserves the numbers) so
   the harness's resolved solver flows through.

2. **Provenance single source of truth in the installed package.**
   `diffsim.diagnostics.provenance` holds the real collector (the research code
   can use it); `common/provenance.py` thins over it and adds the tutorial
   context manager. Avoids `src/` importing `tutorials/`.

3. **`device`/`solver` are run-level arguments, not config fields.** The
   harness records them in `metadata.json` and exposes them via `ctx`, but does
   NOT inject them into the config, so a chapter's schema stays about physics.
   A chapter that genuinely wants `device` in its config declares the field.

4. **Baseline gated to `--mode reference`.** `baseline.yaml` / `EXPECTED.md`
   encode reference-mode numbers; quick/research use different meshes and
   horizons, so the numeric gate only fires in reference mode. Quick mode is
   the fast CI smoke (no tight numeric gate); the cross-mode invariants (mass
   conservation, monotone energy, phase bounds) hold in every mode.

5. **Doc-vs-data staleness via macro parsing + a mapping YAML.** The course
   cites numbers as `\newcommand` macros in `latex/numbers/<ch>.tex`.
   `check_doc_staleness.py` parses them and compares against a freshly
   generated `results.json` through a per-chapter `doc_numbers.yaml` map
   (macro -> results path + tolerance). `docs-build.yml` also has a GPU-free
   consistency check (macros vs baseline references) so a stale edit is caught
   before the expensive GPU job.

6. **Makefile `tutorials-*` targets cover only the ported chapters** (00 smoke
   + P1) today; later phases extend them as more chapters adopt the harness.

## Verification (reference box: RTX 6000 Ada, CUDA 12.9 / driver 13.2)

- `pytest tests/test_diagnostics.py tests/test_course_common.py` — 35 pass.
- `doctor.py` — `READY: full CUDA`.
- P1 reference regression through the harness reproduces EXPECTED.md
  (POLY F 0.256->0.0494, FH F -0.0619->-0.101, mass drift ~machine eps,
  monotone) and the `baseline.yaml` check passes.

## Not bit-reproducibility

Every numeric gate is a scientific invariant within a documented tolerance
(rtol on energies, machine-eps bound on mass drift), never bitwise GPU
identity — two correct cards differ in the last digits.
