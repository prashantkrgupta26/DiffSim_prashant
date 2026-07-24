# HANDOFF — NS-SBM course

*Everything a future Claude Code session needs to complete and extend this
course. Read this first, then the course `README.md`.*

---

## 1. What this course is, and the design conventions

`tutorials/ns-sbm-course/` teaches DiffSim's two incompressible-NS engines
(monolithic VMS saddle vs pressure-projection / Helmholtz–Leray), weak
Dirichlet imposition (Nitsche), and the Shifted Boundary Method (SBM). It is
modeled **exactly** on the sibling `tutorials/orgelmorph-course/` — same
file set per module, same README structure, same tone, same
config/provenance/check harness pattern.

### The mandatory per-module template (copy it faithfully)

Every **real** module folder (`00`–`05`) contains, in this shape:

| file | what it is |
|------|------------|
| `README.md` | orientation + how to run + the mandatory template: *why-it-matters, learning objectives, prerequisites, expected cost, model + assumptions, where-in-diffsim, baseline, verification, failure modes, guided exercises, required deliverable* |
| a **core** module (e.g. `cavity.py`, `square.py`, `shifted.py`, `sphere.py`) | the importable driver you read first — **it calls the real production steppers / SBM bricks**, no toy re-implementation |
| `run.py` | the driver the student runs; wraps the core in the `common/` harness, prints a SELF-CHECK TABLE, runs the tolerance gate |
| `gen_figures.py` | regenerates the figure(s) + `../latex/numbers/<c>.tex` (the macros the course document cites) |
| `EXPECTED.md` | the reference numbers your run reproduces (MEASURED on gpubox) |
| `baseline.yaml` | the tolerance baseline `check_results` gates `results.json` against |
| `INSTRUCTOR.md` | what it really teaches, typical wrong conclusions, runtime ranges, common CUDA/solver errors |
| `grading_rubric.md` | the course rubric (`ASSESSMENT.md`) specialised to this chapter |
| `solutions/hints.md` | hints for every exploratory question + full solutions for the verification/failure exercises |
| `configs/*.yaml` | the YAML config(s) — the canonical run record; `modes:` block gives quick/reference/research tiers |

### Standing rules (do NOT break these)

- **Real bricks only.** Modules call `diffsim.steppers.{linearized,leray}`,
  `diffsim.sbm.{vector,surrogate}`, and the validated `tests/` ladder /
  sphere-de-risk fixtures. No toy solvers.
- **Reproducible numbers.** Every number in `EXPECTED.md` / a `numbers/*.tex`
  macro / the course document was produced by running on `gpubox`. The
  `baseline.yaml` tolerances are *scientific invariants within documented
  tolerances*, not bitwise GPU identity.
- **Compute on gpubox, not the Mac.** The Mac is orchestration + file-level
  CPU checks only. `gpubox` is UNCONSTRAINED for this course. (Nova, if ever
  used, is ≤4 concurrent jobs — not needed here.)
- **The `common/` harness is engine-agnostic** and copied verbatim from
  orgelmorph (only the docstrings are rebranded). Do not re-derive it.

---

## 2. What is DONE vs PLACEHOLDER

### DONE — Modules 00–05 (full template, run GREEN on gpubox)

| # | module | headline (verified on gpubox, CPU/splu) |
|---|--------|------------------------------------------|
| 00 | `00_setup_and_smoke_test` | doctor `READY`; both-engine cavity smoke, finite/bounded |
| 01 | `01_monolithic_vms` | LDC: max\|mono−Ghia\|=0.0620, max\|proj−mono\|=0.0486 |
| 02 | `02_pressure_projection` | LDC (same fixtures): proj−mono 0.0486, proj−Ghia 0.0164 |
| 03 | `03_weak_dirichlet_nitsche` | square d=0: proj Cd +1.3529 vs mono +1.3941 (2.96%) |
| 04 | `04_shifted_boundary` | shift d≠0: proj Cd +1.5407 vs mono +1.5457 (0.32%), dmax/h=0.80 |
| 05 | `05_three_dimensions` | sphere: MONOLITHIC steady Cd +0.381 (n_free 4907, 64 faces, D/h 3.84) |

### DONE — LaTeX spine

`latex/` builds to `ns_sbm_course.pdf` with `tectonic latex/main.tex`.
Chapters `c0`–`c5` are the full math (governing eqns; monolithic VMS with
τ_M/τ_C, SUPG/PSPG/LSIC, the saddle [[F,G],[D,C]] + C-block; the projection
split with the consistent-PPE #1–#6; Nitsche; SBM shift/traction; the two
engines compared). `c6`–`c8` are the PLANNED placeholder chapters. Measured
numbers flow in via `numbers/c1.tex`…`c5.tex` (written by each module's
`gen_figures.py`); `main.tex` `\providecommand`s fallbacks so it always
compiles even before a run.

### PLACEHOLDER — Modules 06–08 (README stub + outline only)

`06_toward_100M_dof`, `07_multi_gpu`, `08_differentiable_structure` each have
a `README.md` with a **⚠️ PLANNED** banner, an outline of what the chapter
teaches, source-material pointers, and a build checklist. Their LaTeX
chapters (`c6`–`c8`) carry the same PLANNED banner. They are NOT
fake-complete: no `run.py`, `EXPECTED.md`, or `baseline.yaml`.

---

## 3. How to complete each PLACEHOLDER

### 06 — Toward 100M DOF

**Teach:** the SPD-PPE→AMGX scaling result (textbook AMG; 100M fits ~33
GB/GPU), device $K_p$ assembly (removes the ~61 s/step host floor), the
one-time mesh-build wall, the L8→GH200 path; the "iterative-not-direct"
verdict (cuDSS cannot factorize the 15.15M-DOF / 1.6B-nnz hero on GH200).

**Read:** `tests/ppe_amgx_scaling.py` (the harness — the real brick to
drive); `docs/dev/production-code-conventions.md` (scaling-pathway section:
the stage-residency + 100M-DOF budget-line format); the projection-ladder
FINAL synthesis; the device-`K_p` assembler (the merged `#2` work — base
`3560ad8` is "merge(device-assembly): device-resident scalar K_p
assembler").

**Real bricks to call:** `LerayProjectionStepper` PPE with `solver="amgx"`;
the `ppe_amgx_scaling.py` driver. **Run on the box**, AMGX lane, with
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build`:
```bash
python tests/ppe_amgx_scaling.py --levels 5 6 7 --solver amgx --steps 5
python tests/ppe_amgx_scaling.py --parity --levels 4 5
```
**Target numbers to lock:** per-level PPE solve time + AMG iteration count
(L5–L8); the amgx-vs-splu parity residual (L4–L5) matching to solver tol;
measured bytes/dof and the projected ~33 GB at 100M.

### 07 — Multi-GPU

**Teach:** distributed solve beyond one GPU; device-FGMRES + distributed AMG
preconditioner; the halo-exchange/coarse-gather comms pattern; the partition
+ comms budget. **Read:** the roadmap reconciliation (Track S), the P2
milestone spec, the `#49 device-FGMRES` note. **Status: future** — depends
on Module 06 landing (per-GPU baseline) and on a real distributed solve
existing in the tree. Mark clearly as roadmap until then.

### 08 — Differentiable structure

**Teach:** the NS-SBM adjoint (steady = 1 transposed solve, tractable at
100M); three-way-verified gradients (custom adjoint == torch-twin == FD) on
the real operator; shape optimization on the shifted boundary (differentiate
through the distance field). **Read:** `src/diffsim/sbm/{ns_adjoint,
leray_adjoint,ns_shape,transient_adjoint,adjoint}.py`; the SP-1 R1 XDD
adjoint (`src/diffsim/xdd/adjoint.py`, spec
`docs/dev/specs/2026-07-20-sp1-r1-adjoint-design.md`) for the
three-way-verification template and the `Control` interface; the
`orgelmorph-course/differentiable/` modules for the gradient-chapter
template. **Target:** a three-way-verified drag/wake gradient w.r.t. a
control agreeing to the FD floor, plus a small inverse-recovery or
shape-opt step.

---

## 4. The build / validate workflow (how the real modules were made)

1. **Worktree + branch.** Work in the `ns-sbm-course` branch (this worktree
   is `.worktrees/ns-course`). Commit frequently.
2. **Box lane.** `source scripts/remote/config.sh`. The course lane is
   `/home/bglab/Baskar/DiffSim-nscourse` (NOT other lanes). The venv is
   `/home/bglab/Baskar/DiffSim/.venv/bin/python`. Sync the course source:
   ```bash
   source scripts/remote/config.sh
   LANE=/home/bglab/Baskar/DiffSim-nscourse
   rsync -az "${RSYNC_EXCLUDES[@]}" tutorials/ns-sbm-course/ \
       "$GPUBOX_HOST:$LANE/tutorials/ns-sbm-course/"
   ```
   (The lane also needs the repo `src/` and `tests/` — the whole worktree
   was synced once at setup; re-sync `src`/`tests` if you change them.)
3. **Run a module on the box** (PYTHONPATH needs the lane's `src`; the module
   cores self-locate `tests/`):
   ```bash
   PY=/home/bglab/Baskar/DiffSim/.venv/bin/python
   ssh "$GPUBOX_HOST" "cd $LANE/tutorials/ns-sbm-course/NN_module && \
     PYTHONPATH=$LANE/src $PY run.py --config configs/<c>.yaml \
     --mode reference --output outputs/<c> --device cpu --overwrite"
   ```
   The run must print `baseline check PASSED`. Copy the measured numbers into
   `EXPECTED.md` and lock them (with headroom) into `baseline.yaml`.
4. **Figures + numbers.** `python gen_figures.py --run-dir outputs/<c>`
   writes `latex/figures/*.png` and `latex/numbers/<c>.tex`.
5. **Build the PDF.** `cd latex && tectonic main.tex` → `main.pdf`; commit it
   as `latex/ns_sbm_course.pdf`. Common LaTeX gotchas already hit: `p^\*`
   must be `p^{*}`; `\code{...}` (lstinline) is INVALID in math mode — use
   `\texttt{...}` inside `$...$`.
6. **Wire into a site (optional).** `tutorials/build_site.py` currently
   targets orgelmorph; if the course is published, extend it — not required
   for the course to be complete.

### Config-schema gotchas (learned building 00–05)

- The YAML must NOT carry a `name:` key unless it is a declared schema field
  (the schema rejects unknown keys). Use a comment for the name.
- A CLI-only toggle passed via `cli_overrides` (e.g. `--zero-shift`,
  `--pipeline`) must be a **declared schema field** (bool, default) or the
  schema rejects it — see `04_shifted_boundary/run.py` (`_zero_shift`) and
  `05_three_dimensions/run.py` (`pipeline`).
- Every `baseline.yaml` check needs a comparator (`reference`/`min`/`max`/
  `equals`); an empty `{}` check is reported as WARN.

---

## 5. The verified numbers (single source of truth)

Measured on gpubox, CPU/splu, 2026-07-24 (re-confirmed at course build):

| module | fixture | numbers |
|--------|---------|---------|
| 01/02 | LDC L4 Re100 200 steps | proj−mono 0.0486, proj−Ghia 0.0164, mono−Ghia 0.0620; lid station proj +0.8376 / mono +0.8529 / Ghia +0.84123; ‖div‖ mono 0.194 / proj 2.035 |
| 03 | square L4 Re40 weak Nitsche (d=0) | proj Cd +1.3529, mono +1.3941 (rel 2.96%); mean\|u\| 0.9057 / 0.8550 (5.93%) |
| 04 | square L4 Re40 offset 0.05 (d≠0) | dmax 0.0500, dmax/h 0.80, 4 area-corr GPs; proj Cd +1.5407, mono +1.5457 (rel 0.32%); mean\|u\| rel 6.60% |
| 05 | sphere L4 Re100 alpha10 | monolithic steady Cd +0.381; n_free 4907, 64 surrogate faces, D/h 3.84, dmax>0 |

These are the numbers in `EXPECTED.md`, `baseline.yaml`, and the course
document. If a re-run drifts beyond the `baseline.yaml` tolerances,
investigate — do NOT loosen the tolerance without a documented physical
reason.

---

## 6. Where the content came from (provenance)

This course **restructured** already-validated content (it did not invent
numbers): the former `docs/tutorials/ns-sbm-tutorial.md` (the math + code
walkthrough, `file:line`-cited) became the LaTeX spine; the former
`examples/ns_sbm/*.py` (verified) became the module cores; the former
`tests/test_tutorial_examples.py` reference numbers became the
`baseline.yaml`/`EXPECTED.md`. Those three files were removed (`git rm`); the
`tests/ladder_*` and `tests/p2r0_task10_*` fixtures they depended on remain
in `tests/` and are what the module cores import. Deeper math lives in
`docs/dev/2026-07-23-projection-ladder-FINAL-synthesis.md` and
`docs/dev/specs/2026-07-23-consistent-projection-fix-design.md`.
