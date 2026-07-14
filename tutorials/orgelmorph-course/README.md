# OrgElMorph — an onboarding course in organic-electronics morphology simulation

*A self-contained, three-track course that takes a new student from the
physics of phase separation to differentiable, GPU-native morphology
simulation — reading, running, and verifying every step.*

This course is built around **OrgElMorph** (DiffSim's M5 milestone): a
GPU-native, differentiable simulator for the coupled **phase separation
and crystallization** of organic-electronic thin films — the morphology
that sets the performance of organic solar cells, transistors, and
bioelectronic devices.

It is written for someone starting research in the area — the target
reader is a new PhD student (or a curious collaborator) with an
undergraduate background in physics or engineering and some Python. By
the end you should be able to set up, run, understand, and *optimize*
morphology simulations of real material systems.

---

## How the course works

You read the **course document** (`latex/`, shipped as a built
`orgelmorph_course.pdf`), and at each concept it points you to a
**tutorial folder** here. You run the code, and compare your output —
figures and a printed self-check table — against the numbers and
figures in the document. Then you try the **exploratory questions**.

```
read a concept  ─▶  run its tutorial  ─▶  compare with the document  ─▶  explore the questions
```

Every tutorial folder contains:

| file | what it is |
|------|------------|
| `README.md` | orientation + how to run + the **mandatory template** (learning objectives, prerequisites, expected cost, required deliverable) |
| a core module | the importable physics (the object you read first) |
| `run.py` | the driver **you** run; prints the self-check table |
| `gen_figures.py` | regenerates the exact figures the document shows |
| `EXPECTED.md` | the reference numbers your run should reproduce |
| `INSTRUCTOR.md` | instructor companion: what it really teaches, typical incorrect conclusions, runtime ranges, common CUDA/solver errors |
| `grading_rubric.md` | the course rubric specialised to this chapter |
| `solutions/` | hints for every exploratory question + full solutions for the verification exercises (instructor-only) |

The tutorials call the **same production bricks** the research code
uses — there are no toy re-implementations. When a tutorial looks
small, it is because the physics lives in the solver, which is the
point: you are learning the real interface.

---

## The three tracks

### 🧪 Physics — *what is happening, and why*
Cahn–Hilliard and Allen–Cahn phase-field theory, from a binary blend to
the full evaporating, crystallizing, multi-component film. Free
energies, coefficients and their physical ranges, boundary conditions,
phase diagrams, nucleation. It opens with **P00** (the model hierarchy,
thermodynamics, and nondimensionalization) and closes with two
**capstones** — reproducing a real material system end to end (P10) and
extending the model with a new physical mechanism (P11). Questions are
**physics** questions.

### 💻 Computational — *how the equations become a fast, correct solver — and a reproducible instrument*
Discretization and verification: basis order, time integration,
convergence (MMS), boundary conditions, spatial and temporal
adaptivity, the solver ecosystem, and 2-D vs 3-D. It opens with **C00**
(weak form → CUDA) and then turns to the **research skills** that make a
solver trustworthy: reading nonlinear-solver diagnostics and diagnosing
divergence (C6), CUDA profiling and memory (C7), extending DiffSim
safely with a verified new term (C8), running reproducible campaigns
with no silent drops (C9), and validation and an uncertainty budget
(C10). Questions are **numerical / implementation** questions.

### 🔁 Differentiable — *how the simulator becomes an instrument for design*
Gradients through the solver: sensitivity of morphology to parameters,
recovering free-energy parameters from data, learning the free-energy
functional from snapshots, and inverse-designing a process. Every
gradient runs on the genuine three-way-verified adjoint (custom adjoint
== autograd twin == finite difference) through the same operator the
research code marches. Questions are **optimization / inverse-problem**
questions.

Each track is a step-by-step progression; the tracks cross-reference one
another (a physics concept links to the numerics that make it correct
and the gradient that optimizes it).

---

## Prerequisites

- A CUDA GPU (the tutorials are 2-D and sized to run in seconds to a
  couple of minutes on any modern card — an 8 GB laptop GPU is plenty;
  nothing here needs the 48 GB research cards).
- DiffSim installed (see the repository root `README.md`).
- To rebuild the course PDF: a LaTeX toolchain (built with `tectonic`,
  which fetches its own packages and runs BibTeX automatically —
  `tectonic latex/main.tex`; the output `main.pdf` is the deliverable,
  committed as `latex/orgelmorph_course.pdf`). A pre-built copy ships
  with the course, so you do not need LaTeX to take it.

## Running your first tutorial

**Start with Chapter 00** (`00_setup_and_smoke_test/`): run `doctor.py` to
confirm your environment (it ends in `READY: full CUDA` / `READY: reduced (no
cuDSS)` / `NOT READY: <fix>`), then the smoke test. Then:

```bash
cd physics/01_ch_binary_energies
python run.py                 # the student driver: both energies, prints the table
```

Compare the printed numbers with `EXPECTED.md`, then open the course
document to Physics Concept 1 and read the walkthrough.

### The scientific-workflow harness (`common/`)

Every run is a *repeatable workflow*, not a bespoke script. The shared
foundation in `common/` gives each chapter a YAML config (the canonical run
record), a provenance `metadata.json`, a `results.json` checked against a
tolerance baseline, and a standard `outputs/<run>/` layout. P1 ships a
harness-driven entry point (`run_harness.py`) as the reference example:

```bash
python run_harness.py --config configs/p1.yaml --mode reference \
    --output outputs/p1 --overwrite      # reproduces EXPECTED.md + checks it
```

Modes trade cost for fidelity: `--mode quick` (<2 min smoke), `reference`
(the EXPECTED numbers), `research` (finer/longer). The diagnostics library
(`src/diffsim/diagnostics/`) provides the measured quantities — energy budget,
mass conservation, structure factor, convergence order, ensemble statistics —
that the chapters and the research code share. See
`docs/dev/2026-07-14-course-phase0-foundation.md`.

---

## The mandatory template, assessment, and definition of done

Every chapter follows one **mandatory tutorial template** so a student
always knows what to learn, what it costs, and what to hand in. Each
chapter README carries *learning objectives*, *prerequisites*, *expected
cost* (GPU/memory/runtime), and an exact *required deliverable*, on top of
the model → run → verify → explore body.

- **`ASSESSMENT.md`** — the course-level student assessment model: the
  standard eight-item deliverable and the weighted grading rubric every
  chapter's `grading_rubric.md` inherits.
- **`DEFINITION_OF_DONE.md`** — the completion checklist (scientific
  specification / numerical verification / software + CUDA / reproducibility
  / pedagogy) a chapter — and a submission — must satisfy.
- **Instructor companion** — each chapter ships an `INSTRUCTOR.md`,
  `grading_rubric.md`, and a `solutions/` directory (hints for all
  questions, full solutions for the verification exercises only).

**Foundations (opening each track).** Chapter 00 (`00_setup_and_smoke_test`)
is the environment gate. Two conceptual foundation chapters open the
Physics and Computational tracks and are the shared reference the rest
build on: **P00** — the model hierarchy, thermodynamics, and
nondimensionalization (symbol table, the p1-vs-r14 χ conventions, the
dimensional / nondimensional / numerical / accelerated separation); and
**C00** — weak-form → CUDA (strong form → integration by parts → element
residual/Jacobian → quadrature → local-to-global → CSR → constraints →
Newton → linear solve → device), mapped to the exact production files.
Later chapters reference P00/C00 for symbols and conventions rather than
re-deriving them.

## Materials

The tutorials use real material systems where it matters — see
`materials/materials.yaml` for the parameter sets (blend interaction
parameters, degrees of polymerization, evaporation rates,
crystallization energetics) and their provenance in the literature.

## Map

| track | concepts |
|-------|----------|
| Physics | **P00 model hierarchy & nondimensionalization** · P1 binary CH energies · P2 adaptive stepping · P3 substrate & BCs · P4 ternary CH & the phase diagram · P5 evaporation · P6 Allen–Cahn crystallization · P7 coupled CH+AC · P8 noise & nucleation · P9 evaporation-induced crystallization · **P10 material-system case study (capstone)** · **P11 model-extension capstone** |
| Computational | **C00 weak form → CUDA** · C1 convergence (basis & time) · C2 boundary conditions · C3 dynamic adaptivity (AMR) · C4 the solver ecosystem · C5 2-D vs 3-D · **C6 nonlinear-solver diagnostics** · **C7 CUDA profiling & memory** · **C8 extending DiffSim safely** · **C9 reproducible campaigns** · **C10 validation & uncertainty** |
| Differentiable | D1 what differentiable simulation is · D2 morphology sensitivity · D3 recovering parameters · D4 process gradients · D5 crystallinity sensitivity · D6 learning the free energy from snapshots · D7 inverse-design capstone |

> Status. All three tracks are complete builds — **Physics (P00, P1–P11)**,
> **Computational (C00, C1–C10)**, and **Differentiable (D1–D7)**: 30
> concepts plus the Chapter 00 environment gate. Every concept has a
> runnable, self-checking tutorial (core module, `run.py`,
> `gen_figures.py`, `EXPECTED.md`) and a chapter in the course document
> with measured, locked numbers — see the built
> `latex/orgelmorph_course.pdf` (178 pp). The Differentiable track is
> built on a genuine, three-way-verified adjoint through the phase-field
> solver (`src/diffsim/adjoint/`, `tests/test_phasefield_adjoint.py`):
> the gradients are taken through the same operator the research code
> marches, not a toy — so a student learns to differentiate the real
> simulator.
>
> *Scientific-workflow overhaul.* Every chapter carries the full
> **mandatory tutorial template** (objectives, prerequisites, expected
> cost, required deliverable) and an **instructor companion**
> (`INSTRUCTOR.md`, `grading_rubric.md`, `solutions/`); the course-level
> **assessment model** (`ASSESSMENT.md`) and **definition of done**
> (`DEFINITION_OF_DONE.md`) make the deliverable and the completion bar
> explicit. The **P00/C00 foundation chapters** and the `common/`
> scientific-workflow harness (`docs/dev/2026-07-14-course-phase0-foundation.md`)
> close the derive → implement → verify → profile → interpret → perturb →
> reproduce loop. The final phase added the **research-skills** run
> (C6 diagnostics · C7 profiling · C8 safe extension · C9 reproducible
> campaigns · C10 validation & uncertainty), **dynamic AMR** in C3
> (`src/diffsim/adaptivity/`), and the two **capstones** (P10 reproducing
> a real material system, P11 extending the model) — the full evaluation
> and action list is `docs/dev/2026-07-14-course-v2-evaluation.md`.
