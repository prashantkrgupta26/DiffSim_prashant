# NS-SBM — a course in incompressible Navier–Stokes on an immersed-boundary GPU solver

*A self-contained course that takes a student from the incompressible
Navier–Stokes weak form to running, comparing, and verifying DiffSim's
two production NS engines — the **monolithic VMS saddle solver** and the
**pressure-projection (Helmholtz–Leray) solver** — including weak
Dirichlet imposition (Nitsche) and the **Shifted Boundary Method**
(SBM).*

This course is built around DiffSim's incompressible-flow stack: a
GPU-native, equal-order ($P_1/P_1$) residual-based-VMS solver that runs
on an octree mesh, imposes no-slip boundaries *weakly* (Nitsche), and
lets an immersed body sit **off** the grid via the Shifted Boundary
Method. It is written for someone who knows finite elements and
variational multiscale (VMS) stabilization *in general* but not this
codebase — a new PhD student or a collaborator with a graduate CFD
background and some Python.

By the end you can set up, run, understand, and *compare* both NS engines
on lid-driven cavity, flow-past-obstacle, and 3-D sphere problems, and
you understand why the projection engine — not the monolithic oracle — is
the path to 100-million-DOF simulations.

---

## How the course works

You read the **course document** (`latex/`, shipped as a built
`ns_sbm_course.pdf`), and at each concept it points you to a **module
folder** here. You run the code, and compare your output — a printed
self-check table and (where relevant) figures — against the numbers in
the document. Then you try the **exploratory questions**.

```
read a concept  ─▶  run its module  ─▶  compare with the document  ─▶  explore the questions
```

Every module folder contains:

| file | what it is |
|------|------------|
| `README.md` | orientation + how to run + the **mandatory template** (learning objectives, prerequisites, expected cost, required deliverable) |
| a core module | the importable driver (the object you read first) — it calls the **production steppers** |
| `run.py` | the driver **you** run; prints the self-check table and runs the tolerance gate |
| `gen_figures.py` | regenerates the figures / `numbers/<c>.tex` the document shows |
| `EXPECTED.md` | the reference numbers your run should reproduce (measured on `gpubox`) |
| `baseline.yaml` | the tolerance baseline the harness checks `results.json` against |
| `INSTRUCTOR.md` | instructor companion: what it really teaches, typical wrong conclusions, runtime ranges, common CUDA/solver errors |
| `grading_rubric.md` | the course rubric specialised to this chapter |
| `solutions/` | hints for every exploratory question + full solutions for the verification exercises (instructor-only) |

The modules call the **same production bricks** the research code uses —
there are **no toy re-implementations**. When a module looks small, it is
because the physics lives in the steppers (`diffsim.steppers.*`) and the
SBM lives in `diffsim.sbm.*`, which is the point: you are learning the
real interface. The 2-D obstacle modules reuse the validated *ladder
fixtures* (`tests/ladder_fixtures.py`, `ladder_rung*`) and the 3-D module
reuses `tests/p2r0_task10_sphere_derisk.py` — the exact fixtures the
research validation ran.

---

## The two engines (the spine of the course)

DiffSim solves incompressible NS with **equal-order $P_1/P_1$** collocated
velocity–pressure elements. Equal-order pairs violate the inf–sup (LBB)
condition, so **both** engines are residual-based-VMS stabilized (SUPG +
PSPG + grad-div). They differ in how they solve the coupled system each
step:

| | **Monolithic** (the *oracle*) | **Pressure-projection** (the *scalable* engine) |
|---|---|---|
| Class | `LinearizedMonolithicStepper` | `LerayProjectionStepper` / `LeraySBMStepper` |
| Solve | ONE coupled $(u,p)$ **saddle** system per step | predictor → **SPD** pressure-Poisson → velocity correction |
| Operator | indefinite $\begin{bmatrix}F&G\\D&C\end{bmatrix}$ | one nonsymmetric + two SPD sub-solves |
| Scalability | direct (splu/cuDSS) or block-preconditioned FGMRES | the **SPD PPE admits AMG/CG** → the 100M-DOF path |
| Source | `src/diffsim/steppers/linearized.py` | `src/diffsim/steppers/leray.py`, `leray_sbm.py` |

The **monolithic** solver is the *correctness oracle*: robust, and its
steady state is the bar every projection run is measured against ("does
the split reproduce the same-mesh monolithic?"). The **projection**
solver is the *scalability lever*: its pressure step is a symmetric
positive-definite (SPD) Poisson problem that AMG/CG handles at scales
where the indefinite saddle factorization runs out of memory. The whole
course is organized around that comparison.

---

## Prerequisites

- A CUDA GPU is recommended but **not required** for the 2-D modules: they
  run on CPU (scipy SuperLU) in seconds to a couple of minutes on the
  small tutorial meshes. The 3-D sphere is a few minutes on CPU/splu at
  level 4; a GPU (cuDSS/AMGX) is the path to finer meshes.
- DiffSim installed (see the repository root `README.md`): `warp-lang`,
  `numpy`, `scipy`, `torch`. Optional `[cudss]` (GPU direct solver) and
  AMGX (algebraic multigrid) unlock the scalable solver paths in the
  advanced modules.
- To rebuild the course PDF: a LaTeX toolchain (built with `tectonic` —
  `tectonic latex/main.tex`; the output is committed as
  `latex/ns_sbm_course.pdf`). A pre-built copy ships with the course, so
  you do not need LaTeX to take it.

## Running your first module

**Start with Chapter 00** (`00_setup_and_smoke_test/`): run `doctor.py`
to confirm your environment, then the smoke test (a tiny lid-driven
cavity through both engines). Then:

```bash
cd 01_monolithic_vms
python run.py                 # lid-driven cavity, monolithic engine, prints the table
```

Compare the printed numbers with `EXPECTED.md`, then open the course
document to Chapter 1 and read the walkthrough.

### The scientific-workflow harness (`common/`)

Every run is a *repeatable workflow*, not a bespoke script. The shared
foundation in `common/` gives each module a YAML config (the canonical
run record), a provenance `metadata.json`, a `results.json` checked
against a tolerance `baseline.yaml`, and a standard `outputs/<run>/`
layout — the same pattern as the sibling `orgelmorph-course`. Modes trade
cost for fidelity: `--mode quick` (a fast smoke), `reference` (the
EXPECTED numbers), `research` (finer/longer).

---

## The mandatory template, assessment, and definition of done

Every chapter follows one **mandatory module template** so a student
always knows what to learn, what it costs, and what to hand in. Each
chapter README carries *learning objectives*, *prerequisites*, *expected
cost* (GPU/memory/runtime), and an exact *required deliverable*, on top of
the model → run → verify → explore body.

- **`ASSESSMENT.md`** — the course-level student assessment model: the
  standard eight-item deliverable and the weighted grading rubric every
  chapter's `grading_rubric.md` inherits.
- **`DEFINITION_OF_DONE.md`** — the completion checklist (scientific
  specification / numerical verification / software + CUDA /
  reproducibility / pedagogy) a chapter — and a submission — must satisfy.
- **Instructor companion** — each real chapter ships an `INSTRUCTOR.md`,
  `grading_rubric.md`, and a `solutions/` directory.

## Map

| # | module | what it teaches | headline number (verified) |
|---|--------|-----------------|-----------------------------|
| **00** | `00_setup_and_smoke_test` | environment gate + a tiny both-engines NS smoke | `READY` verdict; smoke matches |
| **01** | `01_monolithic_vms` | the monolithic VMS saddle; LDC vs Ghia; `assemble_linear_ns` + `LinearizedMonolithicStepper` | `max|mono−Ghia| = 0.062` |
| **02** | `02_pressure_projection` | the projection engine; LDC (same result); `consistent_projection`; SPD-PPE→AMG scalability | `max|proj−mono| = 0.049` |
| **03** | `03_weak_dirichlet_nitsche` | flow past a square; strong vs weak Nitsche ($d=0$); drag | `proj Cd 1.353` vs `mono 1.394` |
| **04** | `04_shifted_boundary` | flow past a shifted body; the SBM shift ($d\neq0$); surrogate traction | `proj Cd 1.541` vs `mono 1.546` (0.32%) |
| **05** | `05_three_dimensions` | 3-D sphere; the shift in 3-D; cuDSS/AMGX solver path | `monolithic Cd = +0.381` |
| 06 | `06_toward_100M_dof` | **PLANNED** — SPD-PPE→AMGX scaling to 100M DOF; device $K_p$ assembly; the mesh-build wall | — |
| 07 | `07_multi_gpu` | **PLANNED** — distributed solve beyond a single GPU | — |
| 08 | `08_differentiable_structure` | **PLANNED** — the adjoint / shape-optimization path on the NS-SBM engine | — |

> **Status.** Modules **00–05 are complete builds** — each with a runnable,
> self-checking module (core driver, `run.py`, `gen_figures.py`,
> `EXPECTED.md`, `baseline.yaml`) whose numbers were **measured on
> `gpubox`** and are reproduced by the tolerance gate, plus the full
> instructor companion, and a chapter in the course document with locked
> numbers. Modules **06–08 are PLACEHOLDERS** — a README stub with an
> outline and pointers to the source material a future session will use;
> `HANDOFF.md` says exactly how to complete them. See `HANDOFF.md` for the
> design, conventions, and the build/validate workflow.

## SBM in one paragraph

The **Shifted Boundary Method** lets an immersed body sit *off* the mesh
grid. Instead of body-fitting, we keep a grid-aligned octree and pick a
**surrogate boundary** $\tilde\Gamma$ (whole element faces that hug the
true boundary $\Gamma$). A **Taylor shift** $S N_a = N_a + (\nabla
N_a)\cdot d$ transfers the boundary condition from $\tilde\Gamma$ to
$\Gamma$ using a distance vector $d$ (surrogate GP → closest point on
$\Gamma$). When the body *is* grid-aligned, $d=0$ and SBM degenerates to
ordinary Nitsche — which is why the modules build up from $d=0$
(Chapter 03) to $d\neq0$ (Chapters 04, 05).
