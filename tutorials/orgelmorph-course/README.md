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
| `README.md` | the one-paragraph orientation + how to run |
| a core module | the importable physics (the object you read first) |
| `run.py` | the driver **you** run; prints the self-check table |
| `gen_figures.py` | regenerates the exact figures the document shows |
| `EXPECTED.md` | the reference numbers your run should reproduce |

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
phase diagrams, nucleation. Questions are **physics** questions.

### 💻 Computational — *how the equations become a fast, correct solver*
Discretization and verification: basis order, time integration,
convergence (MMS), boundary conditions, spatial and temporal
adaptivity, the solver ecosystem, and 2-D vs 3-D. Questions are
**numerical/implementation** questions.

### 🔁 Differentiable — *how the simulator becomes an instrument for design*
Gradients through the solver: sensitivity of morphology to parameters,
recovering free-energy parameters from data, learning the free-energy
functional from snapshots, and inverse-designing a process. Questions
are **optimization/inverse-problem** questions.

Each track is a step-by-step progression; the tracks cross-reference one
another (a physics concept links to the numerics that make it correct
and the gradient that optimizes it).

---

## Prerequisites

- A CUDA GPU (the tutorials are 2-D and sized to run in seconds to a
  couple of minutes on any modern card — an 8 GB laptop GPU is plenty;
  nothing here needs the 48 GB research cards).
- DiffSim installed (see the repository root `README.md`).
- To rebuild the course PDF: a LaTeX toolchain (`latexmk` + `pdflatex`).
  A pre-built `orgelmorph_course.pdf` ships with the course.

## Running your first tutorial

```bash
cd physics/01_ch_binary_energies
python run.py                 # runs both free energies, prints the table
```

Compare the printed numbers with `EXPECTED.md`, then open the course
document to Physics Concept 1 and read the walkthrough.

---

## Materials

The tutorials use real material systems where it matters — see
`materials/materials.yaml` for the parameter sets (blend interaction
parameters, degrees of polymerization, evaporation rates,
crystallization energetics) and their provenance in the literature.

## Map

| track | concepts |
|-------|----------|
| Physics | P1 binary CH energies · P2 adaptive stepping · P3 substrate & BCs · P4 ternary CH & the phase diagram · P5 evaporation · P6 Allen–Cahn crystallization · P7 coupled CH+AC · P8 noise & nucleation · P9 evaporation-induced crystallization |
| Computational | C1 convergence (basis & time) · C2 boundary conditions · C3 adaptivity · C4 the solver ecosystem · C5 2-D vs 3-D |
| Differentiable | D1 what differentiable simulation is · D2 morphology sensitivity · D3 recovering parameters · D4 process gradients · D5 crystallinity sensitivity · D6 learning the free energy from snapshots · D7 inverse-design capstone |

> Status: this course is under construction. **Physics Concept 1** is
> the complete reference build; the remaining concepts follow its
> template.
