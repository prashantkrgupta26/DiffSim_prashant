# 05 — Three dimensions: the SBM sphere (the hero regime)

The end-to-end SBM-NS engine in 3-D. An immersed **sphere** carved from a
unit-box octree channel, weak no-slip on the sphere, strong inflow/walls,
free outflow. The sphere is not grid-aligned, so the SBM shift $d\neq0$ is
genuinely active in 3-D. This chapter marches the **monolithic** drag path
(the working, stable, physical 3-D drag, $C_d = +0.381$) and optionally
checks the projection+SBM composition invariants.

**Read** the course document, Chapter 5 (*Three dimensions*). Start with
`sphere.py` — the importable core.

**Run:**
```bash
python run.py                                       # monolithic drag, level 4, Re=100
python run.py --config configs/sphere.yaml --mode reference --output outputs/sp
python run.py --pipeline                             # + projection composition invariants
python run.py --config configs/sphere.yaml --mode quick   # level 3 (fast)
```

`run.py` prints a self-check table; compare with [`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `sphere.py` | the core: `run_sphere`, `check_pipeline_invariants` — curates the sphere de-risk fixture (`tests/p2r0_task10_*`); read this first |
| `run.py` | the driver you run; harness + drag self-check table |
| `gen_figures.py` | regenerates the verdict card + `numbers/c5.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The chapter calls the real 3-D SBM bricks via the validated de-risk fixture.
The core self-locates the repo `tests/` directory.

## Learning objectives

By the end of this chapter you can:

- Explain how the surrogate boundary, the shift $d$, and the area correction
  generalize to 3-D on an octree, and read the fixture invariants
  ($n_{\text{free}}$, surrogate faces, $D/h$, $d_{\max}$).
- State the honest 3-D verdict: the monolithic engine is the working drag
  path ($C_d = +0.381$); the projection composition is de-risked
  (finite, axisymmetric, BDF2 engages, PPE-space divergence machine-zero)
  but its long-time drag transient is not yet stable at feasible mesh.
- Describe the solver path to scale: host `splu` here → `cudss` (GPU direct)
  / `blockamgx` (block-preconditioned FGMRES) past the ~level-5 wall.
- Explain why the absolute $C_d$ is confinement-inflated and why the bar is
  the monolithic's physical steady state, not the literature number.

## Prerequisites

- **Concepts:** the 2-D SBM (Chapter 04), 3-D immersed-boundary intuition,
  block-preconditioning / direct-solver memory scaling.
- **Chapters:** 00–04.

## Expected cost

- **Device:** host `splu` at level 4 is ~seconds/step (a few minutes to
  steady). Level 5 (~143k DOF) needs `cudss`/AMGX. **Quick** mode (level 3)
  runs in ~a minute.
- A CUDA GPU is recommended for this chapter; the 2-D chapters do not need
  one.

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The `baseline.yaml` check green.
3. The verdict card (`gen_figures.py`).
4. **Headline:** the monolithic steady $C_d$ (+0.381), positive and
   physical, with the fixture invariants ($n_{\text{free}}$, faces, $D/h$).
5. **Verification:** `--pipeline` — the projection composition is finite,
   BDF2 engages, and the traction is axisymmetric ($|C_{\text{lat}}|\ll
   |C_d|$).
6. **Failure:** attempt the projection *long* march and show the drag
   transient diverges (documented R2 item) — "finite" is not "stable".
7. **Exploration:** if a GPU is available, run level 5 with `--solver cudss`
   and note the step-time and DOF change.
8. **Research bridge:** one paragraph on why the SPD-PPE projection is the
   100M-DOF path even though the monolithic is the 3-D *drag* path today
   (Chapter 06).
