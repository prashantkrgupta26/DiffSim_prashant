# 03 — Weak Dirichlet imposition (Nitsche): flow past a square

The first immersed body. A **cell-aligned** square carved from a channel, so
the SBM shift is OFF ($d=0$) and this is **standard Nitsche**. The chapter
imposes no-slip on the obstacle two ways — **strong** (row replacement) and
**weak** (Nitsche block) — and compares drag $C_d$ against the same-mesh
monolithic oracle.

**Read** the course document, Chapter 3 (*Weak Dirichlet imposition —
Nitsche's method*). Start with `square.py` — the importable core it walks
through.

**Run:**
```bash
python run.py                                       # weak Nitsche, level 4, Re=40
python run.py --config configs/square.yaml --mode reference --output outputs/sq
python run.py --mode-noslip strong                  # row-replacement no-slip
python run.py --alpha 0                              # remove the penalty (anti-vacuity)
```

`run.py` prints a drag self-check table; compare with
[`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `square.py` | the core: `run_square` — curates the validated ladder fixtures (`tests/ladder_*`); read this first |
| `run.py` | the driver you run; harness + drag self-check table |
| `gen_figures.py` | regenerates the drag figure + `numbers/c3.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The chapter calls the real Nitsche brick `sbm_vector_dirichlet`
(`src/diffsim/sbm/vector.py`) and the validated ladder marchers — no toy
solver. The core self-locates the repo `tests/` directory, so only `src`
(or a pip-installed diffsim) needs to be importable.

## Learning objectives

By the end of this chapter you can:

- Write the three Nitsche boundary terms (consistency, adjoint-consistency,
  penalty $\alpha\nu/h$) and say what each does.
- Explain why weak imposition — not strong row replacement — is the
  foundation the Shifted Boundary Method (Chapter 04) builds on.
- Read the drag $C_d = F_x/q_{\text{ref}}$ off `surrogate_traction` and
  compare weak vs strong no-slip against the same-mesh oracle.
- Demonstrate the penalty is load-bearing: $\alpha=0$ removes it and the
  obstacle stops being felt (an anti-vacuity check).
- State why the bar is same-mesh faithfulness, not the literature square
  drag (confinement inflates absolute $C_d$).

## Prerequisites

- **Concepts:** Nitsche's method / weak BC imposition, the notion of a
  consistent boundary integral. Chapters 01–02's two engines.
- **Chapters:** 00, 01, 02.

## Expected cost

- **Device:** CPU runs level 4 in a few minutes (`splu`); a GPU is optional.
- **Quick** (300 steps) / **reference** (600 steps, the EXPECTED numbers) /
  **research** (level 5).

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The `baseline.yaml` check green.
3. The drag figure (`gen_figures.py`) — projection vs monolithic $C_d$.
4. **Headline:** the weak-Nitsche $C_d$ (proj and mono) and the rel-diff.
5. **Verification:** faithfulness — the weak-Nitsche projection matches the
   same-mesh weak-Nitsche monolithic in both $C_d$ and mean $|u|$.
6. **Failure:** run `--alpha 0` and show the obstacle stops being felt
   ($C_d \to 0$) — the anti-vacuity check.
7. **Exploration:** sweep $\alpha \in \{1,5,10,50,100\}$ and plot $C_d$ vs
   $\alpha$ (leak vs conditioning).
8. **Research bridge:** one paragraph on why $d=0$ Nitsche is the stepping
   stone to the genuine SBM shift (Chapter 04).
