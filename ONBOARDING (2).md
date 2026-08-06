# Onboarding to DiffSim

Welcome! DiffSim is a GPU-native, **differentiable** finite-element framework
(octree meshing + shifted/surrogate-boundary immersed FEM + an adjoint stack,
built on NVIDIA Warp). You already know FEM (weak forms, shape functions,
quadrature, adaptivity) — this guide gets you from clone to your first
contribution, and points you at the right tutorials for **solid mechanics +
neural constitutive models**.

Work top to bottom; it's ordered.

---

## 0. What you'll need

- A **GitHub account** — send your username to Baskar; he'll add you as a
  collaborator. (You'll have `write`, but `master` is protected — see §5.)
- **Python ≥ 3.10** (the package requires it; 3.11 is what we run on the cluster).
- A **CUDA GPU** for real runs. The first small tutorials run fine on a laptop
  (Warp has a CPU backend); anything 3-D or at scale needs a GPU — ask Baskar
  about access to **Nova** (the `mech-ai` allocation) or the group's **gpubox**.

## 1. Get the code

```bash
git clone git@github.com:BaskarGS/diffsim.git   # canonical name is lowercase
cd diffsim
```

## 2. Set up your environment

```bash
python3.11 -m venv .venv          # any Python >= 3.10
source .venv/bin/activate
pip install -U pip
pip install -e .                  # core deps: warp-lang, numpy, scipy, torch
pip install pyyaml                # NOTE: currently an undeclared dep — needed to load the YAML configs
pip install -e ".[dev]"           # pytest + ruff (to run the test suite)
# optional GPU direct solver (recommended once you have a GPU):
pip install -e ".[cudss]"         # nvmath-python[cu12]
```

Sanity check:

```bash
python -c "import diffsim, warp, torch; print('diffsim ok', torch.__version__)"
```

## 3. Read first (~30 min)

- **`tutorials/README.md`** — the curriculum and the "chapter contract" every
  brick follows (Learning outcome → Background → code → **Expected results you
  must reproduce** → Explore).
- Build & browse the docs site: `python tutorials/build_site.py && mkdocs serve`
  → open the **Solver projects** overview.

## 4. Your tutorial path

Each chapter is a runnable script. **Run it, reproduce the measured numbers in
its "Expected results", then do the "Performance corner"** (predict a stage's
cost exponent, measure it, explain any gap — keep a lab-notebook ledger; the
habit is the point). You know FEM, so the value is in what's *new*: Warp/GPU
kernels, the **shifted-boundary method (SBM)**, and the **differentiable/adjoint**
stack.

**Phase 1 — the DiffSim way**
| Chapter | Why it's for you |
|---|---|
| `A_foundations/A1_mms_convergence.py` | The verification discipline (p1→order 2, p2→order 3). This *is* our engineering culture — don't skip it. |
| `A_foundations/A2_boundary_conditions.py` | Nitsche / weak Dirichlet (new vs the strong BCs you're used to). |
| `A_foundations/A3_shifted_boundary.py` | **The core method of the framework** — immersed geometry via surrogate boundaries + Taylor shift. |
| `A_foundations/A5_three_dimensions.py` | The same physics at k=3; what changes (cost!) and what doesn't. |
| `A_foundations/A6_complex_geometry.py` | Carving domains (channel → sphere → STL) — the bridge from your quadtree to the production octree. |
| `P_performance/P1_cost_model_and_scaling.py` | The group's cost-model habit. GPU is new to you; build the measure-then-explain instinct early. |

**Phase 2 — Nonlinear + Newton (the backbone of solid mechanics)**
| `B_nonlinear/B1_bratu_newton.py` | Newton–Krylov + quadratic convergence — you'll reuse this pattern verbatim for hyperelasticity. |
| `B_nonlinear/B2_bratu_3d.py` | 3-D + continuation in λ (the load-stepping analogue for finite strain). |

**Phase 3 — Differentiable simulation (essential for neural constitutive models)**
| `E_differentiable/E1_shape_optimization.py` | The adjoint loop: recover a hidden parameter from data — the mechanism behind training constitutive models. |
| `E_differentiable/E2_genie_diffsbm.py` | Differentiable geometry. |
| Then **study** `src/diffsim/adjoint/neural_energy.py` (learned, gauge-anchored energy heads), `src/diffsim/adjoint/torch_twin.py` (the differentiable twin), and run `F_phasefield/F1_allen_cahn.py` | The closest existing pattern to a **learned constitutive model** in this codebase. |

**You can skip for now:** the Flow track (`D_*`, `C2`) and the deeper phase-field
chapters (`F2`, `F3`) — not needed for solid mechanics. Do `C_time/C1_heat_bdf.py`
only if your solid mechanics will be dynamic.

## 5. How we work (please read)

- **`master` is protected.** No direct pushes. Your workflow is:
  ```bash
  git checkout -b your-feature
  # ... work, commit small green commits ...
  git push -u origin your-feature
  gh pr create        # or open the PR on GitHub
  ```
  Every PR **requires Baskar's review** (he's the code owner) before it can
  merge. A new commit after his approval re-requests review. This is by design.
- **Run the tests before you push:** `pytest` (or the relevant subset). Keep the
  output pristine.
- **Verification discipline** (non-negotiable, it's why the code is trustworthy):
  every new brick states its **weak form** in the docstring (term-to-code), is
  **MMS-gated** (observed order within tolerance), works under **BDF1 and BDF2**,
  and is **basis-function agnostic**. Gate tolerances come from *measured* values
  with ≥2× headroom. Gaps get recorded honestly (a ledger), never hidden.

## 6. Your first project (we'll brainstorm the details)

Solid mechanics is **greenfield** in DiffSim — the design spec lists
elasticity/shells as *"prioritized by student need,"* so you'll be the one to
build it. The arc:

1. A **linear-elasticity brick** following the A/B chapter contract (weak form
   documented, MMS-gated p1→2 / p2→3, SBM Dirichlet).
2. **Nonlinear / hyperelastic** (Newton, à la B1/B2; finite-strain via
   continuation/load-stepping).
3. **Neural constitutive models** on top — using the `neural_energy.py` /
   `torch_twin.py` / E-track pattern as the template.

Come to that discussion having done Phases 1–3; we'll scope it together.

## 7. Help & key references

- `docs/` — milestone reports (`docs/dev/m*-milestone-report.md`) and the master
  design spec (`docs/dev/specs/2026-07-02-diffsim-design.md`).
- The tutorial scripts are the source of truth; the docs site is generated from them.
- When stuck: open a draft PR with your question, or ask Baskar directly.

Welcome aboard.
