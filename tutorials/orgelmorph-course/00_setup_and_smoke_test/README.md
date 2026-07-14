# 00 — Setup & smoke test

**Run this chapter first.** It confirms your machine can run the course and
teaches the workflow every later chapter uses (a YAML config, a provenance
`metadata.json`, a `results.json` checked against a tolerance baseline).

## 1. Install DiffSim

The course lives in the DiffSim repo. Install variants (the extras are
defined in the repo-root `pyproject.toml`):

| command | what you get | when |
|---|---|---|
| `pip install -e .` | core: `warp-lang`, `numpy`, `scipy`, `torch` (CUDA build) | minimum to run most chapters |
| `pip install -e ".[cudss]"` | core **+ cuDSS** GPU direct solver (`nvmath-python[cu12]`) | the measured stepping default at scale; C4 solver ecosystem |
| `pip install -e ".[dev]"` | core **+** `pytest`, `ruff` | to run the test suite |
| `pip install -e ".[dev,cudss]"` | everything above | recommended full install |

`torch` and `warp-lang` are **core** dependencies (there is no separate
`[cuda]` extra) — a CUDA-enabled build of PyTorch is what provides the device.

**AMGX** (algebraic multigrid, used in a couple of computational chapters as
an alternative solver) has no PyPI package: build
[NVIDIA/AMGX](https://github.com/NVIDIA/AMGX), then
`pip install pyamgx --no-build-isolation` with `AMGX_DIR` set. It is optional —
the doctor reports it as "not built" and the chapters fall back cleanly.

## 2. Run the doctor

```bash
python doctor.py
```

It probes the whole stack (python, diffsim commit, Warp, torch+CUDA, GPU +
memory, FP64, cuDSS/nvmath, AMGX), then **compiles+runs a tiny Warp kernel**
and **assembles+solves an 8×8 Cahn–Hilliard system**, and ends in exactly one
of:

- `READY: full CUDA` — you can run every chapter as written.
- `READY: reduced (no cuDSS)` — CUDA works but cuDSS is missing; see below.
- `NOT READY: <fix>` — do what the line says, then re-run.

## 3. Run the smoke test

```bash
python run.py --config configs/smoke_cuda.yaml --output outputs/smoke
# no GPU? prove the plumbing on the host:
python run.py --config configs/smoke_cpu.yaml  --output outputs/smoke_cpu --device cpu
```

This runs a tiny binary Cahn–Hilliard march through the standard harness and
checks the result against `baseline.yaml`. Compare with `EXPECTED.md`; it must
print `baseline check PASSED`.

## Which chapters need cuDSS?

Everything runs **without cuDSS** — the harness `--solver auto` falls back to
scipy SuperLU (`splu`, a host direct solver) which is perfectly fast at the
small tutorial sizes. cuDSS matters for:

- **C4 (the solver ecosystem)** — the whole point is to compare backends;
  without cuDSS you still see `splu`/matrix-free/AMGX, just not the GPU direct
  solver.
- **large `research`-mode runs** (finer meshes, 3-D in C5) — cuDSS is the
  measured default at 200k+ dofs; on `splu` these are much slower but still
  correct.

`READY: reduced (no cuDSS)` therefore means "the whole course runs; two
chapters are slower and skip the cuDSS comparison." No chapter is blocked.

## The standard output layout

Every `run.py` writes:

```
outputs/<run>/
  config.resolved.yaml   the exact, re-runnable config (defaults + overrides)
  metadata.json          provenance: commit, GPU, versions, timings, exit reason
  results.json           the scalar results (checked against baseline.yaml)
  history.npz            time series / fields for figures
  run.log                the run log
  checkpoints/           solver checkpoints (for --resume)
  figures/               generated figures
```
