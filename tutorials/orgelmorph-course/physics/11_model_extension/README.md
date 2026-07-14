# Physics P11 — Vitrification (composition-dependent mobility arrest)

A **model-extension capstone**: you extend a solver with a new physical
mechanism, derive its **analytic Jacobian by hand**, and *verify* the whole
thing. The mechanism is **vitrification** — as an organic film dries (P10), the
polymer-rich phase concentrates until its molecular mobility collapses at a
**glass composition** and the morphology **freezes**.

This is a **self-contained, tutorial-local** 1-D Cahn-Hilliard (CH) brick in
`vitrification.py` — it does **not** modify the core `src/`.

## The model (1-D, periodic `[0, L)`, `N` nodes, `h = L/N`)

- Order parameter `phi in [0,1]`; double-well `f(phi) = phi^2 (1-phi)^2` with
  hand-derived `f'`, `f''`, `f'''` (unit-tested against central differences).
- Chemical potential `mu = f'(phi) - kappa Lap(phi)` (periodic 2nd difference).
- **Vitrification mobility** `M(phi) = M0 * a(phi)` with the smooth arrest
  factor `a(phi) = 1/(1 + exp((phi - phi_g)/w)) = sigmoid((phi_g - phi)/w)`:
  `a ~ 1` (mobile) for `phi < phi_g`, `a -> 0` (arrested/vitrified) for
  `phi > phi_g`. Analytic `M'(phi) = -(M0/w) a (1-a)`.
  **Limiting case:** `phi_g -> +inf` gives `a -> 1`, `M -> M0` constant — the
  brick reduces to standard constant-mobility CH (used for verification).
- Face-averaged flux divergence (arithmetic face mean `Mf_{i+1/2}=0.5(M_i+M_{i+1})`)
  gives a clean analytic Jacobian.
- Backward Euler: `R(phi) = (phi - phi_old)/dt - div(M grad mu)`; solved by
  damped Newton with the **hand-derived analytic** dense Jacobian `dR/dphi`.

Everything runs in torch float64, preferring CUDA (CPU fallback).

## Files

| file | purpose |
|------|---------|
| `vitrification.py`   | importable core: `f/fprime/fpp/fppp`, `arrest/Mmob/Mprime`, `residual`, analytic `jacobian`, Newton `step`, `march`, `domain_scale` L(t), and MMS `mms_phi/mms_source`. |
| `test_vitrification.py` | pytest gates: analytic derivatives, **Jacobian FD-check** (+ autograd cross-check), dispersion limiting case, MMS spatial order, temporal order, coarsening-arrest. |
| `run_harness.py`     | course-harness driver (config + provenance + `results.json` checked vs `baseline.yaml`). |
| `configs/p11.yaml`   | canonical run record with `quick`/`reference`/`research` modes. |
| `baseline.yaml`      | tolerance gate on `results["checks"]` (measured on a real reference run). |
| `profile_gpu.py`     | GPU profile: per-Newton-step wall time + peak memory vs N (GPU vs CPU) → `outputs/p11/profile.json`. |
| `gen_figures.py`     | figures → `../../latex/figures/p11_*.png` and numbers → `../../latex/numbers/p11.tex`. |

## The verifications (all real, tolerance-checked)

1. **Analytic derivatives** — `f'`, `f''` vs central differences of `f` (~1e-11).
2. **Jacobian FD-check** *(the contributor-workflow gate)* — the hand-assembled
   analytic `dR/dphi` vs a column-wise central-difference Jacobian on a random
   smooth state: max abs error **3.6e-8** (< 1e-6). Autograd cross-check 2.3e-13.
3. **Dispersion limiting case** — arrest off (`phi_g -> inf`, constant mobility):
   the measured single-mode CH growth rate matches the analytic dispersion
   `sigma(k) = -M0 k^2 (f''(phi_bar) + kappa k^2)` to **0.12%**.
4. **MMS spatial convergence** — a manufactured `phi*` with the exact analytic
   continuous source: L2 error falls **~4x per mesh doubling** (order **1.99**).
5. **Temporal order** — backward Euler is **~1st order** (observed **1.02**).
6. **GPU profile** — `profile_gpu.py` times one Newton step (residual + analytic
   Jacobian + dense solve) vs N. Honest verdict: the **dense** N×N Jacobian only
   beats the CPU at large N (~1.7x at N=1024); at small N the CPU wins — profile
   before assuming the GPU helps.
7. **Scientific result** — coarsening from a random IC at the glass composition:
   **without** arrest domains coarsen (`L` grows to 0.38); **with** the
   polymer-rich matrix vitrified `L` **plateaus** at 0.11 — a domain-scale
   **ratio 3.4** and a flat late-time `L(t)`. Mass is conserved to 1.8e-15.

Note (honest): the arithmetic face-averaged mobility conducts through *any*
mobile neighbour, so a clean halt requires arresting the **transporting**
(matrix) phase — hence the mean composition is seeded at `phi_g` (polymer-rich
matrix vitrifies) and a sharp `w = 0.03` collapses its mobility. A softer
`w = 0.05` still halts coarsening but plateaus less sharply.

## Run

```bash
PYTHONPATH=<repo>/src python -m pytest test_vitrification.py -x -q   # all gates
PYTHONPATH=<repo>/src python run_harness.py --config configs/p11.yaml \
    --mode reference --output outputs/p11 --overwrite                # -> baseline PASSED
PYTHONPATH=<repo>/src python profile_gpu.py --run-dir outputs/p11    # GPU profile
PYTHONPATH=<repo>/src python gen_figures.py --run-dir outputs/p11    # figures + numbers
```

Modes: `quick` (N=64, 200 steps, <2 min smoke), `reference` (N=128, 300 steps —
the checked numbers), `research` (N=256, 400 steps, finer MMS meshes). The brick
uses its own torch dense solver (not the diffsim linsolver).
