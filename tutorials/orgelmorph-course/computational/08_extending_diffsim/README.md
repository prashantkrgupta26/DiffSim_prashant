# C8 — Extending DiffSim safely

**Why it matters.** Adding a physics term is easy; adding it *safely* — so the
next person can trust it — is a discipline. This chapter walks the full
contributor workflow on one small real term (a sixth-order Landau free-energy
term) and every gate is **run and passes**. It is the bridge from tutorial user
to research contributor.

**Objectives.** Add a new free-energy term with correct units; write its residual
and *analytic* Jacobian contribution; verify the tangent against a finite
difference; prove the limiting case (β→0 = base model); confirm convergence
order, mass conservation, and energy dissipation still hold; profile the cost;
document it.

**Prerequisites.** Weak-form→CUDA + the red→green Jacobian gate (C0); convergence
+ MMS (C1); nonlinear diagnostics (C6); profiling (C7); the CH energy (P1).

**Cost.** No GPU required — the worked reference is a transparent readable-NumPy
mixed Cahn–Hilliard solver on the diffsim mesh infrastructure. `run.py` and the
pytest battery each finish in a few seconds.

## The new term

    f(c)  = 1/4 (c^2-1)^2 + (beta/6) c^6      [energy density]
    f'(c) = c^3 - c       + beta c^5          (enters the mu residual)
    f''(c)= 3c^2 - 1       + 5 beta c^4        (enters the mu-c Jacobian block)

`beta` (dimensionless) is the new coefficient; `beta=0` recovers the base model
exactly. Getting the analytic `f'' = 5 beta c^4` right is the crux.

## The workflow (each a passed gate)

1. Equations + units → 2. config schema (`beta`) → 3. residual (+`beta c^5`) →
4. **analytic Jacobian** (+`5 beta c^4`) → 5. **derivative unit test** (FD,
rel ~6e-10) → 6. **limiting case** (β→0 = base) → 7. **MMS convergence** (order
~2.01) → 8. **conservation + energy** (mass drift ~1e-15, energy monotone) →
9. **CUDA profile** (new-term overhead negligible) → 10. **docs + CHANGELOG**.

## Run it

```bash
export PYTHONPATH=<repo>/src
<repo>/.venv/bin/python run.py --config configs/c8.yaml --device cpu \
    --solver splu --output outputs/c8 --overwrite --mode reference
<repo>/.venv/bin/python -m pytest -q test_sextic_ch.py    # the contributor's gate
<repo>/.venv/bin/python gen_figures.py --run-dir outputs/c8
```

`run.py` prints eight `PASS/FAIL` gates; `pytest` runs the 7-check battery;
`results.json` is checked against `baseline.yaml`. Compare with `EXPECTED.md`.

## Files

- `sextic_ch.py` — the transparent mixed-CH solver with the new term (free
  energy, residual, analytic Jacobian, Newton, MMS, march, diagnostics).
- `test_sextic_ch.py` — the pytest gate (term derivatives, FD Jacobian, limiting
  case, MMS, mass, energy).
- `run.py` — harness driver walking all ten workflow steps.
- `gen_figures.py` — figures `c8_{verify,workflow}.png` + `numbers/c8.tex`.
- `configs/c8.yaml`, `baseline.yaml`, `CHANGELOG.md`.
