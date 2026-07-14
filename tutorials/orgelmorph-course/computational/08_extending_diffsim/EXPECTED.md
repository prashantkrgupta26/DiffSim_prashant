# C8 — expected results (self-check)

`run.py --mode reference` and `pytest test_sextic_ch.py` reproduce the following
(deterministic on CPU, fixed seeds). Eight `[PASS]` gates and `ALL GATES: PASS`
must print from `run.py`, the 7-check pytest battery must pass, and the baseline
check must pass.

## The workflow gates

| step | gate | measured | must hold |
|---|---|---|---|
| 1 | new-term f′ vs FD | 3.6e-6 | < 1e-4 |
| 1 | new-term f″ vs FD | 1.1e-5 | < 1e-3 |
| 4 | analytic Jacobian vs FD (β=0.5) | 6.6e-10 | < 1e-5 |
| 4 | analytic Jacobian vs FD (β=0) | 5.0e-10 | < 1e-5 |
| 6 | limiting case β→0 = base | exact | true |
| 7 | MMS order (levels 3,4,5) | 2.01 | in [1.7, 2.3] |
| 8 | mass drift (relative) | ~1e-15 | < 1e-10 |
| 8 | energy start → end | 0.3144 → 0.2500 | non-increasing |
| 8 | energy max +increment | 0 | < 1e-9 |
| 7' | deep-quench max\|c\| (sextic < base) | 0.311 < 0.312 | bounded |
| 9 | new-term assembly overhead | negligible | < 25% |

## pytest battery (7 checks)

```
test_sextic_term_derivatives          PASS
test_jacobian_matches_fd[0.0]         PASS
test_jacobian_matches_fd[0.5]         PASS
test_limiting_case_beta_zero          PASS
test_mms_converges                    PASS
test_mass_conserved                   PASS
test_energy_decreases                 PASS
```

## What must be true regardless of hardware

- The analytic Jacobian matches a finite difference to FD-truncation (~1e-10);
  a wrong `f''` would sit at O(1).
- β→0 reproduces the base double well exactly (no regression).
- The extended discretization still converges at order p+1 = 2.
- Mass is conserved and the Ginzburg–Landau energy is non-increasing.
