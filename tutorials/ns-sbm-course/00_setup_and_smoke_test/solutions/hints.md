# 00 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q (why finite `||div u||`?).** The equal-order $P_1/P_1$ pair is
LBB-unstable; the PSPG term relaxes the *pointwise* incompressibility in
exchange for a well-posed pressure. The scheme controls a *weak* /
PPE-space divergence, not $\nabla\!\cdot u = 0$ node-by-node. Expect
$\mathcal{O}(1)$ pointwise divergence on a coarse mesh; it *shrinks under
refinement* but never reaches machine zero. This is Chapter 01's
`||div u||` sentinel discussion.

**Q (why two engines?).** The monolithic solves one indefinite saddle — the
*oracle* (robust, the reference steady state). The projection splits into a
predictor + an **SPD** pressure-Poisson + a correction — the *scalable*
engine, because the SPD Poisson admits algebraic multigrid at 100M DOF
where the saddle factorization dies. Chapter 06 is the payoff.

## Full solution — the `cudss`-on-the-saddle failure

```bash
python run.py --config configs/smoke_cpu.yaml --solver cudss --output outputs/bad --overwrite
```

On a machine with cuDSS this attempts a non-pivoting LU on the *indefinite*
monolithic block. Expect either a solver error or a `|u|max` that blows past
the `baseline.yaml` bound (`umax_* <= 2.0`) — the gate catches it. The
mechanism: cuDSS assumes a factorization that exists without partial
pivoting (true for SPD systems), but the saddle $\begin{bmatrix}F&G\\D&C
\end{bmatrix}$ is indefinite. `splu` pivots and is exact. **Lesson:** match
the solver to the operator's definiteness — a theme that returns in
Chapter 02 (the SPD PPE *is* an AMG/cuDSS-friendly operator) and Chapter 05
(cuDSS on the saddle works only because it is a *direct* factorization with
enough memory, not because it is SPD).

## Full solution — `nsteps` sweep (the transient collapses)

Run `nsteps ∈ {20, 50, 100, 200}` (edit the config or pass a larger
`nsteps`). `max|proj−mono|` falls monotonically toward ~0.049 (the
Chapter 01/02 steady-state agreement on level 4). Plot it; the point is that
the 20-step gap is *pseudo-time transient*, not a modeling disagreement.
