# Hints — C4 exploratory questions

*Hints for every "Explore on your own" question (course document `c4.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Reproduce the divergence: march the polynomial spinodal with cuDSS
for 5, 10, 15, 20 steps and plot `max|c|` versus step.**
Hint: call `march_divergence(steps=n, energies=("poly",), solvers=("cudss",))`
for `n in (5, 10, 15, 20)` and read `cmax`/`cmin` off each row (or plot the
field's `max(|c|)` at each of the intermediate steps if you march by hand,
one `st.step()` at a time). `EXPECTED.md` only gives you the final,
20-step number (`[−552, 542]`, "~500× blow-up") — the shape of the curve
in between (gradual/exponential-looking vs. a sudden jump) is exactly what
you are asked to measure, not something to look up.

**Q2 — cuDSS survives Flory–Huggins because `f″≥4A>0`. Lower the
regularization floor and re-march.**
Hint: push the initial mean `c̄`, or the log-regularization epsilon, so `c`
sits closer to the wall (`c→0` or `c→1`); `f″=A(1/c+1/(1−c))` grows without
bound there, which worsens conditioning even though the sign stays
positive. Watch for the point where cuDSS's march starts to drift outside
`(0,1)` even though `f″` never actually goes negative — conditioning, not
just sign, is what breaks a solver in practice.

**Q3 — The one-iterate residual for cuDSS was tiny yet the march
diverged. Construct the condition-number × residual argument.**
Hint: for a computed `x̂` solving `Ax̂=b−r`, `x−x̂=−A⁻¹r`, so the relative
forward error is bounded by `κ(A)·‖r‖/‖b‖`. Plug in the *measured*
residual (`~5e-13`) and work out how large `κ(A)` has to be for that bound
to allow the divergence you actually see. See `verification_solutions.md`
for the full worked version — it uses only the numbers in `EXPECTED.md`.

**Q4 — `blockch` splits the saddle into SPD sub-solves. Which naive block
preconditioners also diverge on FH, and why does the two-factor signed-`F`
form survive?**
Hint: read the design-laws comment block in
`src/diffsim/solvers/linsolve.py` before guessing — it documents which
naive Schur-complement or block-diagonal splits fail to stay SPD once the
FH curvature varies across the domain, and what property the signed
factorization preserves that a naive split does not. This is a
*preconditioner* correctness question (does the sub-solve stay SPD?),
contrast it with the cuDSS question above (does the *direct* solve pivot?)
— they are different failure modes even though both show up as "the
solver was wrong."

**Q5 — Estimate the cuDSS memory ceiling on an 80 GB A100.**
Hint: the cited ceiling is ~811k dofs on 48 GB (`CfourCudssCeiling`); if
the factorization's fill scales roughly linearly with available memory
(the same assumption `recommend_solver` makes, `cudss_ceiling = 8e5 *
device_mem_gb/48.0`), scale the ratio. Then note the trap in the question:
even where cuDSS *fits* at that scale, it would still be the raw
indefinite CH block — nothing about more memory fixes the missing
pivoting. The masked pattern or `blockch_dev` is still what you actually
run.
