# Hints — C3 exploratory questions

*Hints for every "Explore on your own" question (course document `c3.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Extend the cost accounting to `t_end = 1.2`; does the Newton
saving grow?**
Hint: call `adaptive_cost(t_end=1.2, ...)` and re-run the matched-`dt`
sweep at the new horizon. The step-doubling overhead (three solves per
accepted step, plus rejects) is a roughly fixed tax paid mostly during
the violent onset; the coarsening tail (where `dt` grows large) is where
the adaptive controller pulls ahead of a fixed step sized for the onset.
Plot the Newton-work ratio against `t_end` and look for the point where
the tail starts to dominate the total step count.

**Q2 — Sweep the droplet radius in the conservative-transfer demo;
confirm injection error → 0 as the feature resolves.**
Hint: `transfer_error_sweep` already runs this sweep
(`radii=(0.5, 0.7, 1.0, 1.5, 2.5)` fine cells) — read off `inj_err` and
`avg_err` at each radius, or extend the tuple to larger radii. Averaging
being exact *at every size*, not just in some favorable regime, is the
property a real AMR restriction operator needs — a transfer that is only
approximately conservative for well-resolved features would still leak
mass exactly when a region is coarsened aggressively (the case that
matters).

**Q3 — Tighten the LTE tolerance from `5e-4` to `1e-4`; does the
adaptive advantage grow or shrink?**
Hint: pass `tol=1e-4` to both `adaptive_cost` and the fixed-`dt` sweep's
target accuracy. A tighter tolerance forces more (and smaller) accepted
steps in the adaptive run *and* forces the matched fixed-`dt` search to a
smaller `dt`, so both sides do more work — the interesting question is
whether the *ratio* changes, not just the absolute cost.

**Q4 — Force the constant-coefficient bug on a *fixed* `dt` (`r≡1`
naturally) and confirm order 2 is recovered.**
Hint: use `_fixed_march` (or drive `_const_march` with `dt0 = dt0/2`
removed so every step is the same size) so that `r = dt/dt_prev = 1`
*is* the correct ratio, not a forced lie. Then the "constant-coefficient"
formula is simply the correct variable-coefficient formula evaluated at
`r=1` — there is no bug to expose. This is exactly why the chapter's
order study uses the *alternating* `(dt0, dt0/2)` sequence: only a
genuinely varying step ratio can distinguish the two coefficient forms.

**Q5 — Sketch what transferring the BDF2 history must satisfy across a
remesh.**
Hint: the variable-coefficient BDF2 formula needs both `c^n` and
`c^{n-1}` *on the same mesh* to form the history weights
`[1+r, -r^2/(1+r)]`. If only `c^n` is transferred and `c^{n-1}` is left
on the old mesh's node layout (or dropped), the next step either can't
evaluate the history term or evaluates it against a mismatched
discretization — silently reintroducing the same order-collapse this
chapter measures, but from a mesh change instead of a step-ratio bug.
Think about what "transfer" must mean for a quantity that isn't even a
field on the new mesh yet.
