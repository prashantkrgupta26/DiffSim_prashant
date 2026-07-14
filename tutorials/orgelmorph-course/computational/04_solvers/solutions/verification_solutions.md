# Selected full solutions — C4 verification exercises

*Full worked solutions for the two **verification** exercises only (the
cuDSS divergence-vs-step-count reproduction, and the residual × condition-
number error argument). Hints for all questions are in `hints.md`.
Instructor-only — do not distribute before the deadline.*

---

## V1 — Reproducing the cuDSS divergence vs. step count (Q1)

**Claim.** Marching the real polynomial spinodal (level 5, `dt=0.02`,
`mu_init="consistent"`) with cuDSS as the linear solver, the field range
grows step by step and the chapter's own `EXPECTED.md` records only the
final, 20-step outcome: `poly + cudss` → **`[−552, 542]`**, described there
as a **"~500× blow-up"** relative to the physical band `[−1, 1]`; `poly +
splu` stays bounded at `[−1.03, 1.01]` over the same 20 steps. The
exercise asks you to fill in *how* the field gets from a small, physical
perturbation to that final blown-up range.

**Procedure.**
1. Call `solvers.march_divergence(level=5, steps=n, dt=0.02,
   energies=("poly",), solvers=("cudss",))` for `n ∈ {5, 10, 15, 20}` (or
   instrument `march_divergence`'s own step loop to record `c.min()`/
   `c.max()` after every `st.step()` inside a single 20-step march, which
   is more informative than four separate restarted runs since it shows
   the trajectory rather than four endpoints).
2. Plot `max(|c.min()|, |c.max()|)` versus step count on a log-linear (or
   log-log) axis.
3. Compare against `poly + splu` over the same steps, which the chapter's
   own numbers say stays inside `[−1.03, 1.01]` throughout.

**Expected result.** The chapter's own reference numbers give you exactly
one anchor point with certainty: at step 20, `poly + cudss` has escaped to
`[−552, 542]`, i.e. `max|c| ≈ 552`, versus a physical scale of `1`. That is
consistent with (though not, by itself, proof of) **geometric/exponential
growth of the error**: if the field's excursion beyond the physical band
grows by a roughly constant multiplicative factor `r` each step starting
from an O(1) physical perturbation, then `r^20 ≈ 552` gives `r ≈ 552^{1/20}
≈ 1.37` — a ~37% amplification of the error *per step*, which compounds
(not adds) into the ~500× final blow-up. The qualitative signature to look
for in your own step-by-step trace is therefore **gradual, compounding
growth** (each Newton solve and each time step's unpivoted factorization
error multiplies the previous one), not a single discontinuous jump at one
step — because the mechanism (an indefinite block getting a wrong
factorization) is present at *every* step once the field enters the
spinodal band `|c|<1/√3`, not just at one of them. `splu`'s trace, by
contrast, should stay essentially flat at the physical scale the entire
time, because pivoting keeps every step's solve correct.

**Common wrong conclusion.** "The field was fine for a while and then
suddenly exploded at step 20, so something special happens at step 20."
Nothing about step 20 is special — it is simply the step count
`EXPECTED.md` happened to march to. The mechanism is present from the
first step the field enters the indefinite spinodal band; a shorter or
longer march would show the same compounding, just stopped (or continued)
at a different point. Do not read a fixed reference step count as a
physical threshold.

---

## V2 — Residual × condition number bounds the error (Q3)

**Claim.** For a linear solve `Ax=b` with a computed solution `x̂`, the
residual `r = Ax̂ − b` and the true error `x − x̂` are related by
`x − x̂ = −A⁻¹ r`, which gives the standard forward-error bound
`‖x − x̂‖/‖x‖ ≤ κ(A) · ‖r‖/‖b‖` (using `‖b‖ ≤ ‖A‖‖x‖` and
`‖A⁻¹‖ = κ(A)/‖A‖`). A **tiny relative residual therefore does not bound
the relative error tightly** unless you also know `κ(A)` is not huge — and
for the indefinite, unpivoted CH saddle, it is.

**Procedure.**
1. Take the chapter's own measured one-iterate relative residual for
   cuDSS on the polynomial saddle: `‖r‖/‖b‖ ≈ 5×10⁻¹³` (`EXPECTED.md`,
   "The trap").
2. Take the chapter's own measured *marched* outcome as the target
   relative error to explain: over the 20-step march the field escapes to
   `[−552, 542]`, a "~500×" excursion beyond the physical scale `O(1)`.
   Because that 500× is the *cumulative* result of 20 compounding steps
   (see V1), the *per-step* relative error only needs to be of order
   `500^{1/20} − 1 ≈ 0.36`, i.e. **~36% per step**, not 500× per step.
3. Solve the bound `‖x−x̂‖/‖x‖ ≤ κ(A)·‖r‖/‖b‖` for the condition number
   that would be needed to *permit* (not force) a ~36% per-step relative
   error given the measured residual:
   `κ(A) ≳ 0.36 / (5×10⁻¹³) ≈ 7×10¹¹`.

**Expected result.** The bound says that a per-step relative error of
~36% is consistent with the measured ~5×10⁻¹³ residual only if the
effective condition number of the captured `(c, μ)` Jacobian is on the
order of **`10¹¹`–`10¹²`** — a very large but entirely plausible number for
an indefinite saddle matrix that a non-pivoting factorization has handled
incorrectly (an unpivoted factorization of an indefinite matrix can behave
as if the effective conditioning were far worse than the matrix's true
condition number, because small/zero pivots are not detected or avoided).
The arithmetic itself is an *estimate*, not a proof that `κ(A)` is exactly
this value — it is a plausibility argument for how a residual this small
and an error this large can coexist, which is precisely the point the
question asks you to construct.

**Common wrong conclusion.** "A residual of `5×10⁻¹³` is machine-precision
small, so the condition number must be irrelevant / the bound must be
loose to the point of being meaningless." The bound is exactly what makes
the coexistence *possible*: for a well-conditioned matrix (`κ(A)` of order
1–100, as `splu`'s pivoted, correct solve effectively sees), a `5×10⁻¹³`
residual *would* certify a comparably tiny error — and that is exactly why
`splu`'s march stays bounded. It is only because the raw indefinite CH
block, mishandled by an unpivoted factorization, behaves as if
`κ(A)` were enormous that the same tiny residual is compatible with a
divergent march. The lesson is not "the bound is useless," it is
"the residual alone never tells you `κ(A)`, so a tiny residual can never
certify correctness by itself" — you must independently check the
solution (the marched `c.min`/`c.max`), which is exactly what
`march_divergence` does and `solver_correctness`'s one-iterate check does
not.
