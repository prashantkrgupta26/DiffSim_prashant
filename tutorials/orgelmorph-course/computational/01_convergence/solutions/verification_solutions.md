# Selected full solutions — C1 verification exercises

*Full worked solutions for two **verification** exercises: the level-7
round-off-floor prediction (Q1) and the `k=2` manufactured-wavenumber
intercept check (Q2). Hints for all five questions are in `hints.md`.
Instructor-only — do not distribute before the deadline.*

---

## V1 — Push the `p=1` spatial sweep to level 7: where does round-off
floor the error? (Q1)

**Claim.** The steady-MMS spatial study is `dt`-independent by
construction, so refining the mesh should keep halving-squaring the
`L2(c)` error (order ≈2) *until* floating-point round-off in the direct
solve becomes comparable to the shrinking discretization error — at
which point the error stops shrinking (or even grows slightly) and the
fitted order degrades. The exercise is to predict roughly where that
crossover sits, not to assume it appears immediately.

**Procedure.**
1. From `EXPECTED.md`, the `p=1` study (levels 3,4,5,6) gives
   `L2(c)`: `1.8e-2 → 2.8e-4`, fitted order `2.01`. Each halving of `h`
   shrinks the error by `2^2.01 ≈ 4.03`.
2. Extrapolate one further halving from the level-6 value to predict
   level 7: `2.8e-4 / 4.03 ≈ 6.9e-5`. This is a *prediction*, not a new
   measurement — run `spatial_mms(1, (3,4,5,6,7))` to get the actual
   number.
3. Separately, estimate where round-off *should* start to matter: `splu`
   solves the linear system to machine precision (fp64, unit round-off
   `~2e-16`) up to a factor set by the system's condition number, which
   for this `p=1` discretization grows roughly as `O(1/h^2)`. At level 7,
   `h = 1/128 ≈ 7.8e-3`, so `1/h^2 ≈ 1.6e4` — the round-off floor this
   implies is of order `2e-16 * 1e4 ≈ 2e-12` (absolute, order-of-magnitude
   only), *far* below the still-large `~7e-5` discretization error
   predicted for level 7.

**Expected result.** Level 7 should still sit cleanly on the `order ≈
2.0` line — the geometric extrapolation (`≈6.9e-5`) and the fitted order
across levels `3–7` should both look unremarkable, i.e. **round-off is
not expected to appear yet at level 7.** The condition-number estimate
above says the crossover is several *more* halvings out (each further
level shrinks the discretization error by ~4× while the round-off floor,
tied to the growing condition number, rises far more slowly) — so the
honest answer to "predict at which level `L2` stops halving-squared" is
"not at level 7 itself for this `p=1` problem; expect it several levels
further, and confirm by actually running those levels rather than
asserting a number." The pedagogical win is the *reasoning*
(discretization error shrinks geometrically, round-off floor rises with
conditioning, the crossover is where they meet) — not a specific
fabricated level.

**Common wrong conclusion.** "The order was 2.01 through level 6, so it
must be exactly 2.01 forever, or round-off must already be visible right
at level 7." Neither extreme is right: refinement does not converge
indefinitely at a fixed order (a round-off/conditioning floor exists for
*any* direct solve), but the floor is set by where the discretization
error's geometric decay meets the conditioning-driven noise floor — for
this well-conditioned 2-D `p=1` problem at level 7 that crossover has
not arrived yet. Concluding "it's floored" from a single extra level
without comparing the observed error against the extrapolated
prediction is exactly the kind of premature conclusion this course's
verification habit is meant to prevent.

---

## V2 — `k=2` manufactured wavenumber: the intercept jumps, the slope
does not (Q2)

**Claim.** Standard finite-element interpolation theory bounds the error
by `C h^{p+1} |u|_{H^{p+1}}`, where the seminorm `|u|_{H^{p+1}}` is
(roughly) the `(p+1)`-th derivative amplitude of the exact field. For
`c*_k = cos(k pi x) cos(k pi y)`, each derivative brings down a factor of
`k pi`, so the error *constant* scales as `~ k^{p+1}` while the *order*
(the exponent on `h`) is a property of the element, not the field, and
does not change with `k`.

**Procedure.**
1. Run `spatial_mms(1, (3,4,5,6), k=2.0)` (the `k` keyword threads through
   `c_star_k`/`mu_star_k`/`f_c_k`/`f_m_k`) — the same `p=1`, levels
   `3–6` study `EXPECTED.md` reports for `k=1`.
2. Compare the raw `L2(c)` values at each level against the `k=1`
   numbers in `EXPECTED.md` (`1.8e-2 → 2.8e-4`, order `2.01`).
3. Fit the order the same way (least squares in log–log across the 4
   levels) and compare to the `k=1` order.

**Expected result.** For `p=1`, the relevant seminorm is the 2nd
derivative (`p+1=2`), which scales as `(k pi)^2`; doubling `k` from 1 to
2 predicts roughly a `2^2 = 4x` jump in the `L2(c)` error at each fixed
level (e.g. extrapolating the `k=1` anchor numbers, `1.8e-2 → 2.8e-4`
becomes very roughly `~7e-2 → ~1.1e-3`), while the fitted **order stays
at `≈2`** — the two lines on a log–log plot are parallel, offset by the
intercept, not by slope. One caveat worth checking: at `k=2`, level 3
(8 cells/side) has only ~2 mesh cells per half-wavelength — noticeably
thinner than `k=1`'s ~4 cells per half-wavelength at the same level —
so the coarsest point may sit slightly pre-asymptotic; refitting on
levels `4–6` only should recover the full order-2 slope more cleanly.

**Common wrong conclusion.** "The `k=2` error is bigger than `k=1`'s, so
the discretization is worse / less accurate at `k=2`." No — comparing
*raw* error values across two different manufactured fields without
normalizing is comparing two different problems, not two different
methods: the field itself is harder to represent (larger high-derivative
content), which raises the constant `C`, not the method's order. The
portable, method-characterizing quantity is the **slope**, and that is
what should be reported unchanged; quoting the bigger raw number as
"worse convergence" is exactly the intercept-vs-slope confusion this
exercise is built to catch.
