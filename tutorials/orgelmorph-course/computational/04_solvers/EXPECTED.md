# C4 — expected results (self-check)

`python run.py` marches the real spinodal with each solver, times the 2-D
solve, and echoes the cited 3-D story. **The divergence is reproducible;
timings vary by card.** Seven `[PASS]` lines and `ALL CHECKS: PASS` must
print.

## 1. March divergence — the real correctness test (level 5, dt 0.02, 20 steps)

| energy | solver | final c range | verdict |
|---|---|---|---|
| poly | splu | [−1.03, 1.01] | ok |
| poly | **cuDSS** | **[−552, 542]** | **DIVERGED** (~500× blow-up) |
| fh | splu | [0.06, 0.94] | ok |
| fh | cuDSS | [0.001, 0.999] | ok |

**cuDSS diverges on the polynomial CH saddle** and is fine on
Flory–Huggins. Why: the poly well's curvature f″=3c²−1 goes **negative**
in the spinodal band |c|<1/√3, so the (c,μ) block is **indefinite** there;
cuDSS does **no partial pivoting**, so its solution is wrong and the error
compounds until the field escapes. FH's entropic curvature
f″=A(1/c+1/(1−c)) ≥ 4A stays positive, so cuDSS survives — but you cannot
rely on that. **splu (pivoted) is safe on both.**

## 2. The trap: a one-iterate residual lies

| solver | one-iterate residual |
|---|---|
| splu | ~1e-12 |
| cuDSS | **~5e-13 (tiny!)** |
| blockch | ~4e-10 |

cuDSS's residual at a single captured Newton iterate is tiny — yet its
march diverges. **A small residual ≠ a small error** on an indefinite,
unpivoted system. Measure the marched *solution*, not a one-shot residual.

## 3. Live 2-D timing (context only — cuDSS is disqualified for CH)

cuDSS is *faster* than splu above ~10⁴ dofs (2–6× at 33k dofs), but that
speed is unusable on the CH saddle because cuDSS returns a wrong answer.
Fastest ≠ correct.

## 4. Cited 3-D scaling (dev notes — NOT re-run)

- cuDSS ceilings at ~811k dofs on 48 GB (>16 min factorization); blockch_dev
  is 8.4× cheaper at slab64 and marches where cuDSS ceilings.
- masked cuDSS cuts the fill (17.7 → 3.70 s/call) below the ceiling.
- AMGX: NO (mass-dominated inners). matrix-free reaches 6.39M dofs on one card.

## 5. Decision support (`recommend_solver`)

small → **splu**; large 2-D → **splu / blockch (not cuDSS)**; small 3-D →
masked cuDSS / blockch_dev; large 3-D → **blockch_dev**; extreme →
matrix-free. Every rationale begins with the correctness constraint and
**never picks cuDSS on the raw CH block**.

**What must be true regardless of hardware:**

- **cuDSS DIVERGES on the polynomial CH march** (field escapes [−1,1] by
  orders of magnitude); **splu is safe on poly and FH**.
- **cuDSS survives the FH march** (better-conditioned), but that is
  energy-specific, not a green light.
- **The `--solver auto` default of splu for CH is correct.**
- **The capture uses a supported API** (`capture_system=True`), not a
  `solve_linear` monkeypatch — and the captured residual is shown to *lie*.
