# Grading rubric — C5 Two dimensions versus three

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "C5 specifics"
column says what to look for.*

| Component | Weight | C5 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Dof growth stated as $2^{\dim}$/level; sparsity given as the **correct** $(2p+1)^{\dim}\times n_{\text{fields}}$ count (18 in 2-D, 54 in 3-D) explicitly contrasted with the **wrong** "$3^{\dim}-1$" FD-style count (8, 26) and *why* it undercounts; memory reported as a complete accounting (all nine buffers), not just CSR values; does **not** claim 2-D generalizes to 3-D physics. |
| **Numerical verification** | 20% | The int32-vs-int64 (`--idx-bytes 8`) memory delta at $256^3$ computed and reported; whether int64 alone clears the 48 GB card, or the solver workspace is still the wall, stated explicitly. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all four figures (`c5_dofgrowth`, `c5_nnz`, `c5_memory`, `c5_physics`) regenerated from saved data; the five `[PASS]` self-checks (`ALL CHECKS: PASS`) reproduce; morphology itself not claimed bit-reproducible. |
| **Software & CUDA** | 10% | Correct solver named for each estimate (`blockch` vs. `splu`/`cudss`); live captures use the supported `capture_system=True` API (no monkeypatch); sensible level/`n` choices; clean run with no swallowed failures. |
| **Failure diagnosis** | 10% | A cited case correctly identified as overflowing int32 (e.g. `mk32 128x128x48 (M=3,K=2)`, $8{,}028{,}160$ dofs, $2.17\times10^9$ nnz $>2^{31}$) with the mechanism explained (32-bit CSR slot arithmetic) and the fix named (int64 build or block-masked pattern) — **not** a claim that the case was actually run. |
| **Exploration & research bridge** | 10% | One "Explore on your own" question answered with evidence (a sweep, plot, or fitted trend); a paragraph connecting the dof/sparsity/memory growth law to why C4's device assembler, `blockch`, and the matrix-free frontier exist. |
| **Communication** | 5% | Labeled axes/units; the two headline counts ($18$/$54$ vs. $8$/$26$) never conflated; estimated vs. measured numbers kept visibly distinct throughout the report. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming the sparsity is "$3^{\dim}-1$" (or reporting 8/26 nnz-per-dof
  as if it were correct) without identifying it as the wrong FD-style
  count.
- Claiming a large 3-D case (e.g. $256^3$, or any point on the cited
  device-scale ladder) was **run** — with a wall-clock time, a captured
  step time, or an observed OOM — when it was only **estimated** or
  **cited** from the dev notes. `run.py`/`estimate_capacity.py` never
  launch those solves in this chapter.
- Describing the int32 CSR ceiling as a hardware or CUDA limit rather
  than an implementation choice (index width).
- Hand-copied numbers that do not match the submitted `results.json` /
  console output of `run.py` or `estimate_capacity.py`.
- Claiming 2-D and 3-D morphologies "agree" at equal resolution when
  `physics_comparison()`'s own output shows differing interfacial-area
  density and $S(q)$ wavelength.

## Partial-credit guidance

- Correct $(2p+1)^{\dim}\times n_{\text{fields}}$ formula but no
  comparison to the wrong "$3^{\dim}-1$" count → cap Scientific
  correctness at three-quarters; the contrast is the point of the
  chapter.
- Complete memory accounting reported, but only for one of $128^3$/
  $256^3$ → half of Numerical verification.
- int32 overflow correctly identified for one cited case but the
  mechanism (nnz $>2^{31}$) not explained → half of Failure diagnosis.
- A beautiful `c5_physics.png` figure with no numeric interfacial-area
  density / $S(q)$ wavelength quoted → Communication credit only; the
  science credit is in the numbers, not the picture.
- Big-3-D numbers correctly labeled "estimated" throughout but the
  int64-vs-int32 delta arithmetic is off → full Reproducibility/
  Communication credit, partial Numerical verification credit.
