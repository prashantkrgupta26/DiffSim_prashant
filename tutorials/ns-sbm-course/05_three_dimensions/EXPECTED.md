# 05 — expected results (self-check)

The automated gate is **`baseline.yaml`**, checked by the harness:

```bash
python run.py --config configs/sphere.yaml --mode reference --output outputs/sp --overwrite
python gen_figures.py --run-dir outputs/sp     # verdict card + numbers/c5.tex
```

Reference mode (level 4, Re=100, alpha=10, dt=0.05) reproduces:

| quantity | value |
|---|---|
| $n_{\text{free}}$ | 4907 |
| sphere surrogate faces | 64 |
| $D/h$ | 3.84 |
| SBM $d_{\max}$ | $>0$ (genuine 3-D shift) |
| **MONOLITHIC steady $C_d$** | **+0.381** (positive, physical) |

The `--pipeline` leg additionally reports `finite=True`, `BDF2
engaged=True`, and an axisymmetric traction ($|C_{\text{lat}}|/|C_d|$ at
machine-noise level).

## What must be true regardless of hardware

- **The monolithic 3-D SBM-NS drag is stable and physical:** it recovers
  from the startup transient (which passes through negative $C_d$) to a
  positive steady $C_d = +0.381$, reproducing the M1b sphere lock. This is
  the hero result for 3-D drag.
- **The SBM shift is genuinely active in 3-D:** the sphere is not
  grid-aligned, so $d_{\max}>0$ and the surrogate faces (64 at level 4) hug
  the carved octree boundary.
- **The projection composition is de-risked but not (yet) a drag path.**
  With `--pipeline` the composition is finite, BDF2 engages, and the
  traction is axisymmetric — but its *long-time* drag transient is unstable
  at feasible 3-D mesh (a documented R2 item). So in 3-D we validate the
  pipeline invariants and take **drag from the monolithic**. "Finite" is not
  "stable" — that distinction is the chapter's headline caveat.
- **The bar is the monolithic's physical steady state, not the literature
  number.** The confined unit-box domain inflates $C_d$ above the unconfined
  sphere value (~0.6–0.7 at Re=300); both engines see the same confinement.

## Solver path

Host `splu` at level 4 is ~seconds/step. To scale up, build the fixture on a
CUDA device and route the monolithic solve through `solver="cudss"` (GPU
direct) or `solver="blockamgx"` (block-preconditioned FGMRES: Cahouet–Chabard
Schur + AMG-on-$F$) past the host-splu wall (~level 5 / 143k DOF).

If your monolithic $C_d$ comes out negative (still in the startup transient),
increase `max_steps`; if it stays negative or blows up, re-read the
walkthrough.
